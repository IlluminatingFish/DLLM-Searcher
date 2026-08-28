# 交接文档：跑通并调试两组COCONUT对比实验

这份文档是给接手这个任务的 Claude 看的。请先完整读一遍再动手，里面有很多之前踩过的坑，重复踩坑会浪费很多时间。

**现在有两个训练代码需要跑，是一组对比实验：**
1. `coconut_latent_only_trainer.py` —— 只训练自指代latent生成+辅助重建loss，不碰CE
2. `coconut_latent_ce_trainer.py` —— 同一套latent生成机制，**加上**CE/mask离散生成任务一起训练

两者的"latent怎么生成"是完全一样的机制（自指代、因果、蒙眼），唯一区别是要不要加CE任务。目的是对比：加不加CE任务，对latent学出来的东西有没有影响、效果好不好。建议两个分别在hermes和dionysos上各跑一个（两台机器现在都空闲）。

## 项目背景（一句话版本）

这是一个基于 LLaDA（一个8B参数、双向attention的masked diffusion语言模型）做 COCONUT（连续隐空间推理）+ SIM-CoT（辅助监督）思路的训练实验。用户在跟另一个Claude会话里，深入分析了现有训练代码（`my_train/coconut_sft_trainer.py`）跟 COCONUT/SIM-CoT 两篇论文原始设计的差异，发现现有代码是"作弊"版本（训练时直接读一份GPT-4o提前写好的计划摘要来生成latent，而不是让模型自己蒙眼生成），这跟推理时（没有这份摘要）严重不一致。

## 这次要跑的新代码是什么

`my_train/coconut_latent_only_trainer.py`（已经写好，刚刚做过冒烟测试，前向+反向传播都验证过没问题）。核心设计，严格照着两篇论文的机制来：

- **不训练CE/mask离散生成那部分**（那是`coconut_sft_trainer.py`干的事，这次不碰）
- **只训练"自指代生成latent + 辅助重建loss"这一件事**：
  1. 对每条样本，`K = 计划句子数`（一般2~4句）
  2. `z_k` = 因果注意力下、看着"context + 前面已经生成的z_0...z_{k-1}"、最后一个位置的hidden state——**z_k生成时绝对看不到对应句子的显式文字**（论文原文机制，蒙眼）
  3. aux loss：把z_k当一个"前缀向量"，用**同一个模型自己**当解码器，因果+teacher forcing地去逐token重建这一步该有的句子文字，标准shift-by-one语言模型loss（SIM-CoT论文Eq.6）——**z_k本身不直接受梯度，只有重建出来的token的loss会反传回去**
  4. **注意：LLaDA默认是双向注意力**（没传attention_bias时`is_causal=False`），所以第2、3步都**必须显式传因果(causal)的attention_bias**（`build_causal_bias`函数），这是最容易漏掉的坑，之前有一次修改就是因为漏了这个。

细节都写在文件顶部的docstring里，遇到不确定的地方先看那段注释。

## 已经做完、验证过的事

1. **两份代码都写好了**：
   - `my_train/coconut_latent_only_trainer.py`（复制自`coconut_sft_trainer.py`，删掉了CE/block diffusion那部分，加了自指代生成+causal重建逻辑）
   - `my_train/coconut_latent_ce_trainer.py`（在latent-only基础上，把`coconut_sft_trainer.py`里的block_forward_process/block_mdm_ce_loss/build_coconut_attention_bias搬回来，加了CE任务；latent注入方式类似原来的`H_plan_n`注入，只是N从固定64变成动态的K）
2. **两份都做过冒烟测试**（单卡，不经过Trainer/DeepSpeed），前向+反向传播都跑通：
   - latent-only: loss约10左右（接近词表随机猜测的合理起点）
   - latent-ce: L_aux约10.5，L_ce约0.4（这个CE数值偏低，样本量小(n_masked=3)，可能是运气，也可能有还没发现的问题，值得在正式训练开始后多看几个例子确认是不是普遍偏低）
   - 两份都确认`model.model.transformer.wte.weight.grad`非零，梯度真的流回共享权重
3. **两份都配了监督/自动重启脚本**（在VRPO目录下）：`coconut_latent_only_supervisor.sh`、`coconut_latent_ce_supervisor.sh`。逻辑跟`coconut_supervisor.sh`（`coconut_sft_trainer.py`那次实验用的）完全一样：自动清理不完整的checkpoint（没有`model.safetensors.index.json`就判定为残档并删除）、自动找最新有效checkpoint续训、训练进程挂了就自动重启，直到跑满`TARGET_STEPS`（脚本里写的1000）。

## 待办：接下来要做什么

1. **分别在两台机器上启动两个训练**（用监督脚本，别直接裸跑`accelerate launch`，这两台环境都不稳定，之前吃过好几次亏，见下面"环境教训"部分）：

   ```bash
   # latent-only，建议放hermes
   ssh hermes
   cd /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO
   nohup setsid bash coconut_latent_only_supervisor.sh > /tmp/coconut_latent_only_nohup.log 2>&1 < /dev/null &
   disown
   ```

   ```bash
   # latent+CE合并版，建议放dionysos
   ssh dionysos   # 如果本来就在dionysos上，直接跳过ssh这行
   cd /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO
   nohup setsid bash coconut_latent_ce_supervisor.sh > /tmp/coconut_latent_ce_nohup.log 2>&1 < /dev/null &
   disown
   ```
   日志分别在 `/tmp/coconut_latent_only_supervisor.log` 和 `/tmp/coconut_latent_ce_supervisor.log`（监督脚本自己的日志，含每次重启记录），accelerate/训练本身的stdout也会追加进同一个文件。

2. **盯着两边跑起来，确认没有新的bug**。这套因果attention的机制是全新写的、只做过单卡冒烟测试，没有跑过真正的多卡DeepSpeed训练，第一次跑起来要重点看：
   - 有没有shape mismatch之类的报错
   - loss数值是不是在合理范围，有没有变成NaN或者爆炸到很离谱的数字
   - **latent-ce那边尤其要多看几步的`loss/ce`，确认冒烟测试里那个偏低的0.4是不是普遍现象**——如果持续偏低，可能是attention bias或者mask_pos哪里有泄漏，需要重新检查`build_coconut_attention_bias`
   - 显存有没有爆（8卡ZeRO-2 + gradient checkpointing应该够用，但这个自指代生成是K次串行前向，比之前的训练更吃显存/时间，如果OOM可能需要调小`max_ctx_len`或者确认gradient_checkpointing真的生效了）
   - checkpoint存档是不是真的存完整（存档目录应该有：4个safetensors分片、`model.safetensors.index.json`、tokenizer相关文件、`trainer_state.json`，一共大概14个文件；如果只有几个文件说明存到一半被打断了）

3. **两边都跑起来之后，对比两组的`loss/aux`（latent-only用的key是`loss/latent_recon`，latent-ce用的是`loss/aux`，注意名字不一样）和整体表现，看加不加CE任务对latent学习有没有影响。**

## 环境教训（一定要看，不然会重复踩坑）

- **hermes/dionysos 这两台共享服务器，之前发生过好几次"整个会话/tmux/进程组被莫名杀掉"的情况**，原因不明确（不是OOM，dmesg也查不到，可能是这台机器某种会话清理机制）。**不要用普通的`ssh hermes "command &"`或者裸的tmux指望它能扛住长时间训练**，必须用`nohup setsid ... & disown`彻底脱离会话（监督脚本已经是这么设计的，直接用现成的脚本，不要自己简化这一步）。
- **训练脚本resume时有两个必须打的补丁**（这个新文件里已经加上了，如果要抄一份改新代码记得别漏掉）：
  1. `deepspeed_load_checkpoint`要打成no-op（因为我们自己的`_save_checkpoint`存的是HF格式，不是DeepSpeed原生格式，HF Trainer默认会尝试加载DeepSpeed原生checkpoint，会报错）
  2. `_load_optimizer_and_scheduler`要打成手动"快进" lr_scheduler 到正确的step数（因为HF Trainer resume时"跳过已训练步数"那个逻辑只跳过数据加载，不会调用`lr_scheduler.step()`，如果不手动快进，每次resume学习率都会从头热身，等于把已经训练好的模型用一个突然变大的学习率去"重创"——这个bug之前真实发生过，直接导致一次训练的loss从30多飙到50多，教训惨痛）
- **`_save_checkpoint`里那个后台线程存checkpoint的写法，有个闭包作用域的坑**：不要在后台线程函数体内写`del cpu_state`，Python会把`cpu_state`当成整个函数的局部变量，导致函数一开始"读"`cpu_state`的地方就报`UnboundLocalError`。这个新文件里已经改好了（没有`del`，靠函数返回自动释放引用），如果要改动这部分代码注意别加回去。
- **训练收尾时，最后一次checkpoint保存是在后台daemon线程里写的，主进程如果不等它写完就退出，会杀掉这个线程，导致最后（往往也是最重要）的那个checkpoint是写了一半的残档**。这个新文件的`main()`最后已经加了`_save_thread.join()`来等它写完，别删掉这段。
- 磁盘配额：`/research/cbim/vast/mz751` 这块盘之前查到用了95%左右（1464G/1536G），比较紧张，如果遇到写文件失败，先查一下这个（`quota -s`）。

## 关键文件路径速查

- latent-only trainer代码：`/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/my_train/coconut_latent_only_trainer.py`
  - 监督脚本：`.../VRPO/coconut_latent_only_supervisor.sh`
  - 输出目录：`/common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/coconut_latent_only`
- latent+CE合并trainer代码：`/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO/my_train/coconut_latent_ce_trainer.py`
  - 监督脚本：`.../VRPO/coconut_latent_ce_supervisor.sh`
  - 输出目录：`/common/users/mz751/Projects/dLLM_trainer/checkpoints/SFT/coconut_latent_ce`
- 数据集：`/research/cbim/vast/mz751/Projects/DLLM-Searcher/Dataroller/data/sft_with_plans.jsonl`（4089条样本）
- 基础模型：`/research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/llada_model`
- 对照组（最早那份"作弊版"COCONUT训练代码，仅供参照结构用，读GPT-4o写好的plan文本而不是自指代生成）：`my_train/coconut_sft_trainer.py`
- accelerate/DeepSpeed配置：`recipes/accelerate_configs/zero2.yaml`

## 论文参考（如果需要重新核对机制细节）

- COCONUT: arXiv:2412.06769，官方代码 `github.com/facebookresearch/coconut`（`coconut.py`里的`forward()`是最权威的实现参考）
- SIM-CoT: arXiv:2509.20317（Eq.6是辅助重建loss的公式来源）
