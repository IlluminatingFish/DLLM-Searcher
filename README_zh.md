  <h1 align="center" style="margin-top: -50px;">🚀 DLLM-Searcher </h1>

  <h5 align="center">若喜欢本项目，请在 GitHub 给我们一个 star ⭐ 以获取最新更新。</h5>


> [!NOTE]
> 本项目包含 dLLM 后训练的代码与数据集：**Agentic SFT** 与 **Agentic VRPO**。
>
> 同时包含从 [SDAR](https://github.com/JetAstra/SDAR) 的 jetengine 改编的 **P-ReAct** 实现。
## ⚡ 一次 P-ReAct 迭代

DLLM-Searcher 在思维区域之前先解码工具调用区域，因此 DLLM-Searcher 在等待工具返回时**始终持续思考**。

  <div align="center">   <img src="visualization/diffusion_generation.gif" width="80%" alt="DLLM-Searcher 扩散生成可视化" /> </div>

## 📋 概述

我们设计了两阶段后训练流程，包括 **Agentic SFT** 与 **Agentic VRPO**，以提升推理与工具调用能力。此外，我们提出了一种新的智能体范式 **P-ReAct**。P-ReAct 引导模型优先解码 **tool_call** 指令，从而使模型在等待工具返回时**持续思考**。  

  <div align="center">  <img src="visualization/main.png" width="80%" alt="DLLM-Searcher 架构" /> </div>


------

  ## 📁 项目结构

  本项目代码结构如下：

  ```text
  .
  ├── Dataroller/              # 📊 数据采集与准备
  │   ├── scripts/     
  │   │   └── test.sh          # 数据采集脚本
  │   ├── prompt.py                              
  │   ├── react_agent.py       # ReAct 智能体实现
  │   ├── run.sh               
  │   ├── run_multi_react.py   # 多轮 ReAct 执行
  │   └── tool_search.py       # 搜索工具定义
  │
  ├── dLLM_trainer/            # 🔧 训练流程
  │   ├── SFT/                 
  │   │   └── dLLM-RL/           
  │   │       ├── train/       # SFT 训练代码
  │   │       ├── data/        # SFT 数据集
  │   │       └── sdar_sft.sh  # SFT 训练脚本
  │   │
  │   └── VRPO/                
  │       ├── my_train/        # VRPO 训练代码
  │       │   └── jetengine/   
  │       └── recipes/         # VRPO 训练配置
  │
  ├── my_eval/                 # 🚀 评估脚本
  │
  └── visualization/           # 🎨 视觉素材
      ├── main.png
      └── diffusion_generation.gif
  ```

------

  ## 🛠️ 安装

  建议为数据采集、Agentic SFT 训练、Agentic VRPO 训练和推理**使用独立环境**，避免依赖冲突。

  ### 📦 数据采集环境

  ```bash
  conda create -n dllmeval python=3.10
  conda activate dllmeval
  pip install torch==2.8
  wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl
  pip install flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl
  pip install sglang[all]
  pip install qwen-agent[gui,rag,code_interpreter,mcp]
  ```

  ### 🎯 SFT 训练环境

  ```bash
  conda create --name dllm-rl python=3.10
  conda activate dllm-rl
  pip install torch==2.6.0
  pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
  flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  cd dLLM_trainer/SFT/dLLM-RL
  pip install -r requirements.txt
  ```

  ### 🎓 VRPO 训练环境

  ```bash
  conda create -n espo python=3.11 -y
  conda activate espo
  pip install torch==2.6.0
  pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
  flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  cd dLLM_trainer/VRPO
  pip install -e ".[code]"
  pip install wandb==0.15.12 protobuf==3.20.3
  ```

### 🚀 推理环境

  ```bash
  # 与 VRPO 环境相同
  ```

------

  ## 📊 数据准备

  使用 `Dataroller` 模块采集与处理训练数据，命令如下：

  ```bash
  cd Dataroller
  bash scripts/test.sh
  ```

  我们发布的 SFT 数据集位于 `dLLM_trainer/SFT/data/data.json`，VRPO 数据集位于 `dLLM_trainer/VRPO/data/train.jsonl`。

  **数据集结构：**

  - 包含推理轨迹、工具调用与搜索结果

------

## 🎯 Agentic SFT 训练

  ### 配置

  开始 SFT 训练前，需在 `sdar.yaml` 和 `sft_sdar.py` 中配置路径：

  **sdar.yaml：**

  ```yaml
  model:
      pretrained_model: "your_path"  # 预训练模型的绝对路径
      optimized_name: "optimized"    # 优化后模型输出名，保存于 sft_sdar/ckpt
  ```

  **sft_sdar.py：**

  ```python
  with open("../data/" + config.dataset.optimization_data + ".json", 'r') as f:
      dataset_load = json.load(f)
  ```

  ### 训练

  配置完成后，按以下方式启动 SFT 训练：

  ```bash
  cd dLLM_trainer/SFT/dLLM-RL
  bash sdar_sft.sh
  ```

------

## 🎓 Agentic VRPO 训练

  ### 配置

  根据你的环境在 `dpo.yaml` 中更新路径配置：

  ```yaml
  # 示例配置结构
  model:
      path: "your_sft_model_path"  # SFT 训练后模型路径
  # 其他 VRPO 相关配置
  ```

  ### 训练

  按以下方式启动 VRPO 训练：

  ```bash
  cd dLLM_trainer/VRPO
  bash run_dpo.sh
  ```

------

## 🚀 使用 P-ReAct 推理

  仅需 11 行代码即可实现完整的 token 预填充与置信度偏置。

```python
# dLLM_trainer/VRPO/my_train/jetengine/engine/scheduler.py
elif 'toolcall_pre_rl' in seq.remasking_strategy:
    if seq.current_denoising_step == 0:
        seq_x0[your_tool_end] = 151658 # </tool_call>
        transfer_index[your_tool_end] = True
        seq_x0[your_tool_start] = 151657 # <tool_call>
        transfer_index[your_tool_start] = True
    else:
        confidence = torch.where(mask_index, seq_x0_p, -np.inf)
        confidence[your_tool_start:your_tool_end + 1] = confidence[your_tool_start:your_tool_end + 1] + 0.5
        _, top_indices = torch.topk(confidence, num_to_transfer)
        transfer_index[top_indices] = True
```

### 特性

  - ⚡ 并行推理与动作执行
  - 🔄 异步工具调用
  - 📈 降低端到端延迟
  - 🎯 提升搜索智能体效率

------

## 📊 评估

  使用训练好的模型运行数据脚本：

  ```bash
  cd inference/my_eval
  bash run_test.sh
  ```

 我们使用的 LLM 作为裁判的提示词为：

```python
    prompt = '''给定一个问题和其标准答案，验证预测答案是否正确。
    若预测在含义与关键信息上与标准答案完全一致，则判定为正确。
    若正确请仅回复 True，否则仅回复 False。
    Question: {question}
    Golden Answer: {reference}
    Predicted Answer: {prediction}
    '''
```

  评估脚本位于 `inference/my_eval/` 目录。

  ```bash
  cd inference/my_eval
  python cal_acc.py
  ```

------

  ## 🙏 致谢

  我们由衷感谢以下开源项目的作者：

  - **训练框架**：[TraceRL](https://github.com/Gen-Verse/dLLM-RL/tree/main)、[ESPO](https://github.com/ML-GSAI/ESPO)
  - **评估与部署**：[WebSailor](https://github.com/abusallam/Websailor)、[R1Searcher](https://github.com/RUCAIBox/R1-Searcher)
  - **基础模型**：[LLaDA](https://github.com/ML-GSAI/LLaDA)、[Dream7B](https://github.com/DreamLM/Dream)、[SDAR](https://github.com/JetAstra/SDAR)

------
