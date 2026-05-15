# NaN 排查清单

按数据流顺序依次检查各 NaN 检测点。训练时若出现 NaN，日志中会打印 `[NaN] CHECK X` 或 `[NaN] ...`，据此定位首次出现位置。

## 可能根因（已排除 mask 转换问题后）

1. **bf16 数值不稳定**：长序列 + bf16 可能累积误差或溢出  
2. **gradient_checkpointing**：重算时数值可能略有差异  
3. **attention 全 -inf 行**：SDPA 对整行 -inf 会 NaN（已加防御性修复）  
4. **超长序列**：max_prompt 648 + max_completion 2048 ≈ 2700 tokens  
5. **flash_rms_norm / 其他 kernel**：第三方 kernel 的数值行为

## 实测结论（2026-02-22）

| 配置 | 结果 |
|------|------|
| 长序列 648+2048 + gradient_checkpointing | **NaN**（logits_nan=True） |
| 短序列 256+512 + gradient_checkpointing | **无 NaN** |
| 短序列 256+512 + 无 gradient_checkpointing | **无 NaN** |
| debug_nan_forward 单样本 | **无 NaN** |

**根因**：**长序列**（648+2048）在 bf16 + block diffusion 下导致 model logits 出现 NaN，与 gradient_checkpointing 无关。  
**建议**：先试 `--max_prompt_length 256 --max_completion_length 512` 或 1024，确认训练正常后再逐步加长。

## 检测点顺序（数据流）

| 顺序 | 位置 | 含义 | 若触发说明 |
|------|------|------|------------|
| **1** | `_get_elbo_blk_with_trainable` 中 model forward 后 | model logits 含 NaN/inf | **模型 forward 输出异常**，根因在模型或输入 |
| **2** | `mdm_ce_loss_with_trainable` 内 | cross_entropy 或 loss[i] 为 NaN | logits 或 targets 有问题，或 valid_positions 异常 |
| **3** | `_get_elbo_blk_with_trainable` 每 block 后 | local_loss 含 NaN/inf | 某 block 的 CE 计算异常 |
| **4** | `_get_elbo_blk_with_trainable` 汇总后 | final loss (ELBO) 含 NaN/inf | 多 block 汇总后异常 |
| **5** | `concatenated_forward` | chosen_elbo / rejected_elbo 为 NaN | 单样本 ELBO 异常 |
| **6** | `get_batch_loss_metrics` policy 后 | chosen_logps / rejected_logps 含 NaN | policy 输出异常 |
| **7** | `get_batch_loss_metrics` ref 后 | ref_chosen_logps / ref_rejected_logps 含 NaN | ref 模型输出异常 |
| **8** | `get_batch_loss_metrics` DPO 后 | losses / chosen_rewards / rejected_rewards 含 NaN | DPO loss 公式或输入异常 |

## 排查步骤

### 1. 运行训练并收集日志

```bash
cd /research/cbim/vast/mz751/Projects/DLLM-Searcher/dLLM_trainer/VRPO
bash recipes/run_dpo.sh 2>&1 | tee dpo_nan_debug.log
```

或使用 `max_steps 3` 快速复现：

```bash
# 在 run_dpo.sh 末尾加 --max_steps 3，或直接：
accelerate launch ... --max_steps 3 2>&1 | tee dpo_nan_debug.log
```

### 2. 搜索首次 NaN 出现位置

```bash
grep -n "\[NaN\]" dpo_nan_debug.log | head -20
```

- 若**最先**出现 `[NaN] _get_elbo_blk: model logits have nan/inf` → 根因在**模型 forward**（bf16 数值、attention mask、输入长度等）
- 若最先出现 `[NaN] mdm_ce_loss` → 检查 logits/targets/valid_positions
- 若最先出现 `[NaN] concatenated_forward` → 检查单样本 ELBO 计算
- 若最先出现 `[NaN] get_batch_loss_metrics: after DPO loss` → 检查 DPO 公式输入（policy/ref logps）

### 3. 数据层预检（无需 GPU）

```bash
cd VRPO
python debug_data_pipeline.py
```

检查：mask 长度、全 False、block 级 trainable=0。

### 4. 单次 forward 诊断（需 GPU + flash_attn）

```bash
cd VRPO
python debug_nan_forward.py
```

按 CHECK 1→4 顺序打印，定位 NaN 首次出现位置。若环境无 flash_attn，此脚本会报错，可跳过。

### 5. 根因排查脚本

```bash
cd VRPO
python debug_nan_causes.py
```

检查 attention mask 是否有全 False 行、扫描多组参数。

### 6. 禁用 bf16 排查

```bash
DEBUG_DPO_FP32=1 bash recipes/run_dpo.sh
```

若 fp32 下 NaN 消失，则根因与 bf16 数值有关。

## 常见根因与对策

| 根因 | 对策 |
|------|------|
| 模型 logits NaN（CHECK 1） | 1) 试 fp32/float32 2) 检查 attention mask 形状 3) 缩短 max_completion_length 4) 关闭 gradient_checkpointing 试一次 |
| bf16 数值不稳定 | 试 `mixed_precision="fp16"` 或 `no` |
| attention 全 -inf 行 → softmax NaN | 已在 modeling_sdar.py 加防御性修复（全 -inf 行替换为 0） |
| 某 batch 样本异常 | 用 `debug_data_pipeline.py` 检查该 step 对应样本 |
| valid_positions 全 0 | mask 与 token 错位，检查 `get_trainable_mask` 与 `tokenize_row` 一致性 |
| DPO 输入含 NaN | 若 policy/ref 正常而 DPO 后 NaN，检查 beta、logps 数值范围 |
| gradient_checkpointing | 试 `--gradient_checkpointing false` |

## 已添加的 Debug 日志位置

- `my_dpo_trainer.py` L178-186: mdm_ce_loss
- `my_dpo_trainer.py` L374-379: model logits
- `my_dpo_trainer.py` L330-334: local_loss
- `my_dpo_trainer.py` L336-338: final loss
- `my_dpo_trainer.py` L416-422: concatenated_forward elbo
- `my_dpo_trainer.py` L586-594: policy output
- `my_dpo_trainer.py` L601-607: ref output
- `my_dpo_trainer.py` L615-622: after DPO loss
