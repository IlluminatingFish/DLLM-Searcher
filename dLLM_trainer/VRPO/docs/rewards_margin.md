# ELBO VRPO 训练中的 rewards margin 与 DPO 公式

## 1. 变量定义

| 符号 | 含义 |
|------|------|
| π_θ | 当前策略（要训练的模型） |
| π_ref | 参考模型（通常为 frozen） |
| chosen_logps | log π_θ(y_chosen\|x)，策略对 chosen 的 log 概率 |
| rejected_logps | log π_θ(y_rejected\|x)，策略对 rejected 的 log 概率 |
| ref_chosen_logps | log π_ref(y_chosen\|x) |
| ref_rejected_logps | log π_ref(y_rejected\|x) |
| β | 温度参数（config 中 `beta`，如 0.1） |

---

## 2. 核心公式（TRL sigmoid loss）

### Log ratios
```
chosen_logratios  = chosen_logps  − ref_chosen_logps
rejected_logratios = rejected_logps − ref_rejected_logps
```

### Scores（默认 reverse_kl）
```
chosen_scores   = chosen_logratios
rejected_scores = rejected_logratios
delta_score     = chosen_scores − rejected_scores
```

### Rewards（logging 用）
```
chosen_rewards   = β · chosen_logratios
rejected_rewards = β · rejected_logratios
margin           = chosen_rewards − rejected_rewards = β · delta_score
```

### Loss（sigmoid 类型）
```
L = −log σ(β · delta_score) = −log σ(margin)
```

其中 σ(x) = 1/(1+e^{-x}) 为 sigmoid。

---

## 3. 指标含义

| 指标 | 公式 |
|------|------|
| rewards/chosen | β · (chosen_logps − ref_chosen_logps) |
| rewards/rejected | β · (rejected_logps − ref_rejected_logps) |
| rewards/margins | chosen_rewards − rejected_rewards |
| rewards/accuracies | 1{chosen_rewards > rejected_rewards} 的均值 |

---

## 4. margin 与 loss 的关系

- margin ≈ 0 → σ(0)=0.5 → L ≈ 0.693
- margin 越大 → σ(margin)→1 → L→0
- margin 越小（越负）→ σ(margin)→0 → L 变大
- 因此训练目标：**增大 margin**

---

## 5. 代码位置

`my_train/my_dpo_trainer.py` 调用 TRL 的 `dpo_loss`，对应实现见 `trl/trainer/dpo_trainer.py`（约 1170–1208 行）。
