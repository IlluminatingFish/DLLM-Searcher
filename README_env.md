# 1

dllmsearch Dataroller环境

   bash scripts/start_sglang.sh
   bash Dataroller/scripts/test.sh

   已完成 example 测试，正式复现需要更大数据量：
准备数据：如 gaia.jsonl，放在 Dataroller/data/，格式为 {"question": "...", "answer": "..."}。
修改 Dataroller/scripts/test.sh：DATASETS=("gaia")。
启动 SGLang 并跑 Dataroller：
   bash scripts/start_sglang.sh   bash Dataroller/scripts/test.sh
输出路径：Dataroller/base/<model>_sglang/gaia/iter1.jsonl。
说明：官方在 dLLM_trainer/SFT/data/data.json 和 dLLM_trainer/VRPO/data/train.jsonl 提供了示例数据，可直接用；自己做采集则需把 Dataroller 输出转换成这两种格式。

# 2

dllm-rl Agentic SFT 训练环境

   cd dLLM_trainer/SFT/dLLM-RL
   bash sdar_sft.sh

   配置 configs/sft_sdar.yaml：
model.pretrained_model：预训练基础模型路径（如 Llama-3.1-8B）。
model.optimized_name：输出名，checkpoint 会保存到 sft_sdar/ckpt/。
dataset.optimization_data：数据集名，对应 ../data/<name>.json，例如 "sft_clean_1" 会加载 ../data/sft_clean_1.json。若用官方数据，可改为 "data"（对应 data.json）。
确认 SFT 数据格式：每条约含 prompt、response，可选 step_map（trace）。
配置并运行分布式训练（需要 8 GPU）：
   cd dLLM_trainer/SFT/dLLM-RL   bash sdar_sft.sh
若 GPU 数量不同，需调整 accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml。
SFT 输出：如 sft_sdar/ckpt/optimized/。

# 3

espo Agentic VRPO 训练环境

   cd dLLM_trainer/VRPO
   bash recipes/run_dpo.sh

   修改 recipes/run_dpo.sh：
model_name_or_path：上一步的 SFT 模型，如 sft_sdar/ckpt/optimized。
dataset_path：VRPO 训练数据，如 data/train.jsonl（每行为 prompt/chosen/rejected 格式）。
修改 recipes/dpo.yaml（如需要）：
model_name_or_path、dataset_path 与 run_dpo.sh 保持一致。
运行 VRPO（需 8 GPU）：
   cd dLLM_trainer/VRPO   bash recipes/run_dpo.sh
VRPO 输出：如 output/dpo_*/checkpoints/。

# 4

Step 4：P-ReAct 评估
使用训练后的 P-ReAct 模型做 rollout：
   cd dLLM_trainer/VRPO
   # 修改 my_eval/run_test.sh 中的：MODEL_PATH、BASE_DATA_DIR、BASE_OUTPUT_DIR、DATASETS
   # 搜索工具需配置环境变量（否则 search 会失败返回 None）：
   #   export SEARCH_API_URL="https://你的搜索API地址"
   #   export GOOGLE_SEARCH_KEY="你的API Key"
   # 四卡运行：GPU_NUM=4 PET_MASTER_PORT=29502 bash ../../my_eval/run_test.sh
   # 快速测试 10 个样本：MAX_SAMPLES=10 bash ../../my_eval/run_test.sh
   bash ../../my_eval/run_test.sh
运行 LLM Judge 算准确率：
   cd my_eval   # 修改 cal_acc.py 中的 data_path 指向 rollout 输出   python cal_acc.py
并确保 config.json 中配置好 LLM Judge 的 API（judge_api_key / judge_api_base / judge_model）。

   cd my_eval
   # 修改 cal_acc.py 中的 data_path 指向 rollout 输出
   python cal_acc.py

   cd my_eval
   # 修改 cal_acc.py 中的 data_path 指向 rollout 输出
   python cal_acc.py