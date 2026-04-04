## Kernel RL

MODEL_FORMAT=qwen3.5 USE_MTP=true bash rollout/sglang_server/run_turbo.sh

python rollout/single_turn_rollout.py --data-path data/eval/kernelbench/kernelbench.jsonl --save-path output/qwen3.5-turbo-test --model qwen3.5-35A3
