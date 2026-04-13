# Unlearning README

## Running the code

```
python scripts/download_datasets.py --config configs/highlr.yaml --dataset all
python scripts/prepare_data.py --config configs/highlr.yaml

python scripts/eval_base.py --config configs/highlr.yaml --output-dir results/base

python scripts/train_unlearn.py --config configs/highlr.yaml --output-dir results/run2/unlearn_hlr

python scripts/infer_unlearned.py --config configs/highlr.yaml --adapter-path results/run2/unlearn_hlr/checkpoint-300 --interactive

python scripts/run_tutoring.py --config configs/highlr.yaml --student-checkpoint results/run2/unlearn_hlr/final_adapter --output-dir results/run2/tutoring_hlr_8B --teacher-base-url http://0.0.0.0:8000/v1 --teacher-model /data/xmy/models/Qwen3-8B

python scripts/train_recover.py --config configs/highlr.yaml --student-checkpoint results/run2/unlearn_hlr/final_adapter --tutoring-data results/run2/tutoring_hlr_14B/teach_ft_data.jsonl --output-dir results/run2/recover_hlr_14B

python scripts/eval_all.py --config configs/highlr.yaml --run-dir results/run2
```

## VLLM Host

```
vllm serve /data/xmy/models/Qwen3-14B

vllm serve /data/xmy/models/Qwen3.5-9B --port 8000 --max-model-len 98304 --reasoning-parser qwen3 --language-model-only --enable-prefix-caching
```