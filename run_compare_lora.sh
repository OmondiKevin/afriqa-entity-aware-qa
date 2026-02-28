#!/bin/bash
set -e
echo "=========================================="
echo " Starting Full vs LoRA Multitask Experiments"
echo "=========================================="

echo "[1/4] Preparing Data (if needed)..."
python scripts/00_download_and_subset.py --config configs/default.yaml
python scripts/01_prepare_qa_data.py --config configs/default.yaml
python scripts/01b_prepare_multitask_data.py --config configs/default.yaml

echo "[2/4] Training & Evaluating Full Fine-Tuning Multitask..."
# Force use_lora to false via sed or passing a temporary config is best,
# but we'll assume default.yaml is set to use_lora: false first.
sed -i.bak 's/use_lora: true/use_lora: false/g' configs/default.yaml
python scripts/03_train_multitask_qa.py --config configs/default.yaml
python scripts/04_eval_predictions.py --config configs/default.yaml --pred_path outputs/predictions/multitask_mt5_test.jsonl

echo "Sleeping for 10 seconds to ensure GPU memory is flushed..."
sleep 10
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

echo "[3/4] Training & Evaluating LoRA Multitask..."
# Temporarily enable LoRA
sed -i.bak 's/use_lora: false/use_lora: true/g' configs/default.yaml
python scripts/03_train_multitask_qa.py --config configs/default.yaml
# We know the script auto-changes the output path to end in _lora_test.jsonl
python scripts/04_eval_predictions.py --config configs/default.yaml --pred_path outputs/predictions/multitask_mt5_lora_test.jsonl

# Revert config
mv configs/default.yaml.bak configs/default.yaml

echo "=========================================="
echo " Backup Phase..."
echo "=========================================="
if [ -d "/content/drive/MyDrive" ]; then
    echo "Google Drive detected. Backing up outputs directory..."
    mkdir -p "/content/drive/MyDrive/afriqa_final_outputs_comparison"
    cp -r outputs/* "/content/drive/MyDrive/afriqa_final_outputs_comparison/"
    echo "Backup complete!"
else
    echo "Warning: Google Drive not mounted at /content/drive/MyDrive. Archiving locally."
    tar -czf outputs_comparison_archive.tar.gz outputs/
    echo "Archived to outputs_comparison_archive.tar.gz"
fi

echo "=========================================="
echo " Experiments finished successfully!"
echo " Check outputs/metrics/ for comparison csv files."
echo "=========================================="
