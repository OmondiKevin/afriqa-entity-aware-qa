#!/bin/bash
set -e
echo "=========================================="
echo " Starting Full AfriQA Experiment Run"
echo "=========================================="

echo "[1/4] Preparing Data..."
python scripts/00_download_and_subset.py --config configs/default.yaml
python scripts/01_prepare_qa_data.py --config configs/default.yaml
python scripts/01b_prepare_multitask_data.py --config configs/default.yaml

echo "[2/4] Training Baseline Model..."
python scripts/02_train_baseline_qa.py --config configs/default.yaml

echo "[3/4] Evaluating Baseline Model..."
python scripts/04_eval_predictions.py --config configs/default.yaml --pred_path outputs/predictions/baseline_mt5_test.jsonl

echo "[4/4] Training Multitask Model..."
python scripts/03_train_multitask_qa.py --config configs/default.yaml

echo "[5/5] Evaluating Multitask Model..."
python scripts/04_eval_predictions.py --config configs/default.yaml --pred_path outputs/predictions/multitask_mt5_test.jsonl

echo "=========================================="
echo " Backup Phase..."
echo "=========================================="
if [ -d "/content/drive/MyDrive" ]; then
    echo "Google Drive detected. Backing up outputs directory..."
    mkdir -p /content/drive/MyDrive/afriqa_final_outputs
    cp -r outputs/* /content/drive/MyDrive/afriqa_final_outputs/
    echo "Backup complete!"
else
    echo "Warning: Google Drive not mounted at /content/drive/MyDrive. Archiving locally."
    tar -czf outputs_archive.tar.gz outputs/
    echo "Archived to outputs_archive.tar.gz"
fi

echo "=========================================="
echo " All experiments finished successfully!"
echo "=========================================="
