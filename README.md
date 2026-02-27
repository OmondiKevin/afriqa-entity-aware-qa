# AfriQA Entity-Aware Multilingual QA

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An open-source research project exploring **direct, translation-free Question Answering** for low-resource African languages (Swahili, Hausa, Yoruba). 

## Overview
Traditional Cross-lingual Question Answering (XOR QA) processes rely heavily on translation pipelines (e.g., Native Language -> English FLAN-T5 -> Native Language). While effective for high-resource languages, this aggressively degrades performance on African languages due to compounding translation errors.

This project implements a **100% direct text-to-text generation** approach using `google/mt5-base`. To overcome mT5's lack of structural entity awareness in these languages, we utilize **Multitask Learning**, interleaving the AfriQA dataset with the **MasakhaNER 2.0** dataset. By jointly training QA and Named Entity Recognition (NER), the model learns robust cross-lingual representations of entity boundaries, bypassing the need for translation.

Additionally, this repository includes a framework for performing **Low-Rank Adaptation (LoRA)** fine-tuning, allowing researchers to compare the efficiency and accuracy trade-offs against full parameter fine-tuning.

---

## 🚀 Features
* **Translation-Free Architecture:** Answer questions directly in Swahili, Hausa, and Yoruba.
* **Multitask Supervision:** Jointly trains on QA (AfriQA) and NER (MasakhaNER) to improve entity boundary detection.
* **LoRA Integration:** Support for highly efficient parameter-efficient fine-tuning via HuggingFace `peft`.
* **Advanced Evaluation:** Computes Exact Match (EM), Token F1, and Semantic Answer Similarity (SAS+) using `LaBSE` embeddings.
* **Colab Ready:** Provided notebooks and bash scripts for 1-click execution on cloud GPUs.

---

## 🛠️ Installation

Requires Python 3.11+ and a CUDA/MPS capable device.

```bash
git clone https://github.com/OmondiKevin/afriqa-entity-aware-qa.git
cd afriqa-entity-aware-qa

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install sentence-transformers # Optional, required for Semantic Similarity eval
```

*(Note: `peft` is included in the project dependencies to support the LoRA training branches).*

---

## 📊 Running Experiments

The project workflow is partitioned into modular scripts located in `scripts/`. You can configure hyperparameters (epochs, batch size, LoRA rank) in `configs/default.yaml`.

### 1. Data Preparation
Downloads and prepares the subsets of the `masakhane/afriqa-gold-passages` and `masakhane/masakhaner2` datasets.
```bash
python scripts/00_download_and_subset.py
python scripts/01_prepare_qa_data.py
python scripts/01b_prepare_multitask_data.py
```

### 2. Standard Multitask Fine-Tuning
Trains `mt5-base` using standard Adafactor/AdamW optimization.
```bash
python scripts/03_train_multitask_qa.py --config configs/default.yaml
python scripts/04_eval_predictions.py --pred_path outputs/predictions/multitask_mt5_test.jsonl
```

### 3. LoRA Comparative Analysis
To run the automated side-by-side comparison between **Full Fine-tuning** and **LoRA**, we provide a dedicated runner script. This will temporarily modify your YAML config, run both methods sequentially, and output evaluation CSVs for easy graphing.
```bash
chmod +x run_compare_lora.sh
./run_compare_lora.sh
```

### 4. Google Colab
If you are running in the cloud, simply load the `run_experiments_colab.ipynb` notebook and execute the cells. The notebook will clone this repo, install dependencies, and run the `run_compare_lora.sh` experiment pipeline.

---

## 📈 Evaluation Metrics

Predictions are scored on three criteria:
1. **Exact Match (EM):** Perfect programmatic token matching against the gold label.
2. **Token F1:** Partial overlap measurement.
3. **Semantic Answer Similarity (SAS+):** Uses Language-agnostic BERT Sentence Embeddings (`LaBSE`) to give models credit for correct answers phrased differently from the gold label (cosine similarity `> 0.75`).

Results are automatically saved to `outputs/metrics/`.

---

## 📝 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
