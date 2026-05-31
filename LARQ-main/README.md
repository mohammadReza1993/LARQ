# LARQ: Lightweight Adapter-based Recovery from Quantization in Large Language Models
Lightweight, architecture-agnostic recovery of quantization-induced accuracy loss in Large Language Models (LLMs)

## Overview
Post-training quantization (PTQ) is a widely used technique to reduce the memory footprint and inference cost of large language models. However, aggressive quantization (e.g., 4-bit) often leads to significant accuracy degradation.

**LARQ (Lightweight Adapter-based Recovery from Quantization)** addresses this problem by:

- Adding **small, low-rank adapters** to a frozen quantized model  
- Using **knowledge distillation + feature matching** for training  
- Requiring **no labeled data**  
- Maintaining **<1% parameter overhead**

LARQ is:
- ✅ Architecture-agnostic  
- ✅ Data-efficient (unsupervised)  
- ✅ Memory-efficient (~60× lower overhead vs prior work)

## Results

### WikiText-2 (Perplexity ↓)

- Consistent improvement across **9 LLMs**
- Works with both:
  - **bitsandbytes (NF4)**
  - **GPTQ (4-bit)**

Example (LLaMA-3 8B):
- GPTQ: **6.65 → 6.61**
- BnB: **6.66 → 6.52**

---

### MMLU (Accuracy ↑)

- Improves performance in **17/18 configurations**
- Larger gains for smaller models

Example (Qwen-2 0.5B):
- GPTQ: **39.49% → 42.74%**
- BnB: **41.75% → 42.91%**

---

## Replicate our Results
To replicate our results, follow the directions below:

### Environment Creation
```
conda create -y -n larq python=3.10
conda activate larq

pip install torch  # follow directions at https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### Running Experiments
For WikiText-2 experiments:
```
python larq.py \
    --model_id Qwen/Qwen2-1.5B \
    --dataset_name wikitext2 \
    --output_dir ./runs/kd_fm_wikitext2
```

For MMLU experiments:
```
python larq.py \
    --model_id Qwen/Qwen2-1.5B \
    --dataset_name mmlu \
    --output_dir ./runs/kd_fm_mmlu
```
