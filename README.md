# KnowLight

> **LLM-to-LightGCN Knowledge Distillation Framework for Recommender Systems**

**KnowLight** is a modular framework designed to distill rich semantic and behavioral knowledge from Large Language Models (LLMs) into an efficient **LightGCN** graph neural recommender.

---

## Key Features

- **Efficient Student Architecture**: Compact, fast graph collaborative filtering powered by LightGCN.
- **Multiple Distillation Paradigms**:
  - **Score Distillation**: Soft-ranking knowledge transfer, compared with **RRD**.
  - **Embedding Distillation**: Latent space alignment via InfoNCE, compared with Distillation Experts (**DE**), and Hierarchical Topology Distillation (**HTD**).
  - **Intent-Augmented Graph**: Enrich sparse user-item bipartite graphs with LLM-guided pseudo-edges.
- **Teacher Flexibility**: Supports open-weight LLMs (e.g. Qwen, LLaMA) with LoRA fine-tuning.
- **Ready-to-use Benchmarks**: Automated download, k-core filtering, and leave-one-out splitting for 5 popular benchmark datasets.

---

## Environment Setup

### Option 1: Conda (Recommended)

```bash
conda env create -f setup/environment.yml
conda activate knowlight
```

### Option 2: Pip

```bash
pip install -r setup/requirements.txt
```

---

## LLM Teachers

All open-source LLM teacher backbones (e.g. `Qwen/Qwen3.5-4B`, `meta-llama/Meta-Llama-3-8B-Instruct`) are loaded and fine-tuned using the **Hugging Face** ecosystem across the entire framework.

Models can be stored locally in the `models/` directory:

```bash
# Example: Download open-weight model weights locally via Hugging Face CLI
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
```

The framework automatically resolves local models in `models/<model_name>` or fetches weights directly using their Hugging Face model repository IDs.

---

## Dataset Preparation

Datasets are automatically downloaded, k-core filtered, split (`train.tsv`, `val.tsv`, `test.tsv`), and exported with textual item metadata (`item_metadata.json`) using `DataRec`:

```bash
# MovieLens 1M
python -m knowlight.datasets.movielens

# Amazon Beauty
python -m knowlight.datasets.amazon_beauty

# Amazon Toys and Games
python -m knowlight.datasets.amazon_toys_and_games

# Amazon CD & Vinyl (2014)
python -m knowlight.datasets.amazon_cd_2014

# Yelp
python -m knowlight.datasets.yelp
```

Processed data will be saved under `data/raw/<dataset_name>/`.

---

## Training & Distillation

### 1. Baseline LightGCN (No Distillation)

Train a standard LightGCN student model without teacher guidance:

```bash
python -m knowlight.training.train_student --dataset movielens --no_distillation

# Grid search over learning rate and weight decay
python -m knowlight.training.train_student --dataset movielens --no_distillation --grid_search
```

### 2. LLM Teacher Fine-Tuning

#### Pipeline 1: Ranking Fine-Tuning (Score Distillation)

Generate TAG-oriented ranking prompts and fine-tune an open-source LLM backbone (e.g. Qwen 3.5 / LLaMA) with LoRA:

```bash
# 1. Build text-attributed graph (if not already generated)
python -m knowlight.graphs.generate_tag_graphs --dataset movielens

# 2. Generate TAG-oriented ranking prompts
python -m knowlight.prompts.prompt_builder --dataset movielens

# 3. Fine-tune LLM Teacher and extract ranking scores
python -m knowlight.teachers.llm.finetune_llm --config config/finetune_full.yaml --dataset movielens --model_path Qwen/Qwen3.5-4B
```

#### Pipeline 2: SCFT (Embedding Distillation)

Prepare dataset for LLMEmb format and run Supervised Contrastive Fine-Tuning (SCFT) to extract items and users semantic representations:

```bash
# Convert dataset into LLMEmb format
python -m knowlight.datasets.prepare_llmemb_data --dataset movielens

# SCFT Contrastive LoRA fine-tuning & automatic embedding extraction
python -m knowlight.teachers.llmemb.train_scft --dataset movielens --do_train --model_name_or_path Qwen/Qwen3.5-4B
```

### 3. Distillation Pipelines

#### A. Score Distillation (KnowLight / RRD)

```bash
# KnowLight score distillation
python -m knowlight.training.train_student_score_distill --dataset movielens --distiller score

# Distillation baseline (RRD)
python -m knowlight.training.train_student_score_distill --dataset movielens --distiller rrd

# Grid search over distillation weights
python -m knowlight.training.train_student_score_distill --dataset movielens --grid
```

#### B. Embedding Distillation (KnowLight / DE / HTD)

```bash
# Contrastive alignment (InfoNCE)
python -m knowlight.training.train_student_contrastive_distill --dataset movielens --distiller contrastive

# Distillation Experts (DE) or Hierarchical Topology Distillation (HTD)
python -m knowlight.training.train_student_contrastive_distill --dataset movielens --distiller de
python -m knowlight.training.train_student_contrastive_distill --dataset movielens --distiller htd

# Grid search over distillation weights
python -m knowlight.training.train_student_contrastive_distill --dataset movielens --grid
```


#### C. Intent-Augmented Graph Training

Extract latent user intents with the LLM teacher and train LightGCN over the augmented interaction graph:

```bash
# 1. Build text-attributed graph (if not already generated)
python -m knowlight.graphs.generate_tag_graphs --dataset movielens

# 2. Extract LLM user intent profiles and embeddings
python -m knowlight.teachers.llm.extract_user_intents --dataset movielens --model Qwen/Qwen3.5-4B

# 3. Train LightGCN over the intent-augmented graph
python -m knowlight.training.train_intent_augmented_lightgcn --dataset movielens --top_k_pseudo 2 --pseudo_weight 0.5
```

---

## Evaluation & Analysis

### 1. Student Model Evaluation

Evaluate a trained LightGCN student checkpoint on the test set by computing **Recall@K**, **NDCG@K**, **Precision@K** for $K \in \{5, 10, 20\}$.

```bash
# Evaluate the latest distilled student checkpoint
python -m knowlight.student.evaluate_student --dataset movielens

# Evaluate a specific model checkpoint
python -m knowlight.student.evaluate_student --dataset movielens --model_path data/checkpoints/distilled_students/lightgcn/movielens/<run_folder>/best_student_lightgcn.pt

# Evaluate baseline LightGCN model (non-distilled)
python -m knowlight.student.evaluate_student --dataset movielens --baseline
```


### 2. Behavioral & Diagnostic Analysis

```bash
# Sparsity evaluation across user interaction groups
python -m knowlight.evaluation.run_sparsity_evaluation --dataset movielens

# Catalog coverage and recommendation diversity evaluation
python -m knowlight.evaluation.run_coverage_diversity_evaluation --dataset movielens
```

Supported beyond accuracy metrics include **Coverage**, **Intra-List Diversity**, **Personalization** and **Novelty**.

---

## Configuration Files

Key configuration files in the `config/` directory:

| Config | Description |
| :--- | :--- |
| `config/train_student.yaml` | Student LightGCN architecture, optimizer settings, and dataset hyperparameter overrides |
| `config/teacher_llm.yaml` | LLM teacher backbone, quantization, device mapping, and prompt parameters |
| `config/finetune_full.yaml` | LoRA fine-tuning hyperparameters for LLM teachers |
| `config/train_student_score_distill.yaml` | Score distillation weights ($\alpha, \beta$) and temperatures |
| `config/train_student_contrastive_distill.yaml` | Contrastive, DE, and HTD representation distillation parameters |
| `config/train_intent_augmented_lightgcn.yaml` | Intent graph augmentation hyperparameters (pseudo-edges, weights) |

---

## Project Structure

```text
KnowLight/
├── config/             # YAML configurations for models and experiments
├── data/               # Raw datasets, graphs, cache, prompts, and checkpoints
├── knowlight/          # Core Python package
│   ├── datasets/       # Data loaders, batching, and preprocessing scripts
│   ├── student/        # LightGCN model implementation and student evaluation
│   ├── teachers/       # LLM teacher models and extraction logic
│   ├── distillation/   # Score, contrastive, RRD, DE, and HTD distillers
│   ├── graphs/         # Graph augmentation, multi-hop subgraphs, and TAG builders
│   ├── prompts/        # Prompt templates and TAG prompt builders
│   ├── samplers/       # Negative sampling and teacher top-k samplers
│   ├── training/       # Trainers, execution pipelines, and grid searches
│   ├── evaluation/     # Sparsity, diversity, and representation analysis
│   ├── utils/          # Helper functions and seeding
│   └── paths.py        # Centralized project paths and directory resolution
├── models/             # LLMs weights and adapters
└── setup/              # Conda environment, requirements.txt, and setup scripts
```

