### Resource-Constrained Multilingual Representation Learning via Custom Lightweight Transformer Architectures on the CulturaX Corpus

### 📌 1. Project Abstract & Overview

This repository contains the complete, end-to-end implementation and empirical validation framework for my **MTech Major Thesis Project**. This work addresses the challenge of pretraining robust multilingual language models within highly **resource-constrained computational environments**, eliminating the dependency on massive, compute-prohibitive infrastructure. 

We design, implement, and train a lightweight **Transformer Encoder architecture from scratch** on a strategically balanced multi-lingual subset of the **CulturaX corpus**. To optimize convergence speed, mitigate gradient instability, and maximize representation quality under tight hardware ceilings, the training lifecycle combines **Masked Language Modeling (MLM)** with a formal **Curriculum Learning** data-scheduling framework. Rather than exposing the network to uniform data complexity randomly, text sequences are dynamically scheduled based on structural difficulty metrics, accelerating cross-lingual alignment and geometric space optimization. 

### 🚀 Key Core Contributions

* **Custom Architecture from Scratch:** Complete PyTorch implementation of a lightweight Transformer encoder optimized specifically for minimal memory foot-prints without compromising hidden-state expressiveness.
* **Deterministic Curriculum Scheduler:** A structured text-difficulty metric scheduling framework based on length dynamics, vocabulary rarity, and syntax features.
* **Optimized Low-Resource Pretraining Pipeline:** A highly efficient Python, PyArrow, and Parquet data stream processing layer designed to minimize disk I/O and RAM overhead.

### 🔬 2. Theoretical Foundations & Algorithmic Mechanics

### 🔹 A. Masked Language Modeling (MLM)

The core self-supervised objective follows a strict bidirectional token prediction framework. Given a multilingual sequence 
𝑋 = (𝑥1, 𝑥2, …, 𝑥𝑛), a subset of tokens Y ⊂ X is replaced with a special [MASK] token (15% corruption rate). The objective function minimizes the cross-entropy loss over the masked positions: 

$$\mathcal{L}_{\text{MLM}}(\theta) = - \sum_{i \in Y} \log P(x_i \mid X_{\setminus Y}; \theta)$$


### 🔹 B. Curriculum Learning Framework

To enforce a progressive learning trajectory, the data loader organizes data into discrete complexity tiers. The difficulty metric 𝒟(𝑆)
 for a text sample S is computed using a multi-factor heuristic balancing: 

1. **Sequence Length:** |S|
2. **Token Rarity Profile:** The ratio of low-frequency tokens within the dynamic vocabulary matrix.

The dataset is partitioned into K stages. As training epochs cross defined loss-convergence thresholds, the scheduler expands the sampling distribution to include higher difficulty tiers: 

$$\mathcal{D}_{\text{allowed}}(t) = f(\text{Epoch}, \mathcal{L}_{\text{val}})$$  

script cap D sub allowed end-sub open paren t close paren equals f of open paren Epoch comma script cap L sub val end-sub close paren

 

### 🛠️ 3. Technical Specifications & Architecture

### 📊 Model Hyperparameters

The structural layout of the custom mini-transformer is built to control token-embedding blowup while keeping multi-head representations rich: 


| Component / Layer | Architectural Specification | Academic Justification |
| :--- | :--- | :--- |
| **Model Type** | Transformer Encoder | Bidirectional contextual extraction |
| **Embedding Dimension ($d_{\text{model}}$)** | *e.g., 256 / 512* | Prevents over-parameterization |
| **Feed-Forward Network ($d_{\text{ff}}$)** | *e.g., 1024 / 2048* | Maintains non-linear projection capability |
| **Attention Heads ($n_{\text{heads}}$)** | *e.g., 8* | Sub-space tracking across diverse language sets |
| **Encoder Layers (N)** | *e.g., 6* | Balances deep representation vs memory footprint |
| **Vocabulary Bounds** | *e.g., 32,000 / 52,000* | Constrains the heavy weight of the token embedding layer |


### 🌍 Data Footprint (CulturaX Subset)

The training partition is built from high-quality deduplicated subsets extracted from the **CulturaX Multilingual Corpus**, balancing geographic and morphological language variants: 

* **High-Resource Anchor:** English (EN)
* **Indo-Aryan/Dravidian Space:** Hindi (HI), Telugu (TE)
* **Morphologically Rich / Low-Resource Target:** Afrikaans (AF)

### 📂 4. System Architecture & Repository Layout

```text
Resource-Constrained-Multilingual-Representation-Learning/
├── notebooks/            # Exploratory Data Analysis & pilot scaling tests
├── preprocessing/        # PyArrow & Parquet pipeline for text cleaning and Unicode handling
├── tokenizer/            # Custom subword tokenizer build scripts & vocab artifacts
├── curriculum_learning/  # Algorithmic difficulty scoring matrices & scheduling modules
├── transformer/          # Native PyTorch module blocks (Attention, FFN, EncoderLayer)
├── training/             # MLM loop, optimization scheduling, and gradient management
├── evaluation/           # Validation scripts, embedding geometry, and downstream probes
├── requirements.txt      # Explicit package version lockfile
└── README.md             # Project documentation
```


Use code with caution.

### 🚀 5. Getting Started & Reproducibility

### 1. Environment Instantiation

Isolate the execution framework inside a clean Python virtual environment or conda container: 

bash

git clone https://github.com/kasinasaaket/Culturax.git
cd Culturax
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Use code with caution.

### 2. Execution Pipeline

* **Phase 1: High-Performance Data Extraction & Normalization**
Extracts raw text streams, enforces rigorous Unicode normalization, and compiles the memory-mapped Parquet corpus tables: 

bash

python preprocessing/run_pipeline.py --languages en hi te af --output_dir ./data/processed

Use code with caution.
* **Phase 2: Tokenizer Vocabulary Compilation**
Trains a high-density subword tokenizer explicitly optimized to treat multilingual tokens without vocabulary dispersion: 

bash

python tokenizer/train_tokenizer.py --input_dir ./data/processed --vocab_size 32000

Use code with caution.
* **Phase 3: Curriculum Pretraining Pipeline**
Launches the core training network, passing parameters directly to the deterministic difficulty step manager: 

bash

python training/run_pretrain.py \
  --config config/mini_transformer.json \
  --use_curriculum True \
  --gradient_accumulation_steps 4 \
  --fp16 True

Use code with caution.

### 📈 6. Quantitative Evaluation Metrics

*To thoroughly satisfy thesis validation requirements, the model performance metrics are benchmarked using strict tracking coordinates:* 

### A. Pretraining Convergence Profile

text

[Insert empirical loss/perplexity curve plotting Curriculum Learning vs. Uniform Baseline here]

Use code with caution.

### B. Computational Benchmarks & Throughput

Training StrategyPeak VRAM Usage (MB)Tokens/SecConvergence Epoch (
𝐿val

≤𝜏
)
****Uniform Baseline****
*e.g., 7800 MB**e.g., 45k**e.g., Epoch 42*
****Curriculum Learning (Ours)****
**e.g., 4200 MB****e.g., 68k****e.g., Epoch 26**

### 🔮 7. Advanced Research Vectors & Future Enhancements

* Incorporating **Byte-Pair Encoding (BPE)** variants and **SentencePiece** tokenization paradigms to scale zero-shot target language vocab injection.
* Implementation of **Mixed-Precision (FP16/BF16)** numerical scaling and **Distributed Data Parallel (DDP)** protocols for scalable computing matrices.
* Downstream evaluation probing across zero-shot cross-lingual text classification and Named Entity Recognition (NER) benchmarks.

### 📚 8. Learning Outcomes & Domain Competencies

* **Advanced Deep Learning Systems Architecture:** Custom design and parameter modeling of neural attention matrices from fundamental matrix blocks.
* **Low-Resource Machine Learning Engineering:** Strategic memory control, gradient step allocation, and dataset optimization.
* **Multilingual Natural Language Processing:** Practical experience manipulating deduplicated international web-scale crawl data (CulturaX).

### 👨‍💻 Author Info & Academic Affiliation

**Kasina Saaket** 

* **Institution:** *[Insert your University / College Name here]*
* **GitHub Profile:** [@kasinasaaket](https://github.com/kasinasaaket)
* **Professional Network:** [LinkedIn Profile](https://linkedin.com/in/kasina-saaket-448442296)

### 📑 9. Citation (BibTeX)

If you utilize this custom architecture framework or the curriculum data scheduler in your academic research, please cite this work as follows: 

bibtex

@mastersthesis{kasinasaaket2026multilingual,
  author    = {Kasina Saaket},
  title     = {Resource-Constrained Multilingual Representation Learning using a Custom Transformer Architecture on the CulturaX Corpus},
  school    = {[Insert University/College Name]},
  year      = {2026},
  month     = {September},
  note      = {MTech First Year Major Thesis Project}
}

Use code with caution.
