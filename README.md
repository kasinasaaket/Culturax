**# 🌍 Resource-Constrained Multilingual Representation Learning using a Custom Transformer on CulturaX**



**## 📌 Project Overview**



**This project presents an end-to-end implementation of multilingual representation learning using a \*\*custom Transformer architecture\*\* trained from scratch on a subset of the \*\*CulturaX multilingual corpus\*\*. The work focuses on developing an efficient pretraining pipeline for multilingual language modeling under \*\*resource-constrained environments\*\*, where GPU memory, storage, and compute resources are limited.**



**Instead of relying on large pretrained language models, this project designs and trains a lightweight Transformer encoder using \*\*Masked Language Modeling (MLM)\*\* as the self-supervised learning objective. To improve convergence and multilingual knowledge acquisition, \*\*Curriculum Learning\*\* is incorporated by progressively exposing the model to increasingly complex multilingual text samples.**



**The complete pipeline includes multilingual data preprocessing, tokenizer construction, curriculum-based data scheduling, Transformer model implementation, masked language modeling, training, evaluation, and multilingual embedding generation.**



**---**



**# 🎯 Objectives**



**- Develop a lightweight multilingual Transformer from scratch.**

**- Train using Masked Language Modeling (MLM).**

**- Implement Curriculum Learning for efficient pretraining.**

**- Learn multilingual representations under limited computational resources.**

**- Evaluate representation quality across multiple languages.**

**- Build a reproducible Transformer pretraining pipeline.**



**---**



**# 📂 Dataset**



**The project utilizes a subset of the \*\*CulturaX multilingual corpus\*\*.**



**Languages included:**



**- English**

**- Hindi**

**- Telugu**

**- Afrikaans**



**Only a subset of the complete CulturaX dataset was used to enable efficient experimentation in a resource-constrained environment.**



**---**



**# 🛠 Technologies Used**



**- Python**

**- PyTorch**

**- Hugging Face Datasets**

**- Transformers**

**- Tokenizers**

**- PyArrow**

**- Pandas**

**- NumPy**

**- Jupyter Notebook**



**---**



**# 🧠 Model Architecture**



**A custom lightweight Transformer encoder was implemented from scratch.**



**The architecture consists of:**



**- Token Embedding Layer**

**- Positional Encoding**

**- Multi-Head Self Attention**

**- Feed Forward Network**

**- Layer Normalization**

**- Residual Connections**

**- Output Projection Layer**



**The model was optimized for efficient multilingual representation learning under limited hardware resources.**



**---**



**# 🔄 End-to-End Pipeline**



**## 1. Dataset Collection**



**- Download CulturaX subset**

**- Select multilingual samples**

**- Convert Arrow files into Parquet**

**- Language-wise filtering**



**## 2. Data Preprocessing**



**- Text cleaning**

**- Unicode normalization**

**- Language balancing**

**- Sequence generation**

**- Tokenization**

**- Vocabulary construction**



**## 3. Curriculum Learning**



**The training corpus was organized according to curriculum learning principles.**



**The model was gradually exposed to increasingly difficult multilingual text, improving convergence and representation quality.**



**## 4. Masked Language Modeling**



**Masked Language Modeling (MLM) was used as the self-supervised learning objective.**



**Random tokens were masked, and the Transformer learned to predict the missing tokens from surrounding context.**



**## 5. Custom Transformer Training**



**The lightweight Transformer model was trained from scratch using:**



**- Mini-batch training**

**- Adam optimizer**

**- Learning rate scheduling**

**- Gradient updates**



**## 6. Representation Learning**



**The trained encoder generated contextual multilingual representations suitable for downstream NLP tasks.**



**## 7. Evaluation**



**The pretrained model was evaluated by monitoring:**



**- Training Loss**

**- Validation Loss**

**- MLM Prediction Accuracy**

**- Learning Curves**



**---**



**# 🚀 Key Features**



**- Custom Transformer implementation**

**- Curriculum Learning**

**- Masked Language Modeling**

**- Multilingual representation learning**

**- Lightweight architecture**

**- Resource-constrained training**

**- Efficient preprocessing pipeline**

**- End-to-end implementation**



**---**



**# 📁 Project Structure**



**```**

**Resource-Constrained-Multilingual-Representation-Learning/**

**│**

**├── notebooks/**

**├── tokenizer/**

**├── transformer/**

**├── preprocessing/**

**├── curriculum\_learning/**

**├── training/**

**├── evaluation/**

**├── README.md**

**├── requirements.txt**

**└── .gitignore**

**```**



**---**



**# 📈 Results**



**The project successfully demonstrated that a lightweight custom Transformer can learn meaningful multilingual representations from a subset of the CulturaX corpus using curriculum learning and masked language modeling while operating under limited computational resources.**



**---**



**# 🔮 Future Work**



**- Larger multilingual corpus**

**- Byte Pair Encoding (BPE)**

**- SentencePiece tokenizer**

**- Mixed precision training**

**- Distributed training**

**- Larger Transformer architectures**

**- Fine-tuning on downstream NLP tasks**



**---**



**# 📚 Learning Outcomes**



**- Transformer Architecture**

**- Self-Supervised Learning**

**- Masked Language Modeling**

**- Curriculum Learning**

**- Representation Learning**

**- Multilingual NLP**

**- Efficient Deep Learning**

**- Resource-Constrained AI**



**---**



**# 👨‍💻 Author**



**\*\*Kasina Saaket\*\***



**- GitHub: https://github.com/kasinasaaket**

**- LinkedIn: https://linkedin.com/in/kasina-saaket-448442296**



**---**



**## ⭐ If you found this project useful, consider giving it a Star!**

