# 🧠 Software Requirements Classification: Ambiguity Detection and FR/NFR Categorization

This repository contains the code and datasets used in a study focused on the classification of software requirements. The work is structured in two main tasks:

1. **Ambiguity Detection** – Identifying whether a requirement statement is ambiguous.
2. **FR/NFR Classification** – For non-ambiguous requirements, categorizing them as either **Functional Requirements (FR)** or **Non-Functional Requirements (NFR)**.

A range of deep learning models and embedding techniques have been evaluated to benchmark classification performance across multiple datasets.

---

## 📊 Datasets

The experiments were conducted on three publicly available datasets:

- **PROMISE**
- **Kaggle Fult Pron SRS**
- **FNFC** – [Hosted on HuggingFace](https://huggingface.co/datasets/MSHD-IAU/FNFC-Functional_Non-Functional_Calssification/tree/main)

All datasets are included in the repository under relevant folders.

---

## 🧪 Models & Embedding Methods

We tested **six neural architectures**, each combined with **four types of word embeddings**, resulting in 24 experimental setups:

### 🧠 Models:
- CNN
- Bi-CNN
- LSTM
- Bi-LSTM
- DNN
- GRU

### 🔤 Embeddings:
- GloVe (42B.300d)
- BERT (via Hugging Face Transformers)
- Word2Vec (Gensim)
- TF-IDF (Scikit-learn)

Each combination is organized in a directory named:  
`[model]-[embedding]`, e.g. `bilstm-glove`, `cnn-bert`, etc.

---

## 📁 Project Structure

```
.
├── BiLSTM
│   └── GloVe
    └── BERT
    └── TFIDF
    └── Word2Vec
├── LSTM
│   └── GloVe
    └── BERT
    └── TFIDF
    └── Word2Vec
├── BiCNN
│   └── GloVe
    └── BERT
    └── TFIDF
    └── Word2Vec
├── CNN
│   └── GloVe
    └── BERT
    └── TFIDF
    └── Word2Vec
├── DNN
│   └── GloVe
    └── BERT
    └── TFIDF
    └── Word2Vec
├── GRU
│   └── GloVe
    └── BERT
    └── TFIDF
    └── Word2Vec
└── Datasets
    └──PROMISE
    └──Kaggle
    └──FNFC
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/mahdikabootari/Software-Requirements-Classification.git
cd Software-Requirements-Classification
```

### 2. Create virtual environment and activate
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📥 Download GloVe Embeddings

Due to file size, the GloVe embeddings are not included in this repository.

➡️ Download `glove.42B.300d.txt` (1.9GB) from:  
🔗 [https://nlp.stanford.edu/data/glove.42B.300d.zip](https://nlp.stanford.edu/data/glove.42B.300d.zip)

After downloading and extracting, place the file in:

```
embeddings/glove.42B.300d.txt
```

---

## 🚀 Running the Experiments

All experiments are provided as Jupyter Notebooks. For example:

```bash
jupyter notebook bilstm-glove/experiments.ipynb
```

Each notebook allows you to:

- Select a dataset
- Load embeddings
- Train and evaluate models
- View performance metrics (accuracy, F1, etc.)

---

## 💻 Hardware Requirements

The models can be trained on both **CPU** and **GPU**.  
Using a GPU (e.g., via Google Colab or local CUDA) is recommended for faster training.

---

## 🛠 Dependencies

Main dependencies include:

- Python 3.8+
- Jupyter Notebook
- TensorFlow / Keras
- PyTorch
- Transformers
- Scikit-learn
- Gensim
- NumPy
- Pandas
- Seaborn
- Matplotlib

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📧 Contact

Created by **Mahdi Kabootari**  
📬 kabootarimahdi2@gmail.com  
🔗 [GitHub Profile](https://github.com/mahdikabootari)

---

## 📄 License
