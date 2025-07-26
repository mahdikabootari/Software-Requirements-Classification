# 🧠 Software Requirements Classification: Ambiguity Detection and FR/NFR Categorization

This repository contains the code and datasets used in a study focused on the classification of software requirements. The work is structured in two main tasks:

1. **Ambiguity Detection** – Identifying whether a requirement statement is ambiguous.
2. **FR/NFR Classification** – For non-ambiguous requirements, categorizing them as either **Functional Requirements (FR)** or **Non-Functional Requirements (NFR)**.

A range of deep learning models and embedding techniques have been evaluated to benchmark classification performance across multiple datasets.

---

## 📊 Datasets

The experiments were conducted on three publicly available datasets:

- **PROMISE** 
- **Kaggle Fult Pron SRS** – [Hosted on Kaggle](https://www.kaggle.com/datasets/corpus4panwo/fault-prone-srs-dataset)
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

## ▶️ Running the Experiments

All experiments in this project are implemented as **Jupyter Notebooks**. To run the project:

### ✅ Step 1: Install Dependencies

First, install all required Python libraries:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install jupyter tensorflow keras scikit-learn torch transformers gensim pandas numpy matplotlib seaborn
```

---

### ✅ Step 2: Launch Jupyter

From the root directory of the project, launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open the desired notebook from one of the model folders, e.g.:

```
bilstm-glove/experiments.ipynb
```

---

### ✅ Step 3: Dataset Configuration

Each notebook is designed to work with one dataset at a time.  
The following datasets are supported:

- `PROMISE`
- `Kaggle Fult Pron SRS`
- `FNFC`

To run the notebook with a different dataset, **you must update the dataset file path inside the notebook**.

Look for a code block similar to this:

```python
df = pd.read_csv("data/fnfc/fnfc_cleaned.csv")
```

And change it to:

```python
df = pd.read_csv("data/promise/promise_dataset.csv")
```

Or:

```python
df = pd.read_csv("data/kaggle_fult_srs/kaggle_cleaned.csv")
```

⚠️ Make sure the file structure and column names match the expected format in the notebook.

---

### ✅ Step 4: GloVe Embedding Setup (if used)

If you're using GloVe embeddings (e.g. in `bilstm-glove`), make sure you have downloaded the following file:

🔗 [Download GloVe.42B.300d.txt](https://nlp.stanford.edu/data/glove.42B.300d.zip)

Then extract and place the file here:

```
embeddings/glove.42B.300d.txt
```

The notebook will automatically load the file if it's in the correct path.

---

### 📝 Notes

- All notebooks are self-contained and include training, validation, and evaluation code.
- You can easily modify the architecture, embeddings, or dataset inside the notebook.
- Models can be trained on both CPU and GPU.
- Results such as accuracy, F1-score, and confusion matrix are displayed at the end of each run.


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

Created by **Mahdi Kabootari & Younes Abdeahad**  
📬 kabootarimahdi2@gmail.com  
🔗 [GitHub Profile](https://github.com/mahdikabootari)

📬 abdeahad.y3@gmail.com  
🔗 [GitHub Profile](https://github.com/Younes-Abdeahad)

📬 e.kheirkhah@gmail.com

🔗 [GitHub Profile](https://github.com/ekheirkhah)


---
