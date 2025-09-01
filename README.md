# Automated Detection and Anonymization of Personally Identifiable Information (PII) in Educational Datasets

This project implements a hybrid machine learning and natural language processing (NLP) system for detecting and anonymizing **Personally Identifiable Information (PII)** in educational datasets. It was developed as part of the MSc in Computing at Atlantic Technological University (ATU).

The system combines:
- **Rule-based regex patterns** for structured entities (emails, phone numbers, URLs).
- **Classical machine learning baselines** (Logistic Regression, SVM, SGD) using TF–IDF features.
- **Transformer-based models** (BERT and DeBERTa-v3-small) fine-tuned for token-level PII detection.
- **Anonymization module** that replaces detected entities with placeholders (e.g., `<NAME>`, `<EMAIL>`).

---

## Features

- Preprocessing pipeline for structured (CSV/JSON) and unstructured text (`TXT/DOCX/PDF/CSV`) inputs.
- Regex-based entity detection for deterministic PII types.
- Baseline ML models with TF–IDF features.
- Fine-tuned transformer models (BERT, DeBERTa) for contextual detection.
- Token-level evaluation metrics: precision, recall, F1-score.
- Confusion Matrix, ROC curves, and Precision–Recall curves.
- Deterministic anonymization (tag replacement).
- Sample input/output examples (see Appendices in thesis).

---

## Requirements

- Python **3.11+**
- Jupyter Notebook / Google Colab (for experiments)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- TensorFlow / Keras
- Scikit-learn
- SpaCy (with `en_core_web_sm` model)
- pandas, numpy, matplotlib, seaborn
- pdfplumber, python-docx (for document parsing)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Setup & Compilation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ManojUmesh/edu-pii-detector.git
   cd edu-pii-detector
   ```

2. **Download dataset**:  
   - Primary dataset: [Kaggle – PII Detection & Removal from Educational Data](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/data)  
   - Place the JSON/CSV files inside the `input/` folder.

3. **Train a model**:
   Update ModelConfiguration.train=True in config.py
   ```bash
   python src/main.py
   ```

4. **Run predictions**:
   If ModelConfiguration.train=False and no --input file is provided:
   ```bash
   python src/main.py
   ```
   This will run predictions on the test dataset JSON defined in config.py

   Outputs:
   1.submission.csv → entity predictions.
   2.processed_data.csv → anonymized version of test data.

5. **Run predictions on a customfile**:
   ```bash
   python src/main.py --input sample.txt
   ```
   Supported formats: .txt, .csv, .docx, .pdf.

---

## 📊 Evaluation

You can reproduce the evaluation results using the provided Jupyter notebooks:

- `notebooks/classical_ml_baseline.ipynb` – Implements baseline models such as Logistic Regression, SVM, and SGD.  
- `notebooks/pii_evaluation.ipynb` – Evaluates transformer-based models and generates confusion matrices, ROC curves, and Precision–Recall curves.

The following metrics are reported:

- **Precision, Recall, and F1-score** (both token-level and entity-level).  
- **Confusion Matrix** (absolute counts and normalized percentages).  
- **Receiver Operating Characteristic (ROC) curves**.  
- **Precision–Recall (PR) curves**.


---

## 📂 Repository Structure

```
project/
├── input/                     # Dataset files
├── models/                   # Saved model weights
├── src/                  # Core Python scripts
│   ├── test/
│   ├── config.py
│   ├── io_ingest.py
│   ├── main.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── processed_data.csv
│   ├── sample.txt
│   ├── submission.csv
│   └── training.py
├── notebooks/                # Jupyter/Colab notebooks
│   ├── classical_ml_baseline.ipynb
│   └── pii_evaluation.ipynb
├── README.md
└── requirements.txt
```

---

## Notes & Limitations

- This project was developed using **public Kaggle datasets** only. No real student data was collected.  
- Current support is limited to English text.  
- Transformer models require GPU resources (Google Colab recommended).  
- GUI and advanced anonymization metrics (e.g., Utility Preservation Index) are identified as **future work**.

---

## Acknowledgments

This project was developed as part of the MSc in Computing at **Atlantic Technological University (ATU), Galway**.  

Generative AI tools (OpenAI’s ChatGPT) were used for scaffolding portions of preprocessing and evaluation code; all code was reviewed, adapted, and validated by the author.
