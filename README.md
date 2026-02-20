# 🛡️ Abusive Language Detection using NLP

A complete text classification pipeline for detecting abusive language using classical Natural Language Processing (NLP) techniques — built as part of a university lab assignment.

---

## 📌 Project Overview

This project implements a structured NLP pipeline that:
- Preprocesses raw text data
- Extracts features using **Bag of Words (BoW)** and **TF-IDF**
- Trains **Naive Bayes** and **Logistic Regression** classifiers
- Evaluates and compares model performance using standard metrics

---

## 📁 Repository Structure

```
abusive-language-detection/
│
├── abusive_language_detection.ipynb   # Main Jupyter Notebook (all exercises)
├── abusive_language_dataset.csv       # Custom balanced dataset (200 samples)
├── model_results.csv                  # Final model evaluation results
│
├── plots/
│   ├── class_distribution.png         # Bar + pie chart of class distribution
│   ├── confusion_matrices.png         # Confusion matrices for all 3 models
│   └── model_comparison.png           # Grouped bar chart of all metrics
│
└── README.md                          # This file
```

---

## 📊 Dataset

A custom dataset was created with **200 text samples**:

| Label | Class | Count |
|-------|-------|-------|
| `1` | Abusive | 100 |
| `0` | Non-Abusive | 100 |

**Examples:**

| Text | Label |
|------|-------|
| You are useless | 1 (Abusive) |
| Have a nice day | 0 (Non-Abusive) |
| Shut up idiot | 1 (Abusive) |
| Thank you very much | 0 (Non-Abusive) |

The dataset is **perfectly balanced** (50/50 split), so no class weighting or oversampling was needed.

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| pandas | Data loading & manipulation |
| scikit-learn | Feature extraction, models, evaluation |
| NLTK | Tokenization, stopwords, lemmatization |
| matplotlib / seaborn | Visualization |
| Jupyter Notebook | Interactive development |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/abusive-language-detection.git
cd abusive-language-detection
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

### 3. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### 4. Run the Notebook
```bash
jupyter notebook abusive_language_detection.ipynb
```

> **Note:** The notebook includes a fallback for environments without NLTK — it will use a built-in stopword list automatically.

---

## 🧪 Exercises Covered

### Exercise 1 — Data Loading & Inspection
- Load dataset with pandas
- Check shape, null values, class distribution
- Determine if dataset is balanced or imbalanced

### Exercise 2 — Text Preprocessing
- Lowercase conversion
- URL removal
- Punctuation removal
- Tokenization
- Stopword removal
- Lemmatization (NLTK)
- Before/after comparison examples

### Exercise 3 — Bag of Words (BoW)
- Applied `CountVectorizer`
- Generated document-term matrix (200 × 258)
- Displayed vocabulary size
- Explained BoW limitations

### Exercise 4 — TF-IDF
- Applied `TfidfVectorizer`
- Generated TF-IDF matrix (200 × 258)
- Compared BoW vs TF-IDF in detail

### Exercise 5 — Model Training
- 80/20 stratified train-test split
- Trained Naive Bayes (BoW)
- Trained Naive Bayes (TF-IDF)
- Trained Logistic Regression (TF-IDF)

### Exercise 6 — Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices (plotted)
- Full model comparison table

---

## 📈 Results

### Final Comparison Table

| Feature Method | Classifier | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| BoW | Naive Bayes | 0.775 | 0.690 | 1.000 | 0.816 |
| TF-IDF | Naive Bayes | 0.775 | 0.690 | 1.000 | 0.816 |
| TF-IDF | **Logistic Regression** | **0.825** | **0.842** | 0.800 | **0.821** |

### Key Observations

- **Logistic Regression + TF-IDF** achieved the best overall performance with **82.5% accuracy** and the most balanced precision/recall tradeoff.
- **Naive Bayes** (both BoW and TF-IDF) achieved perfect recall (1.000) — it caught every abusive sample — but had lower precision (0.690), meaning it also flagged many non-abusive texts as abusive (false positives).
- **TF-IDF did not improve Naive Bayes** over BoW in this dataset because MultinomialNB is inherently count-based and benefits less from TF-IDF normalization.
- **Logistic Regression** generalizes better to borderline cases and produces more reliable probability estimates.

---

## 📉 Visualizations

### Class Distribution
![Class Distribution](plots/class_distribution.png)

### Confusion Matrices
![Confusion Matrices](plots/confusion_matrices.png)

### Model Performance Comparison
![Model Comparison](plots/model_comparison.png)

---

## 📝 Conclusion

This lab demonstrated the full lifecycle of a classical NLP text classification pipeline:

1. **Preprocessing** significantly cleans noisy text and reduces vocabulary size.
2. **TF-IDF** is generally preferred over raw BoW as it assigns meaningful weights to words based on their document frequency.
3. **Logistic Regression** with TF-IDF outperformed Naive Bayes on accuracy and precision, making it the best model for this task.
4. For production-level abusive language detection, future work should explore:
   - **n-gram features** (bigrams/trigrams) to capture multi-word abusive phrases
   - **Deep learning models** (LSTM, BERT, RoBERTa) for contextual understanding
   - **Larger, real-world datasets** (e.g., Jigsaw Toxic Comment Dataset)
   - **Ensemble methods** combining multiple classifiers

---

## 👤 Author

**[Your Name]**  
Department of Computer Science  
[Your University Name]  
[Your Email]

---

## 📄 License

This project is for academic/educational purposes only.

---

## 🙏 Acknowledgements

- [scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- Anthropic Claude — AI assistant used for code structuring and explanation
