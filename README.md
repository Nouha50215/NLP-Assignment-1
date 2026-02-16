# NLP-Assignment-1
# IMDB Movie Reviews Sentiment Analysis using MLP (Neural Network)

This project performs **binary sentiment classification** (Positive / Negative) on the **IMDB 50K Movie Reviews Dataset** using a **Multi-Layer Perceptron (MLP)** neural network.

This project uses **sentence-level sentiment features** extracted from:

* VADER (Valence Aware Dictionary and sEntiment Reasoner)
* TextBlob

The model is trained using `sklearn`'s `MLPClassifier`.

---

## Dataset

Dataset used:

**Kaggle** – IMDB 50K Movie Reviews
Dataset name:
`lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`

* 50,000 movie reviews
* 25,000 positive
* 25,000 negative
* Perfectly balanced dataset

---

## Project Pipeline

### Install Dependencies

```bash
pip install nltk textblob vaderSentiment scikit-learn matplotlib seaborn kagglehub joblib
```

---

### Data Loading

* Dataset downloaded using `kagglehub`
* CSV loaded using `pandas`

---

### Exploratory Data Analysis (EDA)

* Checked:

  * Missing values
  * Duplicates
  * Class balance
* Visualized class distribution using `seaborn`

---

### Text Preprocessing

Basic cleaning:

* Convert text to lowercase
* Remove HTML tags
* Remove extra spaces

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

---

### Feature Engineering

We extracted **5 numerical sentiment features** per review:

From **VADER**:

* Negative score
* Neutral score
* Positive score
* Compound score

From **TextBlob**:

* Polarity score

Final feature vector shape:

```
(50000, 5)
```

---

### Label Encoding

* `positive → 1`
* `negative → 0`

---

### Data Splitting

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 60%        |
| Validation | 20%        |
| Test       | 20%        |

Stratified splitting used to preserve class balance.

---

## Model Architecture

Model: `MLPClassifier`

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=100,
    random_state=42
)
```

### Architecture:

* Input Layer: 5 features
* Hidden Layer 1: 64 neurons (ReLU)
* Hidden Layer 2: 32 neurons (ReLU)
* Output Layer: 1 neuron (Sigmoid – handled internally)
* Loss Function: Binary Cross Entropy (log loss)
* Optimizer: Adam

---

## Training

* Trained for 100 iterations
* Training loss plotted using:

```python
mlp.loss_curve_
```

---

## Results

### Validation Accuracy:

```
77.52%
```

### Test Accuracy:

```
77.42%
```

### Classification Report:

| Metric    | Negative | Positive |
| --------- | -------- | -------- |
| Precision | 0.76     | 0.79     |
| Recall    | 0.80     | 0.75     |
| F1-score  | 0.78     | 0.77     |

The model performs consistently across both classes.

---

## Confusion Matrix

Displayed using `seaborn` heatmap.

---

## Model Saving

Model saved using `joblib`:

```python
joblib.dump(mlp, "mlp_sentiment_model.pkl")
```

---

## Project Structure

```
├── NLP_Assignment1.ipynb
├── mlp_sentiment_model.pkl
├── README.md
```

---


## Library Used

* Python
* pandas
* numpy
* matplotlib
* seaborn
* nltk
* TextBlob
* vaderSentiment
* scikit-learn
* joblib
