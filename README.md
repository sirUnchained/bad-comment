# Offensive Text Classifier for Persian Language
This repository contains an Offensive Text Classifier model trained to detect offensive language in Persian text. The model is built using `scikit-learn` and exported to ONNX format for efficient inference.

# Model Description
This model classifies Persian text into two categories: **Offensive (label 1) and Neutral (label 0)**. It uses a `TfidfVectorizer` for text feature extraction combined with a `Support Vector Machine (SVM)` classifier.

# Dataset
The model was trained on the *[ParsOffensive](https://github.com/golnaz76gh/pars-offensive-dataset)* dataset. This dataset consists of Persian comments labeled as either *Offensive* or *Neutral*.

# Preprocessing
The text data underwent the following preprocessing steps:

* **Normalization**: Using `hazm.Normalizer`.
* **Lemmatization**: Using `hazm.Lemmatizer`.
* **Stop-word Removal**: Common Persian stop words were removed.
* **Label Encoding*: 'Neutral' and 'Offensive' labels were converted to numerical 0 and 1 respectively.
* **Imbalance Handling**: The `ADASYN` technique was applied to address class imbalance during training.

# Performance
Below are the performance metrics on the test set:
```
              precision    recall  f1-score   support

           0       0.80      0.96      0.88      1043
           1       0.91      0.62      0.74       644

    accuracy                           0.83      1687
   macro avg       0.86      0.79      0.81      1687
weighted avg       0.84      0.83      0.82      1687
```
Detailed Metrics:

* **Accuracy**: 0.830
* **Precision (Offensive)**: 0.909
* **Recall (Offensive)**: 0.618
* **F1-score (Offensive)**: 0.736

# How to Use
## Load the model
You can load the ONNX model and use it for inference. You will need to apply the same preprocessing steps as during training.
```python
from hazm import Lemmatizer, Normalizer, stopwords_list

# recreate the preprocessing components (or load them if saved)
stopwords = stopwords_list()
lemmatizer = Lemmatizer()
normalizer = Normalizer()

def clean_sentences(sentence: str) -> str:
    return " ".join(lemmatizer.lemmatize(word) for word in normalizer.normalize(sentence).split(" ") if word not in stopwords)

```

# Dependencies
* pandas
* numpy
* hazm
* scikit-learn
* imblearn
* skl2onnx
* onnxruntime
* huggingface_hub

# Demo
You can see Deom of this project in [https://huggingface.co/spaces/sirunchained/bad-comment-gradio](here).
