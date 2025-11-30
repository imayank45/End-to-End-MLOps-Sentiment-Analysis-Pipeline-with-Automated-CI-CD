import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel
import torch
import nltk
import re
import string
import warnings

warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ========================== CONFIG ==========================
CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/imayank45/End-to-End-MLOps-Sentiment-Analysis-Pipeline-with-Automated-CI-CD.mlflow",
    "dagshub_repo_owner": "imayank45",
    "dagshub_repo_name": "End-to-End-MLOps-Sentiment-Analysis-Pipeline-with-Automated-CI-CD",
    "experiment_name": "BERT Embeddings"
}

# ========================== MLflow + DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== PREPROCESSING ==========================
lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)
    text = " ".join([w for w in nltk.word_tokenize(text) if w not in stop_words])
    text = " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)])
    return text

def load_data(file_path):
    df = pd.read_csv(file_path)
    df["review"] = df["review"].astype(str).apply(preprocess)
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].replace({"negative": 0, "positive": 1})
    return df

# ========================== BERT EMBEDDINGS ==========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()   # [CLS] token
    return cls_embeddings

# ========================== ALGORITHMS ==========================
ALGORITHMS = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

# ========================== TRAIN & EVALUATE ==========================
def train_and_evaluate(df):
    X = get_bert_embeddings(df["review"].tolist())
    y = df["sentiment"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=42
    )

    with mlflow.start_run(run_name="BERT-All-Models") as parent:

        for algo_name, algo in ALGORITHMS.items():
            with mlflow.start_run(run_name=algo_name, nested=True):
                model = algo
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred)
                }

                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, algo_name)

                print(f"\nModel: {algo_name}")
                print(metrics)

# ========================== RUN ==========================
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)
