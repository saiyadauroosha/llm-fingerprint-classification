import re
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "splits_v2"
OUTPUT = ROOT / "results" / "tfidf" / "rq1" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUTPUT.mkdir(parents=True, exist_ok=False)

TOKEN_PATTERN = re.compile(r"\w+|\n|[^\w\s]")
LABELS = ["GPT", "Claude", "Gemini"]

def truncate(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    matches = list(TOKEN_PATTERN.finditer(text))
    if len(matches) <= 512:
        return text
    return text[:matches[511].end()]

train = pd.read_parquet(DATA / "train.parquet")
test = pd.read_parquet(DATA / "test.parquet")

X_train = train["response"].map(truncate)
X_test = test["response"].map(truncate)

experiments = [
    ("word", (1, 2), 30),
    ("char", (3, 5), 10),
]

for name, ngrams, c in experiments:
    print(f"\n--- {name.upper()} TF-IDF ---")

    vectorizer = TfidfVectorizer(
        analyzer=name,
        ngram_range=ngrams,
        min_df=2,
        sublinear_tf=True,
        max_features=100000,
    )

    train_features = vectorizer.fit_transform(X_train)
    test_features = vectorizer.transform(X_test)

    model = LogisticRegression(C=c, max_iter=1500)
    model.fit(train_features, train["family"])
    predictions = model.predict(test_features)

    accuracy = accuracy_score(test["family"], predictions)
    macro_f1 = f1_score(test["family"], predictions, average="macro")

    print(f"C: {c}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")

    matrix = pd.DataFrame(
        confusion_matrix(test["family"], predictions, labels=LABELS),
        index=LABELS,
        columns=LABELS,
    )
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(matrix)

    results = test[["sample_id", "family"]].copy()
    results["predicted_family"] = predictions
    results.to_csv(OUTPUT / f"{name}_predictions.csv", index=False)

    (OUTPUT / f"{name}_metrics.txt").write_text(
        f"data=splits_v2\nC={c}\nngrams={ngrams}\n"
        f"accuracy={accuracy:.6f}\nmacro_f1={macro_f1:.6f}\n"
        f"vocabulary_size={len(vectorizer.vocabulary_)}\n"
        f"train_size={len(train)}\ntest_size={len(test)}\n"
        f"confusion_matrix_rows_actual_columns_predicted=\n{matrix}\n"
    )

print(f"\nResults saved to: {OUTPUT}")
