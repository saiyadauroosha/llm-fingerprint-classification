import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = "data/splits_v2"
TOKEN_PATTERN = re.compile(r"\w+|\n|[^\w\s]")

def truncate_response(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    matches = list(TOKEN_PATTERN.finditer(text))
    if len(matches) <= 512:
        return text
    return text[:matches[511].end()]

train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
val = pd.read_parquet(f"{DATA_DIR}/validation.parquet")

X_train = train["response"].map(truncate_response)
X_val = val["response"].map(truncate_response)

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print("Vocabulary size:", len(vectorizer.vocabulary_))

for c in [10, 30, 100]:
    model = LogisticRegression(C=c, max_iter=1500)
    model.fit(X_train_tfidf, train["family"])
    predictions = model.predict(X_val_tfidf)

    print(
        f"C={c}: "
        f"Accuracy={accuracy_score(val['family'], predictions):.4f}, "
        f"Macro-F1={f1_score(val['family'], predictions, average='macro'):.4f}"
    )
