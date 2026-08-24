import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report


df = pd.read_csv("dataset.csv")

X = df.drop(columns=["label"]).values
y = df["label"].values


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svc",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True
        )
    )
])


model.fit(
    X_train,
    y_train
)


pred = model.predict(X_test)

print(
    classification_report(
        y_test,
        pred
    )
)


joblib.dump(
    model,
    "gesture_svm.joblib"
)

print("Saved gesture_svm.joblib")