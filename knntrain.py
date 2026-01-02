import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

def main():
    df = pd.read_csv("data/gesture_samples.csv")

    X = df[[c for c in df.columns if c.startswith("f")]] 
    y = df["label"] #whether its pinky or thumb or neutral

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

    model = Pipeline([ ("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")) ])

    model.fit(X_train, y_train) 
    prediction = model.predict(X_test)

    print(classification_report(y_test, prediction)) 
    Path("models").mkdir(exist_ok=True) 
    joblib.dump(model, "models/knn_model.joblib") 
    print("Saved to models/knn_model.joblib")


if __name__ == "__main__": 
    main()