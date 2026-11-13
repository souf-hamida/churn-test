# script.py — Entraînement churn + métriques + matrices de confusion
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # pas d'écran sur le runner CI
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 45
DATA_PATH = "dataset.csv"

# ---------------------------- 1. Données ---------------------------- #
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df = df[df["Age"] <= 80]                   # filtre des âges extrêmes

X = df.drop(columns=["Exited"])
y = df["Exited"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=SEED, stratify=y)

# ------------------------- 2. Prétraitement ------------------------- #
num_cols = ["Age", "CreditScore", "Balance", "EstimatedSalary"]
categ_cols = ["Gender", "Geography"]
ready_cols = [c for c in X.columns if c not in num_cols + categ_cols]

preprocess = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(drop="first", sparse_output=False))]),
     categ_cols),
    ("ready", SimpleImputer(strategy="most_frequent"), ready_cols),
])
X_train_final = preprocess.fit_transform(X_train)
X_test_final = preprocess.transform(X_test)

# -------------------- 3. Déséquilibre des classes -------------------- #
freq = np.bincount(y_train) / len(y_train)           # ~80 % / ~20 %
weights = (1 - freq) / np.sum(1 - freq)
dict_weights = {0: weights[0], 1: weights[1]}        # classe rare = poids fort

smote = SMOTE(sampling_strategy=0.7, random_state=SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)

# --------------------- 4. Entraînement + métriques --------------------- #
open("metrics.txt", "w").close()                     # repart d'un fichier vide


def train_model(X_tr, y_tr, plot_name, class_weight=None):
    clf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                 random_state=SEED, class_weight=class_weight)
    clf.fit(X_tr, y_tr)
    y_pred_test = clf.predict(X_test_final)
    score_train = f1_score(y_tr, clf.predict(X_tr))
    score_test = f1_score(y_test, y_pred_test)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test),
                annot=True, fmt="d", cbar=False, cmap="Blues")
    plt.title(plot_name)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.savefig(f"{plot_name}.png", bbox_inches="tight", dpi=150)
    plt.close()

    with open("metrics.txt", "a", encoding="utf-8") as f:   # 4 lignes par modèle
        f.write(f"{clf.__class__.__name__} {plot_name}\n")
        f.write(f"F1-score of Training is: {score_train * 100:.2f} %\n")
        f.write(f"F1-Score of Validation is: {score_test * 100:.2f} %\n")
        f.write("-" * 40 + "\n")
    return clf


train_model(X_train_final, y_train, "without-imbalance")
train_model(X_train_final, y_train, "with-class-weights", class_weight=dict_weights)
train_model(X_train_resampled, y_train_resampled, "with-SMOTE")

# ------------------ 5. Figure combinée conf_matrix.png ------------------ #
names = ["without-imbalance", "with-class-weights", "with-SMOTE"]
plt.figure(figsize=(15, 5))
for i, name in enumerate(names, 1):
    plt.subplot(1, 3, i)
    plt.imshow(Image.open(f"{name}.png"))
    plt.axis("off")
plt.suptitle("RandomForestClassifier", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("conf_matrix.png", bbox_inches="tight", dpi=150)
for name in names:
    os.remove(f"{name}.png")

print(open("metrics.txt", encoding="utf-8").read())
