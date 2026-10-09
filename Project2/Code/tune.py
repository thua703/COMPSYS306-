"""Hyperparameter tuning for the Week 10 task.

Run:  python tune.py
Uses the best preprocessing from preprocess.py (edit BEST_PREPROCESSING below to match what won),
grid-searches the classifier with group-aware CV on the TRAIN set, then evaluates once on TEST.
Saves best_model.pkl and prints the ranges tried + best values for your 1-page summary.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC

from data import load_dataset, split_blocks
from preprocess import EXPERIMENTS, make_pipeline

# Change this to the name of the winning row printed by preprocess.py
BEST_PREPROCESSING = "HSV histogram"

# Ranges to test (report these in the summary). Lists of dicts so linear/rbf each get sensible params.
PARAM_GRID = [
    {"clf__kernel": ["rbf"],
     "clf__C": [0.1, 1, 10, 100],
     "clf__gamma": ["scale", 0.001, 0.01, 0.1, 1]},
    {"clf__kernel": ["linear"],
     "clf__C": [0.01, 0.1, 1, 10]},
]
# If the winning preprocessing uses PCA, also add e.g.:  "pca__n_components": [30, 50, 100]


def main():
    images, labels, groups, paths = load_dataset(verbose=False)
    X, y, groups = np.array(images), np.array(labels), np.array(groups)
    train, test = split_blocks(labels, groups)

    pipe = make_pipeline(**EXPERIMENTS[BEST_PREPROCESSING], classifier=SVC())
    search = GridSearchCV(
        pipe, PARAM_GRID, cv=GroupKFold(n_splits=5), scoring="accuracy",
        n_jobs=-1, refit=True, verbose=1,
    )
    search.fit(X[train], y[train], groups=groups[train])

    res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cols = [c for c in res.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score"]
    print("\nTop 10 settings:")
    print(res[cols].head(10).to_string(index=False))

    print("\nRanges tested:")
    for grid in PARAM_GRID:
        for k, v in grid.items():
            print(f"  {k}: {v}")
    print(f"\nBest params: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.3f}")

    pred = search.predict(X[test])
    print(f"Test accuracy: {accuracy_score(y[test], pred):.3f}")
    print(classification_report(y[test], pred, zero_division=0))

    joblib.dump(search.best_estimator_, "best_model.pkl")
    print("Saved best_model.pkl")


if __name__ == "__main__":
    main()
