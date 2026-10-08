"""Preprocessing experiments for the Week 10 task.

Run:  python preprocess.py
Compares several preprocessing pipelines with group-aware cross-validation on the TRAIN set,
prints a results table (copy it into your 1-page summary), then saves the best preprocessor
and a baseline model to .pkl files.

Keep this file next to data.py: the .pkl files reference the ImageFeatures class defined here,
so this file must be importable whenever you load them.
"""
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import load_dataset, split_blocks
from features import ImageFeatures

def make_pipeline(kind, equalize=False, blur=0, pca=None, classifier=None):
    steps = [("features", ImageFeatures(kind, equalize, blur)), ("scale", StandardScaler())]
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=0)))
    steps.append(("clf", classifier or SVC(kernel="rbf", C=1.0, gamma="scale")))
    return Pipeline(steps)


# name -> kwargs for make_pipeline. Add/remove rows to try other ideas.
EXPERIMENTS = {
    "raw gray":                      dict(kind="raw_gray"),
    "raw gray + equalize":           dict(kind="raw_gray", equalize=True),
    "raw gray + blur":               dict(kind="raw_gray", blur=3),
    "raw colour (BGR)":              dict(kind="raw_color"),
    "raw colour + PCA(50)":          dict(kind="raw_color", pca=50),
    "HSV histogram":                 dict(kind="hsv_hist"),
    "HOG":                           dict(kind="hog"),
    "HOG + equalize":                dict(kind="hog", equalize=True),
    "HOG + HSV histogram":           dict(kind="hog_hsv"),
    "HOG + HSV hist + PCA(100)":     dict(kind="hog_hsv", pca=100),
}


def main():
    images, labels, groups, paths = load_dataset(verbose=False)
    X = np.array(images)
    y = np.array(labels)
    groups = np.array(groups)
    train, test = split_blocks(labels, groups)
    print(f"train={len(train)}  test={len(test)}")

    cv = GroupKFold(n_splits=5)
    results = {}
    print(f"\n{'preprocessing':<30}{'CV acc (train)':>16}{'  +/- std':>10}")
    for name, kw in EXPERIMENTS.items():
        pipe = make_pipeline(**kw)
        scores = cross_val_score(pipe, X[train], y[train], groups=groups[train], cv=cv, n_jobs=-1)
        results[name] = scores.mean()
        print(f"{name:<30}{scores.mean():>16.3f}{scores.std():>10.3f}")

    best = max(results, key=results.get)
    print(f"\nBest preprocessing: {best} ({results[best]:.3f})")

    # Fit the best pipeline on all training data and check it ONCE on the held-out test set.
    pipe = make_pipeline(**EXPERIMENTS[best])
    pipe.fit(X[train], y[train])
    pred = pipe.predict(X[test])
    print(f"Test accuracy: {accuracy_score(y[test], pred):.3f}")
    print(classification_report(y[test], pred, zero_division=0))

    # Save the preprocessing part alone and the full baseline model.
    preprocessor = Pipeline(pipe.steps[:-1])
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(pipe, "initial_model.pkl")
    print("Saved preprocessor.pkl and initial_model.pkl")


if __name__ == "__main__":
    main()
