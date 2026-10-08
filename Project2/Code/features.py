"""Feature extraction transformer. Lives in its own module so pickled models can be loaded anywhere."""
import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

W, H = 64, 48


def _new_hog():
    # Created on demand (not at import time) because cv2.HOGDescriptor cannot be pickled.
    return cv2.HOGDescriptor((W, H), (16, 16), (8, 8), (8, 8), 9)


class ImageFeatures(BaseEstimator, TransformerMixin):
    """Turn an array of BGR images (N, H, W, 3) into a 2-D feature matrix.

    kind: raw_gray | raw_color | hsv_hist | hog | hog_hsv
    equalize: apply histogram equalisation (on gray / V channel) first - helps with lighting
    blur: Gaussian blur kernel size (0 = off) to suppress noise
    """

    def __init__(self, kind="raw_gray", equalize=False, blur=0):
        self.kind = kind
        self.equalize = equalize
        self.blur = blur

    def fit(self, X, y=None):
        return self

    def _one(self, img, hog):
        img = np.asarray(img, dtype=np.uint8)
        if self.blur:
            img = cv2.GaussianBlur(img, (self.blur, self.blur), 0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.equalize:
            gray = cv2.equalizeHist(gray)
            hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if self.kind == "raw_gray":
            return gray.astype(np.float32).ravel() / 255.0
        if self.kind == "raw_color":
            return img.astype(np.float32).ravel() / 255.0
        if self.kind == "hsv_hist":
            return self._hist(hsv)
        if self.kind == "hog":
            return hog.compute(gray).ravel()
        if self.kind == "hog_hsv":
            return np.concatenate([hog.compute(gray).ravel(), self._hist(hsv)])
        raise ValueError(self.kind)

    @staticmethod
    def _hist(hsv):
        parts = []
        for ch, bins, rng in ((0, 16, [0, 180]), (1, 8, [0, 256]), (2, 8, [0, 256])):
            h = cv2.calcHist([hsv], [ch], None, [bins], rng).ravel()
            parts.append(h / (h.sum() + 1e-9))
        return np.concatenate(parts).astype(np.float32)

    def transform(self, X):
        hog = _new_hog() if self.kind.startswith("hog") else None
        return np.vstack([self._one(i, hog) for i in X])
