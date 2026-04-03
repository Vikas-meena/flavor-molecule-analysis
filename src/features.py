import numpy as np

def normalize_features(X):
    """
    Simple normalization
    """
    return (X - X.mean()) / X.std()


def handle_missing_values(X):
    """
    Fill missing values with mean
    """
    return X.fillna(X.mean())