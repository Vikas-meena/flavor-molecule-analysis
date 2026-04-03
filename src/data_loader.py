import pandas as pd

def load_data(path):
    """
    Load dataset from CSV
    """
    df = pd.read_csv(path)
    return df


def split_features_labels(df, label_col="flavor"):
    """
    Split into features and labels
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y