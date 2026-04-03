from sklearn.ensemble import RandomForestClassifier

def get_model():
    """
    Return a baseline Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    return model