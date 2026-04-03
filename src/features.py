def prepare_molecule_data(molecules):
    """
    Extract useful features + label
    """
    # Only filter for rows with a flavor_profile label (our target)
    molecules = molecules[molecules['flavor_profile'].notna()].copy()
    
    # Fill NaN in numeric columns with 0 (for missing molecular descriptors)
    numeric_cols = molecules.select_dtypes(include=['float64', 'int64']).columns
    molecules[numeric_cols] = molecules[numeric_cols].fillna(0)

    return molecules


def encode_labels(df):
    """
    Convert flavor labels to numbers
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['flavor_profile'])

    return df, le