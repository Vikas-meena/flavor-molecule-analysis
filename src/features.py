def prepare_molecule_data(molecules):
    """
    Extract useful features + label
    """
    # Drop unnecessary columns (adjust based on your dataset)
    molecules = molecules.dropna()

    # Example: assume 'flavor_profile' exists
    molecules = molecules[molecules['flavor_profile'].notna()]

    return molecules


def encode_labels(df):
    """
    Convert flavor labels to numbers
    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['flavor_profile'])

    return df, le