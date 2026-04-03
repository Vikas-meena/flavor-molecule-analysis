import pandas as pd

def load_data():
    molecules = pd.read_csv("../data/raw/fdb_molecules.csv")
    entities = pd.read_csv("../data/raw/fdb_entities.csv")
    mapping = pd.read_csv("../data/raw/fdb_molecules_entities.csv")

    return molecules, entities, mapping