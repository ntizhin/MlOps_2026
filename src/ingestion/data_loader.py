import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def generate_batches(df, batch_size):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size]