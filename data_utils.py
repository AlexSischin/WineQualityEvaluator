import pandas as pd
from pandas import DataFrame, Series


def read_data(filename: str, columns: list[str], sep=',', random_state=None):
    df = pd.read_csv(filename, sep=sep)
    df = df[columns]
    df = df.sample(frac=1, random_state=random_state)
    df.reset_index(drop=True, inplace=True)
    return df


def split_x_y(df: DataFrame, target_col: str) -> tuple[DataFrame, Series]:
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y
