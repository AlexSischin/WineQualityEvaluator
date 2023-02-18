import numpy as np
import pandas as pd
from pandas import DataFrame


def read_data(filename: str, columns: list[str], sep=','):
    df = pd.read_csv(filename, sep=sep)
    df = df[columns]
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    return df


def split_x_y(df: DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()
    return x, y
