import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from columns import *

white_wine_file = 'dataset/winequality-white.csv'
red_wine_file = 'dataset/winequality-red.csv'

white_wine_cols = [
    COL_FIXED_ACIDITY,
    COL_VOLATILE_ACIDITY,
    COL_CITRIC_ACID,
    COL_RESIDUAL_SIGAR,
    COL_CHLORIDES,
    COL_FREE_SULFUR_DIOXIDE,
    COL_TOTAL_SULFUR_DIOXIDE,
    COL_DENSITY,
    COL_PH,
    COL_SULPHATES,
    COL_ALCOHOL,
    COL_QUALITY
]

red_wine_cols = [
    COL_FIXED_ACIDITY,
    COL_VOLATILE_ACIDITY,
    COL_CITRIC_ACID,
    COL_RESIDUAL_SIGAR,
    COL_CHLORIDES,
    COL_FREE_SULFUR_DIOXIDE,
    COL_TOTAL_SULFUR_DIOXIDE,
    COL_DENSITY,
    COL_PH,
    COL_SULPHATES,
    COL_ALCOHOL,
    COL_QUALITY
]


def read_data(filename: str, columns: list[str], sep=','):
    df = pd.read_csv(filename, sep=sep)
    df = df[columns]
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    return df


def separate_9th_grade_wine(df: DataFrame):
    mask = df[COL_QUALITY] == 9
    wine9_df = df[mask]
    other_wine = df[~mask]
    return wine9_df, other_wine


def split_x_y(df: DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()
    return x, y


def main():
    _wine = read_data(white_wine_file, white_wine_cols, sep=';')
    _wine9, _wine_other = separate_9th_grade_wine(_wine)
    w_wine_train_df, _wine_other_dev_test = train_test_split(_wine_other, test_size=0.6, shuffle=False)
    _wine_dev_test = pd.concat([_wine_other_dev_test, _wine9], axis=0)
    w_wine_dev_df, w_wine_test_df = train_test_split(_wine_dev_test, test_size=0.5, shuffle=True)
    del _wine, _wine9, _wine_other_dev_test, _wine_dev_test

    w_wine_train_x, w_wine_train_y = split_x_y(w_wine_train_df, COL_QUALITY)
    w_wine_dev_x, w_wine_dev_y = split_x_y(w_wine_dev_df, COL_QUALITY)
    w_wine_test_x, w_wine_test_y = split_x_y(w_wine_test_df, COL_QUALITY)

    w_wine_scaler = StandardScaler().fit(w_wine_train_x)

    w_wine_train_x = w_wine_scaler.transform(w_wine_train_x)
    w_wine_dev_x = w_wine_scaler.transform(w_wine_dev_x)
    w_wine_test_x = w_wine_scaler.transform(w_wine_test_x)


if __name__ == '__main__':
    main()
