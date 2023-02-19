import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from columns import *
from data_utils import read_data, split_x_y

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


def separate_9th_grade_wine(df: DataFrame):
    mask = df[COL_QUALITY] == 9
    wine9_df = df[mask]
    other_wine = df[~mask]
    return wine9_df, other_wine


def get_train_dev_test_white_wine():
    all_wine = read_data(white_wine_file, white_wine_cols, sep=';', random_state=123)
    wine9, wine_not9 = separate_9th_grade_wine(all_wine)
    w_wine_train_df, wine_dev_test_not9 = train_test_split(wine_not9, test_size=0.6, shuffle=False)
    w_wine_not9_dev_df, w_wine_not9_test_df = train_test_split(wine_dev_test_not9, test_size=0.5, shuffle=False)
    wine9_dev, wine9_test = train_test_split(wine9, test_size=0.5, shuffle=False)
    w_wine_dev_df = pd.concat([w_wine_not9_dev_df, wine9_dev], axis=0)
    w_wine_test_df = pd.concat([w_wine_not9_test_df, wine9_test], axis=0)
    return w_wine_train_df, w_wine_dev_df, w_wine_test_df


def compare_poly_degrees(train_x: DataFrame, train_y: Series, dev_x: DataFrame, dev_y: Series):
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    dev_x = dev_x.to_numpy()
    dev_y = dev_y.to_numpy()

    degree_list = list(range(1, 4))
    feature_number_list = []
    train_mse_list = []
    dev_mse_list = []

    for d in degree_list:
        # Add polynomial features
        poly_features = PolynomialFeatures(degree=d, include_bias=False)
        train_x = poly_features.fit_transform(train_x)
        dev_x = poly_features.transform(dev_x)

        # Scale features
        w_wine_scaler = StandardScaler()
        train_x = w_wine_scaler.fit_transform(train_x)
        dev_x = w_wine_scaler.transform(dev_x)

        # Train
        model = LinearRegression()
        model.fit(train_x, train_y)

        # Test
        train_yhat = model.predict(train_x)
        dev_yhat = model.predict(dev_x)
        train_mse = mean_squared_error(train_y, train_yhat)
        dev_mse = mean_squared_error(dev_y, dev_yhat)

        # Save
        feature_number_list.append(poly_features.n_output_features_)
        train_mse_list.append(train_mse)
        dev_mse_list.append(dev_mse)

    fig, (mse_ax, feature_ax) = plt.subplots(1, 2)
    mse_ax.set_title('MSE VS polynomial degree')
    mse_ax.set_xlabel('Polynomial degree')
    mse_ax.set_ylabel('MSE')
    mse_ax.plot(degree_list, train_mse_list, label='Train MSE', marker='.')
    mse_ax.plot(degree_list, dev_mse_list, label='Dev MSE', marker='.')
    mse_ax.legend()
    feature_ax.set_title('Feature number VS polynomial degree')
    feature_ax.set_xlabel('Polynomial degree')
    feature_ax.set_ylabel('Feature number')
    feature_ax.plot(degree_list, feature_number_list, marker='.')
    plt.show()


def main():
    w_wine_train_df, w_wine_dev_df, w_wine_test_df = get_train_dev_test_white_wine()

    w_wine_train_x, w_wine_train_y = split_x_y(w_wine_train_df, COL_QUALITY)
    w_wine_dev_x, w_wine_dev_y = split_x_y(w_wine_dev_df, COL_QUALITY)
    w_wine_test_x, w_wine_test_y = split_x_y(w_wine_test_df, COL_QUALITY)

    compare_poly_degrees(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)


if __name__ == '__main__':
    main()
