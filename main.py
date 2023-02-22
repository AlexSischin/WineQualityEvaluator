import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from analysis_utils import visualize_confusion_matrix
from columns import *
from data_utils import read_data, split_x_y

white_wine_file = 'dataset/winequality-white.csv'
red_wine_file = 'dataset/winequality-red.csv'

# Relevant columns
white_wine_cols = [
    COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_RESIDUAL_SIGAR, COL_CHLORIDES, COL_TOTAL_SULFUR_DIOXIDE,
    COL_DENSITY, COL_PH, COL_ALCOHOL,
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
    all_wine = read_data(white_wine_file, sep=';', random_state=123)
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


def compare_feature_sets(train_x: DataFrame, train_y: Series, dev_x: DataFrame, dev_y: Series):
    feature_sets = {
        'all': [
            COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_CITRIC_ACID, COL_RESIDUAL_SIGAR, COL_CHLORIDES,
            COL_FREE_SULFUR_DIOXIDE, COL_TOTAL_SULFUR_DIOXIDE, COL_DENSITY, COL_PH, COL_SULPHATES, COL_ALCOHOL
        ],
        'no residual sugar': [
            COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_CITRIC_ACID, COL_CHLORIDES, COL_FREE_SULFUR_DIOXIDE,
            COL_TOTAL_SULFUR_DIOXIDE, COL_DENSITY, COL_PH, COL_SULPHATES, COL_ALCOHOL
        ],
        'no free sulfur dioxide': [
            COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_CITRIC_ACID, COL_RESIDUAL_SIGAR, COL_CHLORIDES,
            COL_TOTAL_SULFUR_DIOXIDE, COL_DENSITY, COL_PH, COL_SULPHATES, COL_ALCOHOL
        ],
        'no citric acid': [
            COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_RESIDUAL_SIGAR, COL_CHLORIDES,
            COL_FREE_SULFUR_DIOXIDE, COL_TOTAL_SULFUR_DIOXIDE, COL_DENSITY, COL_PH, COL_SULPHATES, COL_ALCOHOL
        ],
        'no pH': [
            COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_CITRIC_ACID, COL_RESIDUAL_SIGAR, COL_CHLORIDES,
            COL_FREE_SULFUR_DIOXIDE, COL_TOTAL_SULFUR_DIOXIDE, COL_DENSITY, COL_SULPHATES, COL_ALCOHOL
        ],
        'no sulphates': [
            COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_CITRIC_ACID, COL_RESIDUAL_SIGAR, COL_CHLORIDES,
            COL_FREE_SULFUR_DIOXIDE, COL_TOTAL_SULFUR_DIOXIDE, COL_DENSITY, COL_PH, COL_ALCOHOL
        ],
        'optimal': [
            COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_RESIDUAL_SIGAR, COL_CHLORIDES, COL_TOTAL_SULFUR_DIOXIDE,
            COL_DENSITY, COL_PH, COL_ALCOHOL
        ],
    }

    # Y to NumPY
    train_y = train_y.to_numpy()
    dev_y = dev_y.to_numpy()

    feature_set_names = []
    train_mse_list = []
    dev_mse_list = []

    for name, features in feature_sets.items():
        # Get arrays from df
        filtered_train_x, filtered_dev_x = _get_x_arrays(train_x, dev_x, relevant_columns=features)

        # Train
        model = LinearRegression()
        model.fit(filtered_train_x, train_y)

        # Test
        train_yhat = model.predict(filtered_train_x)
        dev_yhat = model.predict(filtered_dev_x)
        train_mse = mean_squared_error(train_y, train_yhat)
        dev_mse = mean_squared_error(dev_y, dev_yhat)

        # Save
        feature_set_names.append(name)
        train_mse_list.append(train_mse)
        dev_mse_list.append(dev_mse)

    fig, mse_ax = plt.subplots()
    ind = np.arange(len(feature_set_names))
    width = 0.3
    mse_ax.set_title('MSE VS feature set')
    mse_ax.set_xlabel('Feature set')
    mse_ax.set_ylabel('MSE')
    mse_ax.bar(ind, train_mse_list, width, label='Train MSE')
    mse_ax.bar(ind + width, dev_mse_list, width, label='Dev MSE')
    mse_ax.set_xticks(ind + width / 2, feature_set_names, rotation=20)
    mse_ax.legend()
    plt.show()


def _get_x_arrays(train_x: DataFrame, *args: DataFrame, relevant_columns=None) -> tuple[np.ndarray, ...]:
    if relevant_columns is None:
        relevant_columns = white_wine_cols

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    w_wine_scaler = StandardScaler()

    train_x = train_x[relevant_columns].to_numpy()
    train_x = poly_features.fit_transform(train_x)
    train_x = w_wine_scaler.fit_transform(train_x)

    other_x_list = []

    for x in args:
        x = x[relevant_columns].to_numpy()
        x = poly_features.transform(x)
        x = w_wine_scaler.transform(x)
        other_x_list.append(x)

    return train_x, *other_x_list


def _get_y_arrays(y: Series, *args: Series) -> tuple[np.ndarray, ...]:
    return y.to_numpy(), *[a.to_numpy() for a in args]


def _get_estimates(yhat: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray, ...]:
    yhat_list = []
    for y in [yhat] + list(args):
        y = (y + .5).astype(int)
        y[y > 10] = 10
        yhat_list.append(y)
    return yhat_list[0], *yhat_list[1:]


def train_polynomial_regression(train_x: DataFrame, train_y: Series, dev_x: DataFrame, dev_y: Series):
    # Relevant columns
    relevant_columns = [
        COL_FIXED_ACIDITY, COL_VOLATILE_ACIDITY, COL_RESIDUAL_SIGAR, COL_CHLORIDES, COL_TOTAL_SULFUR_DIOXIDE,
        COL_DENSITY, COL_PH, COL_ALCOHOL,
    ]

    # Extract data arrays
    train_x = train_x[relevant_columns].to_numpy()
    dev_x = dev_x[relevant_columns].to_numpy()
    train_y = train_y.to_numpy()
    dev_y = dev_y.to_numpy()

    # Add polynomial features
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
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

    # Convert estimates to decisions
    train_yhat = (train_yhat + .5).astype(int)
    train_yhat[train_yhat > 10] = 10
    train_yhat[train_yhat < 0] = 0
    dev_yhat = (dev_yhat + .5).astype(int)
    dev_yhat[dev_yhat > 10] = 10
    dev_yhat[dev_yhat < 0] = 0

    # Collect metrics
    labels = np.arange(0, 11)
    train_conf_mat = confusion_matrix(train_y, train_yhat, labels=labels)
    dev_conf_mat = confusion_matrix(dev_y, dev_yhat, labels=labels)
    train_f1_score = f1_score(train_y, train_yhat, labels=labels, average='weighted', zero_division=1)
    dev_f1_score = f1_score(dev_y, dev_yhat, labels=labels, average='weighted', zero_division=1)
    train_precision_score = precision_score(train_y, train_yhat, labels=labels, average='weighted', zero_division=1)
    dev_precision_score = precision_score(dev_y, dev_yhat, labels=labels, average='weighted', zero_division=1)
    train_mse = mean_squared_error(train_y, train_yhat)
    dev_mse = mean_squared_error(dev_y, dev_yhat)

    # Visualize
    fig, ((train_cm_ax, dev_cm_ax), (train_score_ax, dev_score_ax)) = plt.subplots(2, 2, height_ratios=[10, 1])
    visualize_confusion_matrix(train_cm_ax, train_conf_mat)
    visualize_confusion_matrix(dev_cm_ax, dev_conf_mat)
    train_cm_ax.set_title('Train dataset CM')
    dev_cm_ax.set_title('Dev dataset CM')
    train_table = train_score_ax.table(cellText=[[f'{train_f1_score:.3f}'],
                                                 [f'{train_precision_score:.3f}'],
                                                 [f'{train_mse:.3f}']],
                                       rowLabels=['F1-score', 'Precision score', 'MSE'], loc='center')
    train_table.scale(0.7, 1)
    train_score_ax.axis('off')
    dev_table = dev_score_ax.table(cellText=[[f'{dev_f1_score:.3f}'],
                                             [f'{dev_precision_score:.3f}'],
                                             [f'{dev_mse:.3f}']],
                                   rowLabels=['F1-score', 'Precision score', 'MSE'], loc='center')
    dev_table.scale(0.7, 1)
    dev_score_ax.axis('off')

    plt.show()


def train_logistic_regression(train_x: DataFrame, train_y: Series, dev_x: DataFrame, dev_y: Series):
    # Get arrays from df
    train_x, dev_x = _get_x_arrays(train_x, dev_x)
    train_y, dev_y = _get_y_arrays(train_y, dev_y)

    # Train
    model = LogisticRegression(C=1, max_iter=10000, solver='sag')
    model.fit(train_x, train_y)

    # Test
    train_yhat = model.predict(train_x)
    dev_yhat = model.predict(dev_x)

    # Collect metrics
    labels = np.arange(0, 11)
    train_conf_mat = confusion_matrix(train_y, train_yhat, labels=labels)
    dev_conf_mat = confusion_matrix(dev_y, dev_yhat, labels=labels)
    train_f1_score = f1_score(train_y, train_yhat, labels=labels, average='weighted', zero_division=1)
    dev_f1_score = f1_score(dev_y, dev_yhat, labels=labels, average='weighted', zero_division=1)
    train_precision_score = precision_score(train_y, train_yhat, labels=labels, average='weighted', zero_division=1)
    dev_precision_score = precision_score(dev_y, dev_yhat, labels=labels, average='weighted', zero_division=1)
    train_mse = mean_squared_error(train_y, train_yhat)
    dev_mse = mean_squared_error(dev_y, dev_yhat)

    # Visualize
    fig, ((train_cm_ax, dev_cm_ax), (train_score_ax, dev_score_ax)) = plt.subplots(2, 2, height_ratios=[10, 1])
    visualize_confusion_matrix(train_cm_ax, train_conf_mat)
    visualize_confusion_matrix(dev_cm_ax, dev_conf_mat)
    train_cm_ax.set_title('Train dataset CM')
    dev_cm_ax.set_title('Dev dataset CM')
    train_table = train_score_ax.table(cellText=[[f'{train_f1_score:.3f}'],
                                                 [f'{train_precision_score:.3f}'],
                                                 [f'{train_mse:.3f}']],
                                       rowLabels=['F1-score', 'Precision score', 'MSE'], loc='center')
    train_table.scale(0.7, 1)
    train_score_ax.axis('off')
    dev_table = dev_score_ax.table(cellText=[[f'{dev_f1_score:.3f}'],
                                             [f'{dev_precision_score:.3f}'],
                                             [f'{dev_mse:.3f}']],
                                   rowLabels=['F1-score', 'Precision score', 'MSE'], loc='center')
    dev_table.scale(0.7, 1)
    dev_score_ax.axis('off')

    plt.show()


def compare_neural_networks(train_x: DataFrame, train_y: Series, dev_x: DataFrame, dev_y: Series):
    # Models to compare
    reg_param = 0.01
    models = {
        'C1': tf.keras.Sequential([
            tf.keras.layers.Input(shape=44),
            tf.keras.layers.Dense(units=88, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=352, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=1, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
        ]),
        'C2': tf.keras.Sequential([
            tf.keras.layers.Input(shape=44),
            tf.keras.layers.Dense(units=352, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=88, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=1, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
        ]),
        'C3': tf.keras.Sequential([
            tf.keras.layers.Input(shape=44),
            tf.keras.layers.Dense(units=88, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=44, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=88, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=1, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
        ]),
        'C4': tf.keras.Sequential([
            tf.keras.layers.Input(shape=44),
            tf.keras.layers.Dense(units=176, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=176, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=1, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
        ]),
    }

    # Get arrays from df
    train_x, dev_x = _get_x_arrays(train_x, dev_x)
    train_y, dev_y = _get_y_arrays(train_y, dev_y)

    model_name_list = []
    train_mse_list = []
    dev_mse_list = []

    for name, model in models.items():
        # Train
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError()
        )
        model.fit(train_x, train_y, epochs=100)

        # Test
        train_yhat = model.predict(train_x)
        dev_yhat = model.predict(dev_x)

        # Convert estimates to decisions
        train_yhat = (train_yhat + .5).astype(int)
        train_yhat[train_yhat > 10] = 10
        dev_yhat = (dev_yhat + .5).astype(int)
        dev_yhat[dev_yhat > 10] = 10

        # Metrics
        train_mse = mean_squared_error(train_y, train_yhat)
        dev_mse = mean_squared_error(dev_y, dev_yhat)

        # Save
        model_name_list.append(name)
        train_mse_list.append(train_mse)
        dev_mse_list.append(dev_mse)

    # Visualize
    fig, mse_ax = plt.subplots()
    ind = np.arange(len(model_name_list))
    width = 0.3
    mse_ax.set_title('MSE VS configuration')
    mse_ax.set_xlabel('Configuration')
    mse_ax.set_ylabel('MSE')
    mse_ax.bar(ind, train_mse_list, width, label='Train MSE')
    mse_ax.bar(ind + width, dev_mse_list, width, label='Dev MSE')
    mse_ax.set_xticks(ind + width / 2, model_name_list, rotation=20)
    mse_ax.legend()
    plt.show()


def plot_mse_vs_reg_param(train_x: DataFrame, train_y: Series, dev_x: DataFrame, dev_y: Series):
    train_x, dev_x = _get_x_arrays(train_x, dev_x)
    train_y, dev_y = _get_y_arrays(train_y, dev_y)

    reg_params = np.arange(0.0, 0.03, 0.002)
    train_mse_list = []
    dev_mse_list = []

    for reg_param in reg_params:
        # Define model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=44),
            tf.keras.layers.Dense(units=88, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=352, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
            tf.keras.layers.Dense(units=1, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_param)),
        ])

        # Train
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError()
        )
        model.fit(train_x, train_y, epochs=100)

        # Test
        train_yhat = model.predict(train_x)
        dev_yhat = model.predict(dev_x)

        # Get metrics
        train_yhat, dev_yhat = _get_estimates(train_yhat, dev_yhat)
        train_mse = mean_squared_error(train_y, train_yhat)
        dev_mse = mean_squared_error(dev_y, dev_yhat)

        # Save
        train_mse_list.append(train_mse)
        dev_mse_list.append(dev_mse)

    # Plot
    fig, ax = plt.subplots()
    ax.set_title('MSE vs. regularization param')
    ax.set_xlabel('Regularization param')
    ax.set_ylabel('MSE')
    ax.plot(reg_params, np.array(train_mse_list), label='Train MSE', marker='.')
    ax.plot(reg_params, np.array(dev_mse_list), label='Dev MSE', marker='.')
    plt.show()


def plot_mse_vs_train_data_size(train_x: DataFrame, train_y: Series, dev_x: DataFrame, dev_y: Series):
    # Prepare dataset
    train_x, dev_x = _get_x_arrays(train_x, dev_x)
    train_y, dev_y = _get_y_arrays(train_y, dev_y)

    train_size_total = train_x.shape[0]
    data_set_sizes = np.linspace(0, train_size_total, 20)[1:] - 1
    train_mse_list = []
    dev_mse_list = []

    for size in data_set_sizes:
        # Define model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=44),
            tf.keras.layers.Dense(units=88, activation='relu'),
            tf.keras.layers.Dense(units=352, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='relu'),
        ])

        # Train
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError()
        )

        train_sub_x, _, train_sub_y, _ = train_test_split(train_x, train_y, train_size=(size / train_size_total))
        model.fit(train_sub_x, train_sub_y, epochs=100)

        # Test
        train_sub_yhat = model.predict(train_sub_x)
        dev_yhat = model.predict(dev_x)

        # Get metrics
        train_sub_yhat, dev_yhat = _get_estimates(train_sub_yhat, dev_yhat)
        train_mse = mean_squared_error(train_sub_y, train_sub_yhat)
        dev_mse = mean_squared_error(dev_y, dev_yhat)

        # Save
        train_mse_list.append(train_mse)
        dev_mse_list.append(dev_mse)

    # Plot
    fig, ax = plt.subplots()
    ax.set_title('MSE vs. train data size')
    ax.set_xlabel('Train data size')
    ax.set_ylabel('MSE')
    ax.plot(data_set_sizes, np.array(train_mse_list), label='Train MSE', marker='.')
    ax.plot(data_set_sizes, np.array(dev_mse_list), label='Dev MSE', marker='.')
    ax.legend()
    plt.show()


def train_neural_network(train_x: DataFrame, train_y: Series, dev_x: DataFrame, dev_y: Series):
    # Prepare dataset
    train_x, dev_x = _get_x_arrays(train_x, dev_x)
    train_y, dev_y = _get_y_arrays(train_y, dev_y)

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=44),
        tf.keras.layers.Dense(units=88, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        tf.keras.layers.Dense(units=352, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        tf.keras.layers.Dense(units=1, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
    ])

    # Train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError()
    )
    model.fit(train_x, train_y, epochs=100)

    # Test
    train_yhat = model.predict(train_x)
    dev_yhat = model.predict(dev_x)

    # Get metrics
    train_yhat, dev_yhat = _get_estimates(train_yhat, dev_yhat)
    labels = np.arange(0, 11)
    train_conf_mat = confusion_matrix(train_y, train_yhat, labels=labels)
    dev_conf_mat = confusion_matrix(dev_y, dev_yhat, labels=labels)
    train_f1_score = f1_score(train_y, train_yhat, labels=labels, average='weighted', zero_division=1)
    dev_f1_score = f1_score(dev_y, dev_yhat, labels=labels, average='weighted', zero_division=1)
    train_precision_score = precision_score(train_y, train_yhat, labels=labels, average='weighted', zero_division=1)
    dev_precision_score = precision_score(dev_y, dev_yhat, labels=labels, average='weighted', zero_division=1)
    train_mse = mean_squared_error(train_y, train_yhat)
    dev_mse = mean_squared_error(dev_y, dev_yhat)

    # Visualize
    fig, ((train_cm_ax, dev_cm_ax), (train_score_ax, dev_score_ax)) = plt.subplots(2, 2, height_ratios=[10, 1])
    visualize_confusion_matrix(train_cm_ax, train_conf_mat)
    visualize_confusion_matrix(dev_cm_ax, dev_conf_mat)
    train_cm_ax.set_title('Train dataset CM')
    dev_cm_ax.set_title('Dev dataset CM')
    train_table = train_score_ax.table(cellText=[[f'{train_f1_score:.3f}'],
                                                 [f'{train_precision_score:.3f}'],
                                                 [f'{train_mse:.3f}']],
                                       rowLabels=['F1-score', 'Precision score', 'MSE'], loc='center')
    train_table.scale(0.7, 1)
    train_score_ax.axis('off')
    dev_table = dev_score_ax.table(cellText=[[f'{dev_f1_score:.3f}'],
                                             [f'{dev_precision_score:.3f}'],
                                             [f'{dev_mse:.3f}']],
                                   rowLabels=['F1-score', 'Precision score', 'MSE'], loc='center')
    dev_table.scale(0.7, 1)
    dev_score_ax.axis('off')

    plt.show()


def main():
    w_wine_train_df, w_wine_dev_df, w_wine_test_df = get_train_dev_test_white_wine()

    w_wine_train_x, w_wine_train_y = split_x_y(w_wine_train_df, COL_QUALITY)
    w_wine_dev_x, w_wine_dev_y = split_x_y(w_wine_dev_df, COL_QUALITY)
    w_wine_test_x, w_wine_test_y = split_x_y(w_wine_test_df, COL_QUALITY)

    # compare_poly_degrees(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)
    # compare_feature_sets(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)
    # train_polynomial_regression(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)
    # train_logistic_regression(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)
    # compare_neural_networks(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)
    # plot_mse_vs_reg_param(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)
    # plot_mse_vs_train_data_size(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)
    train_neural_network(w_wine_train_x, w_wine_train_y, w_wine_dev_x, w_wine_dev_y)


if __name__ == '__main__':
    main()
