import pandas as pd

white_wine_file = 'dataset/winequality-white.csv'
red_wine_file = 'dataset/winequality-red.csv'


def read_wine_data(filename):
    df = pd.read_csv(filename, sep=';')
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    return df


def main():
    w_wine_df = read_wine_data(white_wine_file)
    r_wine_df = read_wine_data(red_wine_file)
    pass


if __name__ == '__main__':
    main()
