import pandas as pd

from analysis.analysis import analyze_white_wine, analyze_red_wine

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
    analyze_white_wine(w_wine_df)
    analyze_red_wine(r_wine_df)


if __name__ == '__main__':
    main()
