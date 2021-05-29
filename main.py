import pandas
from IPython.display import display


def id3():
    # loading file
    input_file = "speedDating_trab.csv"
    df = pandas.read_csv(input_file, header=0)
    headers = list(df.columns.values)
    # df = df._get_numeric_data()
    # numeric_headers = list(df.columns.values)

    display(df.head(8377))


# run script
if __name__ == '__main__':
    id3()
