import pandas
from IPython.display import display


def id3():
    # loading file
    input_file = "speedDating_trab.csv"
    df = pandas.read_csv(input_file, header=0)
    headers = list(df.columns.values)
    # df = df._get_numeric_data()
    # numeric_headers = list(df.columns.values)

    with pandas.option_context('display.max_rows', 8377,
                               'display.max_columns', None,
                               'display.width', 8377,
                               'display.precision', 3,
                               'display.colheader_justify', 'left'):
        display(df.head(1000))


# run script
if __name__ == '__main__':
    id3()
