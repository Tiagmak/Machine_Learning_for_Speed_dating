import pandas


def load_file():
    input_file = "speedDating_trab.csv"
    df = pandas.read_csv(input_file, header=0)
    original_headers = list(df.columns.values)
    df = df._get_numeric_data()

    # get headers
    numeric_headers = list(df.columns.values)

    print(f'Headers--, {numeric_headers}')
    print(f'stuff--, {df}')


# run script
if __name__ == '__main__':
    load_file()
