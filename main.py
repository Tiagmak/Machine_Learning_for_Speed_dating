import pandas
import math
from IPython.display import display

input_file = "speedDating_trab.csv"
df = pandas.read_csv(input_file, header=0)
headers = list(df.columns.values)
df = df._get_numeric_data()

df.dropna(inplace=True)                          #elimina todas as linhas que tenham algum valor NA
#print(df['length'][0:200])


#vamos calcular todos os 0's w 1's do match
one_match = 0                       #equivale ao 1
zero_match = 0                       #equivalente a 0
for index, rows in df.iterrows():
    print(rows['match'])
    if rows['match'] == 0.0:
        zero_match = zero_match + 1
    else:
        one_match = one_match + 1


def print_Table():
    # loading file
    
    # numeric_headers = list(df.columns.values)

    with pandas.option_context('display.max_rows', 8377,
                               'display.max_columns', None,
                               'display.width', 8377,
                               'display.precision', 3,
                               'display.colheader_justify', 'left'):
        display(df.head(1000))

#funcion for the entropy
def entropy(one, zero):
    inside_first = one/(one + zero)
    inside_second = zero/(one + zero)
    first_part = -inside_first * math.log2(inside_first)
    second_part = inside_second * math.log2(inside_second)
    return first_part - second_part

# run script
if __name__ == '__main__':
    print_Table()

print("1 nos matches: " + str(one_match) + " 0 nos matches: " + str(zero_match)) 

print("motherfucker: " + str(entropy(one_match, zero_match)))                         #the global entropy