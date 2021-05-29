import math

import numpy as np
import pandas
from IPython.display import display

# load file
input_file = "speedDating_trab.csv"
df = pandas.read_csv(input_file, header=0)
headers = list(df.columns.values)
df = df[df.columns.difference(['Unnamed: 0'])]

# eliminate all NaN values
df.dropna(inplace=True)

# let's get 1's and 0's for global entropy
one_match = 0
zero_match = 0
for index, rows in df.iterrows():
    # print(rows['match'])
    if rows['match'] == 0.0:
        zero_match = zero_match + 1
    else:
        one_match = one_match + 1


def print_table():
    with pandas.option_context('display.max_rows', 8377,
                               'display.max_columns', None,
                               'display.width', 8377,
                               'display.precision', 3,
                               'display.colheader_justify', 'left'):
        display(df.head(10))


# function for entropy
def entropy(one, zero):
    inside_first = one / (one + zero)
    inside_second = zero / (one + zero)
    first_part = -inside_first * math.log2(inside_first)
    second_part = inside_second * math.log2(inside_second)
    return first_part - second_part


# return median from list
def median(lst):
    n = len(lst)
    s = sorted(lst)
    return (sum(s[n // 2 - 1:n // 2 + 1]) / 2.0, s[n // 2])[n % 2] if n else None


def normal_dist(x, mean, sd):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


# for non boolean splits one should find an optimal splitting number.
# this means that bellow this magic number the node goes left; above it, goes right.
# To find it, one needs to split in a way that ***maximizes the likelihood
# estimate of the true probability distribution***. If above a certain number
# you are more likely to get a match then let's use that number.
#
# see the "making predictions" chapter in cmu-lecture-22.pdf
# https://en.wikipedia.org/wiki/Probability_density_function
# https://en.wikipedia.org/wiki/File:Visualisation_mode_median_mean.svg
def magic_number(column_name):
    # Calculate mean and Standard deviation.
    mean = np.mean(df[column_name])
    sd = np.std(df[column_name])

    # Apply function to the data.
    pdf = normal_dist(df[column_name], mean, sd)
    return median(pdf)


def entropy_non_bool(column_name):
    magic_n = magic_number(column_name)

    n = 0
    p = 0

    for n in df[column_name]:
        # print(rows['match'])
        if n <= magic_n:
            n = n + 1
        else:
            p = p + 1

    f = p / (p + n)
    s = n / (p + n)
    fp = -f * math.log2(f)
    sp = s * math.log2(s)
    return fp - sp


# run script
if __name__ == '__main__':
    print("Global Entropy (match):\t\t\t" + str(entropy(one_match, zero_match)))
    print("Age frequency Entropy:\t\t\t" + str(entropy_non_bool('age')))
    print("Pair's age frequency Entropy:\t\t" + str(entropy_non_bool('age_o')))
    # goal
    print("Going out for dates frequency Entropy:\t" + str(entropy_non_bool('date')))
    print("Going out frequency Entropy:\t\t" + str(entropy_non_bool('go_out')))
    # int_corr
    # length
    # met
    print("Liked pair Entropy:\t\t\t" + str(entropy_non_bool('like')))
    print("Pair liked it Entropy:\t\t\t" + str(entropy_non_bool('prob')))
    print("\n")
    print_table()
