import math

# normal stuff
import numpy as np
import pandas
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

'''
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
'''
# load file
input_file = "speedDating_trab.csv"
df = pandas.read_csv(input_file, header=0)
headers = list(df.columns.values)
'''
df = df[df.columns.difference(['Unnamed: 0'])]
# eliminate all NaN values
# df.dropna(inplace=True)
'''


def print_table(table):
    with pandas.option_context('display.max_rows', 8377,
                               'display.max_columns', None,
                               'display.width', 8377,
                               'display.precision', 3,
                               'display.colheader_justify', 'left'):
        display(table.head(10))


# let's get 1's and 0's for global entropy
# function for entropy
def entropy(table):
    one = 0
    zero = 0
    for index, rows in table.iterrows():
        # print(rows['match'])
        if rows['match'] == 0.0:
            zero = zero + 1
        else:
            one = one + 1

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
def magic_number(table, column_name):
    # Calculate mean and Standard deviation.
    mean = np.mean(table[column_name])
    sd = np.std(table[column_name])

    # Apply function to the data.
    pdf = normal_dist(table[column_name], mean, sd)
    return median(pdf)


def entropy_non_bool(table, column_name):
    magic_n = magic_number(table, column_name)

    n = 0.0
    p = 0.0

    for n in table[column_name]:
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


###########################################################
###########################################################
def replace_with_median(pima, column):
    h_nm = headers
    h_nm.remove('match')
    h_nm.remove('Unnamed: 0')
    median_n = magic_number(pima, column)

    for index, rows in pima.iterrows():
        if rows[column] == 'NA':
            rows[column] = median_n

    return pima


def replace_with_partner(pima, column):
    h_nm = headers
    h_nm.remove('match')
    h_nm.remove('Unnamed: 0')

    for index, rows in pima.iterrows():
        if rows['match'] == 1:
            if rows[column] == 'NA':
                partner_value = (pima.loc[pima[column] == pima.index.age])
                rows[column] = partner_value
        else:
            pima.drop(pima.index)

    return pima


def id3_auto():
    pima = pandas.read_csv("speedDating_trab.csv")
    pima.head()
    # pima = pima.iloc[1:]
    pima = pima[pima.columns.difference(['Unnamed: 0'])]

    # Let's remove NaN values according to going out for dates value
    pima = replace_with_partner(pima, 'go_out')
    # pima = replace_with_median(pima, 'age')

    pima.dropna(inplace=True)
    X = pima.drop(['match'], axis=1)
    y = pima.match

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # cenas
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("\nGlobal Entropy (match):\t\t\t" + str(entropy(pima)))
    print("Age frequency Entropy:\t\t\t" + str(entropy_non_bool(pima, 'age')))
    print("Pair's age frequency Entropy:\t\t" + str(entropy_non_bool(pima, 'age_o')))
    print("Going out for dates frequency Entropy:\t" + str(entropy_non_bool(pima, 'date')))
    print("Going out frequency Entropy:\t\t" + str(entropy_non_bool(pima, 'go_out')))
    print("Liked pair Entropy:\t\t\t" + str(entropy_non_bool(pima, 'like')))
    print("Pair liked it Entropy:\t\t\t" + str(entropy_non_bool(pima, 'prob')))
    print("\n")
    print("Accuracy (ID3):", metrics.accuracy_score(y_test, y_pred))
    print("\n")
    print_table(pima)
    print("\n")

    ''' descomentar para graph
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names = feature_cols,class_names=['0', '1'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('id3.png')
    Image(graph.create_png())
    '''


def gnb_auto():
    pima = pandas.read_csv("speedDating_trab.csv")
    pima.head()
    pima.dropna(inplace=True)
    pima = pima.iloc[1:]

    X = pima.drop(['match'], axis=1)
    y = pima.match

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # cenas
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Create Decision Tree classifer object
    gnb = GaussianNB()

    # Train Decision Tree Classifer
    gnb = gnb.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = gnb.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy (GNB):", metrics.accuracy_score(y_test, y_pred))


# run script
if __name__ == '__main__':
    id3_auto()
    # gnb_auto()
