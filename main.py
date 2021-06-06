import numpy as np
import pandas
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import dens
import histo

pima = pandas.read_csv("speedDating_trab.csv")


def print_table():
    global pima
    with pandas.option_context('display.max_rows', 8377,
                               'display.max_columns', None,
                               'display.width', 8377,
                               'display.precision', 3,
                               'display.colheader_justify', 'left'):
        display(pima.head(10))


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
def median_pdf(column_name):
    global pima
    # Calculate mean and Standard deviation.
    mean = np.mean(pima[column_name])
    sd = np.std(pima[column_name])

    # Apply function to the data.
    pdf = normal_dist(pima[column_name], mean, sd)
    return median(pdf)


def replace_with_value(column, value):
    global pima
    for index, row in pima.iterrows():
        if np.isnan(row[column]):
            row[column] = value

    return pima


def replace_with_median(column):
    global pima
    median_n = median_pdf(column)
    for index, row in pima.iterrows():
        if np.isnan(row[column]):
            row[column] = median_n

    return pima


def replace_with_partner(column):
    global pima
    for index, rows in pima.iterrows():
        if rows['match'] == 1:
            if rows[column] == 'NA':
                partner_value = (pima.loc[pima[column] == pima.index.column])
                pima.rows[column] = partner_value

    return pima


def eliminate_undefined_no_match(column):
    global pima
    for index, rows in pima.iterrows():
        if rows['match'] == 0:
            if rows[column] == '':
                pima = pima.drop([index])

    return pima


def remove_na(mod):
    global pima
    if mod:
        pima = pima.fillna(0)
    else:
        pima = pima.dropna()


def entropy(column, base):
    vc = pandas.Series(column).value_counts(normalize=True, sort=False)
    return -(vc * np.log(vc) / np.log(base)).sum()


def remove_na_index_from_col(col):
    global pima
    df2 = pima.dropna(subset=[col])
    pima = pima.drop(df2.index)


def id3_auto():
    global pima

    X = pima.drop(['match', 'Unnamed: 0'], axis=1)
    y = pima.match

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # trains
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    print("\n")
    print("Accuracy (ID3):", metrics.accuracy_score(y_test, y_pred))

    return pima


def gnb_auto():
    global pima
    ''' descomentar para graph
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names = feature_cols,class_names=['0', '1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('id3.png')
        Image(graph.create_png())
        '''

    X = pima.drop(['match', 'Unnamed: 0'], axis=1)
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

    return pima


def na_specify():
    global pima
    # Let's remove NaN values according to going out for dates value
    # If age is the same then it shouldn't matter
    pima = replace_with_partner('age')
    pima = replace_with_partner('age_o')
    # least entropy, shouldn't change pdf
    pima = replace_with_median('length')
    pima = replace_with_median('met')
    # entropy is high
    pima = eliminate_undefined_no_match("int_corr")
    # "other" is neutral
    pima = replace_with_value("goal", 6)

    # remove others
    remove_na(True)


# run script
if __name__ == '__main__':

    # Model Accuracy, how often is the classifier correct?
    base = len(set(pima['age']))
    print("Age frequency Entropy:\t\t\t" + str(entropy(pima['age'], base)))
    base = len(set(pima['age_o']))
    print("Pair's age frequency Entropy:\t\t" + str(entropy(pima['age_o'], base)))
    base = len(set(pima['goal']))
    print("Goal Entropy:\t\t\t\t" + str(entropy(pima['goal'], base)))
    base = len(set(pima['date']))
    print("Going out for dates frequency Entropy:\t" + str(entropy(pima['date'], base)))
    base = len(set(pima['go_out']))
    print("Going out frequency Entropy:\t\t" + str(entropy(pima['go_out'], base)))
    base = len(set(pima['like']))
    print("Liked pair Entropy:\t\t\t" + str(entropy(pima['like'], base)))
    base = len(set(pima['prob']))
    print("Pair liked it Entropy:\t\t\t" + str(entropy(pima['prob'], base)))
    base = len(set(pima['int_corr']))
    print("Interests Entropy:\t\t\t" + str(entropy(pima['int_corr'], base)))
    base = len(set(pima['length']))
    print("Length Entropy:\t\t\t\t" + str(entropy(pima['length'], base)))
    base = len(set(pima['met']))
    print("Met Before Entropy:\t\t\t" + str(entropy(pima['met'], base)))
    base = len(set(pima['like']))
    print("Like Entropy:\t\t\t\t" + str(entropy(pima['like'], base)))
    base = len(set(pima['prob']))
    print("Prob Entropy:\t\t\t\t" + str(entropy(pima['prob'], base)))

    # remove nan values with calculated decisions
    na_specify()

    # new entropy
    print("\nEntropy after pre processing")
    base = len(set(pima['age']))
    print("Age frequency Entropy:\t\t\t" + str(entropy(pima['age'], base)))
    base = len(set(pima['age_o']))
    print("Pair's age frequency Entropy:\t\t" + str(entropy(pima['age_o'], base)))
    base = len(set(pima['goal']))
    print("Goal Entropy:\t\t\t\t" + str(entropy(pima['goal'], base)))
    base = len(set(pima['date']))
    print("Going out for dates frequency Entropy:\t" + str(entropy(pima['date'], base)))
    base = len(set(pima['go_out']))
    print("Going out frequency Entropy:\t\t" + str(entropy(pima['go_out'], base)))
    base = len(set(pima['like']))
    print("Liked pair Entropy:\t\t\t" + str(entropy(pima['like'], base)))
    base = len(set(pima['prob']))
    print("Pair liked it Entropy:\t\t\t" + str(entropy(pima['prob'], base)))
    base = len(set(pima['int_corr']))
    print("Interests Entropy:\t\t\t" + str(entropy(pima['int_corr'], base)))
    base = len(set(pima['length']))
    print("Length Entropy:\t\t\t\t" + str(entropy(pima['length'], base)))
    base = len(set(pima['met']))
    print("Met Before Entropy:\t\t\t" + str(entropy(pima['met'], base)))
    base = len(set(pima['like']))
    print("Like Entropy:\t\t\t\t" + str(entropy(pima['like'], base)))
    base = len(set(pima['prob']))
    print("Prob Entropy:\t\t\t\t" + str(entropy(pima['prob'], base)))

    column_print = input("Name of desired column [press n to skip]: ")
    if column_print != "n":
        gtype = input("Hist | Dens ? [0 .. 1]: ")
        if gtype == "0":
            histo.plot_auto(pima[column_print])
        else:
            dens.plot_auto(pima, column_print)

    algo_print = input("ID3 or GNB? [0 .. 1][press n to skip]: ")
    if algo_print != "n":
        if algo_print == "0":
            pima = id3_auto()
        else:
            pima = gnb_auto()

    print_table()
