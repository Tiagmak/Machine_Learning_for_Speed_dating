from math import e

import matplotlib.pyplot as plt
import numpy as np
import pandas
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

pima = pandas.read_csv("speedDating_trab.csv")


def print_table():
    global pima
    with pandas.option_context('display.max_rows', 8377,
                               'display.max_columns', None,
                               'display.width', 8377,
                               'display.precision', 3,
                               'display.colheader_justify', 'left'):
        display(pima.head(10))


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
def median_pdf(column_name):
    global pima
    # Calculate mean and Standard deviation.
    mean = np.mean(pima[column_name])
    sd = np.std(pima[column_name])

    # Apply function to the data.
    pdf = normal_dist(pima[column_name], mean, sd)
    return median(pdf)


def replace_with_median(column):
    global pima
    median_n = median_pdf(column)
    for index, rows in pima.iterrows():
        if rows[column] == 'NA':
            pima.rows[column] = median_n

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
            if rows[column] == 'NA':
                pima = pima.drop([index])

    return pima


def remove_na(mod):
    global pima
    if mod:
        pima = pima.fillna(0)
    else:
        pima = pima.dropna()


def na_values_data_plot(mod):
    global pima
    copy_df = pima

    ylabel = ""
    method = ""
    if mod:
        ylabel = "Zero'd NA"
        method = "_zero"
        copy_df = copy_df.fillna(0)
    else:
        ylabel = "Dropped NA"
        method = "_drop"
        copy_df = copy_df.dropna()

    copy_df.plot.kde(x="match",
                     y=["age"])
    plt.title(ylabel)
    plt.ylabel("Age \"match\" density")
    plt.xlabel('Participant\'s age')
    plt.savefig('plots/na_values_all/age' + method + '.png')

    copy_df.plot.kde(x="match",
                     y=["age_o"])
    plt.title(ylabel)
    plt.ylabel("Partner's Age \"match\" density")
    plt.xlabel('Partner\'s Age')
    plt.savefig('plots/na_values_all/age_o' + method + '.png')

    copy_df.plot.kde(x="match",
                     y=["date"])
    plt.title(ylabel)
    plt.ylabel("Going out on dates density")
    plt.xlabel('Dating frequency ID')
    plt.savefig('plots/na_values_all/date' + method + '.png')

    copy_df.plot.kde(x="match",
                     y=["go_out"])
    plt.title(ylabel)
    plt.ylabel("Going out on dates density")
    plt.xlabel('Going out frequency ID')
    plt.savefig('plots/na_values_all/go_out' + method + '.png')

    copy_df.plot.kde(x="match",
                     y=["int_corr"])
    plt.title(ylabel)
    plt.ylabel("Interests ratings density")
    plt.xlabel('Interests rating [-1,1]')
    plt.savefig('plots/na_values_all/int_corr' + method + '.png')

    copy_df.plot.kde(x="match",
                     y=["length"])
    plt.title(ylabel)
    plt.ylabel("Date length duration ID density")
    plt.xlabel('Date length duration ID')
    plt.savefig('plots/na_values_all/length' + method + '.png')

    copy_df.plot.kde(x="match",
                     y=["met"])
    plt.title(ylabel)
    plt.ylabel("Met before density")
    plt.xlabel('Boolean \"met\"')
    plt.savefig('plots/na_values_all/met' + method + '.png')

    copy_df.plot.kde(x="match",
                     y=["like"])
    plt.title(ylabel)
    plt.ylabel("Like pair density")
    plt.xlabel('Rating [0-10]')
    plt.savefig('plots/na_values_all/like' + method + '.png')

    copy_df.plot.kde(x="match",
                     y=["prob"])
    plt.title(ylabel)
    plt.ylabel("Meeting again density")
    plt.xlabel('Probability [0-10]')
    plt.savefig('plots/na_values_all/prob' + method + '.png')


def going_out_plot():
    pima.plot.kde(x="match",
                  y=["go_out"])
    plt.title("Going out density correlation with \"match\"")
    plt.xlabel('Frequency ID')
    plt.show()


# returns entropy percentage
def entropy(column, base=None):
    vc = pandas.Series(column).value_counts(normalize=True, sort=False)
    base = e if base is None else base
    return -(vc * np.log(vc) / np.log(base)).sum()


def id3_auto():
    global pima

    na_values_data_plot(True)
    na_values_data_plot(False)

    remove_na(True)
    # Let's remove NaN values according to going out for dates value
    pima = replace_with_partner('go_out')

    # pima = replace_with_partner('age')
    # pima = replace_with_median('prob')
    # pima = replace_with_median('age_o')

    # pima = pima.drop('goal', axis=1)

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

    # situação
    # na_values_data_plot(True)
    # na_values_data_plot(False)
    remove_na(True)

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


# run script
if __name__ == '__main__':
    # Model Accuracy, how often is the classifier correct?
    print("Age frequency Entropy:\t\t\t" + str(entropy(pima['age'])))
    print("Pair's age frequency Entropy:\t\t" + str(entropy(pima['age_o'])))
    print("Going out for dates frequency Entropy:\t" + str(entropy(pima['date'])))
    print("Going out frequency Entropy:\t\t" + str(entropy(pima['go_out'])))
    print("Liked pair Entropy:\t\t\t" + str(entropy(pima['like'])))
    print("Pair liked it Entropy:\t\t\t" + str(entropy(pima['prob'])))
    print("Interests correlation:\t\t\t" + str(entropy(pima['prob'])))

    pima = id3_auto()
    # pima = gnb_auto()
    print_table()
