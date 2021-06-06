import matplotlib.pyplot as plt


def plot_auto(df, name):
    df.plot.kde(x="match",
                y=["age_o"])
    plt.title(name + "'s Density Graph")
    plt.ylabel("y")
    plt.xlabel('x')
    plt.show()


def na_values_data_plot(pima, mod):
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
