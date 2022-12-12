import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''     "genaration",
        "dewpoint",
        "downward_short_wave",
        "azimuth",
        "elevator",
        # "sunrise",
        "temperature",
        "total_boundary_cloud",
        "total_cloud",
        "total_convective_cloud",
        "total_high_cloud",
        "total_low_cloud",
        "total_middle_cloud",
        "precipitation",
        "unknown",
        "upward_short_wave",
'''

def analyze(df: pd.DataFrame = None):
    df = df or pd.read_csv('data_aver.csv', sep=";")


    print(df.head(50))
    print(pd.isnull(df).any())  # аналіз пропуски
    print(df.info())
    print(df.describe())
    print(df.columns)

    # plt.hist(df["genaration"], bins=50, ec='black', color='green')
    # plt.xlabel("gen")
    # plt.ylabel("in 30 sec")
    # plt.savefig("1.png")
    # plt.show()
    #
    #
    plt.scatter(x=df["time_float"], y=df["genaration"], color='green', alpha=0.3)
    plt.xlabel("time")
    plt.ylabel("gen")
    plt.savefig("Генерація за кожну годину на протязі дня.png.png")
    plt.cla()
    plt.clf()

    plt.figure(figsize=(20, 15))
    plt.scatter(x=(183 + df["day_float"])%365, y=df["genaration"], color='green', alpha=0.3)
    plt.xlabel("days of year")
    plt.ylabel("generation")
    plt.savefig("Генерація за кожну годину на протязі року.png")
    plt.cla()
    plt.clf()

    plt.figure(figsize=(20, 15))
    # plt.scatter(x=df['sunDuration_float'], y=df["genaration"], color='green', alpha=0.3)
    plt.xlabel("sun duration")
    plt.ylabel("sum generation")
    sns.lmplot(x="sunDuration_float", y="genaration", fit_reg=True, palette="PuOr", data=df, line_kws={"color":"red"})
    plt.savefig("sun duration - sum generation.png")
    plt.cla()
    plt.clf()



    from tabulate import tabulate
    df_for_days = df.groupby('day_float').sum()
    df_for_days["sunDuration_float"] = df.groupby('day_float').mean()["sunDuration_float"]
    print(df_for_days.columns)
    print(tabulate(df_for_days, headers='keys', tablefmt='psql'))


    plt.figure(figsize=(20, 15))
    plt.scatter(x=df_for_days["sunDuration_float"], y=df_for_days["genaration"], color='green', alpha=0.7)
    plt.xlabel("days of year")
    plt.ylabel("generation")
    plt.savefig("Сумарна генерація в залежності від протяжності дня.png")
    plt.cla()
    plt.clf()

    plt.figure(figsize=(20, 15))
    sns.set(rc={'figure.figsize':(50,30)})
    sns.lmplot(x="sunDuration_float", y="genaration", fit_reg=True, palette="PuOr", data=df_for_days, line_kws={"color": "red"})
    plt.savefig("Сумарна генерація в залежності від протяжності дня2.png")
    plt.cla()
    plt.clf()


    plt.figure(figsize=(20, 15))
    plt.scatter(x=(183 + df_for_days.index) % 365, y=df_for_days["genaration"], color='green', alpha=0.7)
    plt.xlabel("days of year")
    plt.ylabel("generation")
    plt.savefig("Генерація за кожну годину на протязі року(sum of day).png")
    plt.cla()
    plt.clf()







    corr = df.corr()
    print(tabulate(corr, headers='keys', tablefmt='psql'))

    plt.figure(figsize=(20, 15))
    mask = np.zeros_like(corr)
    triangle_indices = np.triu_indices_from(mask)
    mask[triangle_indices] = True
    sns.heatmap(corr, mask=mask, annot=True, annot_kws={"size":17})
    sns.set_style("white")
    plt.xticks(fontsize=17)
    plt.savefig("hitmap.png")
