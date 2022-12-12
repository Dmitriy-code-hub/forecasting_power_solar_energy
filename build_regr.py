import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def build_regr(df=None):
    df = df or pd.read_csv('data_aver.csv', sep=';')
    df = df.drop(columns=["date_time", "day", "time", "sunrise",	"sunset", "sunDuration",])
    df = df.drop('number', axis=1)
    df = df.drop('date_time_float', axis=1)
    df = df.drop('Unnamed: 0', axis=1)



    # model 1 all test
    df_1_x = df.drop('genaration', axis=1)
    df_1_y = df['genaration']
    x = df_1_x
    y = df_1_y
    model_1 = calc_regression(x, y, text_info="test 1", predict=[[10, 115,   150, 40, 13, 8, 90, 10, -1, -2,  10, 2, 16594, 20, 1, 1200, 6, 19, 13]])



    # model 2 no days
    df_2_x = df.drop('genaration', axis=1)
    df_2_x = df_2_x.drop('day_float', axis=1)
    df_2_y = df['genaration']
    x = df_2_x
    y = df_2_y
    model_2 = calc_regression(x, y, text_info="test 2 no days", predict=[[10, 115,   150, 40, 13, 8, 90, 10, -1, -2,  10, 2, 16594, 20, 1, 6, 19, 13]])



    # model 3 no days, no sunrise, no sunset, no sunDuration
    df_3_x = df.drop('genaration', axis=1)
    df_3_x = df_3_x.drop('day_float', axis=1)
    df_3_x = df_3_x.drop('sunrise_float', axis=1)
    df_3_x = df_3_x.drop('sunset_float', axis=1)
    df_3_x = df_3_x.drop('sunDuration_float', axis=1)
    df_3_y = df['genaration']
    x = df_3_x
    y = df_3_y
    model_3 = calc_regression(x, y, text_info="test 3 no days no sunrise sunset sunduration",
                              # predict=[[10, 115, 150, 40, 13, 8, 90, 10, -1, -2, 10, 2, 16594, 20, 1]])
                              # predict=[[14.382266, 2.0, 5.9509549, -17.7604386, 18.502991, 0.0, 96.606429, 0.0, 96.355274, 0.0, 0.141643, 0.0, 4378.3, 0.0, 0.0]])
                              predict=[[19.460348,  751.504508,  167.0676963,  63.7066501,  25.939408, -1.493533,  87.736247,  24.099672,  39.274149,  1.739163, 74.482432,  0.06301571428571429,  4388.3,  103.167586, 12.0]])



    # model 4 days only with Sun
    df_4_x = df.drop('unknown', axis=1)
    df_4_x = df_4_x[(df_4_x['time_float'] > df_4_x["sunrise_float"]) & (df_4_x["time_float"] < df_4_x["sunset_float"])]# & (df_4_x["day_float"] > 624) & (df_4_x["day_float"] < 824)]
    df_4_x = df_4_x.drop('sunrise_float', axis=1)
    df_4_x = df_4_x.drop('sunset_float', axis=1)
    # df_4_x = df_4_x.drop('day_float', axis=1)
    df_4_y = df_4_x['genaration']
    df_4_x = df_4_x.drop('genaration', axis=1)
    x = df_4_x
    y = df_4_y
    model_4 = calc_regression(x, y, text_info="test 4",
                              # predict=[[19.460348,  751.504508,  167.0676963,  63.7066501,  25.939408, -1.493533,  87.736247,  24.099672,  39.274149,  1.739163, 74.482432,  0.06301571428571429,  103.167586, 12.0, 16.666]])
                              predict=[[16.328515,  34.823657,  259.0365113,  40.7347268,  17.646443,  61.983213,  99.999962,  51.507041,  99.99996333333333,37.700945,  96.364475,  24.48272,   3.433861, 16.0, 724, 16.166666666666668]])



    # draw figure
    sns.set(rc={'figure.figsize': (50, 30)})
    sns.lmplot(x="unknown", y="genaration", fit_reg=True, palette="PuOr", data=df, line_kws={"color": "red"})
    plt.savefig("unkn_gener.png")
    plt.cla()   # y = UNKONW * (ax1+bx2+cx3+... +C) + CC
    plt.clf()   # y = ax1+bx2+cx3+dx4.....+C

    # plt.figure(figsize=(20, 15))
    # plt.scatter(x=df['day_float'], y=df["genaration"], color='green', alpha=0.3)
    # plt.xlabel("sun duration")
    # plt.ylabel("sum generation")
    df_4_x["genaration"] = df_4_y
    sns.set(rc={'figure.figsize': (50, 30)})
    sns.lmplot(x="day_float", y="genaration", fit_reg=True, palette="PuOr", data=df_4_x, line_kws={"color": "red"})
    plt.plot([0, 1135], [8500, 47000], linewidth=2, color='green')
    plt.plot(df["day_float"], df['unknown'], linewidth=2, color='yellow')
    plt.savefig("5.png")
    plt.cla()
    plt.clf()
    a, b = 33.92, 8500  # y = (47000-8500)/1135 * x + 8500 = 33.92x+8500
    max_day = 1135

    #  Перетворили піки в одну лінію
    df_scaled = df.drop('unknown', axis=1)
    df_scaled["genaration"] = (df_scaled["genaration"] / (df_scaled["day_float"] * a + b)) * (max_day * a + b) # формула нормалізації
    sns.set(rc={'figure.figsize': (50, 30)})
    sns.lmplot(x="day_float", y="genaration", fit_reg=True, palette="PuOr", data=df_scaled, line_kws={"color": "red"})
    plt.savefig("6.png")
    plt.cla()
    plt.clf()



    #  model 5 lineral
    df_5_x = df_scaled.drop('sunrise_float', axis=1)
    df_5_x = df_5_x.drop('sunset_float', axis=1)
    # df_5_x = df_5_x.drop('day_float', axis=1)
    df_5_y = df_5_x['genaration']
    df_5_x = df_5_x.drop('genaration', axis=1)
    x = df_5_x
    y = df_5_y
    model_5 = calc_regression(x, y, text_info="test 5 scaled",
                              predict=[[16.328515,  34.823657,  259.0365113,  40.7347268,  17.646443,  61.983213,  99.999962,  51.507041,  99.99996333333333,37.700945,  96.364475,  24.48272,   3.433861, 16.0, 724, 16.166666666666668]],
                              mult=(724*a+b)/(max_day * a + b))




    df_6_x = df_scaled[(df_scaled['time_float'] > df_scaled["sunrise_float"]) & (df_scaled["time_float"] < df_scaled["sunset_float"])]
    df_6_x = df_6_x.drop('sunrise_float', axis=1)
    df_6_x = df_6_x.drop('sunset_float', axis=1)
    df_6_x = df_6_x.drop('day_float', axis=1)

    df_6_y = df_6_x['genaration']
    df_6_x = df_6_x.drop('genaration', axis=1)
    x = df_6_x
    y = df_6_y
    model_6 = calc_regression(x, y, text_info="test 6 scaled",
                              predict=[[16.328515, 34.823657, 259.0365113, 40.7347268, 17.646443, 61.983213, 99.999962,
                                        51.507041, 99.99996333333333, 37.700945, 96.364475, 24.48272, 3.433861, 16.0,
                                        16.166666666666668]],
                              mult=(724 * a + b) / (max_day * a + b))
    calc_important(x, y, text_info="Які параметри важливі?", mult=(724 * a + b) / (max_day * a + b))



    df_7_x = df_scaled[(df_scaled['time_float'] > df_scaled["sunrise_float"]) & (df_scaled["time_float"] < df_scaled["sunset_float"])]
    df_7_x = df_7_x.drop('sunrise_float', axis=1)
    df_7_x = df_7_x.drop('sunset_float', axis=1)
    df_7_x = df_7_x.drop('day_float', axis=1)

    df_7_y = df_7_x['genaration']
    df_7_x = df_7_x.drop('genaration', axis=1)
    x = df_7_x
    y = df_7_y
    model_7, test_result = calc_regression(x, y, text_info="test 7 scaled",
                              predict=[[16.328515, 34.823657, 259.0365113, 40.7347268, 17.646443, 61.983213, 99.999962,
                                        51.507041, 99.99996333333333, 37.700945, 96.364475, 24.48272, 3.433861, 16.0,
                                        16.166666666666668]],
                              mult=(724 * a + b) / (max_day * a + b))


    return model_7, list(df_7_x.columns.values), test_result







def calc_regression(x: pd.Series, y: pd.Series, text_info="", predict=None, mult=1) -> LinearRegression:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    model = LinearRegression()
    model.fit(X=X_train, y=y_train)

    print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||", text_info, "||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print()
    print(f"x - {x.keys()}, \ny - {y.keys()}")
    print()
    print(f'Train r-score: {model.score(X_train, y_train)}')
    test_result = model.score(X_test, y_test)
    print(f'Test r-score: {test_result}')
    print(f'Intercept: {model.intercept_}')
    print()
    print(pd.DataFrame(data=model.coef_, index=X_train.columns, columns=['coef']))
    print()

    if predict:
        result = model.predict(X=predict)
        print(result*mult)

    return model, test_result

def calc_important(x: pd.Series, y: pd.Series, text_info="", predict=None, mult=1):
    print("====================================", text_info, "====================================")
    print()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    X_all = sm.add_constant(X_train)
    model = sm.OLS(y_train, X_all)
    result = model.fit()
    print(pd.DataFrame({'coef': result.params, 'p-value': round(result.pvalues, 4)}))


    print("residuals")
    plt.scatter(x=result.fittedvalues, y=result.resid, c='cyan', alpha=0.3)
    plt.xlabel("predict genera")
    plt.ylabel("residuals")
    plt.savefig("resid.png")
    plt.cla()
    plt.clf()
    sns.distplot(result.resid, color='cyan')
    plt.title(f'Gen model: residuals Skew = {round(result.resid.skew(), 3)}, Mean = {round(result.resid.mean(), 3)}')
    plt.savefig("resid2.png")
    plt.cla()
    plt.clf()

    sigma = round(result.mse_resid, 4) # ciгма нормального розподілу помилок середнє квадратичне відхилення
    print("2 Sigma in log", 2*np.sqrt(sigma) * mult)
