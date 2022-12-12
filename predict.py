from keras import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def predict(df,
            regres_model: LinearRegression, regres_columns_x, regres_result,
            neural_model: Sequential, neural_columns_x, neural_result):
    count_of_random = 100

    print("\n")
    print(">>>>>>>>>>>>>>>>>>>>> ansambl <<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"regres_model_clumns {regres_columns_x}")
    print(f"neural_model_clumns {neural_columns_x}")


    # Знаходимо пропорцію точностей
    print(f"(worst)regression_accuracy: {regres_result}")    #   0.8       0.95
    print(f"(worst)neural_accuracy: {neural_result}")
    regres_loss, neural_loss = 1 - regres_result, 1-neural_result    #    1-0.8  : 1 -0.95
    min_loss = min(regres_loss, neural_loss)
    #   0.2  :  0.05
    #   4    :   1
    regres_koef, neural_koef = regres_loss/min_loss, neural_loss/min_loss
    print(f"proportion regres_koef:neural_koef {regres_koef}:{neural_koef}")




    # всі дані із сонячним днем
    df = df or pd.read_csv('data_aver.csv', sep=';')
    df = df.drop(columns=["date_time", "day", "time", "sunrise", "sunset", "sunDuration", ])
    df = df.drop('number', axis=1)
    df = df.drop('date_time_float', axis=1)
    df = df.drop('Unnamed: 0', axis=1)
    df = df[(df['time_float'] > df["sunrise_float"]) & (df["time_float"] < df["sunset_float"])]
    df = df.drop("sunrise_float", axis=1)
    df = df.drop("sunset_float", axis=1)

    print(list(df))

    # min max for normalize for neural
    df_max = df.max()
    df_min = df.min()
    print(f"original min/max для всіх даних із сонячним днем{df_min}, {df_max}")


    # random 100 value for test

    df_rand_values = df.sample(count_of_random)


    # normalize test set of values  для нейронки
    df_for_neural_test = (df_rand_values - df_min) / (df_max - df_min)


    # нормалізуємо для регерсії по лінійному закону
    df_for_regres = df_rand_values.copy(True)
    a, b = 33.92, 8500  # y = (47000-8500)/1135 * x + 8500 = 33.92x+8500      y = 33.92x+8500     x = (y - 8500)/33.92
    max_day = 1135
    df_for_regres["genaration"] = (df_for_regres["genaration"] / (df_for_regres["day_float"] * a + b)) * (max_day * a + b)
    df_for_regres = df_for_regres.drop('unknown', axis=1) # 'unknown', 'day_float'   # для тестування конкретних прикладів
    df_days_for_regr = df_for_regres["day_float"]
    df_for_regres = df_for_regres.drop('day_float', axis=1)


    # for X in XX: sum += (r * model_reg.predict(X) + t * model_neu.predict(X))/(r+t)     -> sum/len(XX)
    # передбачаємо дані обох моделей
    y_regres = regres_model.predict(X=df_for_regres.drop("genaration", axis=1))
    y_neural = neural_model.predict(x=df_for_neural_test.drop("genaration", axis=1))
    print("y_reg", y_regres)
    print("y_neu", y_neural)

    average_loss = 0
    print(f"Number\t\t\tDay\t\t\treal\t\t\t(REGnotNorm)\t\t\tregr\t\t\t(NEUnotNorm)\t\t\tneu\t\t\taverage_koef\t\t\tabsol_loss\t\t\tloss_relative")
    for i in range(count_of_random):
        # y_regres[i] = (df_for_regres["genaration"] / (df_for_regres["day_float"] * a + b)) * (max_day * a + b)
        # df_for_regres["genaration"] = y_regres[i] / (max_day * a + b) * (df_for_regres["day_float"] * a + b))
        print(f"{i:.0f}\t\t\t{float(df_days_for_regr.iloc[[i]]):.0f}\t\t\t{float(df_rand_values['genaration'].iloc[[i]]):.2f}", end='\t\t\t')
        print(f"({float(y_regres[i]):.2f})\t\t\t{float(y_regres[i] * (df_days_for_regr.iloc[[i]]*a+b) / (max_day * a + b)):.2f},", end="\t\t\t")
        print(f"({y_neural[i][0]:.4f}*{float(df['genaration'].max()):.2f})\t\t\t{float(y_neural[i][0] * df['genaration'].max()):.2f}", end="\t\t\t")

        average = (regres_koef * (float(y_neural[i][0] * df['genaration'].max())) + neural_koef * (float(y_regres[i] * (df_days_for_regr.iloc[[i]]*a+b) / (max_day * a + b)))) / (regres_koef + neural_koef)

        print(f"{average}", end="\t\t")
        loss_absolute = abs(float(df_rand_values['genaration'].iloc[[i]]) - average)
        print(f"{loss_absolute}", end="\t\t\t")
        loss_relative = loss_absolute / df['genaration'].max()
        print(f"{loss_relative}")
        average_loss += loss_absolute / df['genaration'].max()
    average_loss /= count_of_random



    # calc_regression(x=df_for_regres.drop("genaration", axis=1), y=df_for_regres["genaration"], predict=df_for_regres_test)
    print("regres\tneural\tansam\treal")
    print(f"{regres_result}\t{neural_result}\t{1 - average_loss}\t{1}")  # accuracy
    print()  # real result

