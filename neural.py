import pandas as pd
import numpy as np
import keras
import tensorflow
from keras import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import Callback
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import random

def neural(df=None):
    epoch = 10
    mass_of_dim_hidden = [800, 200, 500]

    df = df or pd.read_csv('data_aver.csv', sep=';')
    df = df.drop(columns=["date_time", "day", "time", "sunrise",	"sunset", "sunDuration",])
    df = df.drop('number', axis=1)
    df = df.drop('date_time_float', axis=1)
    df = df.drop('Unnamed: 0', axis=1)

    df = df[(df['time_float'] > df["sunrise_float"]) & (df["time_float"] < df["sunset_float"])]
    df = df.drop("sunrise_float", axis=1)
    df = df.drop("sunset_float", axis=1)


    # Нормалізуємо дані
    normalized_df = (df - df.min()) / (df.max() - df.min())



    # модель створюється
    model = Sequential()  # 19 50 20 1    19 100 30 1     19 20 1     19 200 50 20 1
    model.add(Dense(units=mass_of_dim_hidden[0],
                    kernel_initializer='normal',
                    activation="relu",  #'softmax', #"relu", "sigmoid", "tanh",  todo
                    input_dim=len(normalized_df.columns) - 1),)
    for i in range(1, len(mass_of_dim_hidden) - 1):
        model.add(Dense(units=mass_of_dim_hidden[i],
                        activation='relu', #"relu", "sigmoid", "tanh",  todo
                        input_dim=mass_of_dim_hidden[i-1]))
    model.add(Dense(units=1,
                    activation="sigmoid", # 'sigmoid', #"relu", "sigmoid", "tanh",  todo
                    input_dim=mass_of_dim_hidden[-1]))
    model.compile(optimizer='sgd', # "rmsprop" "adam" todo
                  # loss='mse', # keras perseptron loss for predictions
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error']) # keras perseptron loss for predictions



    y = normalized_df["genaration"]  # те з чим порівнює в кінці перцептрона
    x = normalized_df.drop('genaration', axis=1)  # подає на вхід перцептрона все крім генерації
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)  # Розбиває дані на тренувальні і валідаційні

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
    history = LossHistory()

    # тренування моделі
    h = model.fit(X_train, y_train, validation_split=0.8, epochs=epoch,
                  callbacks=[history],
                  # verbose = 0
                  )
    print(h)

    loss_custom = 0
    for i in range(100):
        rand_val = random.randint(1, 28000)
        result = model.predict(normalized_df.drop("genaration", axis=1).iloc[[rand_val]], verbose = 0)
        loss_custom += abs(result[0][0] - normalized_df.iloc[[rand_val]]['genaration'].to_numpy())
        # accuracy += np.square(result[0][0] - normalized_df.iloc[[rand_val]]['genaration'].to_numpy())
    loss_custom /= 100
    print(f">>>>>>>>>>>>>>>>>>>>> {loss_custom}")

    indexes = [45, 27000, 28000, 500, 5000,
               7000, 14004, 200, 400, 700,
               8000, 12000, 15000, 27500,
               24400, 12500, 19000, 22500]
    for i in indexes:
        result = model.predict(normalized_df.drop("genaration", axis=1).iloc[[i]])
        d = df.iloc[[i]]
        print(f"{i}\tday_float: {d['day_float'].to_numpy()} dewpoint:{d['dewpoint'].to_numpy()} genr:{d['genaration'].to_numpy()} | {result[0][0]*df['genaration'].max()} ({result[0][0]}*{df['genaration'].max()}) ")

    model.save(filepath=f"result/{loss_custom}_ep_{epoch}_layers_{mass_of_dim_hidden}")

    return model, list(x.columns.values), 1 - max(history.losses[-1], loss_custom)
    # todo tansig - try