import pandas as pd
import numpy as np
from datetime import datetime
import os


# data.row = r.read ()

# generationValue = pd.read_csv ("dataSet/askoe.small_pv.system.generation.15m.txt", sep=" ")
# print(generationValue)
# print (convert_1 (dataraw))

#DATASET
# askoe.small_pv.system.generation.15m.txt     +      genaration                  15 m
# dewpoint_celsius                             +      dewpoint                    3 h    точка роси
# downward_short_wave                          +      downward_short_wave         3 h
# sun.azi                                      +      azimuth                     15 m
# sun.ele                                      +      elevator                    15 m
# sunriseSunset                                +      sunrise sunset duration     1 d
# temperature                                  +      temperature                 3 h
# total_boundary_cloud.comb                    +      total_boundary_cloud        1 h гранична
# total_cloud                                  +      total_cloud                 3 h
# total_convective_cloud.comb                  +      total_convective_cloud      3 h конвективна
# total_high_cloud.comb                        +      total_high_cloud            3 h
# total_low_cloud.comb                         +      total_low_cloud             3 h
# total_middle_cloud.comb                      +      total_middle_cloud          3 h
# total_precipitation.comb                     +      precipitation               at time опади
# unknowhData                                  +      unknown                     at time
# upward_short_wave.comb                       +      upward_short_wave           3 h

# winter                      summer

# 2019                      26.10-27.10
# 2020                      25.10-26.10                 29.03-30.03
# 2021                      31.10-01.11                 28.03-29.03
# 2022                      27.03-28.03

# cntrl/`+

def read_data(path="dataSet", save_name="data_aver.csv"):
    # Функції конвертації із тексту в норм дані
    def convert_1(text):
        """
        функція обробляє файли перетворює в формат списків

        68 ідем з кінця data_2 рядки -1 бо з 0 номерація
        :param text: текст з файла
        :return: список списків з даних  номер час  значення
        """
        data_1 = text.split("\n")
        # data_1.replace ('"', "")
        data_2 = [d.split("\t") for d in data_1]




        for i in range(len(data_2) - 1, -1, -1): # рендж створює з останнього по кроку мінус 1 до -1

            if data_2[i][2] == '':
                del data_2[i]


        for i in range(len(data_2)): # приведення з строкової з числа або дати
            data_2[i][0] = int(data_2[i][0])
            data_2[i][1] = data_2[i][1].replace("A", "0")
            data_2[i][1] = data_2[i][1].replace("B", "0")
            data_2[i][1] = datetime.strptime(data_2[i][1], "%d.%m.%Y %H:%M:%S")

            data_2[i][2] = data_2[i][2].replace(" ", "")
            data_2[i][2] = data_2[i][2].replace(",", ".")
            data_2[i][2] = float(data_2[i][2])
        return data_2
    def convert_2(text):

        data_1 = text.split("\n")

        # data_1 = data_1[:100] 100:200 :2 крок 2      в обернену сторону -1
        data_2 = [d.split("\t") for d in data_1]

        for i in range(len(data_2)): # приведення з строкової з числа dо дати
            data_2[i][0] = datetime.strptime(data_2[i][0], "%d.%m.%Y") # cntrl D copypaste raw # alt click other data backspace
            data_2[i][1] = datetime.strptime(data_2[i][1], "%H:%M")
            data_2[i][2] = datetime.strptime(data_2[i][2], "%H:%M")
            data_2[i][3] = datetime.strptime(data_2[i][3], "%H:%M")

        return data_2


    # зчитування пошук файлів в папці
    listName = os.listdir(path)  # записуємо в змінну список файлів шляхи файлів закидаємо в список
    # print("all files:", listName)
    listName.remove("sunriseSunset.txt")  # видалення зі списку sunrise


    # відкриття файлів
    g = []  # пустий список відкритих файлів
    for name in listName:
        g.append(open(path+"/" + name))  # зчитуємо всі файли додаємо в список g
    sunrise = open("dataSet/sunriseSunset.txt")


    # зчитування файлів
    dataraw = []
    for gg in g:
        dataraw.append(gg.read())  # зчитуємо з відкритих даних та записуємо в датарав
        gg.close()  # відступи 4 пробіли важливі інакше не сприймає як в циклі


    # Записуємо голі текстові дані в змінну
    dataSort = []# створюємо список голих даних вони вже конвенртовані в дататайм флоат запсиані в список списків
    for d in dataraw:
        dataSort.append(convert_1(d)) # беремо дані кожного текстового файлу і за домомогою функції коверт перетворюємо їх в числа час дататайм
    dataSort.append(convert_2(sunrise.read())) # add sunrise
    sunrise.close()


    # конвертуємо данні із звичайних масивів в масиви даних типу dataFrame
    list_values = [
        "genaration",
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
    ]
    data = []
    for d,k in zip(dataSort[:-1], list_values): # exeption surise еріть той список беріть другий  порівнюйте і робіть один
        data.append(pd.DataFrame(d, columns=["number", "date_time", k]))
    data.append(pd.DataFrame(dataSort[-1], columns=["date_time", "sunrise", "sunset", "sunDuration"]))
    data[-1]["day"] = data[-1].apply(lambda row: row["date_time"].date(), axis=1)
    data[-1]["sunrise"] = data[-1].apply(lambda row: row["sunrise"].time(), axis=1)
    data[-1]["sunset"] = data[-1].apply(lambda row: row["sunset"].time(), axis=1)
    data[-1]["sunDuration"] = data[-1].apply(lambda row: row["sunDuration"].time(), axis=1)
    data[-1] = data[-1].drop(columns="date_time", axis=1)


    # консолідація в одну таблицю DataFrame
    dataConsolidated = data[0]
    dataConsolidated["time"] = dataConsolidated.apply(lambda row: row["date_time"].time(), axis=1)
    dataConsolidated["day"] = dataConsolidated.apply(lambda row: row["date_time"].date(), axis=1)
    for d in data[1:-1]:
        dataConsolidated =pd.merge(dataConsolidated, d.drop(columns=["number"]), how="left", on="date_time")
    dataConsolidated =pd.merge(dataConsolidated, data[-1], how="left", on="day")
    # print(dataConsolidated)


    #  Заповнення пустих дир
    for key in ['sunrise', 'sunset', 'sunDuration']:
        val = dataConsolidated[key].iloc[26329]
        print(val)
        for i in range(26330, 26378):
            dataConsolidated[key].iloc[i] = val


    # Додаємо float дату і час
    def date_time_to_float(val): return np.float64((val - datetime(2019, 7, 2)).days + (val.hour * 60+val.minute/(60*24)))
    def day_to_float(val): return np.float64((val - datetime(2019, 7, 2).date()).days)
    def time_to_float(val): return np.float64(val.hour+val.minute / 60)
    dataConsolidated["date_time_float"] = dataConsolidated['date_time'].apply(lambda val: date_time_to_float(val))
    dataConsolidated["time_float"] = dataConsolidated['time'].apply(lambda val: time_to_float(val))
    dataConsolidated["day_float"] = dataConsolidated['day'].apply(lambda val: day_to_float(val))
    dataConsolidated["sunrise_float"] = dataConsolidated['sunrise'].apply(lambda val: time_to_float(val))
    dataConsolidated["sunset_float"] = dataConsolidated['sunset'].apply(lambda val: time_to_float(val))
    dataConsolidated["sunDuration_float"] = dataConsolidated['sunDuration'].apply(lambda val: time_to_float(val))


    # Зберігаємо в файл
    # print(dataConsolidated.info())
    # dataConsolidated.to_csv("data.csv", sep=";")


    #  Зберігає в файл, але додає між рядками середнє арифметичне
    dataConsolidatedAver = pd.DataFrame(dataConsolidated)
    global counter
    counter = 0
    def average_all(row, x):
        global counter
        if not np.isnan(row):
            counter += 1
            return row

        min_ind = x.iloc[:counter].last_valid_index() or x.iloc[counter:].first_valid_index()
        max_ind = x.iloc[counter:].first_valid_index() or x.iloc[:counter].last_valid_index()
        min_val = x[min_ind]
        max_val = x[max_ind]

        if min_ind == max_ind:
            counter += 1
            return min_val
        else:
            dist = max_ind - min_ind
            number = counter - min_ind

            to_return = min_val + (max_val - min_val) * (number) / dist
            counter += 1
            return to_return
    for column in [
                    "dewpoint",
                    "downward_short_wave",
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
                    ]:
        counter = 0
        dataConsolidatedAver[column] = dataConsolidatedAver[column].apply(lambda row: average_all(row, dataConsolidatedAver[column]))
    dataConsolidatedAver.to_csv(save_name, sep=";")
    return dataConsolidatedAver
