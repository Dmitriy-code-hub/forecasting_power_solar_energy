import consolidate
import analyze
import build_regr
import neural
import predict

data = consolidate.read_data()  # Зчитує і консолідує всі дані в один dataFrame щоб не було пробілів
analyze.analyze()  # Аналізує дані загалом
# test_data = consolidate.read_data(path="testDataSet", save_name="test_data.csv")   # fixme Дані колонок надані не коретно

regres_model, regres_columns_x, regres_result = build_regr.build_regr()  # будує модель
neural_model, neural_columns_x, neural_result = neural.neural()  # build neural network
predict.predict(None, regres_model, regres_columns_x, regres_result, neural_model, neural_columns_x, neural_result)  # Передбачає за вказаними параметрами

# perseptron
