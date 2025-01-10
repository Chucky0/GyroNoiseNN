import numpy as np
import os
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Задайте шлях до папки з обробленими даними
processed_data_folder = "processed_data"

# Параметри
window_size = 100

# Моделі, які будуть використовуватись в ансамблі
model_classes = [LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel]
model_names = ["LSTM", "CNN", "CNNLSTM", "TCN", "GRU"]

# Словник для збереження результатів
ensemble_results = {}

# Отримання списку всіх папок з даними
stage_sensor_folders = [d for d in os.listdir(processed_data_folder) if os.path.isdir(os.path.join(processed_data_folder, d))]

for folder in stage_sensor_folders:
    print(f"Processing {folder} for ensemble...")
    stage, sensor = folder.split("_", 1)

    # Завантаження даних
    X_test = np.load(f"{processed_data_folder}/{folder}/X_test.npy")
    y_test = np.load(f"{processed_data_folder}/{folder}/y_test.npy")

    predictions = []
    for model_name, ModelClass in zip(model_names, model_classes):
        # Завантаження моделі
        model_filepath = f"{model_name}_{stage}_{sensor}.h5"
        model = ModelClass(window_size, 1)
        model.load(model_filepath)

        # Предикшн
        y_pred = model.predict(X_test).flatten()
        predictions.append(y_pred)

    # Усереднення прогнозів
    ensemble_prediction = np.mean(predictions, axis=0)

    # Розрахунок метрик
    mae = mean_absolute_error(y_test, ensemble_prediction)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_prediction))
    r2 = r2_score(y_test, ensemble_prediction)

    # Збереження результатів
    ensemble_results[f"{stage}_{sensor}"] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

# Збереження результатів у файл (наприклад, у форматі .npz)
np.savez("ensemble_results.npz", **ensemble_results)

# Виведення результатів
print("Ensemble Results:")
for stage_sensor, metrics in ensemble_results.items():
    print(f"{stage_sensor}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")