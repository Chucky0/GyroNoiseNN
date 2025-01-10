import numpy as np
import os
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import allantools

# Задайте шлях до папки з обробленими даними
processed_data_folder = "processed_data"

# Параметри
window_size = 100

# Моделі
models = {
    "LSTM": LSTMModel,
    "CNN": CNNModel,
    "CNNLSTM": CNNLSTMModel,
    "TCN": TCNModel,
    "GRU": GRUModel
}

# Словник для збереження результатів
results = {}

# Отримання списку всіх папок з даними
stage_sensor_folders = [d for d in os.listdir(processed_data_folder) if
                        os.path.isdir(os.path.join(processed_data_folder, d))]

for model_name, ModelClass in models.items():
    print(f"Evaluating {model_name}...")
    results[model_name] = {}

    for folder in stage_sensor_folders:
        print(f"  Processing {folder}...")
        stage, sensor = folder.split("_", 1)

        # Завантаження даних
        X_test = np.load(f"{processed_data_folder}/{folder}/X_test.npy")
        y_test = np.load(f"{processed_data_folder}/{folder}/y_test.npy")

        # Завантаження моделі
        model_filepath = f"{model_name}_{stage}_{sensor}.h5"
        model = ModelClass(window_size, 1)
        model.load(model_filepath)

        # Предикшн
        y_pred = model.predict(X_test).flatten()

        # Розрахунок метрик
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Розрахунок Allan Deviation
        rate = 108  # Частота дискретизації
        data_type = "freq"
        taus = np.logspace(0, np.log10(len(y_test) / (2 * rate)), 100)

        try:
            (t2, ad, ade, adn) = allantools.oadev(y_test, rate=rate, data_type=data_type, taus=taus)
            (t2_pred, ad_pred, ade_pred, adn_pred) = allantools.oadev(y_pred, rate=rate, data_type=data_type, taus=taus)
        except Exception as e:
            print(f"Error calculating Allan Deviation for {model_name} - {stage} - {sensor}: {e}")
            ad, ad_pred = None, None  # або присвойте значення за замовчуванням

        # Збереження результатів
        results[model_name][f"{stage}_{sensor}"] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Allan Deviation": ad,
            "Allan Deviation Predicted": ad_pred,
        }

        # Побудова графіка
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title(f"{model_name} - {stage} - {sensor}")
        plt.legend()
        plt.savefig(f"{model_name}_{stage}_{sensor}_plot.png")
        plt.close()
        # Побудова графіка Allan Deviation
        if ad is not None and ad_pred is not None:
            plt.figure(figsize=(12, 6))
            plt.loglog(t2, ad, label="Actual")
            plt.loglog(t2_pred, ad_pred, label="Predicted")
            plt.title(f"Allan Deviation - {model_name} - {stage} - {sensor}")
            plt.xlabel("Tau (s)")
            plt.ylabel("Allan Deviation")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{model_name}_{stage}_{sensor}_allan_deviation.png")
            plt.close()

# Збереження результатів у файл (наприклад, у форматі .npz)
np.savez("evaluation_results.npz", **results)