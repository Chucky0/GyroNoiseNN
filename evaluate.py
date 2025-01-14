import numpy as np
import os
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import allantools
import json
import pandas as pd
import logging

# Налаштування логування
logging.basicConfig(filename='evaluation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

# Перевірка та створення папки evaluation_results
evaluation_results_dir = "evaluation_results"
if not os.path.exists(evaluation_results_dir):
    os.makedirs(evaluation_results_dir)

# Оновлений список папок, що сканує директорію models
def get_stage_sensor_folders(base_dir="models"):
    logging.info(f"Base directory for models: {base_dir}")
    print(f"Base directory for models: {base_dir}")
    folders = []
    for model_name in models.keys():
        model_path = os.path.join(base_dir, model_name)
        logging.info(f"Model path: {model_path}")
        print(f"Model path: {model_path}")
        if os.path.isdir(model_path):
            for folder_name in os.listdir(model_path):
                folder_path = os.path.join(model_path, folder_name)
                logging.info(f"Folder path: {folder_path}")
                print(f"Folder path: {folder_path}")
                if os.path.isdir(folder_path):
                    for subfolder_name in os.listdir(folder_path):
                        subfolder_path = os.path.join(folder_path, subfolder_name)
                        print(f"Subfolder path: {subfolder_path}")
                        if os.path.isdir(subfolder_path):
                            folders.append(subfolder_path)
    logging.info(f"Found folders: {folders}")
    print(f"Found folders: {folders}")
    return folders

stage_sensor_folders = get_stage_sensor_folders()
print(f"Stage sensor folders: {stage_sensor_folders}")

for model_name, ModelClass in models.items():
    logging.info(f"Evaluating {model_name}...")
    print(f"Evaluating {model_name}...")

    # Створення папки для результатів моделі
    model_results_dir = os.path.join(evaluation_results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    # DataFrame для збереження загальних метрик
    metrics_df = pd.DataFrame(columns=["File", "Stage", "Sensor", "MAE", "RMSE", "R2", "Bias Instability", "ARW", "RRW"])

    for folder in stage_sensor_folders:
        # Перевірка, чи належить папка поточному типу моделі та чи це папка (не файл)
        relative_path = os.path.relpath(folder, "models")
        print (f"    Relative path {relative_path}")
        if not relative_path.startswith(model_name) or not os.path.isdir(folder):
          print (f"    Пропускаємо {folder} бо не папка, або не модель")
          continue

        logging.info(f"  Processing {folder}...")
        print(f"  Processing {folder}...")

        # Отримуємо назву папки з даними
        data_folder_name = os.path.basename(folder)
        print(f"Data folder name: {data_folder_name}")

        # Розбиваємо назву папки на частини
        parts = data_folder_name.split("_")
        print(f"Parts of folder name: {parts}")
        if len(parts) < 3:
            logging.warning(f"    Skipping folder {folder} due to incorrect name format.")
            print(f"    Skipping folder {folder} due to incorrect name format.")
            continue

        # Отримуємо назву файлу з якого були взяті дані
        source_file_name = data_folder_name.replace(f"_stage_{parts[-2]}_{parts[-1]}", ".xlsx")
        file_name = source_file_name.split('.')[0]
        print(f"Source file name: {source_file_name}")
        print(f"File name: {file_name}")

        stage = parts[-2].replace('stage', '')
        sensor = parts[-1]
        print(f"Stage: {stage}, Sensor: {sensor}")

        # Формуємо шлях до папки з обробленими даними в processed_data
        folder_path = os.path.join(processed_data_folder, data_folder_name)
        folder_path = folder_path.replace(f'\\{model_name}', '')
        print(f"Full path to data folder: {folder_path}")

        # Перевірка наявності даних
        x_test_path = os.path.join(folder_path, "X_test.npy")
        y_test_path = os.path.join(folder_path, "y_test.npy")
        print(f"  Checking for X_test.npy at: {x_test_path}")
        if not os.path.exists(x_test_path):
            logging.warning(f"    Skipping {folder} as X_test.npy not found.")
            print(f"    Skipping {folder} as X_test.npy not found.")
            continue
        print(f"  Checking for y_test.npy at: {y_test_path}")
        if not os.path.exists(y_test_path):
          logging.warning(f"    Skipping {folder} as y_test.npy not found.")
          print(f"    Skipping {folder} as y_test.npy not found.")
          continue

        # Завантаження даних
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        if X_test.shape[0] == 0 or y_test.shape[0] == 0:
            logging.warning("    Error: X_test or y_test is empty")
            print("    Error: X_test or y_test is empty")
            continue

        # Завантаження моделі
        model_filename = f"{model_name}_{data_folder_name}"
        model_filepath = os.path.join(folder, model_filename)
        print(f"Model file path: {model_filepath}")

        # Завантаження гіперпараметрів
        params_filepath = f"{model_filepath}_params.json"
        print(f"Params file path: {params_filepath}")
        if not os.path.exists(params_filepath):
            logging.warning(f"    Skipping {folder} as no hyperparameters file found.")
            print(f"    Skipping {folder} as no hyperparameters file found.")
            continue

        with open(params_filepath, "r") as f:
            params = json.load(f)

        # Завантаження оригінальних даних для колонки Time
        original_data_path = os.path.join(processed_data_folder,  source_file_name)
        print(f"Path to original data: {original_data_path}")

        if not os.path.exists(original_data_path):
            logging.error(f"    Source file not found: {original_data_path}")
            print(f"    Source file not found: {original_data_path}")
            continue

        try:
            original_df = pd.read_excel(original_data_path)
            time_values = original_df['Time'].values
        except FileNotFoundError:
            logging.error(f"Source file not found: {original_data_path}")
            continue
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            continue

        time_values_test = time_values[-len(y_test):]
        print(f"Time values: {time_values_test[:10]}")

        model = ModelClass(params["window_size"], 1)
        model.build()

        # Перевірка наявності файлу з вагами
        if os.path.exists(f"{model_filepath}.keras"):
            logging.info(f"    Loading existing weights from {model_filepath}.keras")
            print(f"    Loading existing weights from {model_filepath}.keras")
            model.load(f"{model_filepath}.keras")
            print(f"Model loaded successfully from {model_filepath}.keras")
        else:
            logging.warning(f"    No existing weights found for {model_filepath}. Model not loaded.")
            print(f"    No existing weights found for {model_filepath}. Model not loaded.")
            continue

        # Предикшн
        y_pred = model.predict(X_test).flatten()
        print(f"y_pred shape: {y_pred.shape}")

        if y_pred.shape[0] == 0:
            logging.warning("    Error: y_pred is empty")
            print("    Error: y_pred is empty")
            continue

        print(f"y_test: {y_test[:10]}")
        print(f"y_pred: {y_pred[:10]}")

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

            # Знаходження мінімального значення Allan Deviation та відповідного йому tau (Bias Instability)
            bias_instability = np.min(ad_pred)
            optimal_tau_index = np.argmin(ad_pred)
            optimal_tau = t2_pred[optimal_tau_index]

            # Розрахунок ARW (приблизний метод)
            arw = ad_pred[0] * np.sqrt(rate)  # Приблизний розрахунок ARW

            # Розрахунок RRW (приблизний метод)
            rrw_index = np.argmin(np.abs(t2_pred - 3))  # Знаходимо індекс, найближчий до tau=3
            rrw = ad_pred[rrw_index] * np.sqrt(rate / 3)

        except Exception as e:
            logging.error(f"    Error calculating Allan Deviation for {model_name} - {stage} - {sensor}: {e}")
            print(f"    Error calculating Allan Deviation for {model_name} - {stage} - {sensor}: {e}")
            ad, ad_pred = None, None

        # Збереження результатів у DataFrame
        new_row = pd.DataFrame({
            "File": [file_name],
            "Stage": [stage],
            "Sensor": [sensor],
            "MAE": [mae],
            "RMSE": [rmse],
            "R2": [r2],
            "Bias Instability": [bias_instability],
            "ARW": [arw],
            "RRW": [rrw]
        })
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        # Побудова графіка
        plt.figure(figsize=(12, 6))
        plt.plot(time_values_test, y_test, label="Actual")
        plt.plot(time_values_test, y_pred, label="Predicted")
        plt.title(f"{model_name} - {stage} - {sensor}")
        plt.legend()
        plt.savefig(os.path.join(model_results_dir, f"{model_filename}_plot.png")) # Змінено шлях до папки
        plt.close()
        print(f"    Saved plot to {os.path.join(model_results_dir, f'{model_filename}_plot.png')}")

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
            plt.savefig(os.path.join(model_results_dir, f"{model_filename}_allan_deviation.png"))# Змінено шлях до папки
            plt.close()
            print(f"    Saved Allan Deviation plot to {os.path.join(model_results_dir, f'{model_filename}_allan_deviation.png')}")

    # Збереження загальних метрик у CSV файл
    metrics_filepath = os.path.join(model_results_dir, f"{model_name}_metrics.csv")
    print(metrics_df)
    metrics_df.to_csv(metrics_filepath, index=False)
    print(f"Saved metrics for {model_name} to {metrics_filepath}")