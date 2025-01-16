import numpy as np
import os
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import allantools
import json
import pandas as pd
import logging
from tqdm import tqdm
import argparse
import glob

# Налаштування логування
logging.basicConfig(filename='evaluation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Шлях до папки з обробленими даними
processed_data_folder = "processed_data"

# Шлях до папки з експериментальними даними
experimental_data_folder = "experimental_data"

# Параметри
window_size = 100
dt = 1 / 108  # Частота дискретизації

# Моделі
models = {
    "LSTM": LSTMModel,
    "CNN": CNNModel,
    "CNNLSTM": CNNLSTMModel,
    "TCN": TCNModel,
    "GRU": GRUModel
}

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
            processed_data_path = os.path.join(model_path, "processed_data")
            if os.path.isdir(processed_data_path):
                for item_name in os.listdir(processed_data_path):
                    item_path = os.path.join(processed_data_path, item_name)
                    logging.info(f"  Item path: {item_path}")
                    print(f"  Item path: {item_path}")
                    # Перевіряємо, чи є елемент папкою
                    if os.path.isdir(item_path):
                        # Перевіряємо наявність файлів .keras та _params.json В СЕРЕДИНІ папки
                        keras_file = glob.glob(os.path.join(item_path, "*.keras"))
                        params_file = glob.glob(os.path.join(item_path, "*_params.json"))
                        if keras_file and params_file:
                            folders.append(item_path)

    logging.info(f"Found folders: {folders}")
    print(f"Found folders: {folders}")
    return folders


# Шляхи до файлів з експериментальними даними
experimental_file_paths = [
    os.path.join(experimental_data_folder, "ProgessiveOscill0.xlsx"),
    os.path.join(experimental_data_folder, "StableOscill0.xlsx"),
    os.path.join(experimental_data_folder, "1.xlsx"),
    os.path.join(experimental_data_folder, "2.xlsx")
]


# Функція для попередньої обробки даних (якщо потрібно)
def preprocess_data(df, sensor_type):
    # Приклад: видалення викидів (можна адаптувати під свої потреби)
    signal = df[f"{sensor_type}"]
    signal_mean = signal.mean()
    signal_std = signal.std()
    signal_filtered = signal[np.abs(signal - signal_mean) < 3 * signal_std]

    # Правильне присвоєння відфільтрованого сигналу колонці
    df[f"{sensor_type}"] = signal_filtered

    # Заповнення пропусків, якщо вони є
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# Функція для підготовки даних для моделі
def prepare_data_for_model(df, sensor_type, window_size):
    # Приклад: розбиття на вікна (можна адаптувати під свої потреби)
    X = []
    y = []
    for i in range(len(df) - window_size):
        X.append(df[f"{sensor_type}"].values[i:i + window_size])
        y.append(df[f"{sensor_type}"].values[i + window_size])
    return np.array(X), np.array(y)


if __name__ == '__main__':
    # Парсер аргументів командного рядка
    parser = argparse.ArgumentParser(description="Програма для оцінки моделей гіроскопа.")
    parser.add_argument("--models_dir", type=str, default="models", help="Шлях до папки з моделями")
    args = parser.parse_args()

    stage_sensor_folders = get_stage_sensor_folders(args.models_dir)
    print(f"Stage sensor folders: {stage_sensor_folders}")

    for model_name, ModelClass in models.items():
        logging.info(f"Evaluating {model_name}...")
        print(f"Evaluating {model_name}...")

        # Створення папки для результатів моделі
        model_results_dir = os.path.join(evaluation_results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        # DataFrame для збереження загальних метрик
        metrics_df = pd.DataFrame(
            columns=["File", "Stage", "Sensor", "MAE", "RMSE", "R2", "Bias Instability", "ARW", "RRW"])

        for folder in stage_sensor_folders:
            # Перевірка, чи належить папка поточному типу моделі та чи це папка (не файл)
            relative_path = os.path.relpath(folder, args.models_dir)
            print(f"    Relative path {relative_path}")
            if not relative_path.startswith(model_name) or not os.path.isdir(folder):
                print(f"    Пропускаємо {folder} бо не папка, або не модель")
                continue

            logging.info(f"  Processing {folder}...")
            print(f"  Processing {folder}...")

            # Отримуємо назву папки з даними
            data_folder_name = os.path.basename(folder)
            print(f"Data folder name: {data_folder_name}")

            # Видаляємо .keras з назви папки
            data_folder_name = data_folder_name.replace('.keras', '')

            # Розбиваємо назву папки на частини
            parts = data_folder_name.split("_")
            print(f"Parts of folder name: {parts}")
            if len(parts) < 3:
                logging.warning(f"    Skipping folder {folder} due to incorrect name format.")
                print(f"    Skipping folder {folder} due to incorrect name format.")
                continue

            # Отримуємо назву файлу з якого були взяті дані
            source_file_name = data_folder_name.replace(f"_{model_name}_{parts[-3]}_{parts[-2]}_{parts[-1]}", ".xlsx")
            file_name = source_file_name.split('.')[0]
            print(f"Source file name: {source_file_name}")
            print(f"File name: {file_name}")

            stage = parts[-2].replace('stage', '')
            sensor = parts[-1]
            print(f"Stage: {stage}, Sensor: {sensor}")

            for file_path in tqdm(experimental_file_paths,
                                  desc=f"Processing files for {model_name} - {stage} - {sensor}"):
                # Отримуємо ім'я файлу
                file_name = os.path.splitext(os.path.basename(file_path))[0]

                # Завантаження та попередня обробка даних
                try:
                    df = pd.read_excel(file_path)
                    df = preprocess_data(df, sensor)  # При необхідності, додай попередню обробку
                    X, y = prepare_data_for_model(df, sensor, window_size)
                except FileNotFoundError:
                    logging.error(f"    File not found: {file_path}")
                    print(f"    File not found: {file_path}")
                    continue
                except Exception as e:
                    logging.error(f"    Failed to load or preprocess data from {file_path}: {e}")
                    print(f"    Failed to load or preprocess data from {file_path}: {e}")
                    continue

                # Завантаження моделі та гіперпараметрів
                model_filename = f"{model_name}_{data_folder_name}"
                model_filepath = os.path.join(folder, model_filename)
                params_filepath = os.path.join(folder, f"{model_name}_{data_folder_name}_params.json")

                # Видаляємо розширення .keras з model_filepath
                model_filepath = model_filepath.replace('.keras', '')

                if not os.path.exists(params_filepath):
                    logging.warning(f"    Skipping {folder} as no hyperparameters file found.")
                    print(f"    Skipping {folder} as no hyperparameters file found.")
                    continue

                with open(params_filepath, "r") as f:
                    params = json.load(f)

                model = ModelClass(params["window_size"], 1)
                model.build()

                # Перевірка наявності файлу з вагами
                weights_path = f"{model_filepath}.keras"
                if os.path.exists(weights_path):
                    logging.info(f"    Loading existing weights from {weights_path}")
                    print(f"    Loading existing weights from {weights_path}")
                    model.load(weights_path)
                    print(f"Model loaded successfully from {weights_path}")
                else:
                    logging.warning(f"    No existing weights found for {model_filepath}. Model not loaded.")
                    print(f"    No existing weights found for {model_filepath}. Model not loaded.")
                    continue

                # Предикшн
                y_pred = model.predict(X).flatten()

                # Обрізаємо predicted_values до довжини time_values
                y_pred = y_pred[:len(y)]

                # Розрахунок метрик
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                r2 = r2_score(y, y_pred)

                # Розрахунок Allan Deviation
                rate = 1 / dt  # Частота дискретизації
                data_type = "freq"
                taus = 'all'

                try:
                    (t2, ad, ade, adn) = allantools.oadev(y, rate=rate, data_type=data_type, taus=taus)
                    (t2_pred, ad_pred, ade_pred, adn_pred) = allantools.oadev(y_pred, rate=rate, data_type=data_type,
                                                                              taus=taus)

                    # Знаходження мінімального значення Allan Deviation та відповідного йому tau (Bias Instability)
                    min_ad_index = np.argmin(ad_pred)
                    min_ad = ad_pred[min_ad_index]
                    min_tau = t2_pred[min_ad_index]

                    # Розрахунок Bias Instability (BI)
                    bi = ad_pred[min_ad_index] * 0.664  # bias instability coefficient

                    # Розрахунок ARW (приблизний метод)
                    arw = ad_pred[0] * 60  # Коефіцієнт перетворення з (rad/s)/sqrt(Hz) в (deg/sqrt(hr))

                    # Розрахунок RRW (приблизний метод)
                    rrw_index = np.argmin(np.abs(t2_pred - 3))  # Знаходимо індекс, найближчий до tau=3
                    rrw = ad_pred[rrw_index] * np.sqrt(rate / 3)

                except Exception as e:
                    logging.error(
                        f"    Error calculating Allan Deviation for {model_name} - {stage} - {sensor} - {file_name}: {e}")
                    print(
                        f"    Error calculating Allan Deviation for {model_name} - {stage} - {sensor} - {file_name}: {e}")
                    ad, ad_pred = None, None
                    min_ad, min_tau, bi, arw, rrw = None, None, None, None, None

                # Збереження результатів у DataFrame
                new_row = pd.DataFrame({
                    "File": [file_name],
                    "Stage": [stage],
                    "Sensor": [sensor],
                    "MAE": [mae],
                    "RMSE": [rmse],
                    "R2": [r2],
                    "Bias Instability": [bi],
                    "ARW": [arw],
                    "RRW": [rrw]
                })
                metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

                # Побудова графіка
                plt.figure(figsize=(12, 6))
                plt.plot(y, label="Actual")
                plt.plot(y_pred, label="Predicted")
                plt.title(f"{model_name} - {stage} - {sensor} - {file_name}")
                plt.legend()
                plt.savefig(os.path.join(model_results_dir, f"{model_filename}_{file_name}_plot.png"))
                plt.close()
                print(f"    Saved plot to {os.path.join(model_results_dir, f'{model_filename}_{file_name}_plot.png')}")

                # Побудова графіка Allan Deviation
                if ad is not None and ad_pred is not None:
                    plt.figure(figsize=(12, 6))
                    plt.loglog(t2, ad, label="Actual")
                    plt.loglog(t2_pred, ad_pred, label="Predicted")
                    plt.title(f"Allan Deviation - {model_name} - {stage} - {sensor} - {file_name}")
                    plt.xlabel("Tau (s)")
                    plt.ylabel("Allan Deviation")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(model_results_dir, f"{model_filename}_{file_name}_allan_deviation.png"))
                    plt.close()
                    print(
                        f"    Saved Allan Deviation plot to {os.path.join(model_results_dir, f'{model_filename}_{file_name}_allan_deviation.png')}")

            # Збереження загальних метрик у CSV файл
            metrics_filepath = os.path.join(model_results_dir, f"{model_name}_metrics.csv")
            metrics_df.to_csv(metrics_filepath, index=False)
            print(f"Saved metrics for {model_name} to {metrics_filepath}")