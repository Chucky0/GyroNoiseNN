import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import allantools
import argparse
import glob

# Параметри
window_size = 100
dt = 1 / 108  # Частота дискретизації
num_processes = 4  # Кількість процесів (ядер)

# Моделі
models = {
    "LSTM": LSTMModel,
    "CNN": CNNModel,
    "CNNLSTM": CNNLSTMModel,
    "TCN": TCNModel,
    "GRU": GRUModel
}

# Шлях до папки з експериментальними даними та моделями
data_and_models_folder = "models_for_predict"

# Перевірка та створення папки evaluation_results
evaluation_results_dir = "predictions"
if not os.path.exists(evaluation_results_dir):
    os.makedirs(evaluation_results_dir)


def calculate_allan_deviation(data, rate, taus="octave"):
    """
    Розраховує Allan Deviation для часового ряду даних.
    """
    if len(data) < 2:
        print("Not enough data points to calculate Allan Deviation.")
        return None, None

    try:
        (t2, ad, ade, adn) = allantools.oadev(data, rate=rate, data_type="freq", taus=taus)
        return t2, ad
    except Exception as e:
        print(f"Error in Allan Deviation calculation: {e}")
        return None, None


def prepare_data_for_model(df, sensor_type, window_size):
    """
    Готує дані для подачі в модель, формуючи вікна заданого розміру.
    """
    X = np.lib.stride_tricks.sliding_window_view(df[f"{sensor_type}"].values, window_size)
    y = df[f"{sensor_type}"].values[window_size - 1:]
    return X, y


def get_stage_sensor_folders(base_dir):
    """
    Знаходить папки з моделями та їхніми конфігураційними файлами.
    Враховує структуру models_for_predict/models_with_predictions/<model_name>/<file_name>_stage_<stage>_<sensor>/
    """
    folders = []
    models_with_predictions_path = os.path.join(base_dir, "models_with_predictions")
    for model_name in models.keys():
        model_path = os.path.join(models_with_predictions_path, model_name)
        print(f"Model path: {model_path}")
        if os.path.isdir(model_path):
            for item_name in os.listdir(model_path):
                item_path = os.path.join(model_path, item_name)
                print(f"  Folder path: {item_path}")
                if os.path.isdir(item_path):
                    # Шукаємо файли .keras та _params.json
                    keras_file = glob.glob(os.path.join(item_path, f"{model_name}_*.keras*"))
                    params_file = glob.glob(os.path.join(item_path, f"{model_name}_*_params.json"))
                    if keras_file and params_file:
                        print(f"    Subfolder path: {item_path}\\{keras_file[0]}")
                        print(f"    Subfolder path: {item_path}\\{params_file[0]}")
                        folders.append(item_path)
    print(f"Found folders: {folders}")
    return folders


def predict_with_model(model, X, sensor, model_name, stage, file_name, model_full_name, params, df, window_size):
    """
    Робить передбачення за допомогою заданої моделі та зберігає результати.
    """
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y_pred = model.predict(X).flatten() # Видалено verbose=0

    predictions_df = pd.DataFrame()
    predictions_df['Time'] = df['Time'][window_size - 1:]
    predictions_df['Encoder Speed'] = df['Encoder Speed'][window_size - 1:]
    for sensor_col in ['NVGyro Z', 'N1Gyro Z', 'N8Gyro Z']:
        predictions_df[f'{sensor_col}_input'] = df[sensor_col][window_size - 1:]

    y_pred = y_pred[:len(df) - window_size + 1]
    predictions_df[f'{sensor}_{model_name}_stage_{stage}_predicted'] = y_pred
    predictions_df['Model'] = model_name
    predictions_df['Stage'] = stage
    predictions_df['File'] = file_name
    predictions_df['Model_Full_Name'] = model_full_name
    predictions_df['Hyperparameters'] = json.dumps(params)

    return predictions_df


def run_predictions(models_dir, experimental_data_folder, output_folder, window_size, dt):
    """
    Запускає передбачення для всіх моделей та файлів, зберігає результати в один CSV.
    """
    all_predictions = []
    sensors = ["N1Gyro Z", "N8Gyro Z", "NVGyro Z"]

    # Отримуємо список файлів з експериментальними даними безпосередньо з models_for_predict
    experimental_file_paths = [
        os.path.join(experimental_data_folder, f)
        for f in os.listdir(experimental_data_folder)
        if f.endswith(".xlsx")
    ]
    print(f"Experimental file paths: {experimental_file_paths}")

    # Отримуємо список моделей для обробки
    model_folders = get_stage_sensor_folders(models_dir)
    print(f"Model folders: {model_folders}")

    for file_path in tqdm(experimental_file_paths, desc="Processing files"):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"  Processing file: {file_name}.xlsx")

        df = pd.read_excel(file_path)
        if 'Time' not in df.columns:
            df['Time'] = np.arange(0, len(df) * dt, dt)

        for model_folder in model_folders:
            model_full_name = os.path.basename(model_folder)
            parts = model_full_name.split("_")
            model_name = parts[0]
            stage = parts[-2].replace('stage', '')
            sensor = parts[-1]
            source_file_name = "_".join(parts[1:-3])

            # Перевірка чи відповідає модель поточному файлу
            if not file_name.startswith(source_file_name):
                print(
                    f"      Skipping model {model_full_name} for file {file_name} as source file names do not match.")
                continue

            # Завантаження моделі та параметрів
            model_filepath = os.path.join(model_folder, f"{model_full_name}.keras")
            if not os.path.exists(model_filepath):
                # Додаткова перевірка на .keras.keras
                model_filepath = os.path.join(model_folder, f"{model_full_name}.keras.keras")

            params_filepath = os.path.join(model_folder, f"{model_full_name}_params.json")

            try:
                with open(params_filepath, "r") as f:
                    params = json.load(f)
            except FileNotFoundError:
                print(f"      Error: File not found: {params_filepath}")
                continue
            except json.JSONDecodeError:
                print(f"      Error: Invalid JSON in file: {params_filepath}")
                continue

            ModelClass = models[model_name]
            model = ModelClass(params["window_size"], 1)
            model.build()

            if os.path.exists(model_filepath):
                model.load(model_filepath)
                print(f"      Model loaded successfully from {model_filepath}")
            else:
                print(f"      No existing weights found for {model_filepath}. Model not loaded.")
                continue

            # Готуємо дані та робимо передбачення
            X, y = prepare_data_for_model(df, sensor, window_size)
            predictions_df = predict_with_model(model, X, sensor, model_name, stage, file_name,
                                                model_full_name, params, df, window_size)
            if predictions_df is not None:
                all_predictions.append(predictions_df)

    # Об'єднуємо передбачення
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        all_predictions_df.to_csv(os.path.join(output_folder, "all_predictions.csv"), index=False)
        print(f"Saved all predictions to {os.path.join(output_folder, 'all_predictions.csv')}")
    else:
        print("No predictions were made.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Програма для передбачень на основі моделей.")
    parser.add_argument("--models_dir", type=str, default="models_for_predict",
                        help="Шлях до папки з моделями та даними")
    parser.add_argument("--output_dir", type=str, default=evaluation_results_dir,
                        help="Шлях до папки для збереження результатів")

    args = parser.parse_args()

    # Перевірка наявності директорій
    if not os.path.exists(args.models_dir):
        print(f"Error: Directory not found: {args.models_dir}")
        exit()

    # Запуск передбачень
    run_predictions(args.models_dir, args.models_dir, args.output_dir, window_size, dt)