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


def predict_and_save_for_combination(model_name, file_name, stage, sensor, models_dir, output_dir):
    """
    Виконує передбачення для однієї комбінації моделі, файлу, етапу та сенсора.
    Зберігає результати в окремий CSV.
    """
    model_folder = os.path.join(models_dir, model_name, f"{file_name}_stage_{stage}_{sensor}")
    model_full_name = os.path.basename(model_folder)

    # Завантаження моделі та параметрів
    model_filepath = os.path.join(model_folder, f"{model_full_name}.keras")
    if not os.path.exists(model_filepath):
        model_filepath = os.path.join(model_folder, f"{model_full_name}.keras.keras")

    params_filepath = os.path.join(model_folder, f"{model_full_name}_params.json")

    try:
        with open(params_filepath, "r") as f:
            params = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {params_filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {params_filepath}")
        return None

    ModelClass = models[model_name]
    model = ModelClass(params["window_size"], 1)
    model.build()

    if os.path.exists(model_filepath):
        model.load(model_filepath)
        print(f"Model loaded successfully from {model_filepath}")
    else:
        print(f"No existing weights found for {model_filepath}. Model not loaded.")
        return None

    # Зчитування даних
    data_path = os.path.join("models_for_predict", f"{file_name}.xlsx")
    try:
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found: {data_path}")
        return None

    if 'Time' not in df.columns:
        df['Time'] = np.arange(0, len(df) * dt, dt)

    # Підготовка даних
    X, y = prepare_data_for_model(df, sensor, window_size)
    # Робимо передбачення
    predictions_df = predict_with_model(model, X, sensor, model_name, stage, file_name, model_full_name, params, df,
                                        window_size)

    # Зберігаємо predictions_df у окремий файл для кожної моделі
    output_file_name = f"{model_full_name}_{file_name}_predictions.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    predictions_df.to_csv(output_file_path, index=False)
    print(f"    Saved predictions to {output_file_path}")

    return predictions_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Програма для передбачень на основі моделей.")
    parser.add_argument("--project_root", type=str, default=".",
                        help="Шлях до кореневої папки проекту")
    args = parser.parse_args()

    # Перевірка наявності директорій
    project_root = args.project_root
    models_dir = os.path.join(project_root, "models_for_predict", "models_with_predictions")
    experimental_data_folder = os.path.join(project_root, "models_for_predict")
    output_dir = os.path.join(project_root, "predictions")

    if not os.path.exists(project_root):
        print(f"Error: Directory not found: {project_root}")
        exit()
    if not os.path.exists(models_dir):
        print(f"Error: Directory not found: {models_dir}")
        exit()
    if not os.path.exists(experimental_data_folder):
        print(f"Error: Directory not found: {experimental_data_folder}")
        exit()

    # Створюємо папку для збереження результатів
    os.makedirs(output_dir, exist_ok=True)

    # Отримуємо список моделей та файлів даних
    model_folders, experimental_files = find_models_and_data(project_root)

    # Всі комбінації файлів, моделей, стадій та сенсорів
    all_combinations = [
        (model_folder, experimental_file, output_dir)
        for experimental_file in experimental_files
        for model_folder in model_folders.values()
        if os.path.splitext(os.path.basename(experimental_file))[0] in os.path.basename(model_folder)
    ]

    # Запуск передбачень
    all_predictions = []
    for file_path, model_folder, output_dir in tqdm(all_combinations, desc="Processing combinations"):
        predictions_df = process_model_for_file((file_path, model_folder, output_dir))
        if predictions_df is not None:
            all_predictions.append(predictions_df)

    # Об'єднуємо передбачення
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        all_predictions_df.to_csv(os.path.join(output_dir, "all_predictions.csv"), index=False)
        print(f"Saved all predictions to {os.path.join(output_dir, 'all_predictions.csv')}")
    else:
        print("No predictions were made.")