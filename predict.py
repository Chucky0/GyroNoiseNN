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


# Папка для збереження результатів

evaluation_results_dir = "evaluation_results_with_integration_and_allan"

# Моделі
models = {
    "LSTM": LSTMModel,
    "CNN": CNNModel,
    "CNNLSTM": CNNLSTMModel,
    "TCN": TCNModel,
    "GRU": GRUModel
}

# Шлях до папки з експериментальними даними та моделями
data_and_models_folder = "models_for_predict"  # Змінено згідно нової структури

# Шляхи до файлів з експериментальними даними
experimental_file_paths = [
    os.path.join(data_and_models_folder, "ProgessiveOscill0.xlsx"),
    os.path.join(data_and_models_folder, "StableOscill0.xlsx"),
    os.path.join(data_and_models_folder, "ProgessiveOscill1.xlsx"),
    os.path.join(data_and_models_folder, "StableOscill1.xlsx")
]

# Папка для збереження результатів
output_folder = "evaluation_results_with_integration_and_allan"
os.makedirs(output_folder, exist_ok=True)


def calculate_allan_deviation(data, rate, taus="octave"):
    # ... (без змін) ...
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
    # ... (без змін) ...
    X = np.lib.stride_tricks.sliding_window_view(df[f"{sensor_type}"].values, window_size)
    y = df[f"{sensor_type}"].values[window_size - 1:]
    return X, y

def predict_with_model(model, X, df, window_size, sensor, model_name, stage, file_name, model_full_name, params):
    """
    Робить передбачення за допомогою заданої моделі та зберігає результати.
    """
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y_pred = model.predict(X, verbose=0).flatten()

    predictions_df = pd.DataFrame()
    predictions_df['Time'] = df['Time'][window_size - 1:]
    predictions_df['Encoder Speed'] = df['Encoder Speed'][window_size - 1:]
    for sensor_col in ['NVGyro Z', 'N1Gyro Z', 'N8Gyro Z']:
        predictions_df[f'{sensor_col}_input'] = df[sensor_col][window_size - 1:]

    y_pred = y_pred[:len(df) - window_size + 1]
    predictions_df[f'{sensor}_{model_name}_predicted'] = y_pred
    predictions_df['Model'] = model_name
    predictions_df['Stage'] = stage
    predictions_df['File'] = file_name
    predictions_df['Model_Full_Name'] = model_full_name
    predictions_df['Hyperparameters'] = json.dumps(params)

    return predictions_df

def run_predictions(models_dir, experimental_data_folder, output_folder, window_size, dt):
    """
    Запускає передбачення для всіх моделей та файлів, зберігає результати.
    """
    all_predictions = []
    sensors = ["N1Gyro Z", "N8Gyro Z", "NVGyro Z"]

    # Отримуємо список файлів з експериментальними даними безпосередньо з models_for_predict
    experimental_file_paths = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".xlsx")
    ]

    for model_name, ModelClass in models.items():
        print(f"Predicting with {model_name}...")
        model_base_path = os.path.join(models_dir, "models_with_predictions", model_name)

        for file_path in experimental_file_paths:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            print(f"  Processing file: {file_name}.xlsx")

            df = pd.read_excel(file_path)
            if 'Time' not in df.columns:
                df['Time'] = np.arange(0, len(df) * dt, dt)

            for stage in range(6):
                print(f"    Stage: {stage}")
                for sensor in sensors:
                    print(f"      Sensor: {sensor}")
                    model_folder = os.path.join(model_base_path, f"{model_name}_{file_name}_stage_{stage}_{sensor}")

                    if not os.path.exists(model_folder):
                        print(f"        Model folder not found: {model_folder}. Skipping.")
                        continue

                    # Завантаження моделі та параметрів
                    model_full_name = os.path.basename(model_folder)
                    model_filepath = os.path.join(model_folder, f"{model_full_name}.keras")
                    params_filepath = os.path.join(model_folder, f"{model_full_name}_params.json")

                    try:
                        with open(params_filepath, "r") as f:
                            params = json.load(f)
                    except FileNotFoundError:
                        print(f"        Error: File not found: {params_filepath}")
                        continue
                    except json.JSONDecodeError:
                        print(f"        Error: Invalid JSON in file: {params_filepath}")
                        continue

                    ModelClass = models[model_name]
                    model = ModelClass(params["window_size"], 1)
                    model.build()

                    if os.path.exists(model_filepath):
                        model.load(model_filepath)
                        print(f"        Model loaded successfully from {model_filepath}")
                    else:
                        print(f"        No existing weights found for {model_filepath}. Model not loaded.")
                        continue

                    # Готуємо дані та робимо передбачення
                    X, y = prepare_data_for_model(df, sensor, window_size)
                    predictions_df = predict_with_model(model, X, sensor, model_name, stage, file_name,
                                                       model_full_name, params, df, window_size)
                    # Перевірка на None перед додаванням до списку
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
    parser.add_argument("--models_dir", type=str, default="models_for_predict/models_with_predictions",
                        help="Шлях до папки з моделями та даними")
    parser.add_argument("--output_dir", type=str, default=evaluation_results_dir,
                        help="Шлях до папки для збереження результатів")

    args = parser.parse_args()

    # Перевірка наявності директорій
    if not os.path.exists(args.models_dir):
        print(f"Error: Directory not found: {args.models_dir}")
        exit()

    # Підготовка даних - тепер ми не готуємо дані заздалегідь, а робимо це всередині process_model_for_file
    # data_dict = prepare_all_data(window_size, dt)

    # Створюємо папку для збереження результатів
    os.makedirs(args.output_dir, exist_ok=True)

    # Запуск передбачень
    run_predictions(args.models_dir, data_and_models_folder, args.output_dir, window_size, dt)