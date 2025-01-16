import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import allantools
from scipy.integrate import cumtrapz
import argparse
import glob
import logging

# Налаштування логування
logging.basicConfig(filename='evaluation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Шлях до папки з експериментальними даними
experimental_data_folder = "experimental_data"

# Параметри
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
    signal = df[f"{sensor_type}Gyro Z"]
    signal_mean = signal.mean()
    signal_std = signal.std()
    signal_filtered = signal[np.abs(signal - signal_mean) < 3 * signal_std]

    # Правильне присвоєння відфільтрованого сигналу колонці
    df[f"{sensor_type}Gyro Z"] = signal_filtered

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
        X.append(df[f"{sensor_type}Gyro Z"].values[i:i + window_size])
        y.append(df[f"{sensor_type}Gyro Z"].values[i + window_size])
    return np.array(X), np.array(y)

def calculate_and_save_metrics(y_true, y_pred, file_name, stage, sensor, model_name, model_results_dir):
    # Розрахунок метрик
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Логування та вивід метрик
    logging.info(f"    Metrics for {model_name} - {file_name} - {stage} - {sensor}:")
    logging.info(f"      MAE: {mae:.4f}")
    logging.info(f"      RMSE: {rmse:.4f}")
    logging.info(f"      R2: {r2:.4f}")
    print(f"    Metrics for {model_name} - {file_name} - {stage} - {sensor}:")
    print(f"      MAE: {mae:.4f}")
    print(f"      RMSE: {rmse:.4f}")
    print(f"      R2: {r2:.4f}")

    return mae, rmse, r2

def calculate_allan_deviation_metrics(y_true, rate, data_type="freq", taus="all"):
    """
    Розраховує метрики на основі Allan Deviation: RND, BI, ARW, RRW.
    """
    try:
        (t2, ad, ade, adn) = allantools.oadev(y_true, rate=rate, data_type=data_type, taus=taus)

        # Знаходження мінімального значення Allan Deviation та відповідного йому tau (Bias Instability)
        min_ad_index = np.argmin(ad)
        min_ad = ad[min_ad_index]
        min_tau = t2[min_ad_index]

        # Розрахунок Bias Instability (BI)
        bi = ad[min_ad_index] * 0.664

        # Розрахунок Angle Random Walk (ARW)
        arw = ad[0] * 60

        # Розрахунок Rate Random Walk (RRW)
        rrw_index = np.argmin(np.abs(t2 - 3))  # Знаходимо індекс, найближчий до tau=3
        rrw = ad[rrw_index] * np.sqrt(rate / 3)

        return min_ad, min_tau, np.mean(ad), bi, arw, rrw, t2, ad

    except Exception as e:
        logging.error(f"Error calculating Allan Deviation: {e}")
        print(f"Error calculating Allan Deviation: {e}")
        return None, None, None, None, None, None, None, None

def create_allan_deviation_plot(t2, ad, t2_pred, ad_pred, rnd, bi, arw, min_ad, min_tau, model_name, file_name, stage, sensor, results_dir):
    """
    Створює та зберігає графік Allan Deviation з анотаціями метрик.
    """
    plt.figure(figsize=(12, 6))
    plt.loglog(t2, ad, label="Actual")
    plt.loglog(t2_pred, ad_pred, label="Predicted")
    plt.title(f"Allan Deviation - {model_name} - {stage} - {sensor} - {file_name}")
    plt.xlabel("Tau (s)")
    plt.ylabel("Allan Deviation")
    plt.legend()
    plt.grid(True)

    # Додавання анотацій з метриками
    plt.text(0.1, 0.8, f"RND: {rnd:.4e} rad/s", transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"BI: {bi:.4e} rad/s", transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"ARW: {arw:.4e} deg/√hr", transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Min AD: {min_ad:.4e} rad/s at tau={min_tau:.1f} s", transform=plt.gca().transAxes)

    # Збереження графіка
    allan_deviation_plot_filepath = os.path.join(results_dir, f"{model_name}_{file_name}_stage_{stage}_{sensor}_allan_deviation_plot.png")
    plt.savefig(allan_deviation_plot_filepath)
    plt.close()
    print(f"    Saved Allan Deviation plot to {allan_deviation_plot_filepath}")

# Оновлений список папок, що сканує директорію models
def get_model_folders_and_info(base_dir="models"):
    model_info = []
    for model_name in models.keys():
        model_path = os.path.join(base_dir, model_name)
        if os.path.isdir(model_path):
            processed_data_path = os.path.join(model_path, "processed_data")
            if os.path.isdir(processed_data_path):
                for item_name in os.listdir(processed_data_path):
                    item_path = os.path.join(processed_data_path, item_name)
                    if os.path.isdir(item_path):
                        keras_file = glob.glob(os.path.join(item_path, "*.keras"))
                        params_file = glob.glob(os.path.join(item_path, "*_params.json"))
                        if keras_file and params_file:
                            # Розбиваємо назву папки на частини для отримання інформації
                            parts = item_name.split("_")
                            if len(parts) >= 5:
                                file_name = parts[1]
                                stage = parts[3]
                                sensor_type = parts[4].replace(".keras", "")  # Видаляємо .keras
                                model_info.append(
                                    {
                                        "model_name": model_name,
                                        "folder": item_path,
                                        "file_name": file_name,
                                        "stage": stage,
                                        "sensor": sensor_type
                                    }
                                )
    return model_info

if __name__ == '__main__':
    # Парсер аргументів командного рядка
    parser = argparse.ArgumentParser(description="Програма для оцінки моделей гіроскопа.")
    parser.add_argument("--models_dir", type=str, default="models", help="Шлях до папки з моделями")
    args = parser.parse_args()

    # Отримуємо список папок з моделями та інформацію про них
    model_folders_info = get_model_folders_and_info(args.models_dir)
    total_tasks = len(model_folders_info) * len(experimental_file_paths)

    # Створюємо загальний DataFrame для збереження всіх результатів
    all_metrics_df = pd.DataFrame(columns=["Model", "File", "Stage", "Sensor", "MAE", "RMSE", "R2", "Bias Instability", "ARW", "RRW"])

    with tqdm(total=total_tasks, desc="Загальний прогрес") as progress_bar:
        # Цикл по моделям, папкам, та файлам
        for model_info in model_folders_info:
            model_name = model_info["model_name"]
            folder = model_info["folder"]
            file_name = model_info["file_name"] # Використовуємо file_name з model_info
            stage = model_info["stage"]
            sensor = model_info["sensor"]
            ModelClass = models[model_name]

            # Створення папки для результатів моделі
            model_results_dir = os.path.join(evaluation_results_dir, model_name)
            os.makedirs(model_results_dir, exist_ok=True)

            for experimental_file_path in experimental_file_paths:
              # Отримуємо ім'я експериментального файлу
              experimental_file_name = os.path.splitext(os.path.basename(experimental_file_path))[0]

              # Завантаження та попередня обробка даних
              try:
                  df = pd.read_excel(experimental_file_path)
                  df = preprocess_data(df, sensor.replace('Gyro Z', ''))
                  X, y = prepare_data_for_model(df, sensor.replace('Gyro Z', ''), window_size)
                  time_values = df['Time'].iloc[window_size:].values

                  # Зберігаємо оригінальні дані
                  original_data_filepath = os.path.join(model_results_dir,
                                                        f"{model_name}_{experimental_file_name}_stage_{stage}_{sensor}_original_data.csv")
                  df.to_csv(original_data_filepath, index=False)
                  print(f"    Saved original data to {original_data_filepath}")

              except FileNotFoundError:
                  logging.error(f"    File not found: {experimental_file_path}")
                  print(f"    File not found: {experimental_file_path}")
                  continue
              except Exception as e:
                  logging.error(f"    Failed to load or preprocess data from {experimental_file_path}: {e}")
                  print(f"    Failed to load or preprocess data from {experimental_file_path}: {e}")
                  continue

              # Завантаження моделі та гіперпараметрів
              model_filename = os.path.basename(folder).replace('.keras', '')
              model_filepath = os.path.join(folder, model_filename)
              params_filepath = os.path.join(folder, f"{model_filename}_params.json")

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

              # Обрізаємо predicted_values до довжини y
              y_pred = y_pred[:len(y)]

              # Розрахунок метрик
              mae = mean_absolute_error(y, y_pred)
              rmse = np.sqrt(mean_squared_error(y, y_pred))
              r2 = r2_score(y, y_pred)

              # Зберігаємо передбачення у CSV
              predictions_df = pd.DataFrame({
                  "Time": df['Time'].iloc[window_size:],
                  "Actual": y,
                  "Predicted": y_pred
              })
              predictions_filepath = os.path.join(model_results_dir,
                                                  f"{model_name}_{experimental_file_name}_stage_{stage}_{sensor}_predictions.csv")
              predictions_df.to_csv(predictions_filepath, index=False)
              print(f"    Saved predictions to {predictions_filepath}")

              # Розрахунок та збереження Allan Deviation
              min_ad, min_tau, rnd, bi, arw, rrw, t2, ad = calculate_allan_deviation_metrics(y,
                                                                                               rate=1 / dt)

              # Розрахунок та збереження компонент декомпозиції
              calculate_and_save_seasonal_decomposition(y, df['Time'].iloc[window_size:], file_name, stage,
                                                        sensor, model_name, model_results_dir)

              # Розрахунок та збереження ACF/PACF
              calculate_and_save_acf_pacf(y, file_name, stage, sensor, model_name, model_results_dir)

              # Збереження метрик у DataFrame
              new_row = pd.DataFrame({
                  "File": [experimental_file_name],
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
              plt.plot(df['Time'].iloc[window_size:], y, label="Actual")
              plt.plot(df['Time'].iloc[window_size:], y_pred, label="Predicted")
              plt.title(f"{model_name} - {stage} - {sensor} - {experimental_file_name}")
              plt.legend()
              plt.savefig(os.path.join(model_results_dir, f"{model_filename}_{experimental_file_name}_plot.png"))
              plt.close()
              print(
                  f"    Saved plot to {os.path.join(model_results_dir, f'{model_filename}_{experimental_file_name}_plot.png')}")
