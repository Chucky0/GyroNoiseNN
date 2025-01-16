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
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# Шлях до папки з експериментальними даними
experimental_data_folder = "experimental_data"

# Шляхи до файлів з експериментальними даними
experimental_file_paths = [
    os.path.join(experimental_data_folder, "ProgessiveOscill0.xlsx"),
    os.path.join(experimental_data_folder, "StableOscill0.xlsx"),
    os.path.join(experimental_data_folder, "1.xlsx"),
    os.path.join(experimental_data_folder, "2.xlsx")
]

# Налаштування логування
logging.basicConfig(filename='evaluation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

# Далі йде решта коду (функції calculate_and_save_metrics, calculate_allan_deviation_metrics, create_allan_deviation_plot, get_model_folders_and_info, calculate_and_save_acf_pacf, calculate_and_save_seasonal_decomposition, if __name__ == '__main__':)
def calculate_and_save_metrics(y_true, y_pred, file_name, stage, sensor, model_name, model_results_dir):
    """
    Розраховує та зберігає метрики MAE, RMSE, R2.
    """
    try:
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

        # Збереження метрик у CSV
        metrics_df = pd.DataFrame({
            "File": [file_name],
            "Stage": [stage],
            "Sensor": [sensor],
            "Model": [model_name],
            "MAE": [mae],
            "RMSE": [rmse],
            "R2": [r2]
        })

        metrics_filepath = os.path.join(model_results_dir, f"{model_name}_metrics.csv")

        # Додавання метрик до існуючого файлу, якщо він вже існує
        if os.path.exists(metrics_filepath):
            existing_metrics_df = pd.read_csv(metrics_filepath)
            metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)

        metrics_df.to_csv(metrics_filepath, index=False)
        print(f"    Saved metrics to {metrics_filepath}")

        return mae, rmse, r2

    except Exception as e:
        logging.error(f"    Error during metrics calculation for {model_name} - {file_name} - {stage} - {sensor}: {e}")
        print(f"    Error during metrics calculation for {model_name} - {file_name} - {stage} - {sensor}: {e}")
        return None, None, None

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

def create_allan_deviation_plot(t2, ad, t2_pred, ad_pred, rnd, bi, arw, min_ad, min_tau, model_name, file_name,
                                        stage, sensor, results_dir):
            """
            Створює та зберігає графік Allan Deviation з анотаціями метрик.
            """
            plt.figure(figsize=(12, 6))
            plt.loglog(t2, ad, label="Actual")
            plt.loglog(t2_pred, ad_pred, label="Predicted")
            plt.title(f"Allan Deviation - {model_name} - {stage} - {sensor} - {file_name}")
            plt.xlabel("Tau (s)")
            plt.ylabel("Allan Deviation (rad/s)")
            plt.legend()
            plt.grid(True)

            # Додавання анотацій з метриками
            plt.text(0.1, 0.9, f"RND: {rnd:.4e} rad/s", transform=plt.gca().transAxes)
            plt.text(0.1, 0.8, f"BI: {bi:.4e} rad/s", transform=plt.gca().transAxes)
            plt.text(0.1, 0.7, f"ARW: {arw:.4e} deg/√hr", transform=plt.gca().transAxes)
            plt.text(0.1, 0.6, f"Min AD: {min_ad:.4e} rad/s at tau={min_tau:.1f} s", transform=plt.gca().transAxes)

            # Збереження графіка
            allan_deviation_plot_filepath = os.path.join(results_dir, "png", model_name, file_name, sensor,
                                                         f"{stage}_allan_deviation_plot.png")
            os.makedirs(os.path.dirname(allan_deviation_plot_filepath),
                        exist_ok=True)  # Створення директорії, якщо її немає
            plt.savefig(allan_deviation_plot_filepath)
            plt.close()
            print(f"    Saved Allan Deviation plot to {allan_deviation_plot_filepath}")

def calculate_and_save_seasonal_decomposition(y_true, time_values, file_name, stage, sensor, model_name, model_results_dir):
    """
    Розраховує та зберігає результати сезонної декомпозиції (тренд, сезонність, залишок).
    """
    try:
        # Виконуємо декомпозицію на сезонну, трендову та залишкову компоненти
        decomposition = seasonal_decompose(y_true, model='additive', period=500)

        # Отримуємо компоненти
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        # Зберігаємо дані у CSV
        decomposition_df = pd.DataFrame({
            "Time": time_values,
            "Trend": trend,
            "Seasonal": seasonal,
            "Residual": residual
        })
        decomposition_filepath = os.path.join(model_results_dir,
                                              f"{model_name}_{file_name}_stage_{stage}_{sensor}_decomposition.csv")
        decomposition_df.to_csv(decomposition_filepath, index=False)
        print(f"    Saved decomposition data to {decomposition_filepath}")

        # Побудова графіків
        fig_decomp = plt.figure(figsize=(12, 8))

        plt.subplot(4, 1, 1)
        plt.plot(time_values, y_true, label='Original')
        plt.legend(loc='best')
        plt.title(f"Original - {model_name} - {stage} - {sensor} - {file_name}")

        plt.subplot(4, 1, 2)
        plt.plot(time_values, trend, label='Trend')
        plt.legend(loc='best')
        plt.title(f"Trend - {model_name} - {stage} - {sensor} - {file_name}")

        plt.subplot(4, 1, 3)
        plt.plot(time_values, seasonal, label='Seasonality')
        plt.legend(loc='best')
        plt.title(f"Seasonality - {model_name} - {stage} - {sensor} - {file_name}")

        plt.subplot(4, 1, 4)
        plt.plot(time_values, residual, label='Residuals')
        plt.legend(loc='best')
        plt.title(f"Residuals - {model_name} - {stage} - {sensor} - {file_name}")

        plt.tight_layout()
        decomposition_plot_filepath = os.path.join(model_results_dir,
                                                  f"{model_name}_{file_name}_stage_{stage}_{sensor}_decomposition_plot.png")
        plt.savefig(decomposition_plot_filepath)
        plt.close(fig_decomp)
        print(f"    Saved decomposition plot to {decomposition_plot_filepath}")

    except Exception as e:
        logging.error(
            f"    Error during seasonal decomposition for {model_name} - {stage} - {sensor} - {file_name}: {e}")
        print(f"    Error during seasonal decomposition for {model_name} - {stage} - {sensor} - {file_name}: {e}")

def calculate_and_save_acf_pacf(y_true, file_name, stage, sensor, model_name, model_results_dir):
    """
    Розраховує та зберігає автокореляційну функцію (ACF) та часткову автокореляційну функцію (PACF).
    """
    try:
        # Автокореляційна функція (ACF)
        acf_values = sm.tsa.stattools.acf(y_true, nlags=int(len(y_true) * 0.1), fft=True)

        # Зберігання даних ACF
        acf_df = pd.DataFrame({"Lag": range(len(acf_values)), "ACF": acf_values})
        acf_df["File"] = file_name
        acf_df["Stage"] = stage
        acf_df["Sensor"] = sensor
        acf_df["Model"] = model_name
        acf_filepath = os.path.join(model_results_dir,
                                    f"{model_name}_{file_name}_stage_{stage}_{sensor}_acf.csv")
        acf_df.to_csv(acf_filepath, index=False)
        print(f"    Saved ACF data to {acf_filepath}")

        # Побудова графіку ACF
        fig_acf = plt.figure(figsize=(12, 6))
        sm.graphics.tsa.plot_acf(y_true, lags=int(len(y_true) * 0.1),
                                 title=f"ACF - {model_name} - {sensor} - {stage} - {file_name}")
        plt.xlabel("Lag")
        plt.ylabel("ACF")
        plt.grid(True)
        acf_plot_filepath = os.path.join(model_results_dir,
                                         f"{model_name}_{file_name}_stage_{stage}_{sensor}_acf_plot.png")
        plt.savefig(acf_plot_filepath)
        plt.close(fig_acf)
        print(f"    Saved ACF plot to {acf_plot_filepath}")

        # Часткова автокореляційна функція (PACF)
        pacf_values = sm.tsa.stattools.pacf(y_true, nlags=int(len(y_true) * 0.1), method='ywm')

        # Зберігання даних PACF
        pacf_df = pd.DataFrame({"Lag": range(len(pacf_values)), "PACF": pacf_values})
        pacf_df["File"] = file_name
        pacf_df["Stage"] = stage
        pacf_df["Sensor"] = sensor
        pacf_df["Model"] = model_name
        pacf_filepath = os.path.join(model_results_dir,
                                     f"{model_name}_{file_name}_stage_{stage}_{sensor}_pacf.csv")
        pacf_df.to_csv(pacf_filepath, index=False)
        print(f"    Saved PACF data to {pacf_filepath}")

        # Побудова графіку PACF
        fig_pacf = plt.figure(figsize=(12, 6))
        sm.graphics.tsa.plot_pacf(y_true, lags=int(len(y_true) * 0.1),
                                  title=f"PACF - {model_name} - {sensor} - {stage} - {file_name}", method='ywm')
        plt.xlabel("Lag")
        plt.ylabel("PACF")
        plt.grid(True)
        pacf_plot_filepath = os.path.join(model_results_dir,
                                          f"{model_name}_{file_name}_stage_{stage}_{sensor}_pacf_plot.png")
        plt.savefig(pacf_plot_filepath)
        plt.close(fig_pacf)
        print(f"    Saved PACF plot to {pacf_plot_filepath}")

    except Exception as e:
        logging.error(
            f"    Error during ACF/PACF calculation for {model_name} - {stage} - {sensor} - {file_name}: {e}")
        print(f"    Error during ACF/PACF calculation for {model_name} - {stage} - {sensor} - {file_name}: {e}")

# ... (Попередні функції: preprocess_data, prepare_data_for_model, calculate_and_save_metrics, calculate_allan_deviation_metrics, create_allan_deviation_plot, calculate_and_save_seasonal_decomposition, calculate_and_save_acf_pacf)

def get_model_folders_and_info(base_dir="models"):
    """
    Повертає список словників з інформацією про моделі, включаючи назву моделі, шлях до папки, назву файлу, етап та сенсор.
    """
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
                                file_name = parts[1] # Змінено індекс
                                stage = parts[3]
                                sensor_type = parts[4].replace(".keras", "")
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
    all_metrics_df = pd.DataFrame(
        columns=["Model", "File", "Stage", "Sensor", "MAE", "RMSE", "R2", "Bias Instability", "ARW", "RRW"])

    with tqdm(total=total_tasks, desc="Загальний прогрес") as progress_bar:
        # Цикл по моделям, папкам, та файлам
        for model_info in model_folders_info:
            model_name = model_info["model_name"]
            folder = model_info["folder"]
            file_name = model_info["file_name"]
            stage = model_info["stage"]
            sensor = model_info["sensor"]
            ModelClass = models[model_name]

            # Створення папки для результатів моделі
            model_results_dir = os.path.join(evaluation_results_dir, model_name)
            os.makedirs(model_results_dir, exist_ok=True)

            # DataFrame для збереження метрик для поточної моделі
            model_metrics_df = pd.DataFrame(
                columns=["File", "Stage", "Sensor", "MAE", "RMSE", "R2", "Bias Instability", "ARW", "RRW"])

            for experimental_file_path in experimental_file_paths:
                # Отримуємо ім'я експериментального файлу
                experimental_file_name = os.path.splitext(os.path.basename(experimental_file_path))[0]

                # Завантаження та попередня обробка даних
                try:
                    df = pd.read_excel(experimental_file_path)
                    df_processed = preprocess_data(df, sensor.replace('Gyro Z', ''))
                    X, y = prepare_data_for_model(df_processed, sensor.replace('Gyro Z', ''), window_size)
                    time_values = df['Time'].iloc[window_size:].values

                    # Зберігаємо оригінальні дані
                    original_data_filepath = os.path.join(model_results_dir,
                                                          f"{model_name}_{experimental_file_name}_stage_{stage}_{sensor}_original_data.csv")
                    df.to_csv(original_data_filepath, index=False)
                    print(f"    Saved original data to {original_data_filepath}")

                except FileNotFoundError:
                    logging.error(f"    File not found: {experimental_file_path}")
                    print(f"    File not found: {experimental_file_path}")
                    progress_bar.update(1)
                    continue
                except Exception as e:
                    logging.error(f"    Failed to load or preprocess data from {experimental_file_path}: {e}")
                    print(f"    Failed to load or preprocess data from {experimental_file_path}: {e}")
                    progress_bar.update(1)
                    continue

                # Завантаження моделі та гіперпараметрів
                model_filename = f"{model_name}_{file_name}_stage_{stage}_{sensor}"
                model_filepath = os.path.join(folder, model_filename)
                params_filepath = os.path.join(folder, f"{model_filename}_params.json")

                if not os.path.exists(params_filepath):
                    logging.warning(f"    Skipping {folder} as no hyperparameters file found.")
                    print(f"    Skipping {folder} as no hyperparameters file found.")
                    progress_bar.update(1)
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
                    progress_bar.update(1)
                    continue

                # Предикшн
                y_pred = model.predict(X).flatten()

                # Обрізаємо predicted_values до довжини y
                y_pred = y_pred[:len(y)]

                # Розрахунок метрик
                mae, rmse, r2 = calculate_and_save_metrics(y, y_pred, file_name, stage, sensor, model_name, model_results_dir)

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
                min_ad, min_tau, rnd, bi, arw, rrw, t2, ad = calculate_allan_deviation_metrics(y, rate=1 / dt)

                # Розрахунок Allan Deviation для передбачених даних
                (t2_pred, ad_pred, ade_pred, adn_pred) = allantools.oadev(y_pred, rate=1 / dt, data_type="freq",
                                                                          taus='all')

                # Розрахунок та збереження компонент декомпозиції
                calculate_and_save_seasonal_decomposition(y, df['Time'].iloc[window_size:], file_name, stage, sensor,
                                                          model_name, model_results_dir)

                # Розрахунок та збереження ACF/PACF
                calculate_and_save_acf_pacf(y, file_name, stage, sensor, model_name, model_results_dir)

                # Збереження метрик у DataFrame
                new_row = pd.DataFrame({
                    "Model": [model_name],
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
                model_metrics_df = pd.concat([model_metrics_df, new_row], ignore_index=True)

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

                # Побудова графіка Allan Deviation
                if ad is not None and t2 is not None:
                    create_allan_deviation_plot(t2, ad, t2_pred, ad_pred, rnd, bi, arw, min_ad, min_tau, model_name,
                                                experimental_file_name, stage, sensor, model_results_dir)

                progress_bar.update(1)

            # Збереження загальних метрик у CSV файл
            metrics_filepath = os.path.join(model_results_dir, f"{model_name}_metrics.csv")
            model_metrics_df.to_csv(metrics_filepath, index=False)
            print(f"Saved metrics for {model_name} to {metrics_filepath}")

            # Додаємо метрики поточної моделі до загального DataFrame
            all_metrics_df = pd.concat([all_metrics_df, model_metrics_df], ignore_index=True)

    # Збереження зведеної таблиці з метриками для всіх моделей
    all_metrics_filepath = os.path.join(evaluation_results_dir, "all_models_metrics.csv")
    all_metrics_df.to_csv(all_metrics_filepath, index=False)
    print(f"\nSaved all models metrics to {all_metrics_filepath}")

    print("\nОцінка моделей завершена.")