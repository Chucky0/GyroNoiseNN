import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import r2_score
from utils_cuda import calculate_allan_variance_cuda, calculate_drift_rate_cuda, calculate_offset_cuda, \
    calculate_psd_cuda
import logging
import yaml
from statsmodels.tsa.ar_model import AutoReg

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def setup_logging(log_file=config["log_file"], log_level="INFO"):
    """Налаштовує логування."""
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    logging.basicConfig(filename=log_file, level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_vrw(taus, allan_var):
    """
    Розраховує Velocity Random Walk (VRW) з графіку Allan Variance.
    """
    log_taus = np.log10(taus)
    log_allan_var = np.log10(allan_var)
    diff_log_taus = np.diff(log_taus)
    diff_log_allan_var = np.diff(log_allan_var)
    slopes = diff_log_allan_var / diff_log_taus
    idx_m05 = np.argmin(np.abs(slopes + 0.5))
    vrw = 10 ** (log_allan_var[idx_m05] - 0.5 * log_taus[idx_m05])
    return vrw


def calculate_bi(taus, allan_var):
    """
    Розраховує Bias Instability (BI) з графіку Allan Variance.
    """
    return np.min(allan_var)


def calculate_rrw(taus, allan_var):
    """
    Розраховує Rate Random Walk (RRW) з графіку Allan Variance.
    """
    log_taus = np.log10(taus)
    log_allan_var = np.log10(allan_var)
    diff_log_taus = np.diff(log_taus)
    diff_log_allan_var = np.diff(log_allan_var)
    slopes = diff_log_allan_var / diff_log_taus
    idx_p05 = np.argmin(np.abs(slopes - 0.5))
    rrw = 10 ** (log_allan_var[idx_p05] + 0.5 * log_taus[idx_p05])
    return rrw


def calculate_correlation_and_rmse(y_true, y_pred):
    """
    Розраховує кореляцію та RMSE.
    """
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return correlation, rmse


# -----------------------------------------------------------------
# Функції для завантаження та обробки даних
# -----------------------------------------------------------------

def load_data(file_path):
    """
    Завантажує дані з Excel файлу.

    Args:
        file_path (str): Шлях до файлу.

    Returns:
        pandas.DataFrame: DataFrame з даними.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    required_columns = ['Time', 'Encoder Speed', "NVGyro Z", "N1Gyro Z", "N2Gyro Z", "N3Gyro Z", "N4Gyro Z", "N5Gyro Z",
                        "N6Gyro Z", "N7Gyro Z", "N8Gyro Z"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Файл {file_path} не містить необхідні колонки")

    df['dt'] = df['Time'].diff().fillna(0)
    return df

    # -----------------------------------------------------------------
    # Збереження результатів
    # -----------------------------------------------------------------


def save_results_to_csv(results, output_dir=config["output_dir"]):
    """
    Зберігає результати у CSV файли.
    """
    logging.info(f"Збереження результатів у {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_path, sensors_data in results.items():
        filename = os.path.splitext(os.path.basename(file_path))[0]
        for sensor_name, models_data in sensors_data.items():
            for model_name, model_data in models_data.items():
                output_path = os.path.join(output_dir, f"{filename}_{sensor_name}_{model_name}.csv")
                if model_name == "metrics":
                    df = pd.DataFrame(model_data)
                    df.to_csv(output_path, index=False)
                else:
                    df = pd.DataFrame()
                    if 'time' in model_data:
                        df['time'] = model_data['time']
                    if 'y_pred' in model_data:
                        df[f'{model_name}_pred'] = model_data['y_pred']

                    # Зберігаємо історію навчання у форматі CSV
                    if 'history' in model_data:
                        hist_df = pd.DataFrame(model_data['history'])
                        hist_df.to_csv(output_path.replace(".csv", "_history.csv"), index=False)

                    df.to_csv(output_path, index=False)
    logging.info("Результати збережено.")


def save_plots(results, stage2_start_time, stage2_end_time, time, output_dir=config["output_dir"]):
    """
    Зберігає графіки у PNG файли.
    """
    logging.info(f"Збереження графіків у {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_path, sensors_data in results.items():
        filename = os.path.splitext(os.path.basename(file_path))[0]
        for sensor_name, models_data in sensors_data.items():
            for model_name, model_data in models_data.items():

                # Додаткові графіки для метрик
                if model_name == "metrics":
                    # Allan Variance
                    if "avar" in model_data:
                        taus_list = model_data["avar"]["taus"]
                        allan_var_list = model_data["avar"]["allan_var"]

                        if len(taus_list) > 0 and len(allan_var_list) > 0:
                            taus = taus_list[0]
                            allan_var = allan_var_list[0]

                            plt.figure(figsize=(12, 6))
                            plt.loglog(taus, np.sqrt(allan_var), marker='o', linestyle='-')
                            plt.title(f"{filename} - {sensor_name} - Allan Variance")
                            plt.xlabel("Averaging Time (s)")
                            plt.ylabel("Allan Deviation")
                            plt.grid(True)
                            output_path = os.path.join(output_dir, f"{filename}_{sensor_name}_avar.png")
                            plt.savefig(output_path)
                            plt.close()

                    # PSD
                    if "psd" in model_data:
                        freqs_list = model_data["psd"]["freqs"]
                        psd_list = model_data["psd"]["psd"]
                        if len(freqs_list) > 0 and len(psd_list) > 0:
                            freqs = freqs_list[0]
                            psd = psd_list[0]

                            plt.figure(figsize=(12, 6))
                            plt.loglog(freqs, psd)
                            plt.title(f"{filename} - {sensor_name} - Power Spectral Density")
                            plt.xlabel("Frequency (Hz)")
                            plt.ylabel("PSD")
                            plt.grid(True)
                            output_path = os.path.join(output_dir, f"{filename}_{sensor_name}_psd.png")
                            plt.savefig(output_path)
                            plt.close()

                # Графік зміни параметрів синусоїди для AR_Sin_Online
                if model_name == "AR_Sin_Online":
                    if all(param in model_data for param in ["A", "f", "phi", "offset"]):
                        plt.figure(figsize=(12, 6))
                        plt.plot(time[(time >= stage2_start_time) & (time <= stage2_end_time)],
                                 model_data["A"], label="Amplitude")
                        plt.plot(time[(time >= stage2_start_time) & (time <= stage2_end_time)],
                                 model_data["f"], label="Frequency")
                        plt.plot(time[(time >= stage2_start_time) & (time <= stage2_end_time)],
                                 model_data["phi"], label="Phase")
                        plt.plot(time[(time >= stage2_start_time) & (time <= stage2_end_time)],
                                 model_data["offset"], label="Offset")
                        plt.title(f"{filename} - {sensor_name} - AR_Sin_Online Parameters")
                        plt.xlabel("Time")
                        plt.ylabel("Parameter Value")
                        plt.legend()
                        plt.grid(True)
                        output_path = os.path.join(output_dir, f"{filename}_{sensor_name}_AR_Sin_Online_params.png")
                        plt.savefig(output_path)
                        plt.close()

                # Додаткові графіки для метрик
                if model_name != "metrics":
                    # Графіки навчання
                    if "history" in model_data:
                        plt.figure(figsize=(12, 6))
                        plt.plot(model_data["history"]['loss'], label='loss')
                        plt.plot(model_data["history"]['val_loss'], label='val_loss')
                        plt.title(f"{filename} - {sensor_name} - {model_name} Training Loss")
                        plt.xlabel("Epoch")
                        plt.ylabel("Loss")
                        plt.legend()
                        plt.grid(True)
                        output_path = os.path.join(output_dir,
                                                   f"{filename}_{sensor_name}_{model_name}_training_loss.png")
                        plt.savefig(output_path)
                        plt.close()

                    # Графік зміни коефіцієнтів для AR_Adaptive
                    if model_name == "AR_Adaptive":
                        if "ar_coefficients" in model_data:
                            p_order = model_data["p"]
                            plt.figure(figsize=(12, 6))
                            for i in range(p_order):
                                plt.plot(time[p_order:], model_data["ar_coefficients"][:, i],
                                         label=f"AR({i + 1})")
                            plt.title(f"{filename} - {sensor_name} - AR_Adaptive Coefficients")
                            plt.xlabel("Time")
                            plt.ylabel("Coefficient Value")
                            plt.legend()
                            plt.grid(True)
                            output_path = os.path.join(output_dir, f"{filename}_{sensor_name}_AR_Adaptive_coeffs.png")
                            plt.savefig(output_path)
                            plt.close()

    logging.info("Графіки збережено.")

    # -----------------------------------------------------------------
    # Збереження та завантаження чекпоінтів
    # -----------------------------------------------------------------


def save_checkpoint(data, filename=config["checkpoint_file"]):
    """Зберігає дані у файл чекпоінту."""
    logging.info(f"Збереження чекпоінту у {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    logging.info("Чекпоінт збережено.")


def load_checkpoint(filename=config["checkpoint_file"]):
    """Завантажує дані з файлу чекпоінту."""
    logging.info(f"Завантаження чекпоінту з {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    logging.info("Чекпоінт завантажено.")
    return data


def prepare_data_for_nn(sensor_data, encoder_data, window_size, n_features):
    """
    Готує дані для навчання нейронної мережі.

    Args:
        sensor_data (np.array): Дані сенсора.
        encoder_data (np.array): Дані енкодера.
        window_size (int): Розмір вікна для вхідних даних.
        n_features (int): Кількість ознак (сенсорів).

    Returns:
        tuple: (X, y) - вхідні та вихідні дані.
    """
    X, y = [], []
    for i in range(len(sensor_data) - window_size):
        X.append(sensor_data[i:i + window_size, :])
        y.append(encoder_data[i + window_size])
    return np.array(X), np.array(y)


# -----------------------------------------------------------------
# Функції для AR + синусоїда моделі
# -----------------------------------------------------------------

def sinusoidal_model(params, t, data=None):
    """
        Синусоїдальна модель для методу найменших квадратів.
        """
    amplitude, frequency, phase, offset = params
    model_values = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
    if data is not None:
        return model_values - data  # Залишки
    else:
        return model_values


def fit_sinusoidal_model(t, data):
    """
    Підганяє параметри синусоїдальної моделі до даних методом найменших квадратів.
    """
    initial_amplitude = np.std(data)
    initial_frequency = 1 / (t[-1] - t[0]) if len(t) > 1 else 0.0
    initial_phase = 0
    initial_offset = np.mean(data)
    initial_params = [initial_amplitude, initial_frequency, initial_phase, initial_offset]

    result = least_squares(sinusoidal_model, initial_params, args=(t, data), loss='soft_l1')
    return result.x


def compensate_noise_ar_sin(data, ar_model, start_time, end_time, times, window_size_sec=5):
    """
    Компенсує шум, використовуючи AR модель та синусоїду з онлайн-адаптацією.
    """
    start_index = np.argmin(np.abs(times - start_time))
    end_index = np.argmin(np.abs(times - end_time))
    compensated_data = data.copy()
    filtered_data = compensate_noise_ar(data, ar_model, start_time, end_time, times)
    sinusoid_signal = np.zeros_like(data)
    sin_params_history = []

    dt = times[1] - times[0] if len(times) > 1 else 0.001
    window_size = int(window_size_sec / dt)

    for i in range(start_index, end_index):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(data), i + window_size // 2)

        current_time_window = times[window_start:window_end]
        current_data_window = filtered_data[window_start:window_end]

        sin_params = fit_sinusoidal_model(current_time_window, current_data_window)
        sinusoid_signal[i] = sinusoidal_model(sin_params, np.array([times[i]]))[0]
        compensated_data[i] = data[i] - sinusoid_signal[i]
        sin_params_history.append(sin_params)

    return compensated_data, sinusoid_signal, np.array(sin_params_history)


# -----------------------------------------------------------------
# Функції для AR моделі
# -----------------------------------------------------------------

def fit_ar_model(data, p_order, start_time, end_time, times):
    """
    Підбирає AR модель до даних на вказаному часовому проміжку.
    """
    start_index = np.argmin(np.abs(times - start_time))
    end_index = np.argmin(np.abs(times - end_time))
    model = AutoReg(data[start_index:end_index], lags=p_order, old_names=False)
    results = model.fit()
    return results


def compensate_noise_ar(data, ar_model, start_time, end_time, times):
    """
    Компенсує шум в даних, використовуючи AR модель.
    """
    start_index = np.argmin(np.abs(times - start_time))
    end_index = np.argmin(np.abs(times - end_time))
    compensated_data = data.copy()
    for i in range(start_index, end_index):
        predicted_noise = ar_model.predict(start=i, end=i)
        compensated_data[i] -= predicted_noise
    return compensated_data


# -----------------------------------------------------------------
# Функція для адаптивної AR моделі (RLS)
# -----------------------------------------------------------------

def rls_filter(data, p_order, forgetting_factor=0.99):
    """
    Реалізує адаптивну AR модель з використанням Recursive Least Squares (RLS).
    """
    P = np.eye(p_order) * (1 / 0.001)
    w = np.zeros(p_order)
    compensated_data = data.copy()
    ar_coefficients = []

    for i in range(p_order, len(data)):
        x = data[i - p_order:i][::-1]
        y_hat = np.dot(w, x)
        error = data[i] - y_hat
        compensated_data[i] = error
        k = np.dot(P, x) / (forgetting_factor + np.dot(np.dot(x.T, P), x))
        w = w + k * error
        P = (P - np.outer(k, np.dot(x.T, P))) / forgetting_factor
        ar_coefficients.append(w.copy())

    return compensated_data, np.array(ar_coefficients)