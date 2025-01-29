import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Параметри
window_size = 100
dt = 1 / 108  # Частота дискретизації

# Шлях до папки з експериментальними даними (зміни, якщо потрібно)
experimental_data_folder = "experimental_data"

# Шляхи до файлів з експериментальними даними (зміни, якщо потрібно)
experimental_file_paths = [
    os.path.join(experimental_data_folder, "ProgessiveOscill0.xlsx"),
    os.path.join(experimental_data_folder, "StableOscill0.xlsx"),
    os.path.join(experimental_data_folder, "ProgessiveOscill1.xlsx"),
    os.path.join(experimental_data_folder, "StableOscill1.xlsx")
]

def preprocess_data(df, sensor_type):
    """
    Попередня обробка даних (видалення викидів, заповнення пропусків).

    Args:
        df (pd.DataFrame): DataFrame з даними.
        sensor_type (str): Назва колонки з даними сенсора.

    Returns:
        pd.DataFrame: Оброблений DataFrame.
    """
    signal = df[f"{sensor_type}"]
    signal_mean = signal.mean()
    signal_std = signal.std()
    signal_filtered = signal[np.abs(signal - signal_mean) < 3 * signal_std]

    df[f"{sensor_type}"] = signal_filtered
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def prepare_data_for_model(df, sensor_type, window_size):
    """
    Готує дані для подачі в модель, формуючи вікна заданого розміру.

    Args:
        df (pd.DataFrame): DataFrame з даними.
        sensor_type (str): Назва колонки з даними сенсора.
        window_size (int): Розмір вікна.

    Returns:
        np.ndarray: Масив даних, підготовлений для моделі.
    """
    X = np.lib.stride_tricks.sliding_window_view(df[f"{sensor_type}"].values, window_size)
    y = df[f"{sensor_type}"].values[window_size - 1:]
    return X, y

def prepare_all_data(window_size=100, dt=1/108):
    """
    Готує дані для всіх файлів та сенсорів.

    Args:
        window_size (int): Розмір вікна.
        dt (float): Частота дискретизації.

    Returns:
        dict: Словник з даними для кожної комбінації файлу та сенсора.
    """
    all_data = {}
    for file_path in experimental_file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df = pd.read_excel(file_path)
            if 'Time' not in df.columns:
                df['Time'] = np.arange(0, len(df) * dt, dt)
            for sensor in ['NVGyro Z', 'N1Gyro Z', 'N8Gyro Z']:
                df_processed = preprocess_data(df.copy(), sensor)
                X, y = prepare_data_for_model(df_processed, sensor, window_size)
                all_data[f"{file_name}_{sensor}"] = {"X": X, "y": y, "time": df['Time'][window_size - 1:].values, 'encoder': df['Encoder Speed'][window_size - 1:].values}
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to load or preprocess data from {file_path}: {e}")
    return all_data