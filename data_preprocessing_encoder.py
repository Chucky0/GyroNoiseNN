import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
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

def preprocess_data(df, sensor_types):
    """
    Попередня обробка даних (видалення викидів, заповнення пропусків).

    Args:
        df (pd.DataFrame): DataFrame з даними.
        sensor_types (list): Список назв колонок з даними сенсорів.

    Returns:
        pd.DataFrame: Оброблений DataFrame.
    """
    for sensor_type in sensor_types:
        signal = df[sensor_type]
        signal_mean = signal.mean()
        signal_std = signal.std()
        signal_filtered = signal[np.abs(signal - signal_mean) < 3 * signal_std]

        df[sensor_type] = signal_filtered

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def prepare_data_for_model(df, input_sensors, window_size):
    """
    Готує дані для подачі в модель, формуючи вікна заданого розміру.

    Args:
        df (pd.DataFrame): DataFrame з даними.
        input_sensors (list): Список назв колонок з даними вхідних сенсорів.
        window_size (int): Розмір вікна.

    Returns:
        np.ndarray, np.ndarray: Масив вхідних даних (X) та цільових значень (y).
    """
    X = np.lib.stride_tricks.sliding_window_view(df[input_sensors].values, (window_size, len(input_sensors)))
    X = X.reshape(X.shape[0], window_size, len(input_sensors))
    y = df['Encoder Speed'].values[window_size - 1:]
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
    input_sensors = ['NVGyro Z', 'N1Gyro Z', 'N8Gyro Z'] # Вхідні сенсори

    for file_path in experimental_file_paths:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df = pd.read_excel(file_path)
            if 'Time' not in df.columns:
                df['Time'] = np.arange(0, len(df) * dt, dt)

            # Перевірка наявності 'Encoder Speed'
            if 'Encoder Speed' not in df.columns:
                logging.warning(f"'Encoder Speed' column not found in {file_path}. Skipping this file.")
                continue

            df_processed = preprocess_data(df.copy(), input_sensors + ['Encoder Speed'])
            X, y = prepare_data_for_model(df_processed, input_sensors, window_size)

            # Розділення на тренувальну, валідаційну та тестову вибірки
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            # Створення унікального ідентифікатора для кожного набору даних
            dataset_id = f"{file_name}_EncoderSpeed"

            # Створення папки для поточного датасету, якщо її не існує
            dataset_path = os.path.join("processed_data", dataset_id)
            os.makedirs(dataset_path, exist_ok=True)

            # Збереження даних у файли
            np.save(f"{dataset_path}/X_train.npy", X_train)
            np.save(f"{dataset_path}/y_train.npy", y_train)
            np.save(f"{dataset_path}/X_val.npy", X_val)
            np.save(f"{dataset_path}/y_val.npy", y_val)
            np.save(f"{dataset_path}/X_test.npy", X_test)
            np.save(f"{dataset_path}/y_test.npy", y_test)

            # Збереження даних у словнику
            all_data[dataset_id] = {
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test,
                "time": df['Time'][window_size - 1:].values
            }

        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
        except Exception as e:
            logging.error(f"Failed to load or preprocess data from {file_path}: {e}")

    return all_data

# Створюємо папку processed_data, якщо вона не існує
os.makedirs("processed_data", exist_ok=True)

# Приклад використання
all_data = prepare_all_data(window_size, dt)