import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_and_preprocess_data(file_paths, window_size=100):
    """
    Завантажує дані з декількох XLSX файлів, розділяє на етапи, формує вікна та зберігає дані.

    Args:
        file_paths (list): Список шляхів до XLSX файлів.
        window_size (int): Розмір вікна для LSTM/CNN.

    Returns:
        dict: Словник з даними, розділеними на навчальну, валідаційну та тестову вибірки
              для кожного етапу та їх комбінацій.
    """

    processed_data = {}

    for file_path in file_paths:
        print(file_path)
        df = pd.read_excel(file_path)

        # Визначення часових меж для кожного етапу (незмінні для всіх файлів)
        stage1_end_time = 277
        stage2_end_time = 1205

        # Розділення на етапи
        stage1_data = df[df['Time'] <= stage1_end_time]
        stage2_data = df[(df['Time'] > stage1_end_time) & (df['Time'] <= stage2_end_time)]
        stage3_data = df[df['Time'] > stage2_end_time]

        # Створення комбінацій етапів
        stage1_stage2_data = pd.concat([stage1_data, stage2_data])
        stage2_stage3_data = pd.concat([stage2_data, stage3_data])
        stage1_stage2_stage3_data = pd.concat([stage1_data, stage2_data, stage3_data])

        # Додавання міток етапів
        stage1_data['Stage'] = 0
        stage2_data['Stage'] = 1
        stage3_data['Stage'] = 2
        stage1_stage2_data['Stage'] = 3
        stage2_stage3_data['Stage'] = 4
        stage1_stage2_stage3_data['Stage'] = 5

        # Список всіх датафреймів
        all_stages_data = [
            stage1_data, stage2_data, stage3_data,
            stage1_stage2_data, stage2_stage3_data, stage1_stage2_stage3_data
        ]

        # Датчики
        sensors = ['NVGyro Z', 'N1Gyro Z', 'N2Gyro Z', 'N3Gyro Z', 'N4Gyro Z', 'N5Gyro Z', 'N6Gyro Z', 'N7Gyro Z',
                   'N8Gyro Z']

        for stage_data in all_stages_data:
            stage_label = stage_data['Stage'].iloc[0]

            # Використовуємо унікальний ідентифікатор для кожного файлу та етапу
            file_id = os.path.splitext(os.path.basename(file_path))[0]  # Отримуємо назву файлу без розширення
            stage_sensor_folder_prefix = f"{file_id}_stage_{stage_label}"

            if stage_label not in processed_data:
                processed_data[stage_label] = {}

            for sensor in sensors:
                # Виділення даних сенсора
                sensor_data = stage_data[['Time', sensor]].copy()

                # Формування вікон
                X, y = [], []
                for i in range(len(sensor_data) - window_size):
                    X.append(sensor_data[sensor].values[i:i + window_size])
                    y.append(sensor_data[sensor].values[i + window_size])

                X, y = np.array(X), np.array(y)

                # Розділення на навчальну, валідаційну та тестову вибірки
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

                # Збереження даних у словнику з урахуванням назви файлу
                if sensor not in processed_data[stage_label]:
                    processed_data[stage_label][sensor] = {}

                if 'X_train' not in processed_data[stage_label][sensor]:
                    processed_data[stage_label][sensor]['X_train'] = X_train
                else:
                    processed_data[stage_label][sensor]['X_train'] = np.concatenate(
                        (processed_data[stage_label][sensor]['X_train'], X_train))

                if 'y_train' not in processed_data[stage_label][sensor]:
                    processed_data[stage_label][sensor]['y_train'] = y_train
                else:
                    processed_data[stage_label][sensor]['y_train'] = np.concatenate(
                        (processed_data[stage_label][sensor]['y_train'], y_train))

                if 'X_val' not in processed_data[stage_label][sensor]:
                    processed_data[stage_label][sensor]['X_val'] = X_val
                else:
                    processed_data[stage_label][sensor]['X_val'] = np.concatenate(
                        (processed_data[stage_label][sensor]['X_val'], X_val))

                if 'y_val' not in processed_data[stage_label][sensor]:
                    processed_data[stage_label][sensor]['y_val'] = y_val
                else:
                    processed_data[stage_label][sensor]['y_val'] = np.concatenate(
                        (processed_data[stage_label][sensor]['y_val'], y_val))

                if 'X_test' not in processed_data[stage_label][sensor]:
                    processed_data[stage_label][sensor]['X_test'] = X_test
                else:
                    processed_data[stage_label][sensor]['X_test'] = np.concatenate(
                        (processed_data[stage_label][sensor]['X_test'], X_test))

                if 'y_test' not in processed_data[stage_label][sensor]:
                    processed_data[stage_label][sensor]['y_test'] = y_test
                else:
                    processed_data[stage_label][sensor]['y_test'] = np.concatenate(
                        (processed_data[stage_label][sensor]['y_test'], y_test))

                # Збереження даних у файли (наприклад, у форматі .npy)
                stage_sensor_folder = f"{stage_sensor_folder_prefix}_{sensor}"
                os.makedirs(stage_sensor_folder, exist_ok=True)  # Створення папки, якщо її немає
                np.save(f"{stage_sensor_folder}/X_train.npy", X_train)
                np.save(f"{stage_sensor_folder}/y_train.npy", y_train)
                np.save(f"{stage_sensor_folder}/X_val.npy", X_val)
                np.save(f"{stage_sensor_folder}/y_val.npy", y_val)
                np.save(f"{stage_sensor_folder}/X_test.npy", X_test)
                np.save(f"{stage_sensor_folder}/y_test.npy", y_test)

    return processed_data


# Приклад використання
import os

file_paths = [
    "C:\\Users\\super\\Desktop\\GyroDataProcessing\\GyroNoiseNet\\GyroNoiseNet\\StableOscill1.xlsx",
    "C:\\Users\\super\\Desktop\\GyroDataProcessing\\GyroNoiseNet\\GyroNoiseNet\\ProgessiveOscill0.xlsx",
    "C:\\Users\\super\\Desktop\\GyroDataProcessing\\GyroNoiseNet\\GyroNoiseNet\\ProgessiveOscill1.xlsx",
    "C:\\Users\\super\\Desktop\\GyroDataProcessing\\GyroNoiseNet\\GyroNoiseNet\\StableOscill0.xlsx"
]
processed_data = load_and_preprocess_data(file_paths)