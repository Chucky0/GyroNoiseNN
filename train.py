import numpy as np
import os
import json
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel
import multiprocessing
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # Додано імпорт

import logging

# Налаштування логування
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Задайте шлях до папки з обробленими даними
processed_data_folder = "processed_data"

# Параметри
window_size = 100
epochs = 50  # Змінено на 50
batch_size = 16
learning_rate = 0.0001

# Моделі
models = {
    "LSTM": LSTMModel,
    "CNN": CNNModel,
    "CNNLSTM": CNNLSTMModel
}


def train_model_for_folder(folder_path, model_name, ModelClass):
    """Функція для навчання моделі на даних з однієї папки."""
    logging.info(f"  Processing {folder_path} for {model_name}...")
    print(f"  Processing {folder_path} for {model_name}...")

    # Отримуємо назву директорії, що містить папку з даними (назва файлу)
    parent_dir = os.path.basename(os.path.dirname(folder_path))

    # Отримуємо назву папки з даними (ProgessiveOscill0_stage_0_N1Gyro Z)
    data_folder_name = os.path.basename(folder_path)

    # Розбиваємо назву папки на частини
    parts = data_folder_name.split("_")

    # Перевіряємо, чи достатньо частин для розбиття
    if len(parts) >= 3:
        # Формуємо назву етапу та сенсора
        stage = parts[-2].replace('stage', '')  # Видаляємо "stage" з назви
        sensor = parts[-1].split(".")[0]
    else:
        logging.error(f"Incorrect folder name format: {data_folder_name}")
        return None, None

    # Завантаження даних
    X_train = np.load(os.path.join(folder_path, "X_train.npy"))
    y_train = np.load(os.path.join(folder_path, "y_train.npy"))
    X_val = np.load(os.path.join(folder_path, "X_val.npy"))
    y_val = np.load(os.path.join(folder_path, "y_val.npy"))
    X_test = np.load(os.path.join(folder_path, "X_test.npy"))
    y_test = np.load(os.path.join(folder_path, "y_test.npy"))

    # Перевірка на NaN
    if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isnan(X_val).any() or np.isnan(y_val).any() or np.isnan(
            X_test).any() or np.isnan(y_test).any():
        logging.error(f"NaN values found in data for {folder_path}!")
        return None, f"{stage}_{sensor}"

    # Ініціалізація моделі
    model = ModelClass(window_size, 1)
    model.build()

    # Створення папки для збереження моделі всередині models/<model_name>/<parent_dir>
    model_dir = os.path.join("models", model_name, parent_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Назва файлу моделі
    model_filename = f"{model_name}_{data_folder_name}"
    model_filepath = os.path.join(model_dir, model_filename)
    print(f"Model Checkpoint filepath: {model_filepath}")

    # Словник з гіперпараметрами
    hyperparameters = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "window_size": window_size,
        "optimizer": "Adam",
        "clipvalue": 1.0,
        "model": model_name
    }

    # Збереження гіперпараметрів у JSON файл
    params_filepath = os.path.join(model_dir, f"{model_filename}_params.json")
    with open(params_filepath, "w") as f:
        json.dump(hyperparameters, f, indent=4)

    # Замість перевірки наявності файлу з вагами ми одразу намагаємось завантажити модель
    try:
        model = ModelClass(window_size, 1)
        model.load(f"{model_filepath}.keras")
        logging.info(f"Loaded existing model from {model_filepath}.keras")
        print(f"Loaded existing model from {model_filepath}.keras")
        # Якщо модель завантажено, історія навчання може бути відсутня, тому повертаємо None
        return None, f"{stage}_{sensor}"
    except:
        logging.info(f"No existing model found for {model_filepath}. Training from scratch.")
        print(f"No existing model found for {model_filepath}. Training from scratch.")
        model.build()

    # Навчання моделі
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    # Зберігаємо найкращу модель
    checkpoint = ModelCheckpoint(f"{model_filepath}.keras", monitor='val_loss', save_best_only=True, mode='min',
                                 verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    history = model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size,
                          model_filepath=model_filepath, optimizer=optimizer)

    return history.history, f"{stage}_{sensor}"


if __name__ == '__main__':
    # Перевірка наявності папки models і створення її, якщо вона відсутня
    if not os.path.exists("models"):
        os.makedirs("models")

    # Словник для збереження історій навчання
    histories = {}

    # Отримання списку всіх папок з даними
    # Оновлено: шлях до папки з даними
    stage_sensor_folders = [os.path.join(processed_data_folder, d) for d in os.listdir(processed_data_folder) if
                            os.path.isdir(os.path.join(processed_data_folder, d))]
    total_tasks = len(stage_sensor_folders) * len(models)

    with tqdm(total=total_tasks, desc="Overall Progress") as progress_bar:
        for model_name, ModelClass in models.items():
            logging.info(f"Training {model_name}...")
            histories[model_name] = {}

            # Створення пулу процесів
            with multiprocessing.Pool(processes=4) as pool:
                # Підготовка аргументів для функції
                # Оновлено: передаємо тільки потрібні аргументи
                args = [(folder, model_name, ModelClass) for folder in stage_sensor_folders]

                # Запуск навчання в пулі процесів
                results = pool.starmap(train_model_for_folder, args)

                # Збір результатів
                for history, stage_sensor in results:
                    if history is not None:
                        histories[model_name][stage_sensor] = history
                    progress_bar.update(1)

    # Збереження історій навчання у файл (наприклад, у форматі .npz)
    np.savez("training_histories.npz", **histories)