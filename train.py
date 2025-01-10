import numpy as np
import os
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel

# Задайте шлях до папки з обробленими даними
processed_data_folder = "processed_data"

# Параметри
window_size = 100
epochs = 50 # зменшено для пришвидшення навчання
batch_size = 32

# Моделі
models = {
    "LSTM": LSTMModel,
    "CNN": CNNModel,
    "CNNLSTM": CNNLSTMModel,
    "TCN": TCNModel,
    "GRU": GRUModel,
}

# Словник для збереження історій навчання
histories = {}

# Отримання списку всіх папок з даними
stage_sensor_folders = [d for d in os.listdir(processed_data_folder) if os.path.isdir(os.path.join(processed_data_folder, d))]

for model_name, ModelClass in models.items():
    print(f"Training {model_name}...")
    histories[model_name] = {}

    for folder in stage_sensor_folders:
        print(f"  Processing {folder}...")
        stage, sensor = folder.split("_", 1)

        # Завантаження даних
        X_train = np.load(f"{processed_data_folder}/{folder}/X_train.npy")
        y_train = np.load(f"{processed_data_folder}/{folder}/y_train.npy")
        X_val = np.load(f"{processed_data_folder}/{folder}/X_val.npy")
        y_val = np.load(f"{processed_data_folder}/{folder}/y_val.npy")
        X_test = np.load(f"{processed_data_folder}/{folder}/X_test.npy") # Завантаження тестових даних
        y_test = np.load(f"{processed_data_folder}/{folder}/y_test.npy") # Завантаження тестових даних

        # Ініціалізація та побудова моделі
        model = ModelClass(window_size, 1)
        model.build()

        # Навчання моделі
        model_filepath = f"{model_name}_{stage}_{sensor}.h5"
        history = model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, model_filepath=model_filepath)

        # Збереження історії навчання
        histories[model_name][f"{stage}_{sensor}"] = history.history

# Збереження історій навчання у файл (наприклад, у форматі .npz)
np.savez("training_histories.npz", **histories)