import numpy as np
import os
import optuna
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam

# Задайте шлях до папки з обробленими даними
processed_data_folder = "processed_data"

# Параметри
window_size = 100
study_name = "hyperparameter_tuning"  # Назва дослідження

# Моделі та гіперпараметри для оптимізації
model_classes = {
    "LSTM": LSTMModel,
    "CNN": CNNModel,
    "CNNLSTM": CNNLSTMModel,
    "TCN": TCNModel,
    "GRU": GRUModel,
}
params = {
    "LSTM": {
        "units": (16, 128),
        "learning_rate": (1e-5, 1e-2),
    },
    "CNN": {
      "filters": (32, 128),
      "kernel_size": (3, 7),
      "dense_units": (16, 128),
      "learning_rate": (1e-5, 1e-2),
    },
    "CNNLSTM": {
      "conv_filters": (32, 128),
      "conv_kernel_size": (3, 5),
      "lstm_units": (16, 128),
      "learning_rate": (1e-5, 1e-2),
    },
    "TCN": {
      "filters": (32, 128),
      "kernel_size": (3, 7),
      "dense_units": (16, 128),
      "learning_rate": (1e-5, 1e-2),
    },
      "GRU": {
        "units": (16, 128),
        "learning_rate": (1e-5, 1e-2),
    },
}

# Отримання списку всіх папок з даними
stage_sensor_folders = [
    d
    for d in os.listdir(processed_data_folder)
    if os.path.isdir(os.path.join(processed_data_folder, d))
]


def objective(trial, model_name, stage_sensor):
    # Вибір моделі та гіперпараметрів
    ModelClass = model_classes[model_name]

    if model_name == "LSTM":
      suggested_params = {
          "units": trial.suggest_int("units", *params[model_name]["units"]),
          "learning_rate": trial.suggest_loguniform("learning_rate", *params[model_name]["learning_rate"]),
      }
      model = ModelClass(window_size, 1)
      model.build_with_params(**suggested_params)
    elif model_name == "CNN":
        suggested_params = {
            "filters": trial.suggest_int("filters", *params[model_name]["filters"]),
            "kernel_size": trial.suggest_int("kernel_size", *params[model_name]["kernel_size"]),
            "dense_units": trial.suggest_int("dense_units", *params[model_name]["dense_units"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", *params[model_name]["learning_rate"]),
        }
        model = ModelClass(window_size, 1)
        model.build_with_params(**suggested_params)
    elif model_name == "CNNLSTM":
        suggested_params = {
            "conv_filters": trial.suggest_int("conv_filters", *params[model_name]["conv_filters"]),
            "conv_kernel_size": trial.suggest_int("conv_kernel_size", *params[model_name]["conv_kernel_size"]),
            "lstm_units": trial.suggest_int("lstm_units", *params[model_name]["lstm_units"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", *params[model_name]["learning_rate"]),
        }
        model = ModelClass(window_size, 1)
        model.build_with_params(**suggested_params)
    elif model_name == "TCN":
        suggested_params = {
            "filters": trial.suggest_int("filters", *params[model_name]["filters"]),
            "kernel_size": trial.suggest_int("kernel_size", *params[model_name]["kernel_size"]),
            "dense_units": trial.suggest_int("dense_units", *params[model_name]["dense_units"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", *params[model_name]["learning_rate"]),
        }
        model = ModelClass(window_size, 1)
        model.build_with_params(**suggested_params)
    elif model_name == "GRU":
        suggested_params = {
            "units": trial.suggest_int("units", *params[model_name]["units"]),
            "learning_rate": trial.suggest_loguniform("learning_rate", *params[model_name]["learning_rate"]),
        }
        model = ModelClass(window_size, 1)
        model.build_with_params(**suggested_params)

    # Завантаження даних
    X_train = np.load(f"{processed_data_folder}/{stage_sensor}/X_train.npy")
    y_train = np.load(f"{processed_data_folder}/{stage_sensor}/y_train.npy")
    X_val = np.load(f"{processed_data_folder}/{stage_sensor}/X_val.npy")
    y_val = np.load(f"{processed_data_folder}/{stage_sensor}/y_val.npy")

    # Компіляція та навчання моделі
    optimizer = Adam(learning_rate=suggested_params['learning_rate'])
    model.model.compile(optimizer=optimizer, loss='mse')
    model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=0)

    # Оцінка моделі
    y_pred = model.predict(X_val).flatten()
    mae = mean_absolute_error(y_val, y_pred)

    return mae


# Створення та запуск дослідження Optuna
for model_name in model_classes:
    for folder in stage_sensor_folders:
        stage, sensor = folder.split("_", 1)
        study = optuna.create_study(
            study_name=f"{study_name}_{model_name}_{stage}_{sensor}",
            direction="minimize",
        )
        study.optimize(
            lambda trial: objective(trial, model_name, folder),
            n_trials=10,
        )  # Змініть кількість ітерацій

        # Виведення результатів
        print(f"Best trial for {model_name} - {stage} - {sensor}:")
        print(f"  Value: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")

        # Збереження найкращих гіперпараметрів
        best_params = study.best_trial.params
        np.savez(
            f"best_params_{model_name}_{stage}_{sensor}.npz", **best_params
        )