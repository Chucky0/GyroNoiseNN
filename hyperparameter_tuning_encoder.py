import numpy as np
import os
import optuna
from model_encoder import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import RMSprop
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tensorflow as tf

# Задайте шлях до папки з обробленими даними
processed_data_folder = "processed_data"

# Параметри
window_size = 100
num_sensors = 3  # Кількість вхідних сенсорів
study_name = "hyperparameter_tuning"  # Назва дослідження
max_memory_percent = 0.85  # Збільшено до 85% від 40 гб
max_batch_size = 256  # Експерементально максимальний batch_size
n_trials = 10 # Кількість випробувань
# Папка для збереження чекпоїнтів
checkpoint_dir = "optuna_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

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
        "learning_rate": (1e-5, 5e-5), # звужено
    },
    "CNN": {
        "filters": (16, 128),
        "kernel_size": (3, 7),
        "dense_units": (16, 128),
        "learning_rate": (1e-5, 5e-5), # звужено
    },
    "CNNLSTM": {
        "conv_filters": (16, 128),
        "conv_kernel_size": (3, 5),
        "lstm_units": (16, 128),
        "learning_rate": (1e-5, 5e-5), # звужено
    },
    "TCN": {
        "filters": (16, 128),
        "kernel_size": (3, 7),
        "dense_units": (16, 128),
        "learning_rate": (1e-5, 5e-5), # звужено
    },
    "GRU": {
        "units": (16, 128),
        "learning_rate": (1e-5, 5e-5), # звужено
    },
}

# Отримання списку всіх папок з даними
stage_sensor_folders = [
    d
    for d in os.listdir(processed_data_folder)
    if os.path.isdir(os.path.join(processed_data_folder, d))
]

# Виключення проблемних наборів даних
stage_sensor_folders = [
    folder for folder in stage_sensor_folders if "stage_1" not in folder
]


def objective(trial, model_name, stage_sensor, num_sensors):
    # Вибір моделі та гіперпараметрів
    ModelClass = model_classes[model_name]
    model = ModelClass(window_size, num_sensors)

    # Конфігурація моделі згідно з гіперпараметрами
    if model_name == "LSTM":
        suggested_params = {
            "units": trial.suggest_int("units", *params[model_name]["units"]),
            "learning_rate": trial.suggest_float("learning_rate", *params[model_name]["learning_rate"], log=True),
        }
    elif model_name == "CNN":
        suggested_params = {
            "filters": trial.suggest_int("filters", *params[model_name]["filters"]),
            "kernel_size": trial.suggest_int("kernel_size", *params[model_name]["kernel_size"]),
            "dense_units": trial.suggest_int("dense_units", *params[model_name]["dense_units"]),
            "learning_rate": trial.suggest_float("learning_rate", *params[model_name]["learning_rate"], log=True),
        }
    elif model_name == "CNNLSTM":
        suggested_params = {
            "conv_filters": trial.suggest_int("conv_filters", *params[model_name]["conv_filters"]),
            "conv_kernel_size": trial.suggest_int("conv_kernel_size", *params[model_name]["conv_kernel_size"]),
            "lstm_units": trial.suggest_int("lstm_units", *params[model_name]["lstm_units"]),
            "learning_rate": trial.suggest_float("learning_rate", *params[model_name]["learning_rate"], log=True),
        }
    elif model_name == "TCN":
        suggested_params = {
            "filters": trial.suggest_int("filters", *params[model_name]["filters"]),
            "kernel_size": trial.suggest_int("kernel_size", *params[model_name]["kernel_size"]),
            "dense_units": trial.suggest_int("dense_units", *params[model_name]["dense_units"]),
            "learning_rate": trial.suggest_float("learning_rate", *params[model_name]["learning_rate"], log=True),
        }
    elif model_name == "GRU":
        suggested_params = {
            "units": trial.suggest_int("units", *params[model_name]["units"]),
            "learning_rate": trial.suggest_float("learning_rate", *params[model_name]["learning_rate"], log=True),
        }

    model.build_with_params(**suggested_params)

    # Завантаження даних
    X_train = np.load(f"{processed_data_folder}/{stage_sensor}/X_train.npy")
    y_train = np.load(f"{processed_data_folder}/{stage_sensor}/y_train.npy")
    X_val = np.load(f"{processed_data_folder}/{stage_sensor}/X_val.npy")
    y_val = np.load(f"{processed_data_folder}/{stage_sensor}/y_val.npy")

    # Розрахунок розміру батчу на основі доступної пам'яті
    dataset_size = X_train.nbytes + y_train.nbytes + X_val.nbytes + y_val.nbytes
    available_memory = max_memory_percent * 40 * (1024 ** 3)  # 40 ГБ у байтах

    batch_size = int(min(max_batch_size, available_memory / dataset_size))
    batch_size = max(1, batch_size)  # Переконайтеся, що batch_size не є нулем

    # Компіляція та навчання моделі
    optimizer = RMSprop(learning_rate=suggested_params['learning_rate'])
    model.model.compile(optimizer=optimizer, loss='mse')

    # Перевірка передбачень до тренування
    y_pred_before = model.predict(X_val[:5]).flatten()
    print(f"Predictions before training ({model_name} - {stage_sensor}): {y_pred_before}")

    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=batch_size)

    # Оцінка моделі
    y_pred = model.predict(X_val).flatten()

    print(f"Predictions ({model_name} - {stage_sensor}): {y_pred}")

    # Перевірка на NaN
    if np.isnan(y_pred).any():
        print(f"WARNING: NaN detected in predictions for {model_name} - {stage_sensor.replace(':', '-')}. Returning a large value to penalize.")
        model.save(f"model_{model_name}_{stage_sensor.replace(':', '-')}_with_nan_predictions.keras")
        return float('inf') # Змінено на нескінченність

    # Перевірка чи повернув train None
    if history is None:
        print(f"WARNING: Training failed for {model_name} - {stage_sensor.replace(':', '-')}. Returning a large value to penalize.")
        return float('inf') # Змінено на нескінченність

    mae = mean_absolute_error(y_val, y_pred)

    print(f"MAE ({model_name} - {stage_sensor}): {mae}")

    # Звільнення пам'яті
    del model
    tf.keras.backend.clear_session()

    return mae


# Функція-обгортка для багатопоточного виклику з підтримкою tqdm та чекпоїнтів
def objective_wrapper(args):
    model_name, folder, num_sensors = args
    study_filepath = os.path.join(checkpoint_dir, f"study_{model_name}_{folder.replace(':', '-')}.db")

    # Створюємо study, якщо воно ще не існує
    try:
        if os.path.isfile(study_filepath):
            # Продовжуємо існуючий
            study = optuna.load_study(study_name=f"{study_name}_{model_name}_{folder}",
                                      storage=f"sqlite:///{study_filepath}")
            print(f"Resuming study {study_name}_{model_name}_{folder} from checkpoint")
        else:
            # Починаємо нове
            study = optuna.create_study(study_name=f"{study_name}_{model_name}_{folder}",
                                          storage=f"sqlite:///{study_filepath}",
                                          direction="minimize")
            print(f"Creating new study {study_name}_{model_name}_{folder}")

        # Запускаємо оптимізацію з обробкою помилок
        try:
            study.optimize(
                lambda trial: objective(trial, model_name, folder, num_sensors),
                n_trials=n_trials,
                catch=(Exception,),  # Обробка виключень
            )
        except Exception as e:
            print(f"Error during optimization: {e}")

        # Виведення результатів
        # ПЕРЕВІРКА НА NONE
        trial = study.best_trial
        if trial is not None:
          print(f"Best trial for {model_name} - {folder}:")
          print(f"  Value: {trial.value}")
          print(f"  Params: {trial.params}")

          # Збереження найкращих гіперпараметрів
          best_params = trial.params
          np.savez(
              f"best_params_{model_name}_{folder.replace(':', '-')}.npz", **best_params
          )
          return f"Finished {model_name} - {folder} (Best Value: {trial.value})"
        else:
          return f"No trials completed successfully for {model_name} - {folder}"
    except Exception as e:
        print(f"Error occurred during optimization for {model_name} - {folder}: {e}")
        return f"Failed {model_name} - {folder}: {e}"


# Створення та запуск дослідження Optuna з багатопоточністю та індикатором прогресу
total_tasks = len(model_classes) * len(stage_sensor_folders)

with ThreadPoolExecutor(max_workers=4) as executor:
    args_list = [
        (model_name, folder, num_sensors)
        for model_name in model_classes
        for folder in stage_sensor_folders
    ]

    # Використовуємо tqdm для відстеження прогресу
    with tqdm(total=total_tasks, desc="Optimizing Hyperparameters") as progress_bar:
        # Виконуємо задачі у пулі потоків та оновлюємо прогрес бар після завершення кожної задачі
        for _ in executor.map(objective_wrapper, args_list):
            progress_bar.update(1)