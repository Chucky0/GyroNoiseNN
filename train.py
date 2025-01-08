import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_data, save_results_to_csv, save_plots, save_checkpoint, load_checkpoint, prepare_data_for_nn, \
    setup_logging
from utils_cuda import calculate_allan_variance_cuda, calculate_drift_rate_cuda, calculate_offset_cuda, \
    calculate_psd_cuda
from models import create_rnn_model, create_lstm_model, create_gru_model, create_cnn_model, create_cnn_lstm_model, \
    ModelWrapper
from metrics import calculate_metrics, generate_metrics_table
import tensorflow as tf
import logging
import yaml

# Завантаження конфігурації
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Налаштування логування
setup_logging(config["log_file"], config["log_level"])

# Параметри
file_paths = config["file_paths"]
sensor_names = config["sensor_names"]
stage1_end_time = config["stage1_end_time"]
stage2_start_time = config["stage2_start_time"]
stage2_end_time = config["stage2_end_time"]
stage3_start_time = config["stage3_start_time"]
window_size_nn = config["window_size_nn"]
epochs = config["epochs"]
batch_size = config["batch_size"]
val_size = config["val_size"]
checkpoint_file = config["checkpoint_file"]
start_from_checkpoint = False  # Змінити на True, якщо хочете почати з чекпоінту

# Ініціалізація словника для результатів
results = {}

# Перевірка наявності чекпоінту
if start_from_checkpoint:
    try:
        results = load_checkpoint(checkpoint_file)
        logging.info("Завантажено чекпоінт.")
    except FileNotFoundError:
        logging.warning("Файл чекпоінту не знайдено. Починаємо з початку.")

# Моделі
model_creation_functions = {
    "RNN": create_rnn_model,
    "LSTM": create_lstm_model,
    "GRU": create_gru_model,
    "CNN": create_cnn_model,
    "CNN_LSTM": create_cnn_lstm_model,
}

# Цикл по файлам та сенсорам
for file_path in file_paths:
    if file_path not in results:
        results[file_path] = {}
    for sensor_name in sensor_names:
        if sensor_name not in results[file_path]:
            results[file_path][sensor_name] = {}

        logging.info(f"Обробка: {file_path}, {sensor_name}")

        # Завантаження даних
        df = load_data(file_path)
        time = df['Time'].values
        sensor_data = df[sensor_name].values
        encoder_data = df['Encoder Speed'].values
        dt = df['dt'].mean() if len(df['dt']) > 0 else 0.001
        window_size = int(window_size_nn / dt)

        # Поділ на stage 1, 2, 3
        stage1_mask = (time <= stage1_end_time)
        stage2_mask = (time > stage2_start_time) & (time <= stage2_end_time)
        stage3_mask = (time > stage3_start_time)

        for stage, stage_mask in zip(["stage1", "stage2", "stage3"], [stage1_mask, stage2_mask, stage3_mask]):
            if stage not in results[file_path][sensor_name]:
                results[file_path][sensor_name][stage] = {}

            logging.info(f"Обробка етапу: {stage}")

            # Підготовка даних для нейронних мереж
            X, y = prepare_data_for_nn(df[[sensor_name]].values, df['Encoder Speed'].values, window_size, n_features=1)

            # Застосування маски етапу
            stage_mask_adjusted = stage_mask[window_size:]
            X_stage = X[stage_mask_adjusted]
            y_stage = y[stage_mask_adjusted]

            # Розбиття на навчальну, валідаційну та тестову вибірки
            X_train, X_temp, y_train, y_temp = train_test_split(X_stage, y_stage, test_size=val_size, random_state=42,
                                                                shuffle=False)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42,
                                                            shuffle=False)

            # Навчання моделей
            for model_name, model_creation_func in model_creation_functions.items():
                if model_name not in results[file_path][sensor_name][stage]:
                    results[file_path][sensor_name][stage][model_name] = {}

                    start_epoch = 0

                    model_wrapper = ModelWrapper(model_creation_func, (X_train.shape[1], X_train.shape[2]),
                                                 f"{model_name}_{stage}")

                    logging.info(f"Навчання {model_name} для {sensor_name} на {file_path} (етап {stage})...")

                    history = model_wrapper.train(X_train, y_train, X_val, y_val, epochs, batch_size)
                    y_pred = model_wrapper.predict(X_test)

                    # Зберігання результатів
                    results[file_path][sensor_name][stage][model_name] = {
                        "history": history.history,
                        "y_pred": y_pred.tolist(),  # Змінено на список
                        "time": time[window_size:][stage_mask_adjusted][len(X_train) + len(X_val):].tolist(),
                        "y_test": y_test.tolist(),
                        "model_path": os.path.join("model_weights",
                                                   f"{os.path.splitext(os.path.basename(file_path))[0]}_{sensor_name}_{model_name}_{stage}.tflite")
                    }
                    # Збереження моделі
                    model_wrapper.save(os.path.join("model_weights",
                                                    f"{os.path.splitext(os.path.basename(file_path))[0]}_{sensor_name}_{model_name}_{stage}"))

        # Збереження чекпоінту результатів
        save_checkpoint(results, checkpoint_file)

        # Розрахунок та збереження метрик
        for stage in ["stage1", "stage2", "stage3"]:
            # Отримуємо дані енкодера
            df = load_data(file_path)
            time = df['Time'].values
            encoder_data = df['Encoder Speed'].values
            dt = df['dt'].mean() if len(df['dt']) > 0 else 0.001
            window_size = int(window_size_nn / dt)

            # Поділ на stage 1, 2, 3
            stage_mask = None
            if stage == "stage1":
                stage_mask = (time <= stage1_end_time)
            elif stage == "stage2":
                stage_mask = (time > stage2_start_time) & (time <= stage2_end_time)
            elif stage == "stage3":
                stage_mask = (time > stage3_start_time)

            # adjusted_mask = stage_mask[:-window_size]
            # print("Довжина time:", len(time))
            # print("Довжина stage_mask:", len(stage_mask))
            # print("Довжина adjusted_mask:", len(adjusted_mask))
            stage_mask_adjusted = stage_mask[window_size:]
            y_test_stage = encoder_data[window_size:][stage_mask_adjusted]

            metrics_result = calculate_metrics(results[file_path][sensor_name][stage], y_test_stage, dt)
            results[file_path][sensor_name][stage]["metrics"] = metrics_result

            # Зберігаємо метрики
            metrics_df = pd.DataFrame(metrics_result)
            metrics_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{sensor_name}_{stage}_metrics.csv"
            metrics_df.to_csv(os.path.join(config["output_dir"], metrics_filename), index=False)

            # Генерація таблиці метрик
            generate_metrics_table(os.path.join(config["output_dir"], metrics_filename))

            # Зберігаємо y_pred та метрики
            y_pred_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{sensor_name}_{stage}_predictions.csv"
            y_pred_df = pd.DataFrame()
            for model_name, model_data in results[file_path][sensor_name][stage].items():
                if model_name != "metrics":
                    y_pred_df[f"{model_name}_pred"] = model_data["y_pred"]
                    y_pred_df["time"] = model_data["time"]

            y_pred_df.to_csv(os.path.join(config["output_dir"], y_pred_filename), index=False)

        # Збереження чекпоінту результатів
        save_checkpoint(results, checkpoint_file)

        # Збереження результатів у CSV та графіки
        save_results_to_csv(results)
        save_plots(results, stage2_start_time, stage2_end_time, time, output_dir=config["output_dir"])

logging.info("Обробка завершена. Результати збережено.")