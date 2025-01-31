import os
import pandas as pd
import numpy as np
import allantools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import glob
import math

# Параметри
dt = 1 / 108  # Частота дискретизації

# Моделі
models = {
    "LSTM": None,
    "CNN": None,
    "CNNLSTM": None,
    "TCN": None,
    "GRU": None
}

stage_boundaries = {
    0: (0, 277),
    1: (277, 1205),
    2: (1205, float('inf'))
}

def calculate_allan_deviation(data, rate, taus="octave"):
    """
    Розраховує Allan Deviation для часового ряду даних.
    """
    if len(data) < 10:
        print("Not enough data points to calculate Allan Deviation.")
        return None, None

    try:
        data_np = data.to_numpy() if isinstance(data, pd.Series) else data # Змінено
        (t2, ad, ade, adn) = allantools.oadev(data_np, rate=rate, data_type="freq", taus=taus) # Змінено
        return t2, ad
    except Exception as e:
        print(f"Error in Allan Deviation calculation: {e}")
        return None, None

def calculate_metrics_for_stage(df, sensor, model_name, stage, file_name, output_folder, dt):
    """
    Розраховує метрики (MAE, RMSE, R2, BI, ARW, RRW) та будує графік Allan Deviation
    для заданого файлу, моделі, етапу та сенсора. Зберігає метрики у CSV та графік у PNG.
    """
    y_true = df[f'{sensor}_input'].dropna()
    y_pred_col = f'{sensor}_{model_name}_stage_{stage}_predicted'
    y_pred = df[y_pred_col].dropna()

    # Перевірка на однакову довжину масивів
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if len(y_true) != len(y_pred):
        print(
            f"Skipping metrics calculation for model {model_name}, stage {stage}, sensor {sensor} in file {file_name} due to mismatched data lengths.")
        return None

    # Розрахунок метрик
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Розрахунок Allan Deviation
    taus_pred, allan_dev_pred = calculate_allan_deviation(y_pred, rate=dt)
    taus_actual, allan_dev_actual = calculate_allan_deviation(y_true, rate=dt)

    # Розрахунок Allan Deviation для різниці між передбаченим сигналом та сигналом з енкодера
    if 'Encoder Speed' in df.columns:
        diff_pred_encoder = y_pred - df['Encoder Speed'].values[:min_len]
        taus_diff, allan_dev_diff = calculate_allan_deviation(diff_pred_encoder, rate=dt)
    else:
        taus_diff, allan_dev_diff = None, None

    # Знаходження Bias Instability, ARW, RRW
    if allan_dev_pred is not None:
        min_ad_index = np.argmin(allan_dev_pred)
        bi = allan_dev_pred[min_ad_index] / 0.664
        arw_index = np.argmin(np.abs(taus_pred - 1))
        arw = allan_dev_pred[arw_index] * 60
        rrw_index = np.argmin(np.abs(taus_pred - 3600))
        rrw = allan_dev_pred[rrw_index] * 3600
    else:
        bi, arw, rrw = None, None, None

    # Збереження метрик
    metrics_data = {
        "File": file_name,
        "Stage": stage,
        "Sensor": sensor,
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Bias Instability": bi,
        "ARW": arw,
        "RRW": rrw
    }

    # Побудова графіка Allan Deviation
    if taus_pred is not None and allan_dev_pred is not None:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.loglog(taus_pred, allan_dev_pred, label="Predicted", marker='o')
        if taus_actual is not None and allan_dev_actual is not None:
            plt.loglog(taus_actual, allan_dev_actual, label="Actual", marker='x')
        plt.title(f"Allan Deviation - {model_name} - {stage} - {sensor} - {file_name}")
        plt.xlabel("Tau (s)")
        plt.ylabel("Allan Deviation")
        plt.legend()
        plt.grid(True)

        # Побудова графіка Allan Deviation (різниця)
        plt.subplot(1, 2, 2)
        if taus_diff is not None and allan_dev_diff is not None:
            plt.loglog(taus_diff, allan_dev_diff, label="Predicted - Encoder", marker='^')
        plt.title(f"Allan Deviation (Pred-Encoder) - {model_name} - {stage} - {sensor} - {file_name}")
        plt.xlabel("Tau (s)")
        plt.ylabel("Allan Deviation")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_filename = f"{model_name}_{file_name}_stage_{stage}_{sensor}_allan_deviation.png"
        plt.savefig(os.path.join(output_folder, plot_filename))
        plt.close()
        print(
            f"        Saved Allan Deviation plot to {os.path.join(output_folder, plot_filename)}")

    return metrics_data

def process_predictions(predictions_dir, output_dir):
    """
    Обробляє всі папки з передбаченнями, розраховує метрики та зберігає результати.
    """
    all_metrics = []

    # Створюємо папки для кожного етапу
    for stage in stage_boundaries.keys():
        os.makedirs(os.path.join(output_dir, f"stage_{stage}"), exist_ok=True)

    for folder_name in tqdm(os.listdir(predictions_dir), desc="Processing predictions"):
        folder_path = os.path.join(predictions_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name.endswith("_predictions"):
            file_name = folder_name.replace("_predictions", "")
            print(f"Processing folder: {folder_name}")

            # Зчитуємо всі файли з передбаченнями в цій папці
            combined_predictions_df = pd.DataFrame()
            for file in glob.glob(os.path.join(folder_path, "*_predictions.csv")):
                try:
                    df = pd.read_csv(file)
                    combined_predictions_df = pd.concat([combined_predictions_df, df], ignore_index=False)
                except pd.errors.EmptyDataError:
                    print(f"        Warning: Empty CSV file encountered: {file}. Skipping.")
                    continue
                except Exception as e:
                    print(f"        Error reading {file}: {e}")
                    continue

            if combined_predictions_df.empty:
                print(f"        Warning: No valid data found in {folder_name}. Skipping metrics calculation.")
                continue

            # Для кожного етапу та сенсора розраховуємо метрики
            for stage, (start_time, end_time) in stage_boundaries.items():
              print(f"  Processing stage: {stage}")
              stage_metrics = []
              for sensor in ["N1Gyro Z", "N8Gyro Z", "NVGyro Z"]:
                print(f"    Sensor: {sensor}")
                for model_name in models:

                  # Фільтруємо дані для поточного етапу та сенсора
                  stage_data = combined_predictions_df[
                      (combined_predictions_df['Time'] > start_time) & (combined_predictions_df['Time'] <= end_time)
                  ]

                  if stage_data.empty:
                      print(f"      No data found for Stage: {stage}, Sensor: {sensor}. Skipping.")
                      continue

                  # Перевірка, чи є дана модель в поточному датафреймі
                  if model_name not in stage_data['Model'].unique():
                      print(f"        Model {model_name} not found for Stage: {stage}, Sensor: {sensor}. Skipping.")
                      continue
                  # Фільтруємо дані для поточної моделі
                  model_data = stage_data[stage_data['Model'] == model_name]

                  # Розраховуємо метрики
                  metrics_data = calculate_metrics_for_stage(model_data, sensor, model_name, stage, file_name,
                                                             os.path.join(output_dir, f"stage_{stage}"), dt)
                  if metrics_data is not None:
                      stage_metrics.append(metrics_data)

              # Зберігаємо метрики для поточного етапу в окремий файл
              if stage_metrics:
                stage_metrics_df = pd.DataFrame(stage_metrics)
                output_file_name = f"{file_name}_stage_{stage}_metrics.csv"
                output_file_path = os.path.join(output_dir, f"stage_{stage}", output_file_name)
                stage_metrics_df.to_csv(output_file_path, index=False)
                print(f"    Saved metrics for stage {stage} to {output_file_path}")

                all_metrics.extend(stage_metrics)

    # Зберігаємо зведені метрики для всіх етапів
    if all_metrics:
        all_metrics_df = pd.DataFrame(all_metrics)
        all_metrics_filepath = os.path.join(output_dir, "all_metrics.csv")
        all_metrics_df.to_csv(all_metrics_filepath, index=False)
        print(f"Saved all metrics to {all_metrics_filepath}")
    else:
        print("No metrics were calculated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics from predictions.")
    parser.add_argument("--predictions_dir", type=str, default="predictions",
                        help="Path to the directory containing predictions folders.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_by_stage",
                        help="Path to the directory to save results.")
    args = parser.parse_args()

    if not os.path.exists(args.predictions_dir):
        print(f"Error: Predictions directory not found: {args.predictions_dir}")
        exit()

    os.makedirs(args.output_dir, exist_ok=True)
    for stage in range(6):
        os.makedirs(os.path.join(args.output_dir, f"stage_{stage}"), exist_ok=True)

    process_predictions(args.predictions_dir, args.output_dir)