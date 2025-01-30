import argparse
import os
import pandas as pd
import numpy as np
import allantools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import json

# Параметри
dt = 1 / 108  # Частота дискретизації

def calculate_allan_deviation(data, rate, taus="octave"):
    """
    Розраховує Allan Deviation для часового ряду даних.
    """
    if len(data) < 2:
        print("Not enough data points to calculate Allan Deviation.")
        return None, None

    try:
        (t2, ad, ade, adn) = allantools.oadev(data, rate=rate, data_type="freq", taus=taus)
        return t2, ad
    except Exception as e:
        print(f"Error in Allan Deviation calculation: {e}")
        return None, None

def calculate_metrics_for_group(group, sensor, model_name, stage, file_name):
    """
    Розраховує метрики для однієї групи (модель, етап, сенсор, файл).
    """
    y_true = group[f'{sensor}_input'].dropna()

    # Перевіряємо наявність і відповідність індексу
    if len(y_true) != len(group[f'{sensor}_{model_name}_stage_{stage}_predicted'].dropna()):
        print(f"Skipping metrics calculation for model {model_name}, stage {stage}, sensor {sensor} in file {file_name} due to mismatched data lengths.")
        return None

    y_pred = group[f'{sensor}_{model_name}_stage_{stage}_predicted'].dropna()

    # Розрахунок метрик
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Розрахунок Allan Deviation
    taus_pred, allan_dev_pred = calculate_allan_deviation(y_pred, rate=1 / dt)

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

    return {
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

def calculate_metrics(predictions_filepath, output_folder):
    """
    Зчитує файл з передбаченнями, розраховує метрики та зберігає їх.
    """
    try:
        all_predictions_df = pd.read_csv(predictions_filepath)
    except FileNotFoundError:
        print(f"Error: Predictions file not found: {predictions_filepath}")
        return

    all_metrics = []

    for (file_name, model_name, stage), group in tqdm(
        all_predictions_df.groupby(['File', 'Model', 'Stage']),
        desc="Calculating metrics"
    ):
        for sensor in ["N1Gyro Z", "N8Gyro Z", "NVGyro Z"]:
            metrics_data = calculate_metrics_for_group(group, sensor, model_name, stage, file_name)
            if metrics_data is not None:
                all_metrics.append(metrics_data)

    all_metrics_df = pd.DataFrame(all_metrics)

    # Зберігаємо зведені метрики
    all_metrics_df.to_csv(os.path.join(output_folder, "all_metrics.csv"), index=False)
    print(f"Saved all metrics to {os.path.join(output_folder, 'all_metrics.csv')}")

    # Групуємо метрики за моделлю та етапом і зберігаємо в окремі CSV файли
    for (model_name, stage), group in all_metrics_df.groupby(['Model', 'Stage']):
        output_file_name = f"{model_name}_stage_{stage}_metrics.csv"
        group.to_csv(os.path.join(output_folder, output_file_name), index=False)
        print(f"Saved metrics for {model_name} (Stage {stage}) to {output_file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics from predictions.")
    parser.add_argument("--predictions_file", type=str, default="predictions/all_predictions.csv",
                        help="Path to the CSV file containing predictions.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_with_integration_and_allan",
                        help="Path to the directory to save results.")
    args = parser.parse_args()

    if not os.path.exists(args.predictions_file):
        print(f"Error: Predictions file not found: {args.predictions_file}")
        exit()

    os.makedirs(args.output_dir, exist_ok=True)

    calculate_metrics(args.predictions_file, args.output_dir)