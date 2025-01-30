import argparse
import glob
import os
import pandas as pd
import numpy as np
import allantools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def calculate_metrics_for_file(predictions_folder, output_folder, dt):
    """
    Розраховує метрики для всіх моделей в одному файлі predictions.csv.
    """
    metrics_list = []
    predictions_files = glob.glob(os.path.join(predictions_folder, "*_predictions.csv"))

    for predictions_file in predictions_files:
        try:
            predictions_df = pd.read_csv(predictions_file)

            model_full_name = predictions_df['Model_Full_Name'][0]
            parts = model_full_name.split("_")
            model_name = parts[0]
            file_name = parts[1]
            stage = parts[-2].replace('stage', '')
            sensor = parts[-1]

            # Розрахунок метрик
            y_true = predictions_df[f'{sensor}_input']
            y_pred_col = [col for col in predictions_df.columns if col.endswith('_predicted')]
            y_pred = predictions_df[y_pred_col[0]]

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            # Розрахунок Allan Deviation
            taus_pred, allan_dev_pred = calculate_allan_deviation(y_pred, rate=dt)
            taus_actual, allan_dev_actual = calculate_allan_deviation(y_true, rate=dt)

            # Розрахунок Allan Deviation для різниці між передбаченим сигналом та сигналом з енкодера
            if 'Encoder Speed' in predictions_df.columns:
                diff_pred_encoder = y_pred - predictions_df['Encoder Speed'].values
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
            metrics_list.append(metrics_data)

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
                plt.savefig(os.path.join(predictions_folder, f"{model_full_name}_{file_name}_allan_deviation.png"))
                plt.close()
                print(
                    f"    Saved Allan Deviation plot to {os.path.join(predictions_folder, f'{model_full_name}_{file_name}_allan_deviation.png')}")

        except Exception as e:
            print(f"Error processing {predictions_file}: {e}")

    return pd.DataFrame(metrics_list)

def process_predictions(predictions_dir, output_dir):
    """
    Обробляє всі папки з передбаченнями, розраховує метрики та зберігає результати.
    """
    all_metrics = []

    for folder_name in tqdm(os.listdir(predictions_dir), desc="Processing predictions"):
        folder_path = os.path.join(predictions_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name.endswith("_predictions"):
            file_name = folder_name.replace("_predictions", "")
            print(f"Processing folder: {folder_name}")

            metrics_df = calculate_metrics_for_file(folder_path, output_dir, dt)
            if metrics_df is not None:
                all_metrics.append(metrics_df)

    # Зберігаємо зведені метрики
    if all_metrics:
        all_metrics_df = pd.concat(all_metrics)
        all_metrics_filepath = os.path.join(output_dir, "all_metrics.csv")
        all_metrics_df.to_csv(all_metrics_filepath, index=False)
        print(f"Saved all metrics to {all_metrics_filepath}")
    else:
        print("No metrics were calculated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics from predictions.")
    parser.add_argument("--predictions_dir", type=str, default="predictions",
                        help="Path to the directory containing predictions.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results_with_integration_and_allan",
                        help="Path to the directory to save results.")
    args = parser.parse_args()

    if not os.path.exists(args.predictions_dir):
        print(f"Error: Predictions directory not found: {args.predictions_dir}")
        exit()

    os.makedirs(args.output_dir, exist_ok=True)
    process_predictions(args.predictions_dir, args.output_dir)