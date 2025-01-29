import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def calculate_metrics(predictions_df, sensor, model_name):
    """
    Розраховує метрики MAE, RMSE та R2 для заданої моделі та сенсора.

    Args:
        predictions_df (pd.DataFrame): DataFrame з передбаченнями та реальними даними.
        sensor (str): Назва сенсора.
        model_name (str): Назва моделі.

    Returns:
        dict: Словник з розрахованими метриками.
    """
    y_true = predictions_df[f'{sensor}_input']
    y_pred = predictions_df[f'{sensor}_{model_name}_predicted']

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def calculate_metrics_for_all_models(predictions_file, output_folder):
    """
    Розраховує метрики для всіх моделей, збережених у файлі predictions_file.

    Args:
        predictions_file (str): Шлях до CSV файлу з передбаченнями.
        output_folder (str): Шлях до папки для збереження результатів.
    """

    predictions_df = pd.read_csv(predictions_file)
    metrics_df = pd.DataFrame(
            columns=["File", "Stage", "Sensor", "Model", "MAE", "RMSE", "R2"])

    for (file_name, stage, sensor, model_name), group in predictions_df.groupby(['File', 'Stage', 'Sensor', 'Model']):
        metrics = calculate_metrics(group, sensor, model_name)
        metrics_data = {
            "File": file_name,
            "Stage": stage,
            "Sensor": sensor,
            "Model": model_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_data])], ignore_index=True)

    # Збереження загальних метрик у CSV файл
    metrics_filepath = os.path.join(output_folder, f"all_metrics.csv")
    metrics_df.to_csv(metrics_filepath, index=False)
    print(f"Saved all metrics to {metrics_filepath}")

if __name__ == '__main__':
    predictions_file = os.path.join(evaluation_results_dir, "all_predictions.csv")
    calculate_metrics_for_all_models(predictions_file, evaluation_results_dir)