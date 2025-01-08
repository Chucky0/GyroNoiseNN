import numpy as np
import pandas as pd
from utils import calculate_vrw, calculate_bi, calculate_rrw, calculate_correlation_and_rmse
from utils_cuda import calculate_allan_variance_cuda, calculate_drift_rate_cuda, calculate_offset_cuda, calculate_psd_cuda

def calculate_metrics(model_data, y_true, dt):
    """
    Рахує метрики для кожної моделі.

    Args:
        model_data (dict): Словник з результатами передбачень моделей.
        y_true (np.array): Реальні значення (Encoder Speed).
        dt (float): Крок дискретизації.

    Returns:
        dict: Словник з метриками.
    """
    metrics_results = {}
    for model_name, data in model_data.items():
        if model_name != "metrics":
            y_pred = data["y_pred"]

            min_len = min(len(y_pred), len(y_true))
            y_pred_trimmed = y_pred[:min_len]
            y_true_trimmed = y_true[:min_len]

            # Розрахунок метрик
            correlation, rmse = calculate_correlation_and_rmse(y_true_trimmed, y_pred_trimmed)

            taus, allan_var = calculate_allan_variance_cuda(y_pred, dt)
            vrw = calculate_vrw(taus, allan_var)
            bi = calculate_bi(taus, allan_var)
            rrw = calculate_rrw(taus, allan_var)
            drift_rate = calculate_drift_rate_cuda(y_pred, dt)
            offset = calculate_offset_cuda(y_pred)
            freqs, psd = calculate_psd_cuda(y_pred, dt)

            metrics_results[model_name] = {
                "correlation": correlation,
                "rmse": rmse,
                "vrw": vrw,
                "bi": bi,
                "rrw": rrw,
                "drift_rate": drift_rate,
                "offset": offset,
                "psd_freqs": freqs.tolist(),  # Перетворення в list для серіалізації
                "psd_values": psd.tolist(),  # Перетворення в list для серіалізації
                "allan_taus": taus.tolist(),
                "allan_vars": allan_var.tolist()
            }

    return metrics_results

def generate_metrics_table(metrics_file):
    """
    Генерує зведену таблицю метрик з файлу CSV.

    Args:
        metrics_file (str): Шлях до CSV файлу з метриками.
    """

    try:
        df = pd.read_csv(metrics_file)
        # Видалення непотрібних колонок з psd данними, бо вони дуже великі
        df = df.drop(columns=["psd_freqs", "psd_values", "allan_taus", "allan_vars"])
    except FileNotFoundError:
        print(f"Файл {metrics_file} не знайдено.")
        return
    except KeyError:
        print(f"У файлі {metrics_file} відсутні необхідні колонки. Перевірте наявність колонок 'psd_freqs', 'psd_values', 'allan_taus', 'allan_vars'")
        return

    # Створення зведеної таблиці
    pivot_table = df.pivot_table(index=df.columns[0]).T

    # Збереження таблиці у новий CSV файл
    output_file = metrics_file.replace(".csv", "_summary.csv")
    pivot_table.to_csv(output_file)
    print(f"Зведена таблиця метрик збережена у {output_file}")