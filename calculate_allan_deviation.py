import os
import numpy as np
import pandas as pd
from statsmodels.sandbox.regression.sympy_diff import df
from tqdm import tqdm
import allantools
import matplotlib.pyplot as plt

from prepare_data import window_size


def calculate_allan_deviation(data, rate, taus="octave"):
    """
    Розраховує Allan Deviation для часового ряду даних.

    Параметри:
    data (np.ndarray): Вхідний часовий ряд.
    rate (float): Частота дискретизації даних.
    taus (str або np.ndarray): Точки, у яких обчислюється Allan Deviation.
                               Якщо 'octave', то використовуються степені двійки.

    Повертає:
    (taus, allan_dev): Кортеж з точками tau та відповідними значеннями Allan Deviation.
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


def calculate_allan_params_for_all_models(predictions_file, output_folder, dt):
    """
    Розраховує Allan Deviation та похідні метрики для всіх моделей та зберігає результати.

    Args:
        predictions_file (str): Шлях до CSV файлу з передбаченнями.
        output_folder (str): Шлях до папки для збереження результатів.
        dt (float): Крок дискретизації.
    """
    predictions_df = pd.read_csv(predictions_file)

    allan_params_df = pd.DataFrame(columns=["File", "Stage", "Sensor", "Model", "Bias Instability", "ARW", "RRW"])

    for (file_name, stage, sensor, model_name), group in predictions_df.groupby(['File', 'Stage', 'Sensor', 'Model']):
        y_pred = group[f'{sensor}_{model_name}_predicted'].dropna().values

        # Розрахунок Allan Deviation для передбаченого сигналу
        taus_pred, allan_dev_pred = calculate_allan_deviation(y_pred, rate=1 / dt)

        # Розрахунок Allan Deviation для сигналу з енкодера
        if 'Encoder Speed' in df.columns:
            taus_encoder, allan_dev_encoder = calculate_allan_deviation(df['Encoder Speed'][window_size - 1:],
                                                                        rate=1 / dt)

            # Розрахунок Allan Deviation для різниці між передбаченим сигналом та сигналом з енкодера
            diff_pred_encoder = y_pred - df['Encoder Speed'].values[window_size - 1:]
            taus_diff, allan_dev_diff = calculate_allan_deviation(diff_pred_encoder, rate=1 / dt)

        else:
            taus_encoder, allan_dev_encoder = None, None
            taus_diff, allan_dev_diff = None, None

        # Знаходження мінімального значення Allan Deviation та відповідного йому tau (Bias Instability)
        if allan_dev_pred is not None:
            min_ad_index = np.argmin(allan_dev_pred)
            bi = allan_dev_pred[min_ad_index] / 0.664