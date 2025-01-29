# plot_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import re

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import pacf, acf

from prepare_data import window_size


def plot_from_csv(output_folder="evaluation_results_with_integration_and_allan"):
    """
    Читає CSV файли з папки output_folder, будує та зберігає графіки.

    Args:
        output_folder (str): Шлях до папки з результатами.
    """
    for filename in os.listdir(output_folder):
        if filename.endswith("_predictions.csv"):
            df = pd.read_csv(os.path.join(output_folder, filename))

            model_full_name = df['Model_Full_Name'][0]
            file_name = df['File'][0]
            stage = df['Stage'][0]

            parts = model_full_name.split("_")
            model_name = parts[0]

            # Extracting the sensor name using regular expression
            sensor_match = re.search(r"(N\dGyro [XYZ])", model_full_name)
            sensor = sensor_match.group(1) if sensor_match else "Unknown Sensor"

            # Time Series Plot
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.plot(df['Time'], df[f'{sensor}_input'], label="Actual")
            plt.plot(df['Time'], df[f'{sensor}_{model_name}_predicted'], label="Predicted")
            plt.title(f"Time Series - {model_name} - Stage {stage} - {sensor} - {file_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Value")
            plt.legend()

            # Allan Deviation Plot
            plt.subplot(1, 3, 2)
            # Assuming Allan Deviation data is saved in the CSV,
            # we need to regenerate it from predictions
            from evaluate_gyro_models import calculate_allan_deviation, dt
            taus_pred, allan_dev_pred = calculate_allan_deviation(
                df[f'{sensor}_{model_name}_predicted'].dropna().values, rate=1 / dt)
            taus_actual, allan_dev_actual = calculate_allan_deviation(df[f'{sensor}_input'].dropna().values,
                                                                      rate=1 / dt)
            taus_encoder, allan_dev_encoder = calculate_allan_deviation(df['Encoder Speed'].dropna().values,
                                                                        rate=1 / dt)

            if allan_dev_pred is not None and taus_pred is not None:
                plt.loglog(taus_pred, allan_dev_pred, label="Predicted", marker='o')
            if allan_dev_actual is not None and taus_actual is not None:
                plt.loglog(taus_actual, allan_dev_actual, label="Actual", marker='x')
            if allan_dev_encoder is not None and taus_encoder is not None:
                plt.loglog(taus_encoder, allan_dev_encoder, label="Encoder Speed", marker='s')
            plt.title(f"Allan Deviation - {model_name} - Stage {stage} - {sensor} - {file_name}")
            plt.xlabel("Tau (s)")
            plt.ylabel("Allan Deviation")
            plt.legend()
            plt.grid(True)

            # Integrated Signal Plot
            plt.subplot(1, 3, 3)
            time = df['Time'].values
            encoder_integrated = np.cumsum(df['Encoder Speed'].values) * dt
            sensor_integrated = np.cumsum(df[f'{sensor}_input'].values) * dt
            predicted_integrated = np.cumsum(df[f'{sensor}_{model_name}_predicted'].dropna().values) * dt

            plt.plot(time, encoder_integrated, label="Encoder Integrated")
            plt.plot(time, sensor_integrated, label=f"{sensor} Integrated")
            plt.plot(time[:len(predicted_integrated)], predicted_integrated, label=f"Predicted {sensor} Integrated")
            plt.title(f"Integrated Signal - {model_name} - Stage {stage} - {sensor} - {file_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Integrated Value")
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{model_full_name}_{file_name}_plots.png"))
            plt.close()
            print(f"Saved plots to {os.path.join(output_folder, f'{model_full_name}_{file_name}_plots.png')}")

            # Time Series Decomposition and ACF/PACF
            y_pred = df[f'{sensor}_{model_name}_predicted'].dropna().values
            if len(y_pred) >= 2 * window_size:
                try:
                    result = seasonal_decompose(y_pred, model='additive', period=window_size, extrapolate_trend='freq')

                    # ACF and PACF Calculation
                    lag_acf = acf(y_pred, nlags=20)
                    lag_pacf = pacf(y_pred, nlags=20, method='ols')

                    # Decomposition Plot
                    plt.figure(figsize=(12, 8))
                    plt.subplot(4, 1, 1)
                    plt.plot(y_pred)
                    plt.title(f'Decomposition - {model_name} - Stage {stage} - {sensor} - {file_name}')
                    plt.ylabel('Original')
                    plt.subplot(4, 1, 2)
                    plt.plot(result.trend)
                    plt.ylabel('Trend')
                    plt.subplot(4, 1, 3)
                    plt.plot(result.seasonal)
                    plt.ylabel('Seasonal')
                    plt.subplot(4, 1, 4)
                    plt.plot(result.resid)
                    plt.ylabel('Residual')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, f"{model_full_name}_{file_name}_decomposition.png"))
                    plt.close()
                    print(
                        f"    Saved Decomposition plot to {os.path.join(output_folder, f'{model_full_name}_{file_name}_decomposition.png')}")

                    # ACF/PACF Plots
                    plt.figure(figsize=(12, 4))
                    plt.subplot(121)
                    plt.stem(lag_acf)
                    plt.axhline(y=0, linestyle='--', color='gray')
                    plt.axhline(y=-1.96 / np.sqrt(len(y_pred)), linestyle='--', color='gray')
                    plt.axhline(y=1.96 / np.sqrt(len(y_pred)), linestyle='--', color='gray')
                    plt.title(f'Autocorrelation Function - {model_name} - {stage} - {sensor} - {file_name}')

                    plt.subplot(122)
                    plt.stem(lag_pacf)
                    plt.axhline(y=0, linestyle='--', color='gray')
                    plt.axhline(y=-1.96 / np.sqrt(len(y_pred)), linestyle='--', color='gray')
                    plt.axhline(y=1.96 / np.sqrt(len(y_pred)), linestyle='--', color='gray')
                    plt.title(f'Partial Autocorrelation Function - {model_name} - {stage} - {sensor} - {file_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, f"{model_full_name}_{file_name}_acf_pacf.png"))
                    plt.close()
                    print(
                        f"    Saved ACF/PACF plots to {os.path.join(output_folder, f'{model_full_name}_{file_name}_acf_pacf.png')}")

                except ImportError:
                    print("Module 'statsmodels' not found. Skipping time series decomposition and ACF/PACF.")
                except ValueError as e:
                    print(f"Error during time series decomposition or ACF/PACF calculation: {e}")
            else:
                print(
                    f"Not enough data for decomposition or ACF/PACF for model {model_name} stage {stage} sensor {sensor} in file {file_name}")


if __name__ == "__main__":
    plot_from_csv()