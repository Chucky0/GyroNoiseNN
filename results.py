import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Завантаження результатів
results = np.load("evaluation_results.npz", allow_pickle=True).item()

# Створення DataFrame для кожної метрики
mae_df = pd.DataFrame()
rmse_df = pd.DataFrame()
r2_df = pd.DataFrame()

for model_name, model_results in results.items():
    for stage_sensor, metrics in model_results.items():
        mae_df.loc[stage_sensor, model_name] = metrics["MAE"]
        rmse_df.loc[stage_sensor, model_name] = metrics["RMSE"]
        r2_df.loc[stage_sensor, model_name] = metrics["R2"]

# Створення зведеної таблиці
summary_table = pd.concat([mae_df, rmse_df, r2_df], keys=['MAE', 'RMSE', 'R2'])

# Збереження зведеної таблиці у CSV
summary_table.to_csv("summary_table.csv")

# Виведення зведеної таблиці
print("Summary Table (MAE, RMSE, R2):")
print(summary_table)

# Побудова графіків
for metric in ['MAE', 'RMSE', 'R2']:
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame(
        {model_name: {stage_sensor: results[model_name][stage_sensor][metric] for stage_sensor in results[model_name]}
         for model_name in results})
    df.plot(kind='bar')
    plt.title(f'{metric} Comparison')
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{metric}_comparison.png')
    plt.close()

    # Побудова графіків Allan Deviation (якщо є дані)
    for model_name in results:
        for stage_sensor in results[model_name]:
            if "Allan Deviation" in results[model_name][stage_sensor] and results[model_name][stage_sensor][
                "Allan Deviation"] is not None:
                plt.figure(figsize=(12, 6))

                # Перевірка, чи є дані не None
                if results[model_name][stage_sensor]["Allan Deviation"] is not None:
                    plt.loglog(results[model_name][stage_sensor]["Allan Deviation"], label="Actual")

                if results[model_name][stage_sensor]["Allan Deviation Predicted"] is not None:
                    plt.loglog(results[model_name][stage_sensor]["Allan Deviation Predicted"], label="Predicted")

                plt.title(f"Allan Deviation - {model_name} - {stage_sensor}")
                plt.xlabel("Tau (s)")
                plt.ylabel("Allan Deviation")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{model_name}_{stage_sensor}_allan_deviation_comparison.png")
                plt.close()