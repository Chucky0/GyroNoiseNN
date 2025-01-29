import os
import json
import numpy as np
import pandas as pd
from model import CNNModel  # Змінено: Використовуємо тільки CNN, як у прикладі шляхів
from prepare_data import prepare_data_for_model

# Параметри
window_size = 100
dt = 1 / 108  # Частота дискретизації

# Шляхи до файлу моделі, конфігурації та даних
model_path = r"C:\Users\GreatTomato\Desktop\GyroDataProcessing\NNEval\models_for_predict\models_with_predictions\CNNLSTM\processed_data\ProgessiveOscill0_stage_0_NVGyro Z\CNNLSTM_ProgessiveOscill0_stage_0_NVGyro Z.keras"
params_path = r"C:\Users\GreatTomato\Desktop\GyroDataProcessing\NNEval\models_for_predict\models_with_predictions\CNNLSTM\processed_data\ProgessiveOscill0_stage_0_NVGyro Z\CNNLSTM_ProgessiveOscill0_stage_0_NVGyro Z_params.json"
data_path = r"C:\Users\GreatTomato\Desktop\GyroDataProcessing\NNEval\models_for_predict\ProgessiveOscill0.xlsx"
output_file = r"C:\Users\GreatTomato\Desktop\GyroDataProcessing\NNEval\predictions\predictions.csv"

# Перевірка існування файлів
if not os.path.exists(model_path):
    print(f"Error: Model file not found: {model_path}")
    exit()
if not os.path.exists(params_path):
    print(f"Error: Params file not found: {params_path}")
    exit()
if not os.path.exists(data_path):
    print(f"Error: Data file not found: {data_path}")
    exit()

# Завантаження параметрів моделі
with open(params_path, "r") as f:
    params = json.load(f)

# Створення моделі
model = CNNModel(params["window_size"], 1)  # Використовуємо CNNModel
model.build()

# Завантаження ваг моделі
model.load(model_path)
print(f"Model loaded successfully from {model_path}")

# Зчитування даних
df = pd.read_excel(data_path)
print(f"Дані з файлу {data_path} успішно завантажені.")
print(f"Перші 5 рядків:\n{df.head()}")

# Перевірка наявності необхідних колонок
if 'Time' not in df.columns:
    print("Error: У файлі даних відсутня колонка 'Time'")
    exit()
if 'Encoder Speed' not in df.columns:
    print("Error: У файлі даних відсутня колонка 'Encoder Speed'")
    exit()
if 'N1Gyro Z' not in df.columns:
    print("Error: У файлі даних відсутня колонка 'N1Gyro Z'")
    exit()

# Підготовка даних
X, y = prepare_data_for_model(df, 'N1Gyro Z', window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(f"Дані для моделі підготовлені. Розмір X: {X.shape}, розмір y: {y.shape}")

# Передбачення
y_pred = model.predict(X).flatten()
print(f"Передбачення зроблені. Розмір y_pred: {y_pred.shape}")

# Створення DataFrame для збереження результатів
predictions_df = pd.DataFrame()
predictions_df['Time'] = df['Time'][window_size - 1:]
predictions_df['Encoder Speed'] = df['Encoder Speed'][window_size - 1:]
predictions_df['NVGyro Z_input'] = df['NVGyro Z'][window_size - 1:]
predictions_df['N1Gyro Z_input'] = df['N1Gyro Z'][window_size - 1:]
predictions_df['N8Gyro Z_input'] = df['N8Gyro Z'][window_size - 1:]
predictions_df['N1Gyro Z_CNN_predicted'] = y_pred[:len(df) - window_size + 1] # Обрізаємо передбачення до відповідного розміру
predictions_df['Model'] = 'CNN'
predictions_df['Stage'] = '0'  # Захардкоджено, бо в прикладі шляху stage_0
predictions_df['File'] = 'ProgessiveOscill0'
predictions_df['Model_Full_Name'] = 'CNN_ProgessiveOscill0_stage_0_N1Gyro Z'  # Захардкоджено
predictions_df['Hyperparameters'] = json.dumps(params)

# Збереження результатів
predictions_df.to_csv(output_file, index=False)
print(f"Передбачення збережено у файл: {output_file}")