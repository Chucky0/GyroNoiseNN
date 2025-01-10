import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from model import LSTMModel, CNNModel, CNNLSTMModel, TCNModel, GRUModel

# Задайте шлях до папки з обробленими даними
processed_data_folder = "processed_data"

# Параметри
window_size = 100

# Моделі
model_classes = {
    "LSTM": LSTMModel,
    "CNN": CNNModel,
    "CNNLSTM": CNNLSTMModel,
    "TCN": TCNModel,
    "GRU": GRUModel,
}

# Отримання списку всіх папок з даними
stage_sensor_folders = [d for d in os.listdir(processed_data_folder) if os.path.isdir(os.path.join(processed_data_folder, d))]

for model_name, ModelClass in model_classes.items():
    for folder in stage_sensor_folders:
        print(f"Interpretability analysis for {model_name} - {folder}...")
        stage, sensor = folder.split("_", 1)

        # Завантаження даних
        X_train = np.load(f"{processed_data_folder}/{folder}/X_train.npy")
        X_test = np.load(f"{processed_data_folder}/{folder}/X_test.npy")

        # Завантаження моделі
        model_filepath = f"{model_name}_{stage}_{sensor}.h5"
        model = ModelClass(window_size, 1)
        model.load(model_filepath)

        # SHAP analysis
        if model_name in ["LSTM", "CNN", "CNNLSTM", "TCN", "GRU"]:
            try:
                explainer = shap.DeepExplainer(model.model, X_train[:100])  # Використовуйте частину навчальних даних
                shap_values = explainer.shap_values(X_test[:100])  # Аналізуйте частину тестових даних

                # Побудова графіків SHAP
                shap.summary_plot(shap_values[0], X_test[:100], feature_names=[f"Time {i}" for i in range(window_size)], show=False)
                plt.savefig(f"shap_summary_{model_name}_{stage}_{sensor}.png")
                plt.close()

                # Додатковий приклад: SHAP force plot для одного прикладу
                shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_test[0], feature_names=[f"Time {i}" for i in range(window_size)], show=False, matplotlib=True)
                plt.savefig(f"shap_force_{model_name}_{stage}_{sensor}.png")
                plt.close()
            except Exception as e:
                print(f"Error during SHAP analysis for {model_name} - {stage} - {sensor}: {e}")
        else:
            print(f"Skipping SHAP analysis for {model_name} as it's not supported for this model type.")