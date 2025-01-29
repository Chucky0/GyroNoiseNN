import os
import shutil
import glob

def organize_files_for_prediction(models_dir="models", experimental_data_folder="experimental_data",
                                 output_base_folder="models_for_predict"):
    """
    Організовує файли для роботи скрипта predict.py, створюючи нову структуру папок
    та копіюючи туди необхідні файли (моделі та дані).

    Args:
        models_dir (str): Шлях до папки з моделями.
        experimental_data_folder (str): Шлях до папки з експериментальними даними.
        output_base_folder (str): Шлях до папки, де буде створена нова структура.
    """

    models = ["LSTM", "CNN", "CNNLSTM", "TCN", "GRU"]  # Список моделей
    sensors = ["N1Gyro Z", "N8Gyro Z", "NVGyro Z"]
    stages = range(6)  # 0 до 5 включно
    file_names = ["ProgessiveOscill0", "StableOscill0", "ProgessiveOscill1", "StableOscill1"]

    # Створюємо кореневу папку для нової структури
    new_models_dir = os.path.join(output_base_folder, "models_with_predictions") # Змінена назва підпапки
    os.makedirs(new_models_dir, exist_ok=True)

    # Копіюємо експериментальні дані
    for file_name in file_names:
        source_path = os.path.join(experimental_data_folder, f"{file_name}.xlsx")
        if os.path.exists(source_path):
            destination_path = os.path.join(output_base_folder, f"{file_name}.xlsx") # Змінено шлях куди копіювати
            shutil.copy2(source_path, destination_path)
            print(f"Copied {source_path} to {destination_path}")
        else:
            print(f"Source file not found: {source_path}")

    # Копіюємо моделі та конфігураційні файли
    for model_name in models:
        for file_name in file_names:
            for stage in stages:
                for sensor in sensors:
                    # Назва папки вхідної моделі
                    source_folder_name = f"{file_name}_stage_{stage}_{sensor}"
                    source_folder_path = os.path.join(models_dir, model_name, "processed_data", source_folder_name)

                    # Назва папки вихідної моделі
                    destination_folder_name = f"{file_name}_stage_{stage}_{sensor}" # Виправлено
                    destination_folder_path = os.path.join(new_models_dir, model_name, destination_folder_name) # Виправлено

                    # Перевірка наявності вихідної папки
                    if os.path.isdir(source_folder_path):
                        os.makedirs(destination_folder_path, exist_ok=True)

                        # Копіюємо .keras файл
                        keras_files = glob.glob(os.path.join(source_folder_path, f"{model_name}_*.keras"))
                        if keras_files:
                            keras_file = keras_files[0] # Беремо перший знайдений .keras файл
                            destination_keras_path = os.path.join(destination_folder_path, os.path.basename(keras_file))
                            shutil.copy2(keras_file, destination_keras_path)
                            print(f"Copied {keras_file} to {destination_keras_path}")

                        # Копіюємо _params.json файл
                        params_files = glob.glob(os.path.join(source_folder_path, f"{model_name}_*_params.json"))
                        if params_files:
                            params_file = params_files[0] # Беремо перший знайдений _params.json файл
                            destination_params_path = os.path.join(destination_folder_path, os.path.basename(params_file))
                            shutil.copy2(params_file, destination_params_path)
                            print(f"Copied {params_file} to {destination_params_path}")

                    else:
                        print(f"Source folder does not exist: {source_folder_path}")


if __name__ == "__main__":
    organize_files_for_prediction()