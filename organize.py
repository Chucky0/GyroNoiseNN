import os
import shutil
import glob


def organize_files(project_root=".", output_base_folder="models_for_predict"):
    """
    Організовує файли проекту, знаходячи моделі та їхні конфігураційні файли,
    та переміщуючи їх у нову структуру папок.

    Args:
        project_root (str): Шлях до кореневої папки проекту.
        output_base_folder (str): Шлях до папки, де буде створена нова структура.
    """

    models = ["LSTM", "CNN", "CNNLSTM", "TCN", "GRU"]  # Список моделей
    sensors = ["N1Gyro Z", "N8Gyro Z", "NVGyro Z"]
    stages = range(6)  # 0 до 5 включно
    file_names = ["ProgessiveOscill0", "StableOscill0", "ProgessiveOscill1", "StableOscill1"]

    # Створюємо кореневу папку для нової структури
    new_models_dir = os.path.join(output_base_folder, "models_with_predictions")
    os.makedirs(new_models_dir, exist_ok=True)

    # Копіюємо експериментальні дані
    experimental_data_folder = os.path.join(project_root, "experimental_data")
    for file_name in file_names:
        source_path = os.path.join(experimental_data_folder, f"{file_name}.xlsx")
        if os.path.exists(source_path):
            destination_path = os.path.join(output_base_folder, f"{file_name}.xlsx")
            shutil.copy2(source_path, destination_path)
            print(f"Copied {source_path} to {destination_path}")
        else:
            print(f"Source file not found: {source_path}")

    # Пошук та переміщення моделей та конфігураційних файлів
    for root, dirs, files in os.walk(project_root):
        for model_name in models:
            for file_name in file_names:
                for stage in stages:
                    for sensor in sensors:
                        model_full_name = f"{model_name}_{file_name}_stage_{stage}_{sensor}"

                        # Check for .keras or .keras.keras files
                        keras_files = glob.glob(os.path.join(root, f"{model_name}_*_{stage}_{sensor}.keras*"))
                        params_file = os.path.join(root, f"{model_name}_{file_name}_stage_{stage}_{sensor}_params.json")

                        if keras_files and os.path.exists(params_file):
                            # Take the first .keras file found
                            keras_file = keras_files[0]
                            destination_folder_name = f"{model_name}_{file_name}_stage_{stage}_{sensor}"
                            destination_folder_path = os.path.join(new_models_dir, model_name, destination_folder_name)
                            os.makedirs(destination_folder_path, exist_ok=True)

                            destination_keras_path = os.path.join(destination_folder_path, os.path.basename(keras_file))
                            shutil.copy2(keras_file, destination_keras_path)
                            print(f"Copied {keras_file} to {destination_keras_path}")

                            destination_params_path = os.path.join(destination_folder_path,
                                                                   os.path.basename(params_file))
                            shutil.copy2(params_file, destination_params_path)
                            print(f"Copied {params_file} to {destination_params_path}")


if __name__ == "__main__":
    organize_files()