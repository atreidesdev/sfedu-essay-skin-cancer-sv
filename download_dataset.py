import kagglehub
import os
import shutil
from config import DATASET_PATH, KAGGLE_DATASET_NAME

def download_kaggle_dataset(dataset_name: str = KAGGLE_DATASET_NAME, target_dir: str = DATASET_PATH):
    print(f"[INFO] Проверяю наличие датасета по пути: {os.path.abspath(target_dir)}")

    if os.path.exists(target_dir):
        print("[INFO] Датасет уже существует.")
        os.path.abspath(target_dir)
        return os.path.abspath(target_dir)

    print(f"[INFO] Загрузка датасета '{dataset_name}'...")

    try:
        source_path = kagglehub.dataset_download(dataset_name)
        print(f"[DEBUG] Датасет загружен во временный каталог:\n{source_path}")

        os.makedirs(target_dir, exist_ok=True)

        for item in os.listdir(source_path):
            src_item = os.path.join(source_path, item)
            dst_item = os.path.join(target_dir, item)

            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item, dst_item)

        print(f"[SUCCESS] Датасет успешно скопирован в:\n{os.path.abspath(target_dir)}")
        return os.path.abspath(target_dir)

    except Exception as e:
        print(f"[ERROR] Ошибка при загрузке датасета: {e}")
        return ""