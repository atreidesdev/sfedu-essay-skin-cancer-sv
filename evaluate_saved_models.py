import os
import keras
import numpy as np
from data_loader import load_data
from evaluation import evaluate_models
from config import MODEL_SAVE_PATH

def load_saved_models():
    models_dict = {}
    
    model_files = [f for f in os.listdir(MODEL_SAVE_PATH) if f.endswith('.keras')]
    
    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(MODEL_SAVE_PATH, model_file)
        
        try:
            print(f"[INFO] Загрузка модели: {model_name}")
            model = keras.models.load_model(model_path)
            models_dict[model_name] = model
        except Exception as e:
            print(f"[WARNING] Не удалось загрузить модель {model_name}: {e}")
            continue
    
    return models_dict

def evaluate_saved_models():
    print("\n[INFO] Загрузка сохраненных моделей...")
    models_dict = load_saved_models()
    
    if not models_dict:
        print("[ERROR] Не найдено ни одной модели для оценки")
        return
    
    print("\n[INFO] Загрузка данных для валидации...")
    _, val_gen = load_data()
    
    print("\n[INFO] Начало оценки моделей...")
    evaluate_models(models_dict, val_gen)

if __name__ == "__main__":
    evaluate_saved_models() 