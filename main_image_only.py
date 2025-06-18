from model_trainer_image_only import train_models_image_only
from evaluation import plot_accuracies, plot_losses
from download_dataset import download_kaggle_dataset
from predictor import predict_image
from dataset_visualization import visualize_dataset
import os
from config import MODEL_SAVE_PATH, RESULT_PATH, CSV_PATH

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

if __name__ == "__main__":
    dataset_path = download_kaggle_dataset()

    if not dataset_path or not os.path.exists(CSV_PATH):
        print(dataset_path, os.path.exists(CSV_PATH), CSV_PATH)
        raise FileNotFoundError("Датасет не найден или не был загружен.")

    print("\n[INFO] Визуализация датасета...")
    visualize_dataset()

    models_dict, histories = train_models_image_only()

    plot_accuracies(histories)
    plot_losses(histories)

    sample_image_path = os.path.join("dataset/Skin Cancer/Skin Cancer", "ISIC_0024306.jpg")
    model_path = os.path.join(MODEL_SAVE_PATH, "transfer_learning_image_only.keras")
    pred_class, probs = predict_image(model_path, sample_image_path)
    print(f"\nПрогнозируемый класс изображения {sample_image_path}: {pred_class}") 