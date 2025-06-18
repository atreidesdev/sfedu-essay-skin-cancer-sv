from model_trainer_advanced import train_models_advanced
from evaluation import plot_accuracies, plot_losses
from download_dataset import download_kaggle_dataset
from predictor import predict_image
from dataset_visualization import visualize_dataset
from evaluate_saved_models import evaluate_saved_models
import os
import argparse
from config import MODEL_SAVE_PATH, RESULT_PATH, CSV_PATH

def train_mode():
    print("\n[INFO] Запуск режима обучения...")

    # Шаг 1: Скачать датасет
    dataset_path = download_kaggle_dataset()
    if not dataset_path or not os.path.exists(CSV_PATH):
        print(dataset_path, os.path.exists(CSV_PATH), CSV_PATH)
        raise FileNotFoundError("Датасет не найден или не был загружен.")

    # Шаг 2: Визуализация датасета
    print("\n[INFO] Визуализация датасета...")
    visualize_dataset()

    # Шаг 3: Обучить модели
    models_dict, histories = train_models_advanced()

    # Шаг 4: Визуализация результатов
    plot_accuracies(histories)
    plot_losses(histories)

def predict_mode(image_path=None):
    print("\n[INFO] Запуск режима предсказания...")

    if image_path is None:
        image_path = input("\nВведите путь к изображению для предсказания: ")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображение не найдено по пути: {image_path}")

    available_models = [f for f in os.listdir(MODEL_SAVE_PATH) if f.endswith('.keras')]
    if not available_models:
        raise FileNotFoundError("Обученные модели не найдены. Сначала запустите обучение.")

    print("\nДоступные модели:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    model_choice = input("\nВыберите номер модели для предсказания (или нажмите Enter для использования transfer_learning): ")

    if model_choice.strip():
        try:
            model_name = available_models[int(model_choice) - 1]
        except (ValueError, IndexError):
            print("Неверный выбор. Используется модель mobilenetv2.")
            model_name = "mobilenetv2.keras"
    else:
        model_name = "mobilenetv2.keras"

    model_path = os.path.join(MODEL_SAVE_PATH, model_name)

    try:
        age = int(input("\nВведите возраст пациента (или нажмите Enter для пропуска): ") or "45")
        sex = input("Введите пол пациента (male/female) (или нажмите Enter для пропуска): ") or "male"
        localization = input("Введите локализацию поражения (или нажмите Enter для пропуска): ") or "back"

        pred_class, probs = predict_image(
            model_path,
            image_path,
            age=age,
            sex=sex,
            localization=localization
        )
        print(f"\nМетаданные: возраст={age}, пол={sex}, локализация={localization}")
    except ValueError:
        pred_class, probs = predict_image(
            model_path,
            image_path
        )

    print(f"\nПрогнозируемый класс изображения {image_path}: {pred_class}")

def main():
    parser = argparse.ArgumentParser(description='Программа для классификации кожных заболеваний')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], help='Режим работы: train (обучение), predict (предсказание) или evaluate (оценка сохраненных моделей)')
    parser.add_argument('--image', help='Путь к изображению для предсказания')

    args = parser.parse_args()

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULT_PATH, exist_ok=True)

    if args.mode:
        if args.mode == 'train':
            train_mode()
        elif args.mode == 'predict':
            predict_mode(args.image)
        else:
            evaluate_saved_models()
    else:
        print("\nВыберите режим работы:")
        print("1. Обучение моделей")
        print("2. Предсказание")
        print("3. Оценка сохраненных моделей")

        choice = input("\nВведите номер режима (1, 2 или 3): ")

        if choice == '1':
            train_mode()
        elif choice == '2':
            predict_mode()
        elif choice == '3':
            evaluate_saved_models()
        else:
            print("Неверный выбор. Завершение программы.")

if __name__ == "__main__":
    main()
