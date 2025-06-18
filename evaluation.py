import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import numpy as np
import os
from config import *

def plot_accuracies(histories):
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(hist['val_accuracy'], label=f'Val Accuracy - {name}')
    plt.title('Сравнение точности на валидационных данных')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_PATH, "comparison_accuracy.png"))
    plt.show()

def plot_losses(histories):
    plt.figure(figsize=(10, 6))
    for name, hist in histories.items():
        plt.plot(hist['val_loss'], label=f'Val Loss - {name}')
    plt.title('Сравнение потерь на валидационных данных')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_PATH, "comparison_loss.png"))
    plt.show()

def plot_f1_scores(models_results):
    plt.figure(figsize=(12, 6))
    
    model_names = []
    f1_scores = []
    
    for model_name, (y_true, y_pred, _) in models_results.items():
        f1 = f1_score(y_true, y_pred, average='weighted')
        model_names.append(model_name)
        f1_scores.append(f1)
    
    plt.bar(model_names, f1_scores)
    plt.title('Сравнение F1-score для всех моделей')
    plt.xlabel('Модели')
    plt.ylabel('F1-score')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "f1_scores_comparison.png"))
    plt.show()

def plot_confusion_matrices(models_results):
    for model_name, (y_true, y_pred, _) in models_results.items():
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Матрица ошибок - {model_name}')
        plt.xlabel('Предсказанные классы')
        plt.ylabel('Истинные классы')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_PATH, f"confusion_matrix_{model_name}.png"))
        plt.show()

def plot_roc_curves(models_results):
    for model_name, (y_true, _, y_score) in models_results.items():
        plt.figure(figsize=(10, 8))
        y_test = label_binarize(y_true, classes=range(NUM_CLASSES))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2,
                     label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:0.2f})')
    
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC-кривые - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(RESULT_PATH, f"roc_curves_{model_name}.png"))
        plt.show()

def plot_prediction_histograms(models_results):
    for model_name, (y_true, y_pred, _) in models_results.items():
        plt.figure(figsize=(12, 6))
        plt.hist(y_pred, bins=range(NUM_CLASSES+1), align='left', rwidth=0.8, alpha=0.7, label='Предсказанные')
        plt.hist(y_true, bins=range(NUM_CLASSES+1), align='left', rwidth=0.8, alpha=0.7, label='Истинные')
        plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45)
        plt.title(f'Распределение предсказаний - {model_name}')
        plt.xlabel('Классы')
        plt.ylabel('Частота')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_PATH, f"prediction_histogram_{model_name}.png"))
        plt.show()

def evaluate_models(models_dict, val_gen):
    models_results = {}
    
    print("\n[+] Отчёты по классификации:")
    for model_name, model in models_dict.items():
        print(f"\n--- {model_name} ---")
        
        all_preds = []
        all_true = []
        all_scores = []
        
        for x_val, y_val in val_gen:
            try:
                preds = model.predict(x_val)
                y_pred = np.argmax(preds, axis=1)
                y_true = np.argmax(y_val, axis=1)
                all_preds.extend(y_pred)
                all_true.extend(y_true)
                all_scores.extend(preds)
            except Exception as e:
                print(f"[WARNING] Пропуск батча при оценке модели {model_name}: {e}")
                continue
        
        if not all_preds or not all_true:
            print(f"[ERROR] Не удалось получить предсказания для модели {model_name}")
            continue
            
        print(classification_report(all_true, all_preds, 
                                  target_names=CLASS_NAMES, 
                                  labels=range(NUM_CLASSES)))
        
        models_results[model_name] = (all_true, all_preds, np.array(all_scores))
    
    plot_f1_scores(models_results)
    plot_confusion_matrices(models_results)
    plot_roc_curves(models_results)
    plot_prediction_histograms(models_results)
    
    return models_results

