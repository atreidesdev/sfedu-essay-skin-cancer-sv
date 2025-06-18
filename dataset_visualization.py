import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import CSV_PATH, RESULT_PATH, CLASS_NAMES

def visualize_dataset():
    df = pd.read_csv(CSV_PATH)
    
    required_columns = ['label', 'age', 'sex', 'localization']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] В датасете отсутствуют следующие столбцы: {missing_columns}")
    
    df = df[df['label'].isin(range(len(CLASS_NAMES)))]
    
    os.makedirs(RESULT_PATH, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='label')
    plt.title('Распределение классов в датасете')
    plt.xlabel('Класс')
    plt.ylabel('Количество')
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'class_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', bins=30)
    plt.title('Распределение возраста пациентов')
    plt.xlabel('Возраст')
    plt.ylabel('Количество')
    plt.savefig(os.path.join(RESULT_PATH, 'age_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='sex')
    plt.title('Распределение по полу')
    plt.xlabel('Пол')
    plt.ylabel('Количество')
    plt.savefig(os.path.join(RESULT_PATH, 'gender_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, y='localization')
    plt.title('Распределение локализации поражений')
    plt.xlabel('Количество')
    plt.ylabel('Локализация')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'localization_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(10, 8))
    numeric_df = df[['age']].copy()
    numeric_df['sex'] = pd.Categorical(df['sex']).codes
    numeric_df['localization'] = pd.Categorical(df['localization']).codes
    numeric_df['label'] = df['label']
    
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Корреляция между признаками')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'feature_correlations.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='label', y='age')
    plt.title('Распределение возраста по классам')
    plt.xlabel('Класс')
    plt.ylabel('Возраст')
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, 'age_by_class.png'))
    plt.close()
    
    print("\nСтатистика датасета:")
    print(f"Общее количество изображений: {len(df)}")
    print("\nРаспределение по классам:")
    class_counts = df['label'].value_counts().sort_index()
    for i, count in class_counts.items():
        print(f"{CLASS_NAMES[i]}: {count}")
    print("\nСтатистика по возрасту:")
    print(df['age'].describe())
    print("\nРаспределение по полу:")
    print(df['sex'].value_counts())
    print("\nРаспределение по локализации:")
    print(df['localization'].value_counts())

if __name__ == "__main__":
    visualize_dataset() 