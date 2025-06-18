import pandas as pd
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import IMAGE_DIR, CSV_PATH, CLASS_NAMES, LABEL_MAP

def preprocess_metadata(df):
    df['age'] = df['age'].fillna(df['age'].median())
    df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
    df['localization'] = df['localization'].fillna(df['localization'].mode()[0])
    
    age_scaler = StandardScaler()
    df['age_normalized'] = age_scaler.fit_transform(df[['age']])
    
    sex_encoder = LabelEncoder()
    df['sex_encoded'] = sex_encoder.fit_transform(df['sex'])
    
    loc_encoder = LabelEncoder()
    df['localization_encoded'] = loc_encoder.fit_transform(df['localization'])
    
    return df

def balance_dataset():
    print("[INFO] Загрузка данных...")
    df = pd.read_csv(CSV_PATH)
    df = df[df['dx'].isin(CLASS_NAMES)]
    df['full_path'] = df['image_id'].map(lambda x: os.path.join(IMAGE_DIR, x + ".jpg"))
    df = df[df['full_path'].map(os.path.exists)]
    df['label'] = df['dx'].map(LABEL_MAP)
    
    df = preprocess_metadata(df)
    
    print("[INFO] Применение SMOTE...")
    X = df[['age_normalized', 'sex_encoded', 'localization_encoded']]
    y = df['label']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    balanced_df = df.iloc[X_resampled.index].copy()
    balanced_df['label'] = y_resampled
    
    output_path = 'balanced_dataset.csv'
    balanced_df.to_csv(output_path, index=False)
    print(f"[INFO] Сбалансированный датасет сохранен в {output_path}")
    
    print("\n[INFO] Статистика по классам:")
    print(balanced_df['label'].value_counts())

if __name__ == "__main__":
    balance_dataset() 