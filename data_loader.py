import pandas as pd
import os
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import CLASS_NAMES, CSV_PATH
import albumentations

def preprocess_metadata(df):
    required_columns = ['age', 'sex', 'localization', 'label', 'full_path']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"[ERROR] В датасете отсутствуют следующие столбцы: {missing_columns}")
    
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

def get_augmentation_pipeline():
    return albumentations.Compose([
        albumentations.RandomRotate90(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.0625, 0.0625),
            rotate=(-45, 45),
            p=0.5
        ),
        albumentations.OneOf([
            albumentations.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=0.5
            ),
            albumentations.GridDistortion(p=0.5),
            albumentations.OpticalDistortion(
                distort_limit=1,
                p=0.5
            ),
        ], p=0.3),
        albumentations.OneOf([
            albumentations.GaussNoise(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.RandomGamma(p=0.5),
        ], p=0.3),
    ])

def apply_augmentation(image, augmentation_pipeline):

    augmented = augmentation_pipeline(image=image)
    return augmented['image']

def load_data(img_size=(128, 128), batch_size=32):
    print("[INFO] Загрузка данных...")
    
    if os.path.exists(CSV_PATH):
        print("[INFO] Используем предварительно сбалансированный датасет")
        df = pd.read_csv(CSV_PATH)
        
        required_columns = ['age', 'sex', 'localization', 'label', 'full_path']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"[ERROR] В датасете отсутствуют следующие столбцы: {missing_columns}")
    else:
        raise FileNotFoundError(f"[ERROR] Файл {CSV_PATH} не найден")
    
    df = preprocess_metadata(df)
    
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        
        if len(class_df) == 2:
            train_sample = class_df.sample(1, random_state=42)
            val_sample = class_df.drop(train_sample.index)
        else:
            train_sample, val_sample = train_test_split(
                class_df,
                test_size=0.2,
                random_state=42,
                stratify=None
            )
        
        train_df = pd.concat([train_df, train_sample])
        val_df = pd.concat([val_df, val_sample])

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_labels_set = set(train_df['label'])
    val_labels_set = set(val_df['label'])
    expected_labels = set(range(len(CLASS_NAMES)))
    if not (expected_labels.issubset(train_labels_set) and expected_labels.issubset(val_labels_set)):
        raise ValueError("[ERROR] Не все классы попали в обе выборки.")

    train_paths = train_df['full_path'].values
    train_metadata = train_df[['age_normalized', 'sex_encoded', 'localization_encoded']].values
    train_labels = keras.utils.to_categorical(train_df['label'], num_classes=len(CLASS_NAMES))

    val_paths = val_df['full_path'].values
    val_metadata = val_df[['age_normalized', 'sex_encoded', 'localization_encoded']].values
    val_labels = keras.utils.to_categorical(val_df['label'], num_classes=len(CLASS_NAMES))

    augmentation_pipeline = get_augmentation_pipeline()

    def parse_image(filename, label, metadata, augment=False):

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0
        
        if augment:
            image = tf.py_function(
                lambda x: apply_augmentation(x.numpy(), augmentation_pipeline),
                [image],
                tf.float32
            )
            image.set_shape(img_size + (3,))
        
        return (image, metadata), label

    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_paths,
        train_metadata,
        train_labels
    ))
    train_dataset = train_dataset.map(
        lambda path, meta, label: parse_image(path, label, meta, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        val_paths,
        val_metadata,
        val_labels
    ))
    val_dataset = val_dataset.map(
        lambda path, meta, label: parse_image(path, label, meta, augment=False),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    print(f"[INFO] Размер обучающей выборки: {len(train_df)}")
    print(f"[INFO] Размер валидационной выборки: {len(val_df)}")
    print("[INFO] Распределение классов в обучающей выборке:")
    print(train_df['label'].value_counts())
    print("[INFO] Распределение классов в валидационной выборке:")
    print(val_df['label'].value_counts())

    return train_dataset, val_dataset
