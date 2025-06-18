import pandas as pd
import os
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import IMAGE_DIR, CSV_PATH, CLASS_NAMES, LABEL_MAP

def preprocess_metadata(df):
    age_scaler = StandardScaler()
    df['age_normalized'] = age_scaler.fit_transform(df[['age']])
    
    sex_encoder = LabelEncoder()
    df['sex_encoded'] = sex_encoder.fit_transform(df['sex'])
    
    loc_encoder = LabelEncoder()
    df['localization_encoded'] = loc_encoder.fit_transform(df['localization'])
    
    return df

def load_data(img_size=(128, 128), batch_size=32):
    df = pd.read_csv(CSV_PATH)
    df = df[df['dx'].isin(CLASS_NAMES)]
    df['full_path'] = df['image_id'].map(lambda x: os.path.join(IMAGE_DIR, x + ".jpg"))
    df = df[df['full_path'].map(os.path.exists)]
    df['label'] = df['dx'].map(LABEL_MAP)
    
    df = preprocess_metadata(df)
    
    class_counts = df['label'].value_counts()
    missing_classes = set(range(len(CLASS_NAMES))) - set(class_counts.index)
    if missing_classes:
        raise ValueError(f"[ERROR] Недостаточно данных для классов: {missing_classes}. Пополните выборку.")
    if (class_counts < 2).any():
        raise ValueError("[ERROR] Некоторые классы имеют менее 2 экземпляров.")

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

    def parse_image(filename, label, metadata):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0
        return (image, metadata), label

    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_paths,
        train_metadata,
        train_labels
    ))
    train_dataset = train_dataset.map(
        lambda path, meta, label: parse_image(path, label, meta),
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
        lambda path, meta, label: parse_image(path, label, meta),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset
