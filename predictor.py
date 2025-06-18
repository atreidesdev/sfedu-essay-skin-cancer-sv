import numpy as np
from keras.preprocessing import image
from config import CLASS_NAMES
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def preprocess_metadata(age, sex, localization):
    df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'localization': [localization]
    })
    
    age_scaler = StandardScaler()
    df['age_normalized'] = age_scaler.fit_transform(df[['age']])
    
    sex_encoder = LabelEncoder()
    df['sex_encoded'] = sex_encoder.fit_transform(df['sex'])
    
    loc_encoder = LabelEncoder()
    df['localization_encoded'] = loc_encoder.fit_transform(df['localization'])
    
    return df[['age_normalized', 'sex_encoded', 'localization_encoded']].values[0]

def predict_image(model_path, img_path, age=None, sex=None, localization=None, img_size=(128, 128)):
    model = load_model(model_path)
    
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if all(x is not None for x in [age, sex, localization]):
        metadata = preprocess_metadata(age, sex, localization)
        metadata = np.expand_dims(metadata, axis=0)
        prediction = model.predict([img_array, metadata])
    else:
        prediction = model.predict(img_array)
    
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    
    return predicted_class, prediction