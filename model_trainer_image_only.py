from keras import layers, models, applications, callbacks
from data_loader import load_data
from config import NUM_CLASSES, MODEL_SAVE_PATH
import os
import keras
import numpy as np
from evaluation import evaluate_models

class ImageOnlyModelTrainer:
    def __init__(self, model_type, input_shape=(128, 128, 3)):
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.base_model = None
        
    def build_model(self):
        if self.model_type == 'simple_cnn':
            return self._build_simple_cnn()
        elif self.model_type == 'transfer_learning':
            return self._build_mobilenet()
        elif self.model_type == 'vgg16':
            return self._build_vgg16()
        elif self.model_type == 'resnet50':
            return self._build_resnet50()
        elif self.model_type == 'efficientnet':
            return self._build_efficientnet()
        else:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def _build_simple_cnn(self):
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        
        x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = models.Model(inputs=image_input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_mobilenet(self):
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        
        self.base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        self.base_model.trainable = False
        
        x = self.base_model(image_input)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = models.Model(inputs=image_input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_vgg16(self):
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        
        self.base_model = applications.VGG16(
            include_top=False,
            input_shape=self.input_shape,
            weights='imagenet',
            pooling='avg'
        )
        self.base_model.trainable = False
        
        x = self.base_model(image_input)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = models.Model(inputs=image_input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_resnet50(self):
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        
        self.base_model = applications.ResNet50(
            include_top=False,
            input_shape=self.input_shape,
            weights='imagenet',
            pooling='avg'
        )
        self.base_model.trainable = False
        
        x = self.base_model(image_input)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = models.Model(inputs=image_input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _build_efficientnet(self):
        image_input = layers.Input(shape=self.input_shape, name='image_input')
        
        self.base_model = applications.EfficientNetB0(
            include_top=False,
            input_shape=self.input_shape,
            weights='imagenet',
            pooling='avg'
        )
        self.base_model.trainable = False
        
        x = self.base_model(image_input)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = models.Model(inputs=image_input, outputs=x)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def unfreeze_layers(self, num_layers=10):
        if self.base_model is None:
            return
        
        for layer in self.base_model.layers[-num_layers:]:
            layer.trainable = True
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

def train_models_image_only():
    train_gen, val_gen = load_data()
    
    model_configs = [
        ('simple_cnn', 20, 0),
        ('transfer_learning', 10, 15),
        ('vgg16', 10, 15),
        ('resnet50', 10, 20),
        ('efficientnet', 10, 20)
    ]
    
    histories = {}
    trained_models = {}
    
    print("\n[INFO] Обучение моделей только на изображениях...")
    
    for model_type, epochs_before, epochs_after in model_configs:
        print(f"\n[INFO] Обучение модели: {model_type}")
        
        trainer = ImageOnlyModelTrainer(model_type)
        model = trainer.build_model()
        
        best_model_path = os.path.join(MODEL_SAVE_PATH, f"{model_type}_image_only_best.keras")
        final_model_path = os.path.join(MODEL_SAVE_PATH, f"{model_type}_image_only.keras")
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        checkpoint = callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            save_weights_only=False
        )
        
        print(f"[INFO] Этап 1: Обучение с замороженными слоями ({epochs_before} эпох)")
        history1 = model.fit(
            train_gen,
            epochs=epochs_before,
            validation_data=val_gen,
            callbacks=[early_stop, checkpoint]
        )
        
        if epochs_after > 0:
            print(f"[INFO] Этап 2: Разморозка слоев и дообучение ({epochs_after} эпох)")
            trainer.model = model
            trainer.unfreeze_layers()
            
            history2 = model.fit(
                train_gen,
                epochs=epochs_after,
                validation_data=val_gen,
                callbacks=[early_stop, checkpoint]
            )
            
            history = {
                'loss': history1.history['loss'] + history2.history['loss'],
                'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
                'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
            }
        else:
            history = history1.history
        
        model.save(final_model_path)
        histories[model_type] = history
        trained_models[model_type] = model
    
    print("\n[INFO] Создание ансамблевой модели...")
    ensemble_models = [trained_models[name] for name in ['simple_cnn', 'transfer_learning', 'vgg16']]
    
    def ensemble_predict(inputs):
        predictions = []
        for model in ensemble_models:
            pred = model.predict(inputs)
            predictions.append(pred)
        return np.mean(predictions, axis=0)
    
    class EnsembleModel:
        def __init__(self, predictor):
            self.predictor = predictor
        
        def predict(self, x):
            return self.predictor(x)
    
    ensemble_model = EnsembleModel(ensemble_predict)
    trained_models['ensemble'] = ensemble_model
    
    print("[INFO] Оценка всех моделей...")
    evaluate_models(trained_models, val_gen)
    
    return trained_models, histories 