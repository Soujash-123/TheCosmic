# fine_tune_image.py
"""
Fine-tune VGG16 for 5-class emotion classification on a custom image dataset.
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_finetune_vgg16(input_shape=(224,224,3), num_classes=5):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

def fine_tune_vgg16(train_dir, val_dir, batch_size=32, epochs=10, img_size=(224,224), output_path="./vgg16_finetuned"):
    # Enhanced data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        brightness_range=[0.8,1.2]
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
    model = build_finetune_vgg16(input_shape=img_size+(3,), num_classes=5)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Add callbacks for early stopping and model checkpointing
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(output_path, monitor='val_loss', save_best_only=True)
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    print(f"Model saved to {output_path}")

    # Hyperparameter tuning stub (suggest using keras-tuner or optuna)
    # from kerastuner.tuners import RandomSearch
    # tuner = RandomSearch(
    #     build_finetune_vgg16,
    #     objective='val_accuracy',
    #     max_trials=5,
    #     directory='tuner_dir',
    #     project_name='vgg16_tuning')
    # tuner.search(train_gen, epochs=10, validation_data=val_gen)

    # For transfer learning, try other pre-trained models (ResNet, EfficientNet, etc.)
    # For ensembling, average predictions from multiple models

# Example usage (replace with your data paths):
# fine_tune_vgg16('path/to/train', 'path/to/val')
