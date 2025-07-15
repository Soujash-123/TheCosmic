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
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
    model = build_finetune_vgg16(input_shape=img_size+(3,), num_classes=5)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(output_path)
    print(f"Model saved to {output_path}")

# Example usage (replace with your data paths):
# fine_tune_vgg16('path/to/train', 'path/to/val')
