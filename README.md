# food-calorie-estimate 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# === Step 1: Data Paths ===
data_dir = '/path/to/food-101/images'  # Change this to your extracted images folder

# === Step 2: Data Preprocessing ===
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
class_names = list(train_gen.class_indices.keys())

# === Step 3: Build Model (Transfer Learning) ===
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Freeze the base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Step 4: Train Model ===
history = model.fit(train_gen, epochs=5, validation_data=val_gen)

# === Step 5: Calorie Mapping ===
# (Simplified example with only a few items. Extend as needed.)
calorie_mapping = {
    'apple_pie': 296,
    'baby_back_ribs': 400,
    'baklava': 334,
    'beef_carpaccio': 150,
    'beef_tartare': 250,
    'caesar_salad': 180,
    # ... add more as needed
}

# === Step 6: Predict and Estimate Calories ===
def predict_food_and_calories(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    top_idx = np.argmax(prediction)
    food_label = class_names[top_idx]
    calories = calorie_mapping.get(food_label, "Unknown")

    print(f"Predicted Food: {food_label}")
    print(f"Estimated Calories: {calories} kcal")
    
    # Visual Output
    plt.imshow(img)
    plt.title(f"{food_label} - {calories} kcal")
    plt.axis('off')
    plt.show()

# Example Usage
predict_food_and_calories('/path/to/food-image.jpg')
