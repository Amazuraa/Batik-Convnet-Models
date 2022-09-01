import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import random


# ----------------- CUSTOM AUGMENTATION -----------------

class RandomInvert(layers.Layer):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)

    def call(self, x):
      p = 0.5
      if  tf.random.uniform([]) < p: x = (255-x)
      else: x
      return x

class RandomSaturation(layers.Layer):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)
        
    def call(self, images):
      seeds = (random.randrange(1, 10), 1)
      images = tf.image.stateless_random_saturation(images, .1, 5, seeds)
      return images

class RandomHue(layers.Layer):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)
        
    def call(self, images):
      seeds = (random.randrange(1, 10), 1)
      images = tf.image.stateless_random_hue(images, .35, seeds)
      return images

class RandomBrightness(layers.Layer):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)
        
    def call(self, images):
      seeds = (random.randrange(1, 10), 1)
      images = tf.image.stateless_random_brightness(images, 0.4, seeds)
      return images

class RandomContrast(layers.Layer):
    def __init__(self, **kwargs):
      super().__init__(**kwargs)
        
    def call(self, images):
      seeds = (3, 1)
      images = tf.image.stateless_random_contrast(images, lower=.1, upper=1, seed=seeds)
      return images


# ----------------- MODEL LOGIC -----------------

# ---- load Dataset ----
data_dir = pathlib.Path("./Batik Dataset Custom Sobel")
# image_count = len(list(data_dir.glob('*/*.jpg')))

# ---- create Dataset ----
batch_size = 35
img_height = 240
img_width = 240

# ---- split data for training & validation 80:20 ----
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split = 0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=64)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split = 0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

# ---- Configure dataset for performance to keep images in memory ----
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---- apply data augmentation ----
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal_and_vertical", input_shape=(img_height, img_width, 3)),
    # layers.RandomRotation(135),
    layers.RandomZoom(0.3),
    RandomInvert(),
    # RandomContrast(),
    # RandomBrightness(),
    # RandomHue(),
    # RandomSaturation(),
  ]
)

# ---- create the model ----
num_classes = len(class_names)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu', dilation_rate=1),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu', dilation_rate=2),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu', dilation_rate=3),
  layers.MaxPooling2D(),
  layers.Dropout(0.63),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# ---- compile the model ----
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()

# ---- train the model ----
epochs = 100
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

m_name = "cnn-batik-5-sobel"

# ---- save the model ----
model.save(m_name + ".h5")

# ---- Training Result ----
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
# plt.show()

plt.savefig(m_name + ".png")

print("Evaluate")
res_loss, res_acc = model.evaluate(val_ds)
print(res_acc)