import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback

from pythonProject.script.model import create_model


class PrintProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch+1}/{self.params['epochs']}")
        print(f" - loss: {logs.get('loss'):.4f} - accuracy: {logs.get('accuracy'):.4f}")
        print(f" - val_loss: {logs.get('val_loss'):.4f} - val_accuracy: {logs.get('val_accuracy'):.4f}")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

images_path = os.path.join(project_root, 'images.npy')
labels_path = os.path.join(project_root, 'labels.csv')
images = np.load(images_path)
labels = pd.read_csv(labels_path)['label']

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 32, 128, 1)
X_val = X_val.reshape(X_val.shape[0], 32, 128, 1)

model_path = os.path.join(project_root, 'handwriting_recognition_model.h5')
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Loaded existing model.")
else:
    input_shape = (32, 128, 1)
    num_classes = len(label_encoder.classes_)
    model = create_model(input_shape, num_classes)
    print("Created new model.")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [PrintProgress()]

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=200, callbacks=callbacks)
model.save(model_path)
