import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tkinter import Tk, filedialog, Label, Button, Toplevel
from tkinter import ttk

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

model_path = os.path.join(project_root, 'handwriting_recognition_model.h5')
model = tf.keras.models.load_model(model_path)

labels_path = os.path.join(project_root, 'labels.csv')
labels = pd.read_csv(labels_path)['label']
label_encoder = LabelEncoder()
label_encoder.fit_transform(labels)

def preprocess_image(img):
    img = cv2.resize(img, (128, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = img.reshape(1, 32, 128, 1)
    return img

def select_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("PNG files", "*.png"), ("All files", "*.*"))
    )
    if file_path:
        process_image(file_path)

def process_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        result_label.config(text="Failed to read the selected image.")
    else:
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_label = np.argmax(prediction, axis=1)
        predicted_text = label_encoder.inverse_transform(predicted_label)
        result_label.config(text=f"texto predecido: {predicted_text[0]}")

def create_main_window():
    root = Tk()
    root.title("Conversor de texto")
    root.geometry("500x300")

    style = ttk.Style()
    style.configure('TButton', font=('Helvetica', 12), padding=10)

    title_label = Label(root, text="Algoritmica Conversor de texto", font=("Helvetica", 18, "bold"))
    title_label.pack(pady=20)

    select_button = ttk.Button(root, text="Seleccionar Imagen", command=select_image)
    select_button.pack(pady=20)

    global result_label
    result_label = Label(root, text="", font=("Helvetica", 14))
    result_label.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_main_window()
