import os
import cv2
import numpy as np
import pandas as pd


def load_words_data(txt_file):
    data = []
    with open(txt_file, 'r') as file:
        for line in file:

            if line.startswith('#') or len(line.strip()) == 0:
                continue
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            if parts[1] == 'ok':
                img_id = parts[0]
                transcription = parts[-1]
                data.append((img_id, transcription))
    print(f"Loaded {len(data)} valid entries from {txt_file}")
    return data


def load_images(data, base_folder):
    images = []
    labels = []
    total_images = len(data)
    for idx, (img_id, transcription) in enumerate(data):
        form_id = img_id.split('-')[0]
        subfolder_id = img_id.split('-')[1]
        img_path = os.path.join(base_folder, form_id, form_id + '-' + subfolder_id, f"{img_id}.png")
        img_path = img_path.replace('\\', os.sep).replace('/', os.sep)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image {img_path}")
                continue
            img = cv2.resize(img, (128, 32))
            img = img / 255.0  # normalize
            images.append(img)
            labels.append(transcription)
        else:
            print(f"Image {img_path} not found.")


        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            print(f"Processed {idx + 1}/{total_images} images")

    print(f"Loaded {len(images)} images from {base_folder}")
    return np.array(images), labels


def preprocess_data(txt_file, base_folder):
    data = load_words_data(txt_file)
    images, labels = load_images(data, base_folder)
    return images, labels


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    words_txt_path = os.path.join(project_root, 'archive/words.txt')
    images_dir = os.path.join(project_root, 'archive/words/')

    print(f"Words file: {words_txt_path}")
    print(f"Images directory: {images_dir}")

    images, labels = preprocess_data(words_txt_path, images_dir)
    np.save(os.path.join(project_root, 'images.npy'), images)
    pd.DataFrame(labels, columns=['label']).to_csv(os.path.join(project_root, 'labels.csv'), index=False)