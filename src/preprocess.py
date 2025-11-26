import os
import cv2
import numpy as np
from tqdm import tqdm

def load_images_and_labels(dataset_path="dataset/plantvillage dataset/color", size=(128, 128)):
    images = []
    labels = []

    # لیست فولدرهای کلاس‌ها (Tomato___healthy و ...)
    classes = [cls for cls in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cls))]

    print("Classes found:", classes)

    for label in classes:
        class_path = os.path.join(dataset_path, label)

        for img_name in tqdm(os.listdir(class_path), desc=f"Loading {label}"):
            img_path = os.path.join(class_path, img_name)

            # خواندن تصویر
            img = cv2.imread(img_path)
            if img is None:
                continue  # اگر تصویر خراب بود، رد کن

            # resize
            img = cv2.resize(img, size)

            # تبدیل به grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # تبدیل تصویر به یک بردار
            flat = gray.flatten()

            images.append(flat)
            labels.append(label)

    return np.array(images), np.array(labels)
