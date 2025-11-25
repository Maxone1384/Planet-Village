import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# --- LBP parameters ---
LBP_P = 8
LBP_R = 1

def extract_hog_features(image):
    features, _ = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )
    return features

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, LBP_P, LBP_R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_color_histogram(image):
    hist = []
    for i in range(3):  
        channel_hist = cv2.calcHist([image], [i], None, [64], [0, 256])
        channel_hist = cv2.normalize(channel_hist, None).flatten()
        hist.extend(channel_hist)
    return np.array(hist)

def extract_features(X_images):
    feature_list = []

    for img in X_images:
        # ----- مرحله مهم: تبدیل به uint8 -----
        img = img.astype("uint8")
        
        # تبدیل به Gray برای HOG و LBP
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype("uint8")

        # استخراج ویژگی‌ها
        hog_features = extract_hog_features(gray)
        lbp_features = extract_lbp_features(gray)
        color_hist = extract_color_histogram(img)

        # ترکیب همه ویژگی‌ها
        final_features = np.hstack([hog_features, lbp_features, color_hist])
        feature_list.append(final_features)

    return np.array(feature_list, dtype="float32")
