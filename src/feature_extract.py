import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

# HOG parameters
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP parameters
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

def extract_hog_features(images):
    hog_features = []
    for img in images:
        img_reshaped = img.reshape(128,128)  # reshape back to 2D
        features = hog(img_reshaped,
                       orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK,
                       block_norm='L2-Hys')
        hog_features.append(features)
    return np.array(hog_features)

def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        img_reshaped = img.reshape(128,128)
        lbp = local_binary_pattern(img_reshaped, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
        # histogram LBP
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS+3), range=(0, LBP_POINTS+2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        lbp_features.append(hist)
    return np.array(lbp_features)

def extract_features(images):
    print("Extracting HOG features...")
    hog_feat = extract_hog_features(images)
    print("Extracting LBP features...")
    lbp_feat = extract_lbp_features(images)
    
    # ترکیب همه ویژگی‌ها
    features = np.concatenate([hog_feat, lbp_feat], axis=1)
    print("Feature extraction done. Shape:", features.shape)
    return features
