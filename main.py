from src.preprocess import load_images_and_labels
from src.feature_extract import extract_features
from src.train import train_models
from src.evaluate import evaluate_models
import os

def main():
    print("🌱 Plant Disease Detection Project Started 🌱\n")

    # --- مرحله 1: بارگذاری تصاویر ---
    print("Step 1: Loading images...")
    X, y = load_images_and_labels(dataset_path="dataset/plantvillage dataset/color")
    print("Images loaded:", X.shape, "Labels:", len(y))

    # --- مرحله 2: استخراج ویژگی‌ها ---
    print("\nStep 2: Extracting features...")
    X_features = extract_features(X)
    print("Features shape:", X_features.shape)

    # --- مرحله 3: آموزش مدل‌ها ---
    print("\nStep 3: Training models...")
    best_model, X_test, y_test = train_models(X_features, y)

    # --- مرحله 4: ارزیابی مدل ---
    print("\nStep 4: Evaluating best model...")
    evaluate_models(best_model, X_test, y_test)

    print("\n✅ Project Finished Successfully!")

if __name__ == "__main__":
    # ساخت فولدر مدل در صورت نبودن
    if not os.path.exists("model"):
        os.makedirs("model")
    main()
