# ----------- [1] Cài đặt thư viện -----------
# !pip install opencv-python scikit-learn scikit-image joblib tqdm argparse matplotlib imagehash thundersvm

# ----------- [2] Import thư viện -----------
import os
import cv2
import numpy as np
import argparse
import logging
import hashlib
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage import exposure
from skimage.feature import hog, local_binary_pattern
from thundersvm import SVC  # Sử dụng ThunderSVM thay cho sklearn.svm.SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from imagehash import average_hash

# ----------- [3] Hàm phụ trợ -----------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", encoding="utf-8")
        ]
    )

def parse_arguments():
    """Xử lý tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Lung Cancer Detection Training & Inference Script')
    parser.add_argument('--grid_search', action='store_true', help='Kích hoạt tối ưu tham số SVM')
    parser.add_argument('--visualize_features', action='store_true', help='Hiển thị HOG & LBP features')
    parser.add_argument('--augment', action='store_true', help='Bật tăng cường dữ liệu (Data Augmentation)')
    parser.add_argument('--custom_weights', action='store_true', help='Sử dụng class weights tùy chỉnh cho SVM')
    parser.add_argument('--group_split', action='store_true', help='Phân chia dữ liệu theo patient bằng GroupShuffleSplit')
    parser.add_argument('--detailed', action='store_true', help='Sử dụng chế độ trích xuất đặc trưng chi tiết hơn')
    parser.add_argument('--predict', type=str, default=None, help='Đường dẫn ảnh hoặc thư mục cần dự đoán')
    return parser.parse_args()

# ----------- [4] Hàm chính -----------
def main():
    args = parse_arguments()
    setup_logging()
    
    try:
        # Nếu tham số --predict được cung cấp, load model, scaler và PCA để dự đoán trên dữ liệu mới
        if args.predict is not None:
            model_files = [f for f in os.listdir("models") if f.startswith("lung_cancer_svm")]
            scaler_files = [f for f in os.listdir("models") if f.startswith("scaler")]
            pca_files = [f for f in os.listdir("models") if f.startswith("pca")]
            if not model_files or not scaler_files or not pca_files:
                raise ValueError("Không tìm thấy model, scaler hoặc PCA đã lưu.")
            model_files.sort()
            scaler_files.sort()
            pca_files.sort()
            MODEL_SAVE_PATH = os.path.join("models", model_files[-1])
            SCALER_SAVE_PATH = os.path.join("models", scaler_files[-1])
            PCA_SAVE_PATH = os.path.join("models", pca_files[-1])
            model = load(MODEL_SAVE_PATH)
            scaler = load(SCALER_SAVE_PATH)
            pca = load(PCA_SAVE_PATH)
            logging.info(f"Loaded model from {MODEL_SAVE_PATH}")
            logging.info(f"Loaded scaler from {SCALER_SAVE_PATH}")
            logging.info(f"Loaded PCA from {PCA_SAVE_PATH}")
            
            if os.path.isdir(args.predict):
                predict_on_new_folder(args.predict, model, scaler, pca, detailed=args.detailed)
            else:
                predict_new_image(args.predict, model, scaler, pca, detailed=args.detailed)
            return

        # ----------- [5] Cấu hình tham số -----------
        DATA_DIR = "dataset"
        TEST_DIR = os.path.join(DATA_DIR, "Testcases")  # Dữ liệu test nằm trong thư mục Testcases
        # Sử dụng tên file cố định cho model, scaler và PCA
        MODEL_SAVE_PATH = os.path.join("models", "lung_cancer_svm.joblib")
        SCALER_SAVE_PATH = os.path.join("models", "scaler.joblib")
        PCA_SAVE_PATH = os.path.join("models", "pca.joblib")
        os.makedirs("models", exist_ok=True)

        # ----------- [6] Xử lý dữ liệu -----------
        logging.info("Bắt đầu quá trình xử lý dữ liệu huấn luyện")
        # Sửa img_size từ (1536, 1536) thành (512, 512)
        images, labels = load_dataset(DATA_DIR, img_size=(512, 512), augment=args.augment)

        print("Số lượng ảnh mỗi lớp sau cân bằng:")
        print("- Normal:", np.sum(labels == 0))
        print("- Cancer:", np.sum(labels == 1))
        
        visualize_samples(images, labels)

        all_image_paths = [os.path.join(root, f) for root, _, files in os.walk(DATA_DIR) for f in files]
        duplicates = find_duplicates(all_image_paths)
        logging.info(f"Phát hiện {len(duplicates)} ảnh trùng lặp")
        
        check_patient_leakage(all_image_paths)
        
        # ----------- [7] Trích xuất đặc trưng -----------
        X = extract_combined_features(images, detailed=args.detailed)
        y = labels

        if args.visualize_features:
            idx_normal = np.where(labels == 0)[0][0]
            idx_cancer = np.where(labels == 1)[0][0]
            visualize_feature_maps(images[idx_normal], title="Normal", detailed=args.detailed)
            visualize_feature_maps(images[idx_cancer], title="Cancer", detailed=args.detailed)
        
        # ----------- [8] Chuẩn bị dữ liệu -----------
        if args.group_split:
            patient_groups = [path.split('_')[0] for path in all_image_paths]
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, y, groups=patient_groups))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.1, 
                random_state=42, 
                shuffle=True,
                stratify=y
            )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Giữ lại nhiều thông tin hơn: tăng số thành phần của PCA
        pca = PCA(n_components=150, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # ----------- [9] Huấn luyện mô hình -----------        
        logging.info("Bắt đầu huấn luyện mô hình với ThunderSVM")
        if args.grid_search:
            # ThunderSVM có giao diện tương tự SVC nên vẫn có thể sử dụng GridSearchCV
            model = train_with_grid_search(X_train, y_train)
        else:
            if args.custom_weights:
                model = train_custom_weight_model(X_train, y_train)
            else:
                model = train_default_model(X_train, y_train)

        # ----------- [10] Đánh giá mô hình -----------        
        evaluate_model(model, X_train, X_test, y_train, y_test)

        # ----------- [11] Lưu model -----------        
        validate_production_requirements(model, scaler)
        dump(model, MODEL_SAVE_PATH)
        dump(scaler, SCALER_SAVE_PATH)
        dump(pca, PCA_SAVE_PATH)
        logging.info(f"Model đã được lưu tại: {MODEL_SAVE_PATH}")

    except Exception as e:
        logging.error(f"Lỗi nghiêm trọng: {str(e)}", exc_info=True)
        raise

# ----------- [12] Các hàm chức năng -----------

def load_dataset(data_dir, img_size=(512, 512), augment=False):
    """Tải và tiền xử lý dataset với tùy chọn tăng cường dữ liệu."""
    images = []
    labels = []
    
    for class_name in ["normal", "cancer"]:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Thiếu thư mục: {class_dir}")
            
        for img_name in tqdm(os.listdir(class_dir), desc=f"Đang tải {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            validate_image(img)
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Chuẩn hóa [0, 1]
            images.append(img)
            labels.append(0 if class_name == "normal" else 1)
            
            if augment:
                # Augmentation gốc (lật, xoay)
                flipped = cv2.flip(img, 1)
                images.append(flipped)
                labels.append(0 if class_name == "normal" else 1)
                
                (h, w) = img.shape
                center = (w // 2, h // 2)
                for angle in [-5, 5]:
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(img, M, (w, h))
                    images.append(rotated)
                    labels.append(0 if class_name == "normal" else 1)

                # Thêm biến đổi W/L ngẫu nhiên
                for _ in range(2):  # Thêm 2 biến thể W/L cho mỗi ảnh
                    alpha = np.random.uniform(0.5, 1.5)    # Mô phỏng thay đổi W (contrast)
                    beta = np.random.uniform(-0.3, 0.3)    # Mô phỏng thay đổi L (brightness)
                    augmented_img = img * alpha + beta
                    augmented_img = np.clip(augmented_img, 0, 1)  # Giữ giá trị trong [0, 1]
                    images.append(augmented_img)
                    labels.append(0 if class_name == "normal" else 1)
    
    return np.array(images), np.array(labels)

def extract_hog_features_enhanced(images, detailed=False):
    """Trích xuất đặc trưng HOG.
       Nếu detailed=True, sử dụng tham số chi tiết hơn để nắm bắt đặc trưng vi mô.
    """
    hog_features = []
    for img in tqdm(images, desc="Trích xuất HOG"):
        if detailed:
            features = hog(
                img,
                orientations=32,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                channel_axis=None
            )
        else:
            features = hog(
                img,
                orientations=16,
                pixels_per_cell=(16, 16),
                cells_per_block=(3, 3),
                channel_axis=None
            )
        hog_features.append(features)
    return np.array(hog_features)

def extract_lbp_features(images, detailed=False, radius=3, n_points=24):
    """Trích xuất đặc trưng LBP và tính histogram.
       Nếu detailed=True, sử dụng radius nhỏ hơn để thu nhận đặc trưng vi mô.
    """
    lbp_features = []
    for img in tqdm(images, desc="Trích xuất LBP"):
        if detailed:
            lbp = local_binary_pattern(img, n_points, 2, method="uniform")
            bins = np.arange(0, n_points + 3)
            range_val = (0, n_points + 2)
        else:
            lbp = local_binary_pattern(img, n_points, radius, method="uniform")
            bins = np.arange(0, n_points + 3)
            range_val = (0, n_points + 2)
        (hist, _) = np.histogram(lbp.ravel(), bins=bins, range=range_val)
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        lbp_features.append(hist)
    return np.array(lbp_features)

def extract_combined_features(images, detailed=False):
    """Kết hợp đặc trưng HOG và LBP thành vector đặc trưng duy nhất."""
    hog_feats = extract_hog_features_enhanced(images, detailed=detailed)
    lbp_feats = extract_lbp_features(images, detailed=detailed)
    combined = np.hstack([hog_feats, lbp_feats])
    return combined

def visualize_samples(images, labels, n_samples=5):
    """Trực quan hóa mẫu ảnh của mỗi lớp."""
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        idx = np.random.choice(np.where(labels == 0)[0]) if i < n_samples//2 else np.random.choice(np.where(labels == 1)[0])
        plt.imshow(images[idx], cmap='gray')
        plt.title("Normal" if labels[idx] == 0 else "Cancer")
        plt.axis('off')
    plt.show()

def visualize_feature_maps(img, title="Feature Map", detailed=False):
    """Hiển thị trực quan HOG và LBP của ảnh."""
    if detailed:
        hog_image, _ = hog(
            img,
            orientations=32,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=None,
            visualize=True
        )
    else:
        hog_image, _ = hog(
            img,
            orientations=16,
            pixels_per_cell=(16, 16),
            cells_per_block=(3, 3),
            channel_axis=None,
            visualize=True
        )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    if detailed:
        lbp = local_binary_pattern(img, 24, 2, method="uniform")
    else:
        lbp = local_binary_pattern(img, 24, 3, method="uniform")
    
    plt.figure(figsize=(10, 4))
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.imshow(hog_image_rescaled, cmap='gray')
    plt.title("HOG")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(lbp, cmap='gray')
    plt.title("LBP")
    plt.axis('off')
    plt.show()

def train_with_grid_search(X_train, y_train):
    """Tối ưu tham số SVM bằng GridSearchCV với phạm vi tham số mở rộng."""
    logging.info("Khởi chạy GridSearchCV")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01],
        'kernel': ['rbf', 'linear', 'poly']
    }
    grid_search = GridSearchCV(
        SVC(class_weight='balanced', probability=True, random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1_weighted',
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Tham số tối ưu: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_default_model(X_train, y_train):
    """Huấn luyện mô hình SVM với tham số mặc định giảm overfitting, sử dụng ThunderSVM."""
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_custom_weight_model(X_train, y_train):
    """Huấn luyện SVM sử dụng class weights tùy chỉnh, sử dụng ThunderSVM."""
    class_weights = {0: 1, 1: 2}
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight=class_weights,
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Đánh giá mô hình và kiểm tra overfitting."""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    logging.info(f"Độ chính xác - Train: {train_acc:.2f}, Test: {test_acc:.2f}")
    print(f"\nĐộ chính xác - Train: {train_acc:.2f}, Test: {test_acc:.2f}")
    
    if train_acc - test_acc > 0.15:
        logging.warning("Cảnh báo: Mô hình có thể bị overfitting!")
    
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, model.predict(X_test)))

def validate_image(img):
    """Kiểm tra chất lượng ảnh đầu vào."""
    if img is None:
        raise ValueError("Không thể đọc ảnh")
    # Kiểm tra độ tương phản để phát hiện ảnh không hợp lệ
    if img.std() < 5:  # Độ lệch chuẩn thấp -> ảnh ít tương phản
        raise ValueError("Ảnh có độ tương phản quá thấp")

def validate_production_requirements(model, scaler):
    """Kiểm tra yêu cầu triển khai production."""
    if not hasattr(model, 'predict_proba'):
        raise ValueError("Model cần hỗ trợ xác suất dự đoán")
    if not hasattr(scaler, 'mean_'):
        raise ValueError("Scaler không hợp lệ")

def find_duplicates(image_paths):
    """Tìm ảnh trùng lặp."""
    hashes = {}
    duplicates = []
    for path in image_paths:
        with open(path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        img = Image.open(path)
        img_hash = str(average_hash(img))
        combined_hash = file_hash + img_hash
        if combined_hash in hashes:
            duplicates.append((path, hashes[combined_hash]))
        else:
            hashes[combined_hash] = path
    return duplicates

def check_patient_leakage(image_paths):
    """Kiểm tra data leakage theo patient ID."""
    patient_ids = [path.split('_')[0] for path in image_paths]
    unique_patients = len(set(patient_ids))
    print(f"Số lượng bệnh nhân duy nhất: {unique_patients}")
    print(f"Số lượng ảnh trung bình mỗi bệnh nhân: {len(image_paths)/unique_patients:.1f}")

# ----------- [13] Các hàm hỗ trợ dự đoán trên dữ liệu mới -----------
def preprocess_image_from_path(image_path, img_size=(512, 512)):
    """Đọc và tiền xử lý một ảnh từ file."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    validate_image(img)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    
    # Thêm cân bằng histogram để chuẩn hóa độ tương phản
    img = exposure.equalize_hist(img)
    
    return img

def extract_feature_single(img, detailed=False):
    """Trích xuất đặc trưng kết hợp HOG và LBP cho một ảnh đơn lẻ."""
    if detailed:
        hog_feat = hog(
            img,
            orientations=32,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=None
        )
        lbp = local_binary_pattern(img, 24, 2, method="uniform")
        bins = np.arange(0, 24 + 3)
        range_val = (0, 24 + 2)
    else:
        hog_feat = hog(
            img,
            orientations=16,
            pixels_per_cell=(16, 16),
            cells_per_block=(3, 3),
            channel_axis=None
        )
        lbp = local_binary_pattern(img, 24, 3, method="uniform")
        bins = np.arange(0, 24 + 3)
        range_val = (0, 24 + 2)
    (hist, _) = np.histogram(lbp.ravel(), bins=bins, range=range_val)
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    combined = np.hstack([hog_feat, hist])
    return combined

def predict_new_image(image_path, model, scaler, pca, img_size=(512,512), detailed=False):
    """
    Dự đoán nhãn cho một ảnh đơn lẻ:
      - Đọc ảnh, tiền xử lý và trích xuất đặc trưng.
      - Chuẩn hóa qua scaler và giảm chiều qua PCA.
      - Dự đoán nhãn và tính độ tin cậy.
    """
    img = preprocess_image_from_path(image_path, img_size)
    feature = extract_feature_single(img, detailed=detailed)
    feature = np.array(feature).reshape(1, -1)
    feature = scaler.transform(feature)
    feature = pca.transform(feature)
    pred = model.predict(feature)[0]
    prob = model.predict_proba(feature)[0]
    conf = prob[int(pred)]
    label_name = "Normal" if pred == 0 else "Cancer"
    print(f"Ảnh: {image_path}, Dự đoán: {label_name}, Độ tin cậy: {conf*100:.2f}%")
    return pred, conf

def predict_on_new_folder(folder_path, model, scaler, pca, img_size=(512,512), detailed=False):
    """Dự đoán cho tất cả ảnh trong một thư mục."""
    for file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file)
        try:
            predict_new_image(image_path, model, scaler, pca, img_size, detailed=detailed)
        except Exception as e:
            print(f"Không thể xử lý ảnh {image_path}: {e}")

if __name__ == "__main__":
    main()
