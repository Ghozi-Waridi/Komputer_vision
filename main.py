import numpy as np
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from utils.load import load_all_images
from layers.Conv import Conv
from layers.NN import NeuralNetwork
from utils.logger import Logger


def extract_or_load_features(conv_layer, X_train, X_test, logger, save_dir='data/extract/numpy'):
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, 'train_features.npy')
    test_path = os.path.join(save_dir, 'test_features.npy')

    if os.path.exists(train_path) and os.path.exists(test_path):
        logger.info("Memuat fitur dari cache...")
        X_train_feat = np.load(train_path)
        X_test_feat = np.load(test_path)
    else:
        logger.info("Mengekstrak fitur training...")
        X_train_feat = conv_layer.extract_all(X_train, desc="Training features")
        logger.info("Mengekstrak fitur testing...")
        X_test_feat = conv_layer.extract_all(X_test, desc="Testing features")

        np.save(train_path, X_train_feat)
        np.save(test_path, X_test_feat)
        logger.info(f"Fitur disimpan ke {save_dir}/")

    return X_train_feat, X_test_feat


def log_confusion_matrix(y_true, y_pred, class_names, logger):
    cm = confusion_matrix(y_true, y_pred)
    header = f"{'':>14}" + "".join(f"{c:>12}" for c in class_names)
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for i, name in enumerate(class_names):
        row = f"  {name:>12}" + "".join(f"{cm[i, j]:>12}" for j in range(len(class_names)))
        logger.info(row)


def main():
    logger = Logger(log_dir='logs')

    IMAGE_SIZE = (128, 128)
    POOL_SIZE = 2
    HIDDEN_LAYERS = [256, 64]
    LEARNING_RATE = 0.01
    EPOCHS = 100
    DATA_ROOT = 'data/raw'

    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    logger.log_hyperparameters({
        'image_size': IMAGE_SIZE,
        'pool_size': POOL_SIZE,
        'hidden_layers': HIDDEN_LAYERS,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
    })

    logger.info("[1/4] Memuat dataset...")
    images_per_class = load_all_images(
        data_root=DATA_ROOT,
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
    )

    all_images = []
    all_labels = []
    for class_name in CLASS_NAMES:
        imgs = images_per_class[class_name]
        all_images.append(imgs)
        all_labels.extend([CLASS_TO_INDEX[class_name]] * len(imgs))

    X_all = np.concatenate(all_images, axis=0)
    y_all_labels = np.array(all_labels)

    num_classes = len(CLASS_NAMES)
    y_all = np.zeros((len(y_all_labels), num_classes))
    for i, label in enumerate(y_all_labels):
        y_all[i, label] = 1.0

    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
        X_all, y_all, y_all_labels,
        test_size=0.2,
        stratify=y_all_labels,
    )

    logger.info(f"Total data: {X_all.shape[0]} samples")
    logger.info(f"Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test (test_size=0.2, stratified)")

    logger.log_dataset_info({
        'total_samples': X_all.shape[0],
        'train_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'image_size': IMAGE_SIZE,
        'num_classes': len(CLASS_NAMES),
        'classes': str(CLASS_NAMES),
    })

    logger.info("[2/4] Feature extraction Convolution...")
    conv = Conv(pool_size=POOL_SIZE)
    conv.info(logger=logger)
    feature_dim = conv.get_feature_dim(IMAGE_SIZE)
    logger.info(f"Feature dimension: {feature_dim}")

    t0 = time.time()
    X_train_feat, X_test_feat = extract_or_load_features(conv, X_train, X_test, logger)
    logger.info(f"Waktu feature extraction: {time.time() - t0:.1f}s")
    logger.info(f"Train features: {X_train_feat.shape}")
    logger.info(f"Test  features: {X_test_feat.shape}")

    logger.info("[3/4] Training Neural Network...")
    layer_sizes = [feature_dim] + HIDDEN_LAYERS + [len(CLASS_NAMES)]
    nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=LEARNING_RATE)
    nn.info(logger=logger)

    logger.log_model_architecture({
        'type': 'Static Gaussian CNN + Fully-Connected NN',
        'conv_kernels': 1,
        'feature_dim': feature_dim,
        'nn_layers': str(layer_sizes),
    })

    logger.log_training_start()
    start_time = time.time()

    history = nn.fit(
        X_train_feat, y_train,
        X_val=X_test_feat, y_val=y_test,
        epochs=EPOCHS,
        logger=logger,
    )

    duration = time.time() - start_time
    logger.log_training_end(duration)

    logger.info("[4/4] Evaluasi model pada data testing...")
    test_loss, test_acc = nn.evaluate(X_test_feat, y_test, y_test_labels)
    logger.info(f"Test Loss    : {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    y_pred = nn.predict(X_test_feat)
    logger.info("Akurasi per kelas:")
    for i, name in enumerate(CLASS_NAMES):
        mask = y_test_labels == i
        if mask.sum() > 0:
            class_acc = np.mean(y_pred[mask] == i)
            logger.info(f"  {name:>12}: {class_acc:.4f} ({class_acc*100:.2f}%)")

    logger.info("Confusion Matrix:")
    log_confusion_matrix(y_test_labels, y_pred, CLASS_NAMES, logger)

    logger.info("Classification Report:")
    report = classification_report(y_test_labels, y_pred, target_names=CLASS_NAMES)
    for line in report.strip().split('\n'):
        logger.info(line)

    logger.log_metrics({
        'test_loss': test_loss,
        'test_accuracy': test_acc,
    })

    logger.info("Selesai!")


if __name__ == '__main__':
    main()
