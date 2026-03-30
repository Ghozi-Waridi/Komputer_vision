import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utils.load import load_all_images, extract_load_features, log_confusion_matrix
from layers.Conv import Conv
from layers.NN import NeuralNetwork
from utils.logger import Logger





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
        random_state=42,
    )

    logger.info(f"Total data: {X_all.shape[0]} samples")
    logger.info(
        f"Split data: {X_train.shape[0]} train, {X_test.shape[0]} test "
        "(test_size=0.2, stratified)"
    )

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
    X_train_feat, X_test_feat = extract_load_features(conv, X_train, X_test, logger)
    logger.info(f"Waktu feature extraction: {time.time() - t0:.1f}s")
    logger.info(f"Train features: {X_train_feat.shape}")
    logger.info(f"Test  features: {X_test_feat.shape}")

    logger.info("[3/4] Training Neural Network...")
    nn = NeuralNetwork(
        input_dim=feature_dim,
        hidden_layers=HIDDEN_LAYERS,
        output_dim=num_classes,
        learning_rate=LEARNING_RATE
    )
    nn.info(logger=logger)

    t1 = time.time()
    nn.train(
        X_train_feat,
        y_train,
        epochs=EPOCHS,
        logger=logger,
    )
    logger.info(f"Waktu training: {time.time() - t1:.1f}s")

    logger.info("[4/4] Evaluasi Model...")
    test_loss, test_acc = nn.evaluate(X_test_feat, y_test, y_labels=y_test_labels)
    y_pred = nn.predict(X_test_feat)

    logger.log_metrics({
        'test_loss': test_loss,
        'test_accuracy': test_acc,
    })

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test_labels, y_pred, target_names=CLASS_NAMES))

    logger.info("\nConfusion Matrix:")
    log_confusion_matrix(y_test_labels, y_pred, CLASS_NAMES, logger)

    logger.info("✓ Project selesai!")


if __name__ == '__main__':
    main()
