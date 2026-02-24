# Contoh Penggunaan Logger

from utils.logger import create_logger
import time

# Buat logger instance
logger = create_logger(log_dir='logs', log_name='example.log')

# 1. Log informasi dataset
dataset_info = {
    'Total samples': 1000,
    'Training samples': 800,
    'Validation samples': 100,
    'Test samples': 100,
    'Number of classes': 10,
    'Image size': '224x224'
}
logger.log_dataset_info(dataset_info)

# 2. Log hyperparameters
hyperparams = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropy'
}
logger.log_hyperparameters(hyperparams)

# 3. Log model architecture
model_info = {
    'Type': 'CNN',
    'Total layers': 10,
    'Total parameters': 1250000,
    'Input shape': '(3, 224, 224)',
    'Output classes': 10
}
logger.log_model_architecture(model_info)

# 4. Log training dimulai
logger.log_training_start()

# 5. Simulasi training epochs
start_time = time.time()
for epoch in range(1, 6):
    train_loss = 1.5 / epoch
    train_acc = 0.3 * epoch
    val_loss = 1.7 / epoch
    val_acc = 0.28 * epoch
    
    logger.log_epoch(epoch, 5, train_loss, train_acc, val_loss, val_acc)
    
    # Log checkpoint setiap 2 epoch
    if epoch % 2 == 0:
        checkpoint_path = f'checkpoints/model_epoch_{epoch}.pth'
        metrics = {'loss': val_loss, 'accuracy': val_acc}
        logger.log_checkpoint(checkpoint_path, epoch, metrics)
    
    time.sleep(0.5)

# 6. Log training selesai
duration = time.time() - start_time
logger.log_training_end(duration)

# 7. Log metrics evaluasi
metrics = {
    'Test Accuracy': 0.95,
    'Test Loss': 0.25,
    'Precision': 0.94,
    'Recall': 0.93,
    'F1-Score': 0.935
}
logger.log_metrics(metrics)

# 8. Log prediksi
logger.log_prediction('image_001.jpg', 'cat', confidence=0.98)
logger.log_prediction('image_002.jpg', 'dog', confidence=0.87)

# 9. Log data augmentation
augmentation = {
    'Random Flip': True,
    'Random Rotation': '±15 degrees',
    'Random Crop': True,
    'Color Jitter': True
}
logger.log_data_augmentation(augmentation)

# 10. Log exception (contoh)
try:
    result = 10 / 0
except Exception as e:
    logger.log_exception(e, context="Division operation")

# 11. Log pesan custom
logger.info("Custom info message")
logger.warning("This is a warning message")
logger.error("This is an error message")

# Tutup logger
logger.close()

print(f"\n✅ Log berhasil disimpan di: {logger.log_file}")
