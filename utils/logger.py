import logging
import os
from datetime import datetime
from pathlib import Path


class Logger:
    """
    Logger class untuk mencatat semua aktivitas training dan inference
    """
    
    def __init__(self, log_dir='logs', log_name=None):
        """
        Inisialisasi logger
        
        Args:
            log_dir: Direktori untuk menyimpan log files
            log_name: Nama file log (default: timestamp)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Buat nama file log dengan timestamp jika tidak diberikan
        if log_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = f'training_log_{timestamp}.log'
        
        self.log_file = self.log_dir / log_name
        
        # Setup logger
        self.logger = logging.getLogger('ComputerVision')
        self.logger.setLevel(logging.DEBUG)
        
        # Hapus handler yang sudah ada untuk menghindari duplikasi
        self.logger.handlers.clear()
        
        # File handler - menyimpan semua log ke file
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler - menampilkan log di console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # Tambahkan handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.info(f"Logger initialized. Log file: {self.log_file}")
    
    def debug(self, message):
        """Log pesan debug"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log pesan info"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log pesan warning"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log pesan error"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log pesan critical"""
        self.logger.critical(message)
    
    def log_epoch(self, epoch, total_epochs, train_loss, train_acc, val_loss=None, val_acc=None):
        """
        Log hasil training per epoch
        
        Args:
            epoch: Epoch saat ini
            total_epochs: Total epochs
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss (optional)
            val_acc: Validation accuracy (optional)
        """
        msg = f"Epoch [{epoch}/{total_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        if val_loss is not None and val_acc is not None:
            msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        self.info(msg)
    
    def log_model_architecture(self, model_info):
        """
        Log arsitektur model
        
        Args:
            model_info: Dictionary atau string berisi informasi model
        """
        self.info("="*50)
        self.info("MODEL ARCHITECTURE")
        self.info("="*50)
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                self.info(f"{key}: {value}")
        else:
            self.info(str(model_info))
        self.info("="*50)
    
    def log_hyperparameters(self, params):
        """
        Log hyperparameters
        
        Args:
            params: Dictionary berisi hyperparameters
        """
        self.info("="*50)
        self.info("HYPERPARAMETERS")
        self.info("="*50)
        for key, value in params.items():
            self.info(f"{key}: {value}")
        self.info("="*50)
    
    def log_dataset_info(self, dataset_info):
        """
        Log informasi dataset
        
        Args:
            dataset_info: Dictionary berisi informasi dataset
        """
        self.info("="*50)
        self.info("DATASET INFORMATION")
        self.info("="*50)
        for key, value in dataset_info.items():
            self.info(f"{key}: {value}")
        self.info("="*50)
    
    def log_training_start(self):
        """Log saat training dimulai"""
        self.info("="*50)
        self.info("TRAINING STARTED")
        self.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("="*50)
    
    def log_training_end(self, duration=None):
        """
        Log saat training selesai
        
        Args:
            duration: Durasi training dalam detik (optional)
        """
        self.info("="*50)
        self.info("TRAINING COMPLETED")
        self.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if duration:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            self.info(f"Duration: {hours}h {minutes}m {seconds}s")
        self.info("="*50)
    
    def log_prediction(self, input_data, prediction, confidence=None):
        """
        Log hasil prediksi
        
        Args:
            input_data: Data input
            prediction: Hasil prediksi
            confidence: Confidence score (optional)
        """
        msg = f"Prediction: {prediction}"
        if confidence is not None:
            msg += f" (Confidence: {confidence:.4f})"
        self.info(msg)
    
    def log_metrics(self, metrics):
        """
        Log metrics evaluasi
        
        Args:
            metrics: Dictionary berisi metrics
        """
        self.info("="*50)
        self.info("EVALUATION METRICS")
        self.info("="*50)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.info(f"{key}: {value:.4f}")
            else:
                self.info(f"{key}: {value}")
        self.info("="*50)
    
    def log_checkpoint(self, checkpoint_path, epoch, metrics):
        """
        Log saat menyimpan checkpoint
        
        Args:
            checkpoint_path: Path checkpoint yang disimpan
            epoch: Epoch saat ini
            metrics: Metrics yang dicapai
        """
        self.info(f"Checkpoint saved at epoch {epoch}")
        self.info(f"Path: {checkpoint_path}")
        self.info(f"Metrics: {metrics}")
    
    def log_data_augmentation(self, augmentation_info):
        """
        Log informasi data augmentation
        
        Args:
            augmentation_info: Dictionary atau string berisi info augmentasi
        """
        self.info("Data Augmentation Applied:")
        if isinstance(augmentation_info, dict):
            for key, value in augmentation_info.items():
                self.info(f"  {key}: {value}")
        else:
            self.info(f"  {augmentation_info}")
    
    def log_exception(self, exception, context=""):
        """
        Log exception yang terjadi
        
        Args:
            exception: Exception object
            context: Konteks tambahan (optional)
        """
        if context:
            self.error(f"Exception in {context}: {type(exception).__name__}: {str(exception)}")
        else:
            self.error(f"Exception: {type(exception).__name__}: {str(exception)}")
    
    def close(self):
        """Tutup semua handlers"""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)


# Convenience function untuk membuat logger dengan mudah
def create_logger(log_dir='logs', log_name=None):
    """
    Factory function untuk membuat logger instance
    
    Args:
        log_dir: Direktori untuk menyimpan log files
        log_name: Nama file log (default: timestamp)
    
    Returns:
        Logger instance
    """
    return Logger(log_dir=log_dir, log_name=log_name)
