import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from activation.Activation import relu


class Conv:


    # KERNEL = np.array([
    #     [ -1, 0,  1],
    #     [-2,  0, 2],
    #     [ -1, 0, 1]
    # ], dtype=np.float64)
    
    # KERNEL = np.array([
    #     [ -1, 0,  1],
    #     [-2,  0, 2],
    #     [ -1, 0, 1]
    # ], dtype=np.float64)
    
    KERNEL = np.array([
    [0.0625, 0.1250, 0.0625],
    [0.1250, 0.2500, 0.1250],
    [0.0625, 0.1250, 0.0625]
])

    def __init__(self, pool_size=2):
        self.kernel_size = self.KERNEL.shape[0]
        self.pool_size = pool_size
        self.kernel = self.KERNEL

    def _convolve2d(self, image, kernel):

        kh, kw = kernel.shape
        ih, iw = image.shape
        oh = ih - kh + 1
        ow = iw - kw + 1

        kernel_flat = kernel.flatten()
        output = np.zeros((oh, ow))

        for i in range(oh):
            for j in range(ow):
                patch = image[i:i+kh, j:j+kw].flatten()
                output[i, j] = np.dot(patch, kernel_flat)

        return output

    def _max_pool(self, image, pool_size):
        h, w = image.shape
        new_h = h // pool_size
        new_w = w // pool_size
        output = np.zeros((new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                region = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                output[i, j] = np.max(region)

        return output
     
     
    def _save_image(self, array, path):
       
        arr = array - array.min()
        max_val = arr.max()
        if max_val > 0:
            arr = arr / max_val
        img = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
        img.save(path)

    def extract_features(self, image, idx=None, save_dir=None):
         conv_out = self._convolve2d(image, self.kernel)
         conv_out = relu(conv_out)
         pooled = self._max_pool(conv_out, self.pool_size)
         conv_out = self._convolve2d(pooled, self.kernel)
         conv_out = relu(conv_out)
         pooled = self._max_pool(conv_out, self.pool_size)

         if save_dir is not None and idx is not None:
             conv_dir = os.path.join(save_dir, 'conv')
             pool_dir = os.path.join(save_dir, 'pool')
             os.makedirs(conv_dir, exist_ok=True)
             os.makedirs(pool_dir, exist_ok=True)
             self._save_image(conv_out, os.path.join(conv_dir, f'img_{idx:04d}.png'))
             self._save_image(pooled, os.path.join(pool_dir, f'img_{idx:04d}.png'))

         return pooled.flatten()

    def extract_all(self, images, desc="Feature extraction", save_dir=None):

        n = len(images)
        all_features = []
        for i in tqdm(range(n), desc=desc):
            feat = self.extract_features(images[i], idx=i, save_dir=save_dir)
            all_features.append(feat)
        return np.array(all_features)

    def get_feature_dim(self, image_size=(128, 128)):
        oh = image_size[0] - self.kernel_size + 1
        ow = image_size[1] - self.kernel_size + 1
        pooled_h = oh // self.pool_size
        pooled_w = ow // self.pool_size
        return pooled_h * pooled_w

    def info(self, logger=None):
        lines = [
            "Conv Layer:",
            f"  Kernel size : {self.kernel_size}x{self.kernel_size}",
            f"  Kernel      : {self.kernel.tolist()}",
            f"  Pool size   : {self.pool_size}",
        ]
        for line in lines:
            if logger:
                logger.info(line)
            else:
                print(line)
