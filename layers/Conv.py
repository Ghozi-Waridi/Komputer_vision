import numpy as np
from tqdm import tqdm
from activation.Activation import relu


class Conv:


    KERNEL = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64)

    def __init__(self, pool_size=8):
        self.kernel_size = self.KERNEL.shape[0]
        self.pool_size = pool_size
        self.kernel = self.KERNEL

    def _convolve2d(self, image, kernel):

        kh, kw = kernel.shape
        ih, iw = image.shape
        oh = ih - kh + 1
        ow = iw - kw + 1

        kernel_flat = kernel
        output = np.zeros((oh, ow))

        for i in range(oh):
            for j in range(ow):
                patch = image[i:i+kh, j:j+kw]
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
     
     
    def extract_features(self, image):
         conv_out = self._convolve2d(image, self.kernel)
      #   conv_out = np.maximum(conv_out, 0) 
         conv_out = relu(conv_out)
         pooled = self._max_pool(conv_out, self.pool_size)
         
         return pooled.flatten()

    def extract_all(self, images, desc="Feature extraction"):

        n = len(images)
        all_features = []
        for i in tqdm(range(n), desc=desc):
            feat = self.extract_features(images[i])
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
