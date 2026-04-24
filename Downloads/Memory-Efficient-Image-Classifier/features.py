import cv2
import numpy as np

class HOGFeatureExtractor:
    def __init__(self):
        self.hog = cv2.HOGDescriptor(
            _winSize=(32, 32),
            _blockSize=(8, 8),
            _blockStride=(4, 4),
            _cellSize=(8, 8),
            _nbins=9
        )

    def extract(self, images):
        features = []

        for img in images:
            # CIFAR is 32x32 RGB → convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            hog_feat = self.hog.compute(img)
            features.append(hog_feat.flatten())

        return np.array(features)