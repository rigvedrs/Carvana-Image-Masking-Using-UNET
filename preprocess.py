import cv2
import albumentations as A
#import numpy as np
from albumentations.pytorch import ToTensorV2


def preprocess(img_filepath):
    image = cv2.imread(img_filepath)
    
    test_transform= A.Compose([
    A.Resize(572,572),
    A.Normalize(
        mean = [0.0,0.0,0.0],
        std = [1.0,1.0,1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2()
    ])
    
    aug = test_transform(image=image)
    
    image = aug['image']
    
    return image