import cv2
import numpy as np

def preprocess_image(image):
    """
    Apply Lab-Color Space CLAHE to enhance underwater images.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        
    Returns:
        numpy.ndarray: Enhanced image in BGR format.
    """
    if image is None:
        return None

    # 1. Convert BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2. Split channels (L: Lightness, A: Green-Red, B: Blue-Yellow)
    l, a, b = cv2.split(lab)
    
    # 3. Apply CLAHE to the L-channel
    # Clip limit 3.0 is a good starting point for underwater contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 4. Merge the enhanced L-channel back with A and B channels
    limg = cv2.merge((cl, a, b))
    
    # 5. Convert back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_image
