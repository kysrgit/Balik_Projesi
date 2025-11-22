import cv2
import numpy as np

class ImageEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Initializes the CLAHE object.
        
        Args:
            clip_limit (float): Threshold for contrast limiting. Higher = more contrast.
            tile_grid_size (tuple): Size of grid for histogram equalization.
        """
        # Create CLAHE object once to save initialization time in loop
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply_clahe(self, frame):
        """
        Applies Contrast Limited Adaptive Histogram Equalization to a BGR image.
        
        Args:
            frame (numpy.ndarray): Input image in BGR format.
            
        Returns:
            numpy.ndarray: Enhanced image in BGR format.
        """
        # 1. Convert to LAB color space (L channel contains lightness)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # 2. Split channels
        l, a, b = cv2.split(lab)
        
        # 3. Apply CLAHE to L-channel
        l_enhanced = self.clahe.apply(l)
        
        # 4. Merge channels back
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        
        # 5. Convert back to BGR
        frame_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return frame_enhanced

    def preprocess_for_onnx(self, frame, input_shape=(640, 640)):
        """
        Prepares the frame for YOLO ONNX inference.
        
        Args:
            frame (numpy.ndarray): Input BGR image.
            input_shape (tuple): Target size (width, height).
            
        Returns:
            numpy.ndarray: Preprocessed input tensor (1, 3, 640, 640).
            float: Scale factor used for resizing (to map boxes back).
        """
        # Resize while maintaining aspect ratio (letterbox) is ideal, 
        # but for speed on RPi, simple resizing is often acceptable if aspect ratio is close.
        # Here we implement simple resize for maximum FPS. 
        # If accuracy drops, switch to letterboxing.
        
        original_h, original_w = frame.shape[:2]
        
        # Resize
        img_resized = cv2.resize(frame, input_shape)
        
        # Normalize & Transpose
        img_data = img_resized.transpose(2, 0, 1) # HWC -> CHW
        img_data = np.expand_dims(img_data, axis=0) # Add batch dimension
        img_data = img_data.astype(np.float32) / 255.0
        
        scale_x = original_w / input_shape[0]
        scale_y = original_h / input_shape[1]
        
        return img_data, (scale_x, scale_y)
