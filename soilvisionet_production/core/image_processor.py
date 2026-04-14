"""
Image Processor
Handles image upload, validation, preprocessing, and augmentation
"""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Tuple, Union, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Image handling and preprocessing for ViT inference"""
    
    # Standard ViT input size
    INPUT_SIZE = 224
    # Match training data_loader normalization (ImageNet-style)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    MAX_FILE_SIZE_MB = 50
    
    @staticmethod
    def validate_image(image_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate image format and size
        
        Args:
            image_path: Path to image file
            
        Returns:
            (is_valid, message)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return False, "File does not exist"
        
        # Check format
        if image_path.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
            return False, f"Unsupported format. Allowed: {ImageProcessor.SUPPORTED_FORMATS}"
        
        # Check file size
        file_size_mb = image_path.stat().st_size / (1024 * 1024)
        if file_size_mb > ImageProcessor.MAX_FILE_SIZE_MB:
            return False, f"File too large ({file_size_mb:.1f}MB). Max: {ImageProcessor.MAX_FILE_SIZE_MB}MB"
        
        # Try to open
        try:
            img = Image.open(image_path)
            img.load()
        except Exception as e:
            return False, f"Cannot read image: {str(e)}"
        
        return True, "Valid"
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (H, W, 3) in RGB format, or None if failed
        """
        try:
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def load_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Load image from bytes
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Image as numpy array in RGB format, or None if failed
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return np.array(img)
        except Exception as e:
            logger.error(f"Error loading image from bytes: {e}")
            return None
    
    @staticmethod
    def resize_image(image: np.ndarray, size: int = INPUT_SIZE, 
                    keep_aspect: bool = True) -> np.ndarray:
        """
        Resize image to specified size
        
        Args:
            image: Input image (H, W, 3)
            size: Target size
            keep_aspect: Whether to keep aspect ratio (with padding)
            
        Returns:
            Resized image
        """
        if keep_aspect:
            # Resize keeping aspect ratio with padding
            h, w = image.shape[:2]
            scale = size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply padding
            pad_h = (size - new_h) // 2
            pad_w = (size - new_w) // 2
            pad_h_r = size - new_h - pad_h
            pad_w_r = size - new_w - pad_w
            
            padded = cv2.copyMakeBorder(resized, pad_h, pad_h_r, pad_w, pad_w_r,
                                       cv2.BORDER_CONSTANT, value=(128, 128, 128))
            return padded
        else:
            # Simple resize
            return cv2.resize(image, (size, size), interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def normalize_image(image: np.ndarray, mean: np.ndarray = MEAN, 
                       std: np.ndarray = STD) -> np.ndarray:
        """
        Normalize image (ImageNet normalization)
        
        Args:
            image: Input image (H, W, 3) with values in [0, 255]
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Normalized image (H, W, 3) with values in range per normalization
        """
        # Convert to float [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        image = (image - mean) / std
        
        return image
    
    @staticmethod
    def to_tensor(image: np.ndarray) -> torch.Tensor:
        """
        Convert image to PyTorch tensor
        
        Args:
            image: Image as numpy array (H, W, 3)
            
        Returns:
            Image as tensor (3, H, W) in range appropriate for normalization
        """
        # Convert HWC to CHW
        image = image.transpose(2, 0, 1)
        
        # Convert to tensor
        return torch.FloatTensor(image)
    
    @staticmethod
    def preprocess(image: np.ndarray, size: int = INPUT_SIZE,
                  normalize: bool = True) -> torch.Tensor:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image (H, W, 3) with values in [0, 255]
            size: Target resolution
            normalize: Whether to apply normalization
            
        Returns:
            Preprocessed tensor (3, H, W)
        """
        # Resize to fixed square to match training transforms (no aspect padding)
        image = ImageProcessor.resize_image(image, size=size, keep_aspect=False)
        
        # Normalize
        if normalize:
            image = ImageProcessor.normalize_image(image)
        else:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = ImageProcessor.to_tensor(image)
        
        return tensor
    
    @staticmethod
    def batch_preprocess(images: list, size: int = INPUT_SIZE) -> torch.Tensor:
        """
        Preprocess multiple images
        
        Args:
            images: List of images as numpy arrays
            size: Target resolution
            
        Returns:
            Batch tensor (B, 3, H, W)
        """
        tensors = [ImageProcessor.preprocess(img, size=size) for img in images]
        return torch.stack(tensors)
    
    @staticmethod
    def apply_augmentation(image: np.ndarray, augmentation_type: str = 'none') -> np.ndarray:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image (H, W, 3)
            augmentation_type: Type of augmentation ('none', 'brightness', 'contrast', 
                             'flip', 'rotate', 'blur', 'random')
            
        Returns:
            Augmented image
        """
        if augmentation_type == 'none':
            return image
        
        elif augmentation_type == 'brightness':
            alpha = np.random.uniform(0.7, 1.3)
            return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        elif augmentation_type == 'contrast':
            alpha = np.random.uniform(0.7, 1.3)
            beta = -50 * (alpha - 1)
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        elif augmentation_type == 'flip':
            if np.random.rand() > 0.5:
                return cv2.flip(image, 1)  # Horizontal flip
            return image
        
        elif augmentation_type == 'rotate':
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))
        
        elif augmentation_type == 'blur':
            kernel_size = np.random.choice([3, 5, 7])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        elif augmentation_type == 'random':
            # Apply random augmentation
            aug_type = np.random.choice(['brightness', 'contrast', 'flip', 'rotate'])
            return ImageProcessor.apply_augmentation(image, aug_type)
        
        return image
    
    @staticmethod
    def get_image_stats(image: np.ndarray) -> Dict:
        """
        Get statistics about image
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with image stats
        """
        return {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'size_mb': image.nbytes / (1024 * 1024)
        }


if __name__ == '__main__':
    # Test image processor
    test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    tensor = ImageProcessor.preprocess(test_image)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    # Stats
    stats = ImageProcessor.get_image_stats(test_image)
    print(f"Image stats: {stats}")
