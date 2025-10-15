import cv2
import numpy as np
from ultralytics import YOLO

class ShipDetector:
    """
    A simple YOLO-based ship detection wrapper for satellite imagery.
    Provides methods to predict binary masks and contour-overlaid images.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.1, device: str = None):
        """
        Args:
            model_path (str): Path to YOLO model weights (.pt file)
            conf_threshold (float): Confidence threshold for predictions
            device (str): 'cpu' or 'cuda' (auto if None)
        """
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.device = device

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Returns a binary mask where ships are detected.
        Args:
            image: RGB image (np.ndarray)
        Returns:
            np.ndarray: Binary mask (0 or 255)
        """
        # Run YOLO inference
        # NOTE: Ignoring batch inference options for simplicity
        results = self.model.predict(source=image, conf=self.conf, device=self.device, verbose=False)[0]

        # Handle no detections
        if results.masks is None or len(results.masks.data) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # Combine all instance masks into one
        mask = (results.masks.data.sum(0) > 0).cpu().numpy().astype(np.uint8) * 255

        # Resize back to original size (YOLO rescales internally)
        h, w = image.shape[:2]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_resized


    def predict_instance_masks(self, image: np.ndarray) -> list[np.ndarray]:
        """
        Returns a list of binary instance masks (one per detected ship).
        Args:
            image: RGB image (np.ndarray)
        Returns:
            list of np.ndarray: Each (H,W) uint8 mask with values {0,255}
        """
        results = self.model.predict(source=image, conf=self.conf, device=self.device, verbose=False)[0]
        if results.masks is None or len(results.masks.data) == 0:
            return []

        h, w = image.shape[:2]
        masks = []
        for m in results.masks.data.cpu().numpy():
            mask = cv2.resize((m > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        return masks



    def predict_contours(self, image: np.ndarray) -> np.ndarray:
        """
        Returns the original image with yellow contours drawn around ships.
        Args:
            image: RGB image (np.ndarray)
        Returns:
            np.ndarray: Contour-overlaid RGB image
        """
        mask = self.predict_mask(image)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoured = image.copy()
        cv2.drawContours(contoured, contours, -1, (255, 255, 0), 2)  # Yellow
        return contoured
