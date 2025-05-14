import base64
import cv2
import numpy as np
import io
from PIL import Image


def decode_image(image_data):
    """Decode image from base64 or bytes"""
    if isinstance(image_data, str):
        # Base64 string
        image_bytes = base64.b64decode(image_data)
    else:
        # Already bytes
        image_bytes = image_data
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def encode_image(image):
    """Encode image to base64"""
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        return None
    
    image_bytes = encoded_image.tobytes()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return base64_image


def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image