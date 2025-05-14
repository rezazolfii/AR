"""
Image processing utilities for the AR Try-on application
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from ar_tryon.models import BiSeNetOfficialWrapper


# ----------------------------
# Color Palette with Standard HSV Values
# ----------------------------
def standard_to_opencv_hsv(h, s_percent, v_percent):
    """Convert standard HSV values to OpenCV HSV format"""
    h_opencv = int(h / 2)  # OpenCV uses 0-179 for hue (half of 360)
    s_opencv = int(s_percent * 255 / 100)  # Convert percentage to 0-255
    v_opencv = int(v_percent * 255 / 100)  # Convert percentage to 0-255
    
    return (h_opencv, s_opencv, v_opencv)

def create_color_palette():
    """Create a dictionary of color palettes for different features"""
    standard_hsv_colors = {
        # Hair colors
        "Black": (330, 18, 4),
        "Copper Penny": (347, 55, 34),
        "Cinnamon": (11, 54, 69),
        "Chocolate": (15, 81, 27),
        "Heavy Brown": (0, 46, 31),
        "Medium Brown": (12, 54, 60),
        "Light Brown": (21, 66, 75),
        "Angel Blonde 6": (28, 56, 63),
        "Angel Blonde 5": (31, 45, 73),
        "Angel Blonde 4": (34, 34, 83),
        "Angel Blonde 2": (44, 21, 90),
        "Strawberry Blonde": (25, 65, 85),
        "Auburn": (10, 70, 50),
        "Ginger": (20, 80, 70),
        "Red": (5, 90, 70),
        "Green": (120, 80, 60),
        "Blue": (220, 70, 60),
        "Purple": (280, 70, 60),
        
        # Lip colors
        "Natural Pink": (340, 40, 80),
        "Coral": (15, 70, 90),
        "Ruby Red": (0, 90, 80),
        "Burgundy": (345, 80, 50),
        
        # Skin tones
        "Fair": (19 , 17, 86),
        "Warm": (25, 20, 95),
        "Medium": (34, 27, 89),
        "Tan": (32, 34, 76),
        "Deep": (32, 44, 64),
        "Deep2": (22, 54, 45)
    }
    
    # Convert to OpenCV HSV format
    color_palette = {}
    for name, (h, s, v) in standard_hsv_colors.items():
        color_palette[name] = standard_to_opencv_hsv(h, s, v)
    return color_palette
# ----------------------------
# Create mask from multiple class indices
# ----------------------------
def create_mask_from_indices(segmentation_map, class_indices):
    """Create a binary mask from multiple class indices"""
    if not isinstance(class_indices, list):
        class_indices = [class_indices]
    mask = np.zeros_like(segmentation_map, dtype=np.uint8)
    for class_index in class_indices:
        mask = np.logical_or(mask, segmentation_map == class_index)
    return mask.astype(np.uint8) * 255

def refine_mask(mask, iterations=1, operation="both"):
    """
    Refine mask with morphological operations
    
    Args:
        mask: Binary mask
        iterations: Number of iterations for morphological operations
        operation: Type of operation ("open", "close", "both")
        
    Returns:
        Refined mask
    """
    kernel = np.ones((3, 3), np.uint8)
    
    if operation == "open" or operation == "both":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    if operation == "close" or operation == "both":
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    return mask

# Function to remove small isolated regions
def remove_small_regions(mask, min_size=50):
    """Remove small isolated regions from mask"""
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create a new mask with only large regions
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 255
            
    return new_mask

# ----------------------------
# Process mask with improved edge smoothing
# ----------------------------
def create_smooth_mask(mask, blur_size=51, expand_pixels=2):
    """
    Create a smooth transition mask for natural edges
    
    Args:
        mask: Binary mask
        blur_size: Size of Gaussian blur kernel (higher = smoother edges)
        expand_pixels: Number of pixels to expand the mask by
        
    Returns:
        Smooth transition mask
    """
    # Optionally expand the mask slightly to avoid gaps
    if expand_pixels > 0:
        kernel = np.ones((expand_pixels, expand_pixels), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Create a smooth transition mask with large Gaussian blur
    smooth_mask = mask.copy().astype(np.float32) / 255.0
    
    # Make sure blur_size is odd
    if blur_size % 2 == 0:
        blur_size += 1
        
    # Apply a large Gaussian blur for ultra-smooth transitions
    smooth_mask = cv2.GaussianBlur(smooth_mask, (blur_size, blur_size), 0)
    
    return smooth_mask


# ----------------------------
# Apply color to a region
# ----------------------------
def apply_color(image, mask, color_name, color_palette, color_strength=1.0):
    """
    Apply color to a region
    
    Args:
        image: Input BGR image
        mask: Binary mask of region to color
        color_name: Name of color from palette
        color_palette: Dictionary of colors
        color_strength: Strength of color application (0.0-1.0)
        
    Returns:
        Colored image
    """
    # Get color values
    if color_name not in color_palette:
        print(f"Color {color_name} not found. Using default.")
        color_name = "Medium Brown"
    
    h, s, v = color_palette[color_name]
    
    # # Special case for black hair - use direct BGR application
    # if color_name == "Black":
    #     result = image.copy()
    #     mask_bool = mask > 0
        
    #     # Apply black color directly in BGR
    #     black_bgr = np.zeros_like(image)
        
    #     # Alpha blend with strength
    #     alpha = color_strength * 0.9  # Slightly stronger for black
    #     result[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * black_bgr[mask_bool]
        
    #     return result
    
    # Special case for lip colors
    if color_name in ["Ruby Red", "Burgundy","Natural Pink", "Cora"]: 
        
        # Increase saturation for lip colors
        s = min(int(s * 1.2), 255)
    
    # Convert image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Create a boolean mask
    mask_bool = mask > 0
    
    # Create a new image with modified color
    result_hsv = image_hsv.copy()
    
    # Apply color with strength control
    if color_strength < 1.0:
        # Blend hue with strength factor
        original_h = image_hsv[mask_bool, 0]
        target_h = np.ones_like(original_h) * h
        result_hsv[mask_bool, 0] = (1 - color_strength) * original_h + color_strength * target_h
        
        # Blend saturation with strength factor
        original_s = image_hsv[mask_bool, 1]
        target_s = np.ones_like(original_s) * s
        result_hsv[mask_bool, 1] = (1 - color_strength) * original_s + color_strength * target_s
        
        # Blend value with strength factor
        blend_factor = 0.7 * color_strength
        result_hsv[mask_bool, 2] = (1 - blend_factor) * image_hsv[mask_bool, 2] + blend_factor * v
    else:
        # Full strength - direct replacement
        result_hsv[mask_bool, 0] = h
        result_hsv[mask_bool, 1] = s
        
        # Preserve some of the original value for lighting #ensure affect of v in original image cut by hailf image_hsv[mask_bool, 2]//2
        blend_factor = 0.7
        result_hsv[mask_bool, 2] = (1 - blend_factor) * image_hsv[mask_bool, 2]//1.3 + blend_factor * v
    
    # Convert back to BGR
    result_hsv = np.clip(result_hsv, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    
    return result_bgr



# ----------------------------
# Blend with original using smooth mask
# ----------------------------
def blend_with_original(original, colored, smooth_mask, is_black=False):
    """Blend colored result with original using smooth mask"""
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    colored_lab = cv2.cvtColor(colored, cv2.COLOR_BGR2LAB)
    
    if is_black:
        l_original = colored_lab[:,:,0]  # Use L from colored, which should be 0 for black
    else:
        l_original = original_lab[:,:,0]  # Use L from original to preserve lighting
    # l_original = original_lab[:,:,0]
    a_colored = colored_lab[:,:,1]
    b_colored = colored_lab[:,:,2]
    
    result_lab = np.zeros_like(original_lab)
    result_lab[:,:,0] = l_original
    result_lab[:,:,1] = a_colored
    result_lab[:,:,2] = b_colored
    
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    
    # Alpha blend with smooth mask
    final = original.astype(np.float32) * (1 - smooth_mask[:,:,np.newaxis]) + \
            result_bgr.astype(np.float32) * smooth_mask[:,:,np.newaxis]
    
    return np.clip(final, 0, 255).astype(np.uint8)

# ----------------------------
# Apply detail preservation
# ----------------------------
def apply_detail_preservation(original, result, mask, detail_factor=0.3):
    """Apply high-frequency details to result"""
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_original, (0, 0), 3)
    high_freq_details = gray_original.astype(np.float32) - blurred.astype(np.float32)
    
    mask_bool = mask > 0
    result_with_details = result.copy()
    
    for c in range(3):
        result_with_details[mask_bool, c] = np.clip(
            result_with_details[mask_bool, c] + high_freq_details[mask_bool] * detail_factor, 
            0, 
            255
        ).astype(np.uint8)
        
    return result_with_details


# ----------------------------
# Main function to apply colors
# ----------------------------
def apply_makeup(image_path, 
                hair_color="Auburn", hair_strength=1.0,
                lip_color="Ruby Red", lip_strength=1.0,
                skin_color="Dark Brown", skin_strength=0.3,
                edge_smoothness=51, detail_factor=0.3):
    """
    Apply makeup with simple parameter controls
    
    Args:
        image_path: Path to input image
        hair_color: Hair color name
        hair_strength: Strength of hair color (0.0-1.0)
        lip_color: Lip color name
        lip_strength: Strength of lip color (0.0-1.0)
        skin_color: Skin color name (None to skip)
        skin_strength: Strength of skin color (0.0-1.0)
        edge_smoothness: Smoothness of edges (higher = smoother)
        detail_factor: Strength of detail preservation
        
    Returns:
        Final image with makeup applied
    """
    # Load BiSeNet model
    model = BiSeNetOfficialWrapper(n_classes=19)
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = (image.shape[1], image.shape[0])  # width, height
    
    # Process image through model
    segmentation_mask = model.process_image(image)
    
    # Create color palette
    color_palette = create_color_palette()
    
    # Create masks for different features
    hair_indices = [17, 2, 3]  # Hair
    lips_indices = [12, 13]  # Upper and lower lips
    skin_indices = [1, 10]  # Skin, nose, neck
    
    hair_mask = create_mask_from_indices(segmentation_mask, hair_indices)
    lips_mask = create_mask_from_indices(segmentation_mask, lips_indices)
    skin_mask = create_mask_from_indices(segmentation_mask, skin_indices)
    
    # Refine masks
    hair_refined = refine_mask(hair_mask, iterations=2, operation="close")
    lips_refined = refine_mask(lips_mask, iterations=2, operation="both")
    skin_refined = refine_mask(skin_mask, iterations=1, operation="both")
    
    # Remove small isolated regions
    hair_refined = remove_small_regions(hair_refined, min_size=100)
    lips_refined = remove_small_regions(lips_refined, min_size=50)
    
    # Create smooth masks
    hair_smooth = create_smooth_mask(hair_refined, blur_size=edge_smoothness)
    lips_smooth = create_smooth_mask(lips_refined, blur_size=edge_smoothness-10)  # Slightly less blur for lips
    skin_smooth = create_smooth_mask(skin_refined, blur_size=edge_smoothness)
    
    # Apply colors
    result = image.copy()
    
    # Apply hair color - IMPORTANT: Use refined masks for both coloring and blending
    if hair_color:
        hair_colored = apply_color(result, hair_refined, hair_color, color_palette, hair_strength)
        result = blend_with_original(result, hair_colored, hair_smooth, is_black=(hair_color == "Black"))
        result = apply_detail_preservation(image, result, hair_refined, detail_factor)
    
    # Apply lip color
    if lip_color:
        lip_colored = apply_color(result, lips_refined, lip_color, color_palette, lip_strength)
        result = blend_with_original(result, lip_colored, lips_smooth, is_black=False)
        result = apply_detail_preservation(image, result, lips_refined, detail_factor)
    
    # Apply skin color
    if skin_color:
        skin_colored = apply_color(result, skin_refined, skin_color, color_palette, skin_strength)
        result = blend_with_original(result, skin_colored, skin_smooth, is_black=False)
        result = apply_detail_preservation(image, result, skin_refined, detail_factor * 0.8)  # Less detail for skin
    
    # Display results
    plt.figure(figsize=(15, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Final Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return result

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Use a local test image
    image_path = "D:/AR-tryon-app/AR-app/00072.png"  # Replace with your test image path
    
    # Apply makeup with default parameters
    result = apply_makeup(
        image_path,
        hair_color="Auburn",
        hair_strength=1.0,
        lip_color="Ruby Red",
        lip_strength=1.0,
        skin_color="Dark Brown",  # Set to None to skip skin coloring
        skin_strength=0.4,
        edge_smoothness=51,  # Higher value = smoother edges
        detail_factor=0.3    # Higher value = more texture
    )

# Example with different parameters
    result_smooth = apply_makeup(
        image_path,
        hair_color="Black",
        hair_strength=1,
        lip_color="Coral",
        lip_strength=1,
        skin_color="Light Brown",
        skin_strength=0.2,
        edge_smoothness=81,  # Very smooth edges
        detail_factor=0.2    # Less texture
    )