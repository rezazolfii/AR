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
        "Black": (330, 18, 4),  #(279, 33.33, 4.71)
        "Copper Penny": (347, 55, 34),
        "Cinnamon": (11, 54, 69),
        "Chocolate": (15, 81, 27),
        "Heavy Brown": (0, 46, 31),
        "Medium Brown": (12, 54, 60),
        "Light Brown": (21, 66, 75),
        "Angel Blonde 6": (26, 37, 36),
        "Angel Blonde 7" : (26, 37, 36),
        "Angel Blonde 5": (24, 25, 75),
        "Angel Blonde 4": (34, 34, 83),
        "Angel Blonde 2": (43, 13, 77),
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
# ----------------------------
# Apply color or texture to a region
# ----------------------------
def apply_color(image, mask, color_name, color_palette, color_strength=1.0, texture_swatches=None):
    """
    Apply color or texture to a region

    Args:
        image: Input BGR image
        mask: Binary mask of region to color
        color_name: Name of color from palette or texture swatch name
        color_palette: Dictionary of solid colors (HSV)
        color_strength: Strength of color/texture application (0.0-1.0)
        texture_swatches: Dictionary mapping texture names to image file paths

    Returns:
        Colored/textured image region
    """
    result = image.copy().astype(np.float32) # Use float for blending calculations
    mask_bool = mask > 0

    if texture_swatches and color_name in texture_swatches:
        # --- Apply Texture ---
        texture_path = texture_swatches[color_name]
        try:
            texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
            if texture is None:
                print(f"Error: Texture image not found at {texture_path}")
                # Fallback to a default color or original image if texture loading fails
                if color_name in color_palette:
                     print(f"Falling back to solid color for {color_name}")
                     h, s, v = color_palette[color_name]
                     # Create a solid color image of the same size as the original
                     solid_color_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0,0]
                     result[mask_bool] = (1 - color_strength) * result[mask_bool] + color_strength * solid_color_bgr

                return result
        except Exception as e:
             print(f"Error loading texture image {texture_path}: {e}")
             # Fallback to a default color or original image if texture loading fails
             if color_name in color_palette:
                  print(f"Falling back to solid color for {color_name}")
                  h, s, v = color_palette[color_name]
                  solid_color_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0,0]
                  result[mask_bool] = (1 - color_strength) * result[mask_bool] + color_strength * solid_color_bgr
             return result


        # Ensure texture is at least 1x1
        if texture.shape[0] == 0 or texture.shape[1] == 0:
             print(f"Error: Loaded texture image {texture_path} is empty or invalid.")
             # Fallback
             if color_name in color_palette:
                  print(f"Falling back to solid color for {color_name}")
                  h, s, v = color_palette[color_name]
                  solid_color_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0,0]
                  result[mask_bool] = (1 - color_strength) * result[mask_bool] + color_strength * solid_color_bgr
             return result


        # Tile the texture to cover the image size
        img_h, img_w, _ = image.shape
        tex_h, tex_w, _ = texture.shape

        # Calculate how many tiles are needed
        tiles_y = (img_h + tex_h - 1) // tex_h
        tiles_x = (img_w + tex_w - 1) // tex_w

        # Create a tiled version of the texture
        tiled_texture = np.tile(texture, (tiles_y, tiles_x, 1))

        # Crop the tiled texture to the original image size
        tiled_texture_cropped = tiled_texture[:img_h, :img_w, :]

        # Apply the tiled texture within the mask
        # Simple alpha blending of the texture onto the original image in the masked area
        # The strength of the texture application is controlled by color_strength
        alpha = color_strength # Use color_strength as the blending alpha
        result[mask_bool] = (1 - alpha) * result[mask_bool] + alpha * tiled_texture_cropped[mask_bool].astype(np.float32)

        # Note: Luminance transfer/texture modulation based on original hair
        # will likely need to happen in blend_with_original after apply_color
        # creates the initial textured layer.

    elif color_name in color_palette:
        # --- Apply Solid Color (Fallback or for non-textured colors) ---
        if color_name not in color_palette:
            print(f"Color {color_name} not found. Using default.")
            color_name = "Medium Brown" # Ensure this default exists or handle appropriately

        h, s, v = color_palette[color_name]

        # Convert image to HSV for solid color application
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        result_hsv = image_hsv.copy()

        # Apply color with strength control
        if color_strength < 1.0:
            # Blend hue, saturation, and value
            original_h = image_hsv[mask_bool, 0]
            target_h = np.ones_like(original_h) * h
            result_hsv[mask_bool, 0] = (1 - color_strength) * original_h + color_strength * target_h

            original_s = image_hsv[mask_bool, 1]
            target_s = np.ones_like(original_s) * s
            result_hsv[mask_bool, 1] = (1 - color_strength) * original_s + color_strength * target_s

            # For value, often a partial blend is better to preserve original lighting
            blend_factor_v = 0.7 * color_strength # Reduced influence for value
            result_hsv[mask_bool, 2] = (1 - blend_factor_v) * image_hsv[mask_bool, 2] + blend_factor_v * v
        else:
            # Full strength - direct replacement (with some value preservation)
            result_hsv[mask_bool, 0] = h
            result_hsv[mask_bool, 1] = s
            # Preserve some of the original value for lighting
            blend_factor_v = 0.7
            result_hsv[mask_bool, 2] = (1 - blend_factor_v) * image_hsv[mask_bool, 2]//1.3 + blend_factor_v * v # Adjusted value preservation

        # Convert back to BGR
        result = cv2.cvtColor(np.clip(result_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    else:
         print(f"Color or texture '{color_name}' not found in palette or swatches.")
         # If color/texture not found, return the original image portion in the mask
         # result[mask_bool] = image[mask_bool].astype(np.float32)
         pass # If not found, result remains the original image portion


    return result # Return the image with color/texture applied in the masked area




# ----------------------------
# Blend with original using smooth mask
# ----------------------------

import cv2
import numpy as np

def blend_with_original(original, colored_layer, smooth_mask, is_black=False):
    """Blend colored/textured layer with original using smooth mask, refining lightness blending for natural highlights."""
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    # Ensure colored_layer is in the correct data type before converting to LAB
    colored_lab = cv2.cvtColor(colored_layer.astype(np.uint8), cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l_orig, a_orig, b_orig = cv2.split(original_lab)
    l_col, a_col, b_col = cv2.split(colored_lab)

    # Ensure smooth_mask is float and scaled correctly
    smooth_mask_float = smooth_mask.astype(np.float32) / 255.0 if smooth_mask.max() > 1 else smooth_mask.astype(np.float32)
    mask_bool = smooth_mask > 0

    # --- Refined Lightness Blending ---
    # Blend the original lightness and the colored layer's lightness
    # Use the smooth mask to control the transition
    l_blend = (1 - smooth_mask_float) * l_orig.astype(np.float32) + smooth_mask_float * l_col.astype(np.float32)

    # Optional: Apply a non-linear adjustment to the blended lightness
    # This can help compress the highlight range and lift the shadow range
    # tune these parameters based on desired effect
    # adjusted_l_blend = 255 * (l_blend / 255.0)**0.9 # Example power transform to lift shadows
    # adjusted_l_blend = 255 * (l_blend / 255.0)**1.1 # Example power transform to compress highlights
    # l_blend = adjusted_l_blend

    # Another approach: Blend the *difference* from the mean lightness
    # This tries to preserve the relative contrast of the colored layer
    # if np.sum(mask_bool) > 0:
    #     mean_l_col = np.mean(l_col[mask_bool])
    #     l_col_centered = l_col.astype(np.float32) - mean_l_col
    #     l_orig_mean_masked = np.mean(l_orig[mask_bool])
    #     l_blend_centered = (1 - smooth_mask_float) * (l_orig.astype(np.float32) - l_orig_mean_masked) + smooth_mask_float * l_col_centered
    #     l_blend = l_orig_mean_masked + l_blend_centered # Add back original mean in masked area

    # A blend that favors the original lightness structure more in shadows
    # while still incorporating the colored lightness
    # You can experiment with weighted averages or other blending functions
    # Example: Use a blend factor that varies with the original lightness
    blend_factor_l = smooth_mask_float * (l_orig.astype(np.float32) / 255.0)**0.5 # More influence of colored in bright areas
    l_blend = (1 - blend_factor_l) * l_orig.astype(np.float32) + blend_factor_l * l_col.astype(np.float32)


    # Ensure lightness stays within valid range [0, 255]
    l_blend = np.clip(l_blend, 0, 255).astype(np.uint8)

    # --- Color (A and B) Channel Blending ---
    # Blend a and b channels using the smooth mask
    # This smoothly transitions the color
    a_blend = (1 - smooth_mask_float) * a_orig.astype(np.float32) + smooth_mask_float * a_col.astype(np.float32)
    b_blend = (1 - smooth_mask_float) * b_orig.astype(np.float32) + smooth_mask_float * b_col.astype(np.float32)

    a_blend = np.clip(a_blend, 0, 255).astype(np.uint8)
    b_blend = np.clip(b_blend, 0, 255).astype(np.uint8)

    # Merge the blended LAB channels
    result_lab = cv2.merge((l_blend, a_blend, b_blend))

    # Convert back to BGR
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # Final alpha blend with original using the smooth mask (primarily for edge refinement)
    smooth_mask_3ch = np.repeat(smooth_mask_float[:, :, np.newaxis], 3, axis=2)
    final = original.astype(np.float32) * (1 - smooth_mask_3ch) + result_bgr.astype(np.float32) * smooth_mask_3ch

    return np.clip(final, 0, 255).astype(np.uint8)
# def blend_with_original(original, colored, smooth_mask, is_black=False, max_lightness_for_black=80):
#     """
#     Blend the colored result with the original using a smooth mask.
#     Handles black hair by reducing lightness to avoid silvery highlights.
    
#     Parameters:
#         original (np.ndarray): Original BGR image.
#         colored (np.ndarray): Colored BGR image.
#         smooth_mask (np.ndarray): Mask with values in [0, 1].
#         is_black (bool): If True, force dark lightness to avoid silver.
#         max_lightness_for_black (int): Max L channel value for black hair.

#     Returns:
#         np.ndarray: Blended BGR image.
#     """
#     # Convert to LAB
#     original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
#     colored_lab = cv2.cvtColor(colored, cv2.COLOR_BGR2LAB)

#     # Split LAB channels
#     l_orig, a_orig, b_orig = cv2.split(original_lab)
#     l_col, a_col, b_col = cv2.split(colored_lab)

#     # Lightness handling
#     if is_black:
#         # Clamp lightness to avoid gray/silver highlights
#         l_blend = np.minimum(l_col, max_lightness_for_black)
#     else:
#         # Use original lightness for natural lighting
#         l_blend = l_orig

#     # Merge new LAB image
#     result_lab = cv2.merge((l_blend, a_col, b_col))
#     result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

#     # Alpha blend using smooth mask
#     smooth_mask_3ch = np.repeat(smooth_mask[:, :, np.newaxis], 3, axis=2)
#     blended = original.astype(np.float32) * (1 - smooth_mask_3ch) + \
#               result_bgr.astype(np.float32) * smooth_mask_3ch

#     return np.clip(blended, 0, 255).astype(np.uint8)

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
                edge_smoothness=51, detail_factor=0.3,
                swatch_path=None):
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
    # --- Define Texture Swatches ---
    # Map color names to your texture image file paths
    texture_swatches = {
        "Angel Blonde 2": "assets/hair_textures/angel_blonde_2_texture.png", # Replace with your actual path
        # Add other textured colors here
        "Angel Blonde 4": "assets/hair_textures/angel_blonde_4_texture.png", # Example for the L'Oreal color
    }

    
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
    # --- Add erosion step for hair mask to prevent bleeding ---
    # Use a small kernel for slight erosion
    erode_kernel = np.ones((3, 3), np.uint8)
    hair_refined_eroded = cv2.erode(hair_refined, erode_kernel, iterations=1)
    # Use the eroded mask for creating the smooth mask and for applying color
    hair_mask_for_coloring = hair_refined_eroded
    # ----------------------------------------------------------


    # Create smooth masks
    # Use the slightly eroded mask to create the smooth mask
    hair_smooth = create_smooth_mask(hair_mask_for_coloring, blur_size=edge_smoothness)
    lips_smooth = create_smooth_mask(lips_refined, blur_size=edge_smoothness-10)  # Slightly less blur for lips
    skin_smooth = create_smooth_mask(skin_refined, blur_size=edge_smoothness)

    # Apply colors
    result = image.copy()

    # Apply hair color - IMPORTANT: Use the potentially eroded mask for coloring
    if hair_color:
        # Use hair_mask_for_coloring here
        # Pass the texture_swatches dictionary to apply_color
        hair_colored = apply_color(result, hair_mask_for_coloring, hair_color, color_palette, hair_strength, texture_swatches=texture_swatches)
        # Blend using the smooth mask derived from the eroded mask
        result = blend_with_original(result, hair_colored, hair_smooth, is_black=(hair_color == "Black"))
        # Apply detail preservation using the mask used for coloring
        result = apply_detail_preservation(image, result, hair_mask_for_coloring, detail_factor)

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
    image_path = "C:/Users/user/Downloads/WhatsApp Image 2025-05-14 at 5.03.10 PM.jpeg"  # Replace with your test image path
   
    # Apply makeup with default parameters
    result = apply_makeup(
        image_path,
        hair_color="Auburn",
        hair_strength=1.0,
        lip_color="Ruby Red",
        lip_strength=1.0,
        skin_color="Dark Brown",  # Set to None to skip skin coloring
        skin_strength=0.4,
        edge_smoothness=81,  # Higher value = smoother edges
        detail_factor=0.3    # Higher value = more texture
    )

# Example with different parameters
    result_smooth = apply_makeup(
        image_path,
        swatch_path,
        hair_color="Black",
        hair_strength=1,
        lip_color="Coral",
        lip_strength=1,
        skin_color="Light Brown",
        skin_strength=0.2,
        edge_smoothness=91,  # Very smooth edges
        detail_factor=0.2 )   # Less texture
