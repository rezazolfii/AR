from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import parse_obj_as
import json
import io
import cv2
import numpy as np
from PIL import Image
import base64

from ar_tryon.models.bisenet_official_wrapper import BiSeNetOfficialWrapper
from ar_tryon.utils.image_processing import (
    create_color_palette,
    create_mask_from_indices,
    create_smooth_mask,
    apply_color,
    blend_with_original,
    apply_detail_preservation,
    refine_mask,
    remove_small_regions
)
from ar_tryon.config import BIESNET_MODEL_PATH
from .models import MakeupRequest, MakeupResponse, MakeupFeatures, MakeupColors
from .utils import decode_image, encode_image, pil_to_cv2, cv2_to_pil

router = APIRouter()

# Initialize BiSeNet model
model = None


def get_model():
    global model
    if model is None:
        model = BiSeNetOfficialWrapper(n_classes=19)
        model.load_weights(BIESNET_MODEL_PATH)
    return model


@router.post("/apply-makeup", response_model=MakeupResponse)
async def apply_makeup(
    file: UploadFile = File(...),
    makeup_data: str = Form(...),
):
    """
    Apply makeup to an uploaded image
    
    - **file**: Image file to process
    - **makeup_data**: JSON string containing makeup parameters
    """
    try:
        # Parse makeup data
        makeup_request = parse_obj_as(MakeupRequest, json.loads(makeup_data))
        
        # Read image file
        image_bytes = await file.read()
        image = decode_image(image_bytes)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get model
        model = get_model()
        
        # Process image
        segmentation_mask = model.process_image(image)
        
        # Apply makeup
        processed_image = apply_makeup_with_features(
            image,
            segmentation_mask,
            makeup_request.selected_features.dict(),
            makeup_request.selected_colors.dict(),
            edge_smoothness=makeup_request.edge_smoothness,
            color_strength=makeup_request.color_strength,
            detail_factor=makeup_request.detail_factor
        )
        
        # Encode processed image
        encoded_image = encode_image(processed_image)
        
        return MakeupResponse(
            success=True,
            message="Makeup applied successfully",
            image_url=f"data:image/png;base64,{encoded_image}"
        )
    
    except Exception as e:
        return MakeupResponse(
            success=False,
            message=f"Error applying makeup: {str(e)}"
        )


def apply_makeup_with_features(image, segmentation_mask, selected_features, selected_colors, 
                       edge_smoothness=51, color_strength=1.0, detail_factor=0.3):
    """Apply makeup to selected features using segmentation mask"""
    # Create color palette
    color_palette = create_color_palette()

    # Define class indices for different features
    hair_indices = [17]  # Hair
    lips_indices = [12, 13]  # Upper and lower lips
    skin_indices = [1, 10]  # Skin, nose, neck

    # Create masks for selected features
    result = image.copy()

    # Process each selected feature
    if selected_features['hair']:
        hair_mask = create_mask_from_indices(segmentation_mask, hair_indices)
        # Refine hair mask
        hair_refined = refine_mask(hair_mask, iterations=2, operation="close")
        # Remove small isolated regions
        hair_refined = remove_small_regions(hair_refined, min_size=100)
        hair_smooth = create_smooth_mask(hair_refined, blur_size=edge_smoothness)
        hair_color = selected_colors['hair']
        
        # Use refined mask for coloring
        hair_colored = apply_color(result, hair_refined, hair_color, color_palette, color_strength)
        result = blend_with_original(result, hair_colored, hair_smooth)
        result = apply_detail_preservation(image, result, hair_refined, detail_factor)

    if selected_features['lips']:
        lips_mask = create_mask_from_indices(segmentation_mask, lips_indices)
        # Refine lips mask to remove artifacts
        lips_refined = refine_mask(lips_mask, iterations=2, operation="both")
        # Remove small isolated regions
        lips_refined = remove_small_regions(lips_refined, min_size=50)
        
        # Create a more conservative mask for lips to avoid artifacts
        kernel = np.ones((3, 3), np.uint8)
        lips_refined = cv2.erode(lips_refined, kernel, iterations=1)
        
        lips_smooth = create_smooth_mask(lips_refined, blur_size=edge_smoothness-10)
        lip_color = selected_colors['lips']
        
        # Use refined mask for coloring
        lips_colored = apply_color(result, lips_refined, lip_color, color_palette, color_strength)
        result = blend_with_original(result, lips_colored, lips_smooth)
        result = apply_detail_preservation(image, result, lips_refined, detail_factor)

    if selected_features['skin']:
        skin_mask = create_mask_from_indices(segmentation_mask, skin_indices)
        # Refine skin mask
        skin_refined = refine_mask(skin_mask, iterations=1, operation="both")
        skin_smooth = create_smooth_mask(skin_refined, blur_size=edge_smoothness)
        skin_color = selected_colors['skin']
        
        # Use refined mask for coloring
        skin_colored = apply_color(result, skin_refined, skin_color, color_palette, color_strength * 0.5)
        result = blend_with_original(result, skin_colored, skin_smooth)
        result = apply_detail_preservation(image, result, skin_refined, detail_factor * 0.5)

    return result


@router.post("/apply-makeup-base64", response_model=MakeupResponse)
async def apply_makeup_base64(request_data: dict):
    """
    Apply makeup to a base64 encoded image
    
    - **image_data**: Base64 encoded image
    - **makeup_data**: Makeup parameters
    """
    try:
        # Extract data
        image_data = request_data.get("image_data")
        makeup_data = request_data.get("makeup_data")
        
        if not image_data:
            raise HTTPException(status_code=400, detail="Missing image data")
        
        if not makeup_data:
            raise HTTPException(status_code=400, detail="Missing makeup data")
        
        # Parse makeup data
        makeup_request = parse_obj_as(MakeupRequest, makeup_data)
        
        # Decode image
        image = decode_image(image_data)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Get model
        model = get_model()
        
        # Process image
        segmentation_mask = model.process_image(image)
        
        # Apply makeup
        processed_image = apply_makeup_with_features(
            image,
            segmentation_mask,
            makeup_request.selected_features.dict(),
            makeup_request.selected_colors.dict(),
            edge_smoothness=makeup_request.edge_smoothness,
            color_strength=makeup_request.color_strength,
            detail_factor=makeup_request.detail_factor
        )
        
        # Encode processed image
        encoded_image = encode_image(processed_image)
        
        return MakeupResponse(
            success=True,
            message="Makeup applied successfully",
            image_url=f"data:image/png;base64,{encoded_image}"
        )
    
    except Exception as e:
        return MakeupResponse(
            success=False,
            message=f"Error applying makeup: {str(e)}"
        )