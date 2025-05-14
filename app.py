"""
Main Streamlit application for AR Try-on
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import torch
import torchvision.transforms as transforms

from ar_tryon.config import (
    STREAMLIT_TITLE,
    STREAMLIT_DESCRIPTION,
    IMAGE_SIZE,
    BIESNET_MODEL_PATH
)
from ar_tryon.models.bisenet_official_wrapper import BiSeNetOfficialWrapper
from ar_tryon.utils.image_processing import (
    create_color_palette,
    create_mask_from_indices,
    create_smooth_mask,
    apply_color,
    blend_with_original,
    apply_detail_preservation,
    apply_makeup,
    refine_mask, 
    remove_small_regions
)

# Set page config
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    page_icon="💇",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #FF4B4B;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #4B4BFF;
    margin-bottom: 1rem;
}
.feature-button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 20px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    margin: 0 10px;
}
.feature-button.selected {
    background-color: #4B4BFF;
}
.color-circle {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    display: inline-block;
    margin: 0 5px;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = {
        'hair': True,
        'lips': True,
        'skin': False
    }
if 'selected_colors' not in st.session_state:
    st.session_state.selected_colors = {
        'hair': "Auburn",
        'lips': "Ruby Red",
        'skin': "Warm"
    }

# Function to load BiSeNet model
@st.cache_resource
def load_model():
    """Load the BiSeNet model"""
    try:
        model = BiSeNetOfficialWrapper(n_classes=19)
        if os.path.exists(BIESNET_MODEL_PATH):
            if model.load_weights(BIESNET_MODEL_PATH):
                st.success("Successfully loaded pre-trained weights")
            else:
                st.warning("Using randomly initialized weights")
        else:
            st.warning(f"Pre-trained weights not found at {BIESNET_MODEL_PATH}")
            st.warning("Using randomly initialized weights")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to apply makeup with selected features
def apply_makeup_with_features(image, segmentation_mask, selected_features, selected_colors, 
                           edge_smoothness=51, color_strength=1.0, detail_factor=0.3):
    """Apply makeup to selected features using segmentation mask"""
    # Create color palette
    color_palette = create_color_palette()

    # Define class indices for different features
    hair_indices = [17, 2, 3]  # Hair
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
        
        # CORRECTED: Use refined mask for coloring
        hair_colored = apply_color(result, hair_refined, hair_color, color_palette, color_strength)
        result = blend_with_original(result, hair_colored, hair_smooth, is_black=(hair_color == "Black"))
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
        
        # CORRECTED: Use refined mask for coloring
        lips_colored = apply_color(result, lips_refined, lip_color, color_palette, color_strength//color_strength *100)
        result = blend_with_original(result, lips_colored, lips_smooth, is_black= False)
        result = apply_detail_preservation(image, result, lips_refined, detail_factor)

    if selected_features['skin']:
        skin_mask = create_mask_from_indices(segmentation_mask, skin_indices)
        # Refine skin mask
        skin_refined = refine_mask(skin_mask, iterations=1, operation="both")
        skin_smooth = create_smooth_mask(skin_refined, blur_size=edge_smoothness)
        skin_color = selected_colors['skin']
        
        # CORRECTED: Use refined mask for coloring
        skin_colored = apply_color(result, skin_refined, skin_color, color_palette, color_strength * 0.5)
        result = blend_with_original(result, skin_colored, skin_smooth, is_black= False)
        result = apply_detail_preservation(image, result, skin_refined, detail_factor * 0.5)

    return result

# Main content
st.title(STREAMLIT_TITLE)
st.write(STREAMLIT_DESCRIPTION)

# Create layout with two columns
col1, col2 = st.columns([1, 3])

# Sidebar (left column)
with col1:
    st.markdown("<h2 class='sub-header'>Choose Try-on Experience</h2>", unsafe_allow_html=True)
    
    # Option buttons
    option = st.radio(
        "Select input method:",
        ["Upload Photo", "Use Sample"],
        index=0
    )
    
    # File uploader
    uploaded_file = None
    if option == "Upload Photo":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="main_uploader")
    else:
        # Sample images
        st.markdown("<h3>Select a sample image:</h3>", unsafe_allow_html=True)
        sample_cols = st.columns(2)
        with sample_cols[0]:
            if st.button("Sample 1"):
                uploaded_file = "sample1.jpg"
        with sample_cols[1]:
            if st.button("Sample 2"):
                uploaded_file = "sample2.jpg"
    
    # Color palette section
    st.markdown("<h2 class='sub-header'>Color Palette</h2>", unsafe_allow_html=True)
    
    # Get color palette
    color_palette = create_color_palette()
    
    # Group colors by category
    color_categories = {
        "Hair Colors": ["Black", "Copper Penny", "Cinnamon", "Chocolate", 
                        "Heavy Brown", "Medium Brown", "Light Brown"
                       "Angel Blonde 6", "Angel Blonde 5", "Angel Blonde 4", 
                       "Angel Blonde 2", "Strawberry Blonde", "Auburn", 
                       "Ginger", "Red", "Green", "Blue", "Purple"],
        "Lip Colors": ["Natural Pink", "Coral", "Ruby Red", "Burgundy"],
        "Skin Tones": ["Fair", "Warm", "Medium", "Tan", "Deep", "Deep2"]
    }
    
    # Create tabs for color categories
    tabs = st.tabs(list(color_categories.keys()))
    
    # Display colors in each tab
    for i, (category, colors) in enumerate(color_categories.items()):
        with tabs[i]:
            # Determine which feature this category applies to
            if category == "Hair Colors":
                feature = "hair"
            elif category == "Lip Colors":
                feature = "lips"
            else:
                feature = "skin"
            
            # Create a grid of colors (3 columns)
            for j in range(0, len(colors), 3):
                cols = st.columns(3)
                for k, color_name in enumerate(colors[j:j+3]):
                    if j+k < len(colors):
                        with cols[k]:
                            if st.button(
                                color_name,
                                key=f"{feature}_{color_name}",
                                help=f"Apply {color_name} color to {feature}"
                            ):
                                st.session_state.selected_colors[feature] = color_name
                            
                            # Show selection indicator
                            if st.session_state.selected_colors[feature] == color_name:
                                st.markdown(f"✓ Selected")
    
    # Sliders for parameters
    st.markdown("<h2 class='sub-header'>Adjustment Controls</h2>", unsafe_allow_html=True)
    
    edge_smoothness = st.slider(
        "Edge Smoothness",
        min_value=11,
        max_value=101,
        value=51,
        step=10,
        help="Higher values create smoother edges"
    )
    
    color_strength = st.slider(
        "Color Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Control the intensity of the color"
    )
    
    detail_factor = st.slider(
        "Detail Preservation",
        min_value=0.1,
        max_value=0.5,
        value=0.3,
        step=0.05,
        help="Control how much original detail is preserved"
    )

# Main content area (right column)
with col2:
    # Feature selection buttons
    st.markdown("<h2 class='sub-header'>Select Features to Modify</h2>", unsafe_allow_html=True)
    
    # Create three buttons for hair, skin, and lips
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        hair_selected = st.session_state.selected_features['hair']
        if st.button(
            "Hair", 
            key="hair_button",
            help="Toggle hair coloring",
            type="primary" if hair_selected else "secondary"
        ):
            st.session_state.selected_features['hair'] = not hair_selected
    
    with feature_cols[1]:
        skin_selected = st.session_state.selected_features['skin']
        if st.button(
            "Skin", 
            key="skin_button",
            help="Toggle skin coloring",
            type="primary" if skin_selected else "secondary"
        ):
            st.session_state.selected_features['skin'] = not skin_selected
    
    with feature_cols[2]:
        lips_selected = st.session_state.selected_features['lips']
        if st.button(
            "Lips", 
            key="lips_button",
            help="Toggle lip coloring",
            type="primary" if lips_selected else "secondary"
        ):
            st.session_state.selected_features['lips'] = not lips_selected
    
    # Display selected features
    st.markdown(
        f"**Selected Features:** " + 
        f"{'Hair ' if st.session_state.selected_features['hair'] else ''}" +
        f"{'Skin ' if st.session_state.selected_features['skin'] else ''}" +
        f"{'Lips' if st.session_state.selected_features['lips'] else ''}"
    )
    
    # Display selected colors
    st.markdown(
        f"**Selected Colors:** " + 
        f"Hair: {st.session_state.selected_colors['hair']} | " +
        f"Lips: {st.session_state.selected_colors['lips']} | " +
        f"Skin: {st.session_state.selected_colors['skin']}"
    )
    
    # Process and display image
    if uploaded_file is not None:
        # Load image
        if isinstance(uploaded_file, str):
            # Load sample image
            if not os.path.exists(uploaded_file):
                st.error(f"Sample image {uploaded_file} not found. Please upload your own image.")
                st.stop()
            image = cv2.imread(uploaded_file)
        else:
            # Load uploaded image
            image_bytes = uploaded_file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Load model if not already loaded
        if st.session_state.model is None:
            with st.spinner("Loading BiSeNet model..."):
                try:
                    st.session_state.model = load_model()
                except Exception as e:
                    st.error(f"Error loading model: {e}")
                    st.error("Please make sure the model path is correct in the code.")
                    st.stop()
        
        # Process image with BiSeNet
        with st.spinner("Processing image..."):
            try:
                segmentation_mask = st.session_state.model.process_image(image)
                
                # Apply makeup with selected features
                processed_image = apply_makeup_with_features(
                    image, 
                    segmentation_mask,
                    st.session_state.selected_features,
                    st.session_state.selected_colors,
                    edge_smoothness=edge_smoothness,
                    color_strength=color_strength,
                    detail_factor=detail_factor
                )
                
                # Display images side by side
                col_orig, col_proc = st.columns(2)
                
                with col_orig:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
                
                with col_proc:
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
                
                # Download button
                result_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                result_pil = Image.fromarray(result_rgb)
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Result",
                    data=byte_im,
                    file_name="processed_image.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.error("Please check your image processing functions and make sure they are working correctly.")
    else:
        # Display instructions when no image is uploaded
        st.info("Please upload an image or select a sample image to see the results.")
        
        # Placeholder image
        st.image("https://via.placeholder.com/800x400.png?text=Upload+an+image+to+see+results", use_container_width=True)