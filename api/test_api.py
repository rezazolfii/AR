import requests
import json
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import os

def test_file_upload(image_path):
    """Test the file upload endpoint"""
    url = "http://127.0.0.1:8000/api/apply-makeup"
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return False
    
    # Prepare makeup data
    makeup_data = {
        "selected_features": {"hair": True, "lips": True, "skin": False},
        "selected_colors": {"hair": "Auburn", "lips": "Ruby Red", "skin": "Warm"},
        "edge_smoothness": 71,
        "color_strength": 0.8,
        "detail_factor": 0.3
    }
    
    # Open the image file
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"makeup_data": json.dumps(makeup_data)}
            
            # Send the request
            print(f"Sending request to {url}...")
            response = requests.post(url, files=files, data=data)
    except Exception as e:
        print(f"Error sending request: {e}")
        return False
    
    # Check the response
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print("Success! Received processed image.")
            
            # Extract and save the image
            image_data = result["image_url"].split(",")[1]
            image_bytes = base64.b64decode(image_data)
            
            # Save the image
            output_path = "processed_image.png"
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            print(f"Saved processed image to {output_path}")
            
            # Display the images
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                original = Image.open(image_path)
                ax1.imshow(original)
                ax1.set_title("Original Image")
                ax1.axis("off")
                
                # Processed image
                processed = Image.open(io.BytesIO(image_bytes))
                ax2.imshow(processed)
                ax2.set_title("Processed Image")
                ax2.axis("off")
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error displaying images: {e}")
            
            return True
        else:
            print(f"API Error: {result['message']}")
    else:
        print(f"HTTP Error: {response.status_code}")
        print(response.text)
    
    return False

if __name__ == "__main__":
    # Replace with the path to your test image
    image_path = input("00072.png")
    
    print("Testing file upload endpoint...")
    test_file_upload(image_path)