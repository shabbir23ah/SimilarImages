import os
from PIL import Image

image_folder = "static/images"  # Change this to your folder path

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.svg')):  # Check image files only
        image_path = os.path.join(image_folder, filename)
        try:
            img = Image.open(image_path)
            if img.mode == "P":
                print(f"Palette mode image found: {filename}")
        except Exception as e:
            print(f"Error with {filename}: {e}")
