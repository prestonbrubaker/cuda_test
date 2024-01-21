import os
import random
from PIL import Image

def transform_image(image_path, save_path):
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
    # Random rotation between -90 and 90 degrees
    rotated = image.rotate(random.uniform(-90, 90), expand=True, fillcolor=(0,))  # Single-element tuple for grayscale
    # Random zoom between 80% and 120%
    zoom_factor = random.uniform(0.8, 1.2)
    width, height = rotated.size
    new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
    zoomed = rotated.resize((new_width, new_height), Image.ANTIALIAS)
    # Crop or pad to get 256x256
    if new_width > 256 or new_height > 256:
        # Crop the center
        left = (new_width - 256) / 2
        top = (new_height - 256) / 2
        right = (new_width + 256) / 2
        bottom = (new_height + 256) / 2
        result = zoomed.crop((left, top, right, bottom))
    else:
        # Pad with black color
        result = Image.new("RGB", (256, 256), (0, 0, 0))
        result.paste(zoomed, (int((256 - new_width) / 2), int((256 - new_height) / 2)))
    result.save(save_path)

def process_images(input_folder, output_folder, synthetic_multiplier):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):  # Check for PNG files
            file_path = os.path.join(input_folder, filename)
            # Save original image
            original = Image.open(file_path)
            original = original.resize((256, 256), Image.ANTIALIAS)
            original.save(os.path.join(output_folder, filename))
            # Create augmented images
            for i in range(synthetic_multiplier):
                save_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_synthetic{i}.png')  # Save as PNG
                transform_image(file_path, save_path)

process_images('photos', 'semi_synthetic_photos', synthetic_multiplier=500)
