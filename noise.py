import os
import cv2
import numpy as np
from pathlib import Path

def add_noise(image, noise_type="gaussian"):
    if noise_type == "gaussian":
        mean = 0
        var = 10
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy = cv2.add(image.astype(np.float32), gauss)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    return image

def invert_colors(image):
    return cv2.bitwise_not(image)

def increase_contrast(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    for filename in os.listdir(input_dir):
        if Path(filename).suffix.lower() in image_extensions:
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Не удалось загрузить {filename}")
                continue

            inverted = invert_colors(image)
            noisy = add_noise(image, noise_type="gaussian")
            contrast = increase_contrast(image)


            base_name = Path(filename).stem
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_inverted.jpg"), inverted)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_noisy.jpg"), noisy)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_contrast.jpg"), contrast)

            print(f"Обработано {filename}")

if __name__ == "__main__":
    input_folder = "/home/dmitriy/PycharmProjects/T-bank/Yellow_for_labeling"
    output_folder = "/home/dmitriy/PycharmProjects/T-bank/yellow_noise"
    process_images(input_folder, output_folder)
