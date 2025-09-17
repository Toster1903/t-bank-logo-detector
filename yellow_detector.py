import cv2
import numpy as np
import os
import shutil
from pathlib import Path

def detect_yellow_percentage(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return 0.0
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Диапазон желтого цвета в HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    
    yellow_percentage = (yellow_pixels / total_pixels) * 100
    return yellow_percentage

def copy_yellow_images(source_folder, destination_folder, threshold=1.0):
    os.makedirs(destination_folder, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    count_copied = 0
    
    for filename in os.listdir(source_folder):
        if Path(filename).suffix.lower() in image_extensions:
            filepath = os.path.join(source_folder, filename)
            yellow_percent = detect_yellow_percentage(filepath)
            if yellow_percent >= threshold:
                shutil.copy2(filepath, os.path.join(destination_folder, filename))
                count_copied += 1
                print(f"Copied {filename} ({yellow_percent:.2f}% желтого)")
    
    print(f"\nОбщее количество скопированных изображений с желтым цветом: {count_copied}")

if __name__ == "__main__":
    source_dir = "/home/dmitriy/data_sirius/"
    dest_dir = "/home/dmitriy/vs_code/T-bank/images"
    yellow_threshold_percent = 5.0  
    
    copy_yellow_images(source_dir, dest_dir, yellow_threshold_percent)
