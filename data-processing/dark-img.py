import os
import cv2
import numpy as np

def adjust_gamma(image, gamma=0.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def darken_images(input_folder, output_folder, gamma=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Could not open or find the image {img_path}. Skipping.")
                continue

            darkened_image = adjust_gamma(image, gamma=gamma)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, darkened_image)
            print(f"Processed and saved: {output_path}")

input_folder = "./051024-session1-0person"
output_folder = "./dark-051024-session1-0person"
gamma_value = 0.2  
darken_images(input_folder, output_folder, gamma=gamma_value)
