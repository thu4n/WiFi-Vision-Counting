import cv2
import os
import numpy as np

def get_brightness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)
    return brightness

# origin = "./051024-session1-5person/frame_1728096039.8494563.jpg"
#dark = "./dark-051024-session1-5person/frame_1728096039.8494563.jpg"
folder = "./dark-051024-session1-5person"
images = os.listdir(folder)

for image_path in images:
    image = cv2.imread(os.path.join(folder, image_path))

    if image is not None:
        brightness = get_brightness(image)
        if brightness < 50:
            print(f"detect dark image: {image_path}")
        else:
            print(f"The brightness of the image {image_path} is: {brightness}")
    else:
        print("Image not found.")

# for image_path in [origin, dark]:
#     image = cv2.imread(image_path)

#     if image is not None:
#         brightness = get_brightness(image)
#         print(f"The brightness of the image {image_path} is: {brightness}")
#     else:
#         print("Image not found.")