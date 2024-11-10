import os
import shutil

src_dir = ""
dst_dir = ""

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

files = os.listdir(src_dir)
image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))][:100]

for file_name in image_files:
    src_file = os.path.join(src_dir, file_name)
    dst_file = os.path.join(dst_dir, file_name)
    shutil.copy(src_file, dst_file)

print("Copied 100 images from", src_dir, "to", dst_dir)