import os
import shutil
from PIL import Image

def process_directory(src_dir, dst_dir, min_pixels, min_files):
    marked_dirs = []

    for root, dirs, files in os.walk(src_dir):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        if len(image_files) >= min_files:
            large_images = []
            for image_file in image_files:
                image_path = os.path.join(root, image_file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width * height > min_pixels:
                        large_images.append(image_file)

            if len(large_images) >= min_files:
                marked_dirs.append((root, large_images))

    for marked_dir, large_images in marked_dirs:
        rel_path = os.path.relpath(marked_dir, src_dir)
        new_dir = os.path.join(dst_dir, rel_path)
        os.makedirs(new_dir, exist_ok=True)

        for image_file in large_images:
            src_image_path = os.path.join(marked_dir, image_file)
            dst_image_path = os.path.join(new_dir, image_file)
            shutil.copy2(src_image_path, dst_image_path)

            txt_file = os.path.splitext(image_file)[0] + '.txt'
            src_txt_path = os.path.join(marked_dir, txt_file)
            if os.path.exists(src_txt_path):
                dst_txt_path = os.path.join(new_dir, txt_file)
                shutil.copy2(src_txt_path, dst_txt_path)

# 使用示例
source_directory = r'F:\SDXL_large'
destination_directory = r'F:\SDXL_large_highres'
minimum_pixels = 1535 * 1536
minimum_files = 15
process_directory(source_directory, destination_directory, minimum_pixels, minimum_files)