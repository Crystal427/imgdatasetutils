import os
import shutil
import concurrent.futures
from PIL import Image

def convert_image(src_file, dst_file):
    try:
        with Image.open(src_file) as img:
            img = img.convert('RGB')
            img.thumbnail((1024, 1024))
            img.save(dst_file, 'webp', quality=90)
    except Exception as e:
        print(f"Error converting {src_file}: {str(e)}")

def process_file(src_file, src_dir, dst_dir):
    rel_path = os.path.relpath(src_file, src_dir)
    dst_file = os.path.join(dst_dir, rel_path)
    
    if src_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        dst_file = os.path.splitext(dst_file)[0] + '.webp'
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        convert_image(src_file, dst_file)
    elif src_file.lower().endswith('.txt') or os.path.basename(src_file) == 'results.json':
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)

def convert_images(src_dir, dst_dir, max_workers=32):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                src_file = os.path.join(root, file)
                futures.append(executor.submit(process_file, src_file, src_dir, dst_dir))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {str(e)}")

# 使用示例
src_directory = r'E:\Datasets\dataset\SDXL_large'
dst_directory = r'F:\Datasets\SDXL2'
convert_images(src_directory, dst_directory)