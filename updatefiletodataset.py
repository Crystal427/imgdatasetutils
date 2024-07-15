import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.fftpack import dct
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # This removes the limit on the image size
Image.LOAD_TRUNCATED_IMAGES = True  # This allows to load truncated images
class PHash:
    def __init__(self):
        self.target_size = (32, 32)
        self.__coefficient_extract = (8, 8)

    def _array_to_hash(self, hash_mat):
        return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

    def encode_image(self, image_file):
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"图片文件 {image_file} 不存在")
        
        img = Image.open(image_file).convert("L").resize(self.target_size)
        img_array = np.asarray(img)
        return self._hash_algo(img_array)

    def _hash_algo(self, image_array):
        dct_coef = dct(dct(image_array, axis=0), axis=1)
        
        dct_reduced_coef = dct_coef[: self.__coefficient_extract[0], : self.__coefficient_extract[1]]

        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        hash_mat = dct_reduced_coef >= median_coef_val
        return self._array_to_hash(hash_mat)

    @staticmethod
    def hamming_distance(hash1, hash2):
        hash1_bin = bin(int(hash1, 16))[2:].zfill(64) 
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return sum(i != j for i, j in zip(hash1_bin, hash2_bin))

def deduplicate_against_folder(temp_folder, existing_folder, similarity_threshold):
    phash = PHash()
    temp_hashes = {img: phash.encode_image(os.path.join(temp_folder, img)) for img in os.listdir(temp_folder) if img.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))}
    existing_hashes = {img: phash.encode_image(os.path.join(existing_folder, img)) for img in os.listdir(existing_folder) if img.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))}

    to_remove = []
    for temp_img, temp_hash in temp_hashes.items():
        for existing_img, existing_hash in existing_hashes.items():
            if phash.hamming_distance(temp_hash, existing_hash) <= similarity_threshold:
                to_remove.append(temp_img)
                break

    for temp_img in to_remove:
        temp_path = os.path.join(temp_folder, temp_img)
        json_path = temp_path + ".json"
        try:
            os.remove(temp_path)
            print(f"Removed duplicate image from temp folder: {temp_img}")
            if os.path.exists(json_path):
                os.remove(json_path)
                print(f"Removed corresponding JSON file: {os.path.basename(json_path)}")
        except OSError as e:
            print(f"Error removing file {temp_img}: {e}")

def get_year_from_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").year
    except:
        return None

def get_folder_by_year(year):
    if year is None:
        return "undefined"
    elif year <= 2010:
        return "2010s"
    elif 2011 <= year <= 2017:
        return "2017s"
    elif 2018 <= year <= 2020:
        return "2020s"
    elif 2021 <= year <= 2022:
        return "2022s"
    else:
        return "new"

def process_image(img_path, artist_folder, json_folder):
    json_path = img_path + ".json"
    year = None

    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                year = get_year_from_date(data.get('date'))
        except:
            pass

    if year is None:
        try:
            mtime = os.path.getmtime(img_path)
            year = datetime.fromtimestamp(mtime).year
        except:
            pass

    target_folder = os.path.join(artist_folder, get_folder_by_year(year))
    os.makedirs(target_folder, exist_ok=True)

    new_img_path = os.path.join(target_folder, os.path.basename(img_path))
    shutil.move(img_path, new_img_path)

    if os.path.exists(json_path):
        new_json_path = os.path.join(json_folder, os.path.basename(json_path))
        shutil.move(json_path, new_json_path)

    if year is None:
        with open(os.path.join(artist_folder, "need_check.delme"), 'a') as f:
            f.write(f"{os.path.basename(img_path)}\n")

def process_artist_folder(artist_path, dataset_dir):
    artist_name = os.path.basename(artist_path)
    dataset_artist_path = os.path.join(dataset_dir, artist_name)
    
    if os.path.exists(dataset_artist_path):
        deduplicate_against_folder(artist_path, dataset_artist_path, similarity_threshold=10)
    else:
        os.makedirs(dataset_artist_path)

    for subfolder in ['2010s', '2017s', '2020s', '2022s', 'new', 'undefined', 'jsons']:
        os.makedirs(os.path.join(dataset_artist_path, subfolder), exist_ok=True)

    json_folder = os.path.join(dataset_artist_path, 'jsons')

    # 复制 crawler.txt 文件
    crawler_txt_path = os.path.join(artist_path, 'crawler.txt')
    if os.path.exists(crawler_txt_path):
        shutil.copy2(crawler_txt_path, dataset_artist_path)

    for img in os.listdir(artist_path):
        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(artist_path, img)
            process_image(img_path, dataset_artist_path, json_folder)

    if not os.listdir(artist_path):
        os.rmdir(artist_path)

def update_dataset(update_dir, dataset_dir):
    for root, dirs, files in os.walk(update_dir):
        for dir_name in dirs:
            artist_path = os.path.join(root, dir_name)
            if any(file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) for file in os.listdir(artist_path)):
                process_artist_folder(artist_path, dataset_dir)

if __name__ == "__main__":
    update_dir = r'F:\Crystal\Downloads\OneDrive_1_7-10-2024\todo_ex'
    dataset_dir = r'E:\Datasets\SDXL_large_Modified'
    update_dataset(update_dir, dataset_dir)
    print("数据集更新完成！")