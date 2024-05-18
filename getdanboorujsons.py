import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.fftpack import dct
import subprocess
import json

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

    def find_duplicates(self, image_dir1, image_dir2, max_distance_threshold=1):
        if not os.path.isdir(image_dir1):
            raise NotADirectoryError(f"{image_dir1} 不是一个有效的目录")
        if not os.path.isdir(image_dir2):
            raise NotADirectoryError(f"{image_dir2} 不是一个有效的目录")

        encoding_map1 = {}
        for img_path in Path(image_dir1).glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                try:
                    hash_str = self.encode_image(str(img_path))
                    encoding_map1[str(img_path)] = hash_str
                except Exception as e:
                    print(f"处理图像 {img_path} 出错: {str(e)}")

        encoding_map2 = {}
        for img_path in Path(image_dir2).glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                try:
                    hash_str = self.encode_image(str(img_path))
                    encoding_map2[str(img_path)] = hash_str
                except Exception as e:
                    print(f"处理图像 {img_path} 出错: {str(e)}")

        duplicates = {}
        for img1, hash1 in encoding_map1.items():
            for img2, hash2 in encoding_map2.items():
                if self.hamming_distance(hash1, hash2) <= max_distance_threshold:
                    duplicates[img1] = img2
        return duplicates
    
    def remove_duplicates(self, image_dir, max_distance_threshold=0.4):
        if not os.path.isdir(image_dir):
            raise NotADirectoryError(f"{image_dir} 不是一个有效的目录")

        encoding_map = {}
        for img_path in Path(image_dir).glob("*"):
            if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
                try:
                    hash_str = self.encode_image(str(img_path))
                    encoding_map[str(img_path)] = hash_str
                except Exception as e:
                    print(f"处理图像 {img_path} 出错: {str(e)}")

        duplicates = {}
        for img1, hash1 in encoding_map.items():
            for img2, hash2 in encoding_map.items():
                if img1 != img2 and self.hamming_distance(hash1, hash2) <= max_distance_threshold:
                    duplicates.setdefault(img1, []).append(img2)
        
        for original, duplicate_list in duplicates.items():
            for duplicate in duplicate_list:
                try:
                    os.remove(duplicate)
                    json_file = os.path.splitext(duplicate)[0] + '.json'
                    if os.path.exists(json_file):
                        os.remove(json_file)
                    print(f"删除重复图片: {os.path.basename(duplicate)} (保留: {os.path.basename(original)})")
                except FileNotFoundError:
                    print(f"文件 {duplicate} 不存在,跳过...")
        
    @staticmethod
    def hamming_distance(hash1, hash2):
        hash1_bin = bin(int(hash1, 16))[2:].zfill(64) 
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return sum(i != j for i, j in zip(hash1_bin, hash2_bin))


def process_subfolder(target_folder, subfolder_name):
    print(f"开始处理子文件夹: {subfolder_name}")
    
    # 下载图片到temp文件夹
    temp_folder = os.path.join(os.getcwd(),"gallery-dl", "danbooru", subfolder_name)
    print(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)
    gallery_dl_path = os.path.join(os.getcwd(), "gallery-dl", "gallery-dl.exe")
    command = f'"{gallery_dl_path}" "https://danbooru.donmai.us/posts?tags={subfolder_name}" --write-metadata'
    subprocess.run(command, shell=True)
    
    # 检查下载的图片数量
    image_files = [f for f in os.listdir(temp_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    if len(image_files) < 10:
        with open(os.path.join(target_folder, subfolder_name, "nodanbooru.txt"), 'w') as f:
            f.write("Not enough images from Danbooru")
        print(f"子文件夹 {subfolder_name} 图片数量不足,跳过")
        return
    
    # 去除temp文件夹中的重复图片
    phash = PHash()
    phash.remove_duplicates(temp_folder)
    
    # 匹配temp文件夹和目标文件夹中的图片
    duplicates = phash.find_duplicates(temp_folder, os.path.join(target_folder, subfolder_name))
    print(duplicates)
    for temp_image, target_image in duplicates.items():
        print(f"temp_image"+temp_image)
        json_file = temp_image + '.json'
        if os.path.exists(json_file):
            target_json = os.path.splitext(target_image)[0] + '.json'
            shutil.copy(json_file, target_json)
    
    # 删除temp文件夹
    shutil.rmtree(temp_folder)
    print(f"子文件夹 {subfolder_name} 处理完成")
    
def main(target_folder):
    for subfolder_name in os.listdir(target_folder):
        subfolder_path = os.path.join(target_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            process_subfolder(target_folder, subfolder_name)
    
if __name__ == "__main__":
    target_folder = r"E:\Datasets\dataset\Last\123"
    main(target_folder)