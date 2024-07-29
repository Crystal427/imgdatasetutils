import os
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from imgutils.tagging import get_wd14_tags
from PIL import Image
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="Unsupported Windows version (11). ONNX Runtime supports Windows 10 and above, only.")
# Increase the maximum image size to avoid DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None 

def process_image_file(image_path, model_name_wd14='SmilingWolf/wd-eva02-large-tagger-v3'):
    image = Image.open(image_path)

    # Resize image to be close to 1024x1024 while maintaining aspect ratio
    base_size = 1280
    img_ratio = image.size[0] / image.size[1]  # width / height
    if img_ratio > 1:  # Width is greater than height
        new_size = (base_size, int(base_size / img_ratio))
    else:  # Height is greater than width or equal
        new_size = (int(base_size * img_ratio), base_size)
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    rating, features, chars = get_wd14_tags(image, general_threshold=0.2, model_name=model_name_wd14)

    return rating, features, chars

def update_artist_results(artist_dir):
    results_path = artist_dir / 'results.json'
    if not results_path.exists():
        return 0, artist_dir.name

    with open(results_path, 'r', encoding='utf-8') as json_file:
        results = json.load(json_file)

    image_folders = ['2010s', '2017s', '2020s', '2022s', 'new', 'unknown', 'undefined']
    updated_count = 0

    for file_name, data in results.items():
        image_path = None
        for folder in image_folders:
            potential_path = artist_dir / folder / file_name
            if potential_path.exists():
                image_path = potential_path
                break

        if image_path:
            try:
                rating, features, chars = process_image_file(image_path)
                data['rating'] = rating
                data['features'] = features
                data['character'] = chars
                updated_count += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    with open(results_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    
    return updated_count, artist_dir.name

def walk_and_update_dataset(directory, max_workers):
    print("Initializing.....")
    dataset_path = Path(directory)
    artist_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    total_updated = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(update_artist_results, artist_dir) for artist_dir in artist_dirs]
        
        for future in tqdm(as_completed(futures), total=len(artist_dirs), desc="Updating artists", unit="artist"):
            updated_count, artist_name = future.result()
            total_updated += updated_count
            tqdm.write(f"Updated {updated_count} images for artist: {artist_name}")

    print(f"Total images updated: {total_updated}")

if __name__ == '__main__':
    directory_path = r'E:\Datasets\SDXL_large_Modified'
    walk_and_update_dataset(directory_path, max_workers=4)