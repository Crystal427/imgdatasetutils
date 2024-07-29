import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from imgutils.validate import anime_classify_score, anime_real_score
from imgutils.generic import classify_predict_score
from imgutils.tagging import get_wd14_tags
from PIL import Image, ImageFile
from functools import lru_cache
from huggingface_hub import HfFileSystem, hf_hub_download
from natsort import natsorted
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="Unsupported Windows version (11). ONNX Runtime supports Windows 10 and above, only.")
# Increase the maximum image size to avoid DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True 

hf_fs = HfFileSystem()

_REPOSITORY = 'deepghs/anime_aesthetic'
_DEFAULT_MODEL = 'swinv2pv3_v0_448_ls0.2_x'
_MODELS = natsorted([
    os.path.dirname(os.path.relpath(file, _REPOSITORY))
    for file in hf_fs.glob(f'{_REPOSITORY}/*/model.onnx')
])

LABELS = ["worst", "low", "normal", "good", "great", "best", "masterpiece"]

@lru_cache()
def _get_mark_table(model):
    df = pd.read_csv(hf_hub_download(
        repo_id=_REPOSITORY,
        repo_type='model',
        filename=f'{model}/samples.csv',
    ))
    df = df.sort_values(['score'])
    df['cnt'] = list(range(len(df)))
    df['final_score'] = df['cnt'] / len(df)

    x = np.concatenate([[0.0], df['score'], [6.0]])
    y = np.concatenate([[0.0], df['final_score'], [1.0]])
    return x, y

def _get_percentile(x, y, v):
    idx = np.searchsorted(x, np.clip(v, a_min=0.0, a_max=6.0))
    if idx < x.shape[0] - 1:
        x0, y0 = x[idx], y[idx]
        x1, y1 = x[idx + 1], y[idx + 1]
        return np.clip((v - x0) / (x1 - x0) * (y1 - y0) + y0, a_min=0.0, a_max=1.0)
    else:
        return y[idx]

def _fn_predict(image, model):
    scores = classify_predict_score(
        image=image,
        repo_id=_REPOSITORY,
        model_name=model,
    )
    weighted_mean = sum(i * scores[label] for i, label in enumerate(LABELS))
    x, y = _get_mark_table(model)
    percentile = _get_percentile(x, y, weighted_mean)
    return weighted_mean, percentile, scores

def process_image_file(image_path, model_name_classify='mobilenetv3_v1.3_dist', model_name_real='mobilenetv3_v1.2_dist', aesthetic_model=_DEFAULT_MODEL):
    image = Image.open(image_path)

    # Resize image to be close to 1024x1024 while maintaining aspect ratio
    base_size = 1280
    img_ratio = image.size[0] / image.size[1]  # width / height
    if img_ratio > 1:  # Width is greater than height
        new_size = (base_size, int(base_size / img_ratio))
    else:  # Height is greater than width or equal
        new_size = (int(base_size * img_ratio), base_size)
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    classify_scores = anime_classify_score(image, model_name=model_name_classify)
    real_scores = anime_real_score(image, model_name=model_name_real)

    rating, features, chars = get_wd14_tags(image, general_threshold = 0.2)

    # highres aesthetic rating consume a lot of performance, the model is trained by 448*448, Resize again
    aesthetic_image = image.resize((image.size[0] // 2, image.size[1] // 2), Image.Resampling.LANCZOS)
    weighted_mean, percentile, scores_by_class = _fn_predict(aesthetic_image, aesthetic_model)

    result = {
        "imgscore": classify_scores,
        "anime_real_score": real_scores,
        "aesthetic_score": weighted_mean,
        "percentile": percentile,
        "scores_by_class": scores_by_class,
        "rating": rating,
        "features": features,
        "character": chars,
        "is_AI": False,
        "Comment": "",
        "additional_tags": "",
        "folder_repnum_offset": "0",
        "placeholder": "",
        "datasetver": "2.0"
    }

    return image_path.name, result

def process_artist_directory(artist_dir):
    results_path = artist_dir / 'results.json'
    old_results = {}
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as json_file:
            old_results = json.load(json_file)

    image_folders = ['2010s', '2017s', '2020s', '2022s', 'new', 'unknown', 'undefined']
    image_files = []
    for folder in image_folders:
        folder_path = artist_dir / folder
        if folder_path.exists():
            image_files.extend([f for f in folder_path.glob('*') if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')])

    new_images = [img for img in image_files if img.name not in old_results]

    processed_count = 0
    if new_images:
        for img in new_images:
            file_name, result = process_image_file(img)
            old_results[file_name] = result
            processed_count += 1

        with open(results_path, 'w', encoding='utf-8') as json_file:
            json.dump(old_results, json_file, ensure_ascii=False, indent=4)
    
    return processed_count, artist_dir.name

def walk_and_process_dataset(directory, max_workers):
    print("Initializing.....")
    dataset_path = Path(directory)
    artist_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    total_processed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_artist_directory, artist_dir) for artist_dir in artist_dirs]
        
        for future in tqdm(as_completed(futures), total=len(artist_dirs), desc="Processing artists", unit="artist"):
            processed_count, artist_name = future.result()
            total_processed += processed_count
            tqdm.write(f"Processed {processed_count} images for artist: {artist_name}")

    print(f"Total images processed: {total_processed}")

if __name__ == '__main__':
    directory_path = r'E:\Datasets\SDXL_large_Modified'
    walk_and_process_dataset(directory_path, max_workers=4)