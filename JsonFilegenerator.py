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
# Increase the maximum image size to avoid DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None  # This removes the limit on the image size
ImageFile.LOAD_TRUNCATED_IMAGES = True  # This allows to load truncated images

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
    start_time = time.time()
    image = Image.open(image_path)
    
    # Resize image to be close to 1024x1024 while maintaining aspect ratio
    base_size = 1024
    img_ratio = image.size[0] / image.size[1]  # width / height
    if img_ratio > 1:  # Width is greater than height
        new_size = (base_size, int(base_size / img_ratio))
    else:  # Height is greater than width or equal
        new_size = (int(base_size * img_ratio), base_size)
    image = image.resize(new_size, Image.Resampling.LANCZOS) 
    classify_scores = anime_classify_score(image, model_name=model_name_classify)
    real_scores = anime_real_score(image, model_name=model_name_real)

    # 添加get_wd14_tags函数
    rating, features, chars = get_wd14_tags(image)

    # 在这里添加一次分辨率压缩
    aesthetic_image = image.resize((image.size[0]//2, image.size[1]//2), Image.Resampling.LANCZOS)
    weighted_mean, percentile, scores_by_class = _fn_predict(aesthetic_image, aesthetic_model)
    
    processing_time = time.time() - start_time
    print(f"Processed {image_path} in {processing_time:.2f} seconds.")
    return image_path, {
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
        "folder_repnum_offset":"0"
    }

def walk_and_process_images(directory, max_workers=1):
    print("Initializaing.....")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for root, dirs, files in os.walk(directory):
            results_path = Path(root) / 'results.json'
            if results_path.exists():
                print(f"Skipping {root} as results.json already exists.")
                continue

            futures = []
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = Path(root) / file
                    futures.append(executor.submit(process_image_file, file_path))

            directory_results = {}
            for future in as_completed(futures):
                file_path, result = future.result()
                directory_results[file_path.name] = result

            if directory_results:
                with open(results_path, 'w', encoding='utf-8') as json_file:
                    json.dump(directory_results, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    directory_path = r'F:\New folder'
    walk_and_process_images(directory_path, max_workers=1)