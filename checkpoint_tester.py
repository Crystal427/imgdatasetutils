import os
import csv
import shutil
import webuiapi
import json
import random
from PIL import Image
import pandas as pd
import numpy as np
from imgutils.generic import classify_predict_score
from functools import lru_cache
from huggingface_hub import HfFileSystem, hf_hub_download
from natsort import natsorted

# 设置features_threshold
features_threshold = 0.27

Image.MAX_IMAGE_PIXELS = None  # This removes the limit on the image size
Image.LOAD_TRUNCATED_IMAGES = True  # This allows to load truncated images
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

def process_image_file(image,aesthetic_model=_DEFAULT_MODEL):
    base_size = 1024
    img_ratio = image.size[0] / image.size[1]  # width / height
    if img_ratio > 1:  # Width is greater than height
        new_size = (base_size, int(base_size / img_ratio))
    else:  # Height is greater than width or equal
        new_size = (int(base_size * img_ratio), base_size)
    image = image.resize(new_size, Image.Resampling.LANCZOS) 
    weighted_mean, percentile, scores_by_class =  _fn_predict(image, aesthetic_model)
    return image, {
            "aesthetic_score": weighted_mean,
            "percentile": percentile,
            "scores_by_class": scores_by_class
        }


def process_image(root, filename):
    # 获取图片文件名(不包括扩展名)
    name, _ = os.path.splitext(filename)
    
    # 检查是否存在对应的danboorujson文件
    danbooru_json = None
    json_path = os.path.join(root, 'jsons', name + '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            danbooru_json = json.load(f)
    
    # 读取results.json
    results_json = None
    results_path = os.path.join(root, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results_json = json.load(f)
    
    # 生成final_artist_tag
    final_artist_tag = os.path.basename(root).replace('_', ' ') + ','
    
    # 生成final_copyright_tag
    final_copyright_tag = ''
    if danbooru_json and 'tags_copyright' in danbooru_json:
        final_copyright_tag = ','.join(danbooru_json['tags_copyright']).replace('_', ' ') + ','
        final_character_tag = final_copyright_tag.replace('original,','')
    
    # 生成final_character_tag
    final_character_tag = ''
    if danbooru_json and 'tags_character' in danbooru_json and danbooru_json['tags_character']:
        final_character_tag = ','.join(danbooru_json['tags_character']).replace('_', ' ') + ','
    elif results_json and filename in results_json and 'character' in results_json[filename]:
        final_character_tag = ','.join(results_json[filename]['character'].keys()).replace('_', ' ') + ','
    
    # 生成final_features_tag
    final_features_tag = ''
    if results_json and filename in results_json and 'features' in results_json[filename]:
        features = [k.replace('_', ' ') for k, v in results_json[filename]['features'].items() if v > features_threshold]
        final_features_tag = ','.join(features) + ','
    
    # 生成final_rating_tag
    final_rating_tag = ''
    if results_json and filename in results_json:
        if results_json[filename].get('is_AI'):
            final_rating_tag += 'ai-generated,'
        
        final_rating_tag += "masterpiece" + ','

        
        rating = results_json[filename].get('rating', {})
        if rating:
            max_rating = max(rating, key=rating.get)
            if max_rating == 'general':
                max_rating = 'safe'
            elif max_rating == 'questionable':
                max_rating = 'nsfw'
            final_rating_tag += max_rating + ','
        
        # 获取图片分辨率并判断是否添加lowres或absurdres标签
        final_rating_tag += 'absurdres,'
    
    # 生成additional_tags
    additional_tags = ''
    if results_json and filename in results_json and 'additional_tags' in results_json[filename]:
        additional_tags = results_json[filename]['additional_tags'].replace('_', ' ')
    
    # 组合finaltag
    finaltag = f"{final_artist_tag}{final_copyright_tag}{final_character_tag}{final_features_tag}{final_rating_tag}{additional_tags}"
    print(finaltag)
    return finaltag

def process_artist_images(artist_path, output_path):
    image_scores = []
    best_image = None
    best_score = float('-inf')

    for image_file in os.listdir(artist_path):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            image_path = os.path.join(artist_path, image_file)
            with Image.open(image_path) as img:
                processed_img, scores = process_image_file(img)
                
                aesthetic_score = scores['aesthetic_score']
                image_scores.append(aesthetic_score)

                if aesthetic_score > best_score:
                    best_score = aesthetic_score
                    best_image = (image_file, processed_img)

    artist_name = os.path.basename(artist_path)
    artist_score = sum(image_scores) / len(image_scores) if image_scores else 0

    # 保存最佳图片
    if best_image:
        best_folder = os.path.join(output_path, 'best')
        os.makedirs(best_folder, exist_ok=True)
        best_image_path = os.path.join(best_folder, f"{artist_name}_best.jpg")
        best_image[1].save(best_image_path)

    return artist_name, artist_score

def select_images(root):
    test_prompt_pic = []
    results_path = os.path.join(root, 'results.json')
    
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results_json = json.load(f)
        
        # 处理2010s和2017s文件夹
        old_images = []
        for folder in ['2010s', '2017s']:
            folder_path = os.path.join(root, folder)
            if os.path.exists(folder_path):
                old_images.extend([os.path.join(folder, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        if len(old_images) >= 15:
            old_scores = {img: results_json[os.path.basename(img)]['aesthetic_score'] for img in old_images if os.path.basename(img) in results_json}
            sorted_old = sorted(old_scores.items(), key=lambda x: x[1], reverse=True)
            top_50_percent = sorted_old[:len(sorted_old)//2]
            test_prompt_pic.extend(random.sample([img for img, _ in top_50_percent], min(4, len(top_50_percent))))
        
        # 处理2020s, 2022s和new文件夹
        new_images = []
        for folder in ['2020s', '2022s', 'new']:
            folder_path = os.path.join(root, folder)
            if os.path.exists(folder_path):
                new_images.extend([os.path.join(folder, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        new_scores = {img: results_json[os.path.basename(img)]['aesthetic_score'] for img in new_images if os.path.basename(img) in results_json}
        sorted_new = sorted(new_scores.items(), key=lambda x: x[1], reverse=True)
        top_50_percent = sorted_new[:len(sorted_new)//2]
        test_prompt_pic.extend(random.sample([img for img, _ in top_50_percent], min(8, len(top_50_percent))))
    
    return test_prompt_pic

class SDWebUIGenerator:
    def __init__(self, host, port, model='13.4-6e'):
        self.api = webuiapi.WebUIApi(host=host, port=port)
        #self.api = webuiapi.WebUIApi(host=host, port=port, use_https=True)  FOR HTTPS SITUATIONS
        self.negative_prompt = "lowres,bad hands,worst quality,watermark,censored,jpeg artifacts"
        self.cfg_scale = 4.5
        self.steps = 28
        self.sampler_name = 'Euler a'
        self.scheduler = 'SGM Uniform'
        self.width = 1024
        self.height = 1024
        self.seed = 47

        self.set_model(model)

    def set_model(self, model):
        self.api.util_set_model(model)
        print("Model set to:" + model)

    def generate(self, prompt):
        result = self.api.txt2img(
            prompt=prompt,
            steps=self.steps,
            negative_prompt=self.negative_prompt,
            cfg_scale=self.cfg_scale,
            sampler_name=self.sampler_name,
            scheduler=self.scheduler,
            width=self.width,
            height=self.height,
            seed=self.seed
        )
        return result.image
def append_to_csv(csv_path, artist_name, artist_score):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['artistname', 'artistname_score'])
        writer.writerow([artist_name, artist_score])

def main():
    dataset_path = r'E:\Datasets\testerEXample'
    output_path = r'F:\temp'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    generator = SDWebUIGenerator(host='localhost', port=7860)  # 使用本地主机和默认端口
    
    csv_path = os.path.join(output_path, 'artist_scores.csv')

    for artist_folder in os.listdir(dataset_path):
        artist_path = os.path.join(dataset_path, artist_folder)
        if os.path.isdir(artist_path):
            process_fin_path = os.path.join(artist_path, 'process.fin')
            
            # 如果存在 process.fin 文件，跳过这个 artist
            if os.path.exists(process_fin_path):
                print(f"Skipping {artist_folder} as it has already been processed.")
                continue

            selected_images = select_images(artist_path)
            
            artist_output_path = os.path.join(output_path, artist_folder)
            os.makedirs(artist_output_path, exist_ok=True)

            for image in selected_images:
                finaltag = process_image(artist_path, os.path.basename(image))
                generated_image = generator.generate(finaltag)
                if generated_image:
                    output_filename = f"{artist_folder}_{os.path.basename(image)}"
                    generated_image.save(os.path.join(artist_output_path, output_filename))
                    print(f"Generated image for {output_filename}")

            # 处理生成的图片并计算分数
            artist_name, artist_score = process_artist_images(artist_output_path, output_path)
            
            # 立即将分数写入 CSV 文件
            append_to_csv(csv_path, artist_name, artist_score)
            
            # 创建 process.fin 文件
            with open(process_fin_path, 'w') as f:
                f.write(f"Processed on: {pd.Timestamp.now()}")

            print(f"Completed processing {artist_folder}")

    print(f"All artists have been processed. Scores saved to {csv_path}")

if __name__ == "__main__":
    main()