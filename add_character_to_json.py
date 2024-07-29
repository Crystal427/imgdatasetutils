import os
import json
import regex as re
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_artist_folder(artist_path):
    modified_tags_path = os.path.join(artist_path, 'modifiedtags.txt')
    results_json_path = os.path.join(artist_path, 'results.json')

    # 检查modifiedtags.txt是否存在
    if not os.path.exists(modified_tags_path):
        logging.info(f"Skipping {artist_path}: modifiedtags.txt not found")
        return

    try:
        # 读取modifiedtags.txt
        with open(modified_tags_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
        # 解析modifiedtags.txt内容
        pattern_char_pairs = []
        for pair in content.split('\n\n'):
            if pair.strip():
                parts = pair.strip().split('\n')
                if len(parts) == 2:
                    pattern, char_name = parts
                    try:
                        pattern_char_pairs.append((re.compile(pattern, re.UNICODE), char_name))
                    except re.error as e:
                        logging.error(f"Invalid regex in {modified_tags_path}: {pattern}. Error: {str(e)}")
    
        # 读取results.json
        with open(results_json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
        # 处理每个图片条目
        for img_name, img_data in results.items():
            # 确保img_name是Unicode字符串
            img_name = os.fsdecode(img_name)
            for pattern, char_name in pattern_char_pairs:
                if pattern.match(img_name):
                    if 'character' not in img_data:
                        img_data['character'] = {}
                    img_data['character'][char_name] = 1
    
        # 写回results.json
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Successfully processed {artist_path}")

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in {results_json_path}: {str(e)}")
    except IOError as e:
        logging.error(f"IO error processing {artist_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error processing {artist_path}: {str(e)}")

def process_dataset(dataset_path):
    for artist_folder in os.listdir(dataset_path):
        artist_path = os.path.join(dataset_path, os.fsdecode(artist_folder))
        if os.path.isdir(artist_path):
            process_artist_folder(artist_path)


dataset_path = r'E:\Datasets\SDXL_tester'
process_dataset(dataset_path)