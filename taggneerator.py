import os
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# 设置features_threshold
features_threshold = 0.27
ROOT = Path(__file__).parent
PATTERN_ESCAPED_BRACKET = r"\\([\(\)\[\]\{\}])"  # match `\(` and `\)`

def search_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None


OVERLAP_TABLE_PATH = search_file('overlap_tags.json', ROOT)


def init_overlap_table(table_path=OVERLAP_TABLE_PATH):
    global OVERLAP_TABLE
    if OVERLAP_TABLE is not None:
        return True
    try:
        import json
        with open(table_path, 'r') as f:
            table = json.load(f)
        table = {entry['query']: (set(entry.get("has_overlap") or []), set(entry.get("overlap_tags") or [])) for entry in table}
        table = {k: v for k, v in table.items() if len(v[0]) > 0 or len(v[1]) > 0}
        OVERLAP_TABLE = table
        return True
    except Exception as e:
        OVERLAP_TABLE = None
        print(f'failed to read overlap table: {e}')
        return False

def unescape(s):
    return re.sub(PATTERN_ESCAPED_BRACKET, r'\1', s)

def fmt2danbooru(tag):
    r"""
    Process a tag to:
    - lower case
    - replace spaces with underscores
    - unescape brackets
    """
    tag = tag.lower().replace(' ', '_').strip('_').replace(':_', ':')
    tag = unescape(tag)
    return tag

def deoverlap_tags(tag_str):
    # 将tag字符串转换为tag列表
    tags = tag_str.split(", ")
    
    # 初始化overlap表
    init_overlap_table()
    
    # 构建dan2tag和tag2dan字典
    dan2tag = {fmt2danbooru(tag): tag for tag in tags}
    tag2dan = {v: k for k, v in dan2tag.items()}
    
    # 获取overlap表
    ovlp_table = OVERLAP_TABLE
    
    # 记录需要删除的tag
    tags_to_remove = set()
    tagset = set(tags)
    
    # 遍历每个tag
    for tag in tagset:
        dantag = tag2dan[tag]
        if dantag in ovlp_table and tag not in tags_to_remove:
            parents, children = ovlp_table[dantag]
            parents = {dan2tag[parent] for parent in parents if parent in dan2tag}
            children = {dan2tag[child] for child in children if child in dan2tag}
            tags_to_remove |= tagset & children
    
    # 删除需要删除的tag
    deoverlaped_tags = [tag for tag in tags if tag not in tags_to_remove]
    
    # 将处理后的tag列表转换为字符串并返回
    return ", ".join(deoverlaped_tags)



def process_image(root, filename):
    # 获取图片文件名(不包括扩展名)
    name, _ = os.path.splitext(filename)
    
    # 检查是否存在对应的danboorujson文件
    danbooru_json = None
    for ext in ['.json', '.png.json', '.jpg.json', '.jpeg.json', '.webp.json']:
        json_path = os.path.join(root, name + ext)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                danbooru_json = json.load(f)
            break
    
    # 读取results.json
    results_json = None
    results_path = os.path.join(root, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            results_json = json.load(f)
    
    # 生成final_artist_tag
    final_artist_tag = os.path.basename(root).replace('_', ' ') + ', '
    
    # 生成final_copyright_tag
    final_copyright_tag = ''
    if danbooru_json and 'tags_copyright' in danbooru_json:
        final_copyright_tag = ','.join(danbooru_json['tags_copyright']).replace('_', ' ') + ', '
    
    # 生成final_character_tag
    final_character_tag = ''
    if danbooru_json and 'tags_character' in danbooru_json and danbooru_json['tags_character']:
        final_character_tag = ', '.join(danbooru_json['tags_character']).replace('_', ' ') + ', '
    elif results_json and filename in results_json and 'character' in results_json[filename]:
        final_character_tag = ', '.join(results_json[filename]['character'].keys()).replace('_', ' ') + ', '
    
    # 生成features_tag
    features_tag = ''
    if results_json and filename in results_json and 'features' in results_json[filename]:
        features = [k.replace('_', ' ') for k, v in results_json[filename]['features'].items() if v > features_threshold]
        features_tag = ', '.join(features) + ', '
    
    #进行语义去重处理

    final_features_tag = deoverlap_tags(features_tag)

    # 生成final_rating_tag
    final_rating_tag = ''
    if results_json and filename in results_json:
        if results_json[filename].get('is_AI'):
            final_rating_tag += 'ai-generated, '
        
        scores_by_class = results_json[filename].get('scores_by_class', {})
        if scores_by_class:
            max_class = max(scores_by_class, key=scores_by_class.get)
            final_rating_tag += max_class + ', '
        
        rating = results_json[filename].get('rating', {})
        if rating:
            max_rating = max(rating, key=rating.get)
            if max_rating == 'general':
                max_rating = 'safe'
            elif max_rating == 'questionable':
                max_rating = 'nsfw'
            final_rating_tag += max_rating + ', '
        
        # 获取图片分辨率并判断是否添加lowres或absurdres标签
        img_path = os.path.join(root, filename)
        with Image.open(img_path) as img:
            width, height = img.size
            if width * height <= 589824:
                final_rating_tag += 'lowres, '
            elif width * height >= 1638400:
                final_rating_tag += 'absurdres, '
    
    # 生成additional_tags
    additional_tags = ''
    if results_json and filename in results_json and 'additional_tags' in results_json[filename]:
        additional_tags = results_json[filename]['additional_tags'].replace('_', ' ')
    
    # 组合finaltag

    finaltag = f"{final_artist_tag}{final_character_tag}{final_copyright_tag}|||{final_features_tag}{final_rating_tag}{additional_tags}"

    # 写入文本文件
    txt_path = os.path.join(root, name + '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(finaltag)
    
    print(f"Processed: {os.path.join(root, filename)}")

def process_folder(folder):
    with ThreadPoolExecutor() as executor:
        futures = []
        for root, _, files in os.walk(folder):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    future = executor.submit(process_image, root, filename)
                    futures.append(future)
        
        for future in as_completed(futures):
            future.result()

# 指定目标文件夹
target_folder = r'F:\SDXL_large'

# 处理目标文件夹
process_folder(target_folder)