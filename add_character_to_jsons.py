import json
import re
import os

def process_patch(json_file, patch_file, directory):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 读取patch文件
    with open(os.path.join(directory, 'patch.txt'), 'r', encoding='utf-8') as f:
        patch_lines = f.readlines()

    matched_items = []

    # 处理每一行patch
    for line in patch_lines:
        line = line.strip()
        if line:
            regex, field, value = line.split('|')
            
            # 遍历JSON中的每个项
            for item_name, item_data in data.items():
                if re.search(regex, item_name):
                    # 如果是character字段
                    if field == 'character':
                        if 'character' not in item_data:
                            item_data['character'] = {}
                        item_data['character'][value] = 1
                    
                    matched_items.append(item_name)

    # 将更新后的数据写回JSON文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return matched_items

# 使用函数
json_file = 'results.json'
directory = '.'  # 当前目录，你可以根据需要修改
matched_items = process_patch(json_file, 'patch.txt', directory)

print("匹配的项：")
for item in matched_items:
    print(item)s