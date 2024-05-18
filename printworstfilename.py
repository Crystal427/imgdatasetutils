import os
import json

def find_max_low_worst_scores(directory):
    max_pictures = []  # 记录包含最大low和worst评分的图片名

    # 遍历给定目录下的所有子目录
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file == 'results.json':
                filepath = os.path.join(subdir, file)
                with open(filepath, 'r',encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    max_low_score = 0
                    max_worst_score = 0
                    current_max_low = []
                    current_max_worst = []

                    # 遍历JSON文件中的所有图片
                    for picturename, content in data.items():
                        scores = content.get('scores_by_class', {})
                        low_score = scores.get('low', 0)
                        worst_score = scores.get('worst', 0)

                        # 更新最大low评分和对应图片名
                        if low_score > max_low_score:
                            max_low_score = low_score
                            current_max_low = [picturename]
                        elif low_score == max_low_score:
                            current_max_low.append(picturename)

                        # 更新最大worst评分和对应图片名
                        if worst_score > max_worst_score:
                            max_worst_score = worst_score
                            current_max_worst = [picturename]
                        elif worst_score == max_worst_score:
                            current_max_worst.append(picturename)

                    # 将当前文件的最大low和worst图片名添加到总列表
                    max_pictures.extend(current_max_low)
                    max_pictures.extend(current_max_worst)

    return max_pictures

# 假设你的顶级目录是'path_to_directory'
directory = r'E:\Datasets\dataset\SDXL_large'
resulting_pictures = find_max_low_worst_scores(directory)
print("Pictures with the highest 'low' or 'worst' scores:")
for pic in resulting_pictures:
    print(pic)