import json
from utils.utils import train_test_split


# 加载数据集
data_path = '../data'
data_split_path = '../data_split'
f = open('./char_dict.json', 'r')
category = json.loads(f.read())
# 将原数据集按0.1的比例分割成train与test
train_test_split(data_path, data_split_path, save_split_path, 0.1)

