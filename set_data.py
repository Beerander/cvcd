import json
from utils.utils import train_test_split


# �������ݼ�
data_path = '../data'
data_split_path = '../data_split'
f = open('./char_dict.json', 'r')
category = json.loads(f.read())
# ��ԭ���ݼ���0.1�ı����ָ��train��test
train_test_split(data_path, data_split_path, save_split_path, 0.1)

