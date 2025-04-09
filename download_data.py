'''
数据集下载
'''
from datasets import load_dataset, load_from_disk

# 加载在线数据
datasets = load_dataset(path="Ayaka/ORCHESTRA-simple-1M", cache_dir="data/")
print(datasets)

# datasets = load_dataset(path="qgyd2021/chinese_porn_novel", cache_dir="data/")
# print(datasets)

# datasets = load_dataset(path="larryvrh/Chinese-Poems", cache_dir="data/")
# print(datasets)

# datasets = load_dataset(
#     "./data/larryvrh___chinese-poems/default/0.0.0/2b9da0fc4b724531ddd85f58f0dca83dd26b44cc")
# print(datasets)

train_data = datasets['train']
print(len(train_data))
for data in train_data:
    print(data)

# 加载 csv 格式数据
# datasets = load_dataset(path="csv", data_files="")
# print(datasets)
