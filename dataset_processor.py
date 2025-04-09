from datasets import load_dataset
from torch.utils.data import Dataset


class DatasetProcessor(Dataset):
    def __init__(self):
        datasets = load_dataset(
            "./data/larryvrh___chinese-poems/default/0.0.0/2b9da0fc4b724531ddd85f58f0dca83dd26b44cc")
        train_data = datasets['train']
        lines = [i['content'].strip() for i in train_data]
        self.lines = lines

        datasets = load_dataset(
            "./data/Ayaka___orchestra-simple-1_m/default/0.0.0/d0a766b1b2ad1d3e4ef39fa6faff628811a14041")
        train_data = datasets['train']
        lines = [i['content'].strip() for i in train_data]
        self.lines.append(lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        return self.lines[item]


if __name__ == '__main__':
    datasets = DatasetProcessor()
    for data in datasets:
        print(data)
