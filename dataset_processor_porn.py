from datasets import load_dataset
from torch.utils.data import Dataset


class DatasetProcessorPorn(Dataset):
    def __init__(self):
        datasets = load_dataset(
            "./data/qgyd2021___chinese_porn_novel/xbookcn_short_story/0.0.0/170c125e168cf58400ad3b31300c88ed8a1c978a")
        train_data = datasets['train']
        lines = [i['content'].strip() for i in train_data]
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        return self.lines[item]


if __name__ == '__main__':
    datasets = DatasetProcessorPorn()
    for data in datasets:
        print(data)
