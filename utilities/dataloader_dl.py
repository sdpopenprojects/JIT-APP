import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels

    def readData(self, data, labels):
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index],
                                                                                 dtype=torch.float32)