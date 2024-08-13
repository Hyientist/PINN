from torch.utils.data import Dataset


class DatasetBurgers(Dataset):
    def __init__(self, input_chunk, output_chunk):
        self.data = input_chunk.clone().detach().requires_grad_(True)
        self.labels = output_chunk

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
