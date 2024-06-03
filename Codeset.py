from torch.utils.data import Dataset
import pandas as pd
class CodeDataset(Dataset):
    def __init__(self, dataframe):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data.iloc[idx].values
        return code
