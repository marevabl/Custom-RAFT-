import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
        self.std[self.std == 0] = 1

    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class Dataset_Custom(Dataset):
    """
    Custom dataset class for RAFT-compatible multivariate time-series.
    Handles:
        - non-numeric columns
        - date columns
        - scaling
        - window slicing
        - returning dataset indices for RAFT retrieval
    """
    def __init__(self, root_path, data_path, flag='train',
                 seq_len=96, label_len=48, pred_len=24,
                 scale=True, inverse=False, features='S', target='OT',
                 timeenc=0, freq='h'):
        
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.scale = scale
        self.inverse = inverse
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # Load raw dataframe
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ------------------------------
        # 1. HANDLE DATE COLUMN
        # ------------------------------
        datetime_col = None
        for col in df_raw.columns:
            if df_raw[col].dtype == object and "20" in str(df_raw[col].iloc[0]):
                datetime_col = col
                break
        
        if datetime_col is not None:
            df_raw[datetime_col] = pd.to_datetime(df_raw[datetime_col], errors='coerce')
            df_raw = df_raw.dropna(subset=[datetime_col])
            df_raw = df_raw.set_index(datetime_col)

        # ------------------------------
        # 2. KEEP ONLY NUMERIC COLUMNS
        # ------------------------------
        df_raw = df_raw.select_dtypes(include=['float32','float64','int32','int64'])
        df_raw = df_raw.astype('float32')

        # If dataset author intended target column but it's missing:
        if self.target not in df_raw.columns:
            # fallback: last column is target
            self.target = df_raw.columns[-1]

        num_train = int(len(df_raw) * 0.7)
        num_val = int(len(df_raw) * 0.1)
        num_test = len(df_raw) - num_train - num_val

        # Data splits
        if flag == 'train':
            border1, border2 = 0, num_train
        elif flag == 'val':
            border1, border2 = num_train, num_train + num_val
        else:
            border1, border2 = num_train + num_val, len(df_raw)

        self.border1 = border1
        self.border2 = border2

        # Extract data
        self.df_raw = df_raw
        df_data = df_raw.copy()

        # ------------------------------
        # 3. SCALE NUMERIC DATA
        # ------------------------------
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values)
            df_data = self.scaler.transform(df_data.values)
        else:
            df_data = df_data.values

        self.data_x = df_data
        self.data_y = df_data  # same for forecasting

        # ------------------------------
        # 4. Generate time features (dummy, RAFT doesn't use markers heavily)
        # ------------------------------
        # To keep compatibility with main RAFT code
        self.data_stamp = np.zeros((len(df_data), 1))  # dummy marker

    def __len__(self):
        return self.border2 - self.border1 - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s_begin = self.border1 + idx
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # RAFT needs the raw index for retrieval
        return (
            idx,                                       # <---- index (critical for RAFT)
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(self.data_stamp[s_begin:s_end], dtype=torch.float32),
            torch.tensor(self.data_stamp[r_begin:r_end], dtype=torch.float32)
        )


def data_provider(args, flag):
    """
    Wrapper for dataset + dataloader.
    RAFT-specific structure preserved.
    """
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    dataset = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        scale=True,
        inverse=args.inverse,
        features=args.features,
        target=args.target,
        timeenc=args.timeenc,
        freq=args.freq
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return dataset, dataloader
