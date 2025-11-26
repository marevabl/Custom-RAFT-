from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader


def data_provider(args, flag):
    """
    Unified data provider for RAFT and non-RAFT models.

    Returns:
        dataset (Dataset)
        dataloader (DataLoader)

    Ensures:
        - consistent batch format
        - correct splitting (train/val/test)
        - correct batch_size and shuffle settings
        - returns index in __getitem__ for RAFT
    """

    # ------------------------------------
    # Split configuration
    # ------------------------------------
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size

    # ------------------------------------
    # Build dataset
    # ------------------------------------
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

    # ------------------------------------
    # Build dataloader
    # ------------------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return dataset, dataloader
