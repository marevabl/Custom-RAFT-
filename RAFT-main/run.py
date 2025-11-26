import argparse
import os
import torch
import pandas as pd  
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='RAFT Time-Series Forecasting')

    # --------------------------------------------------
    # Basic settings
    # --------------------------------------------------
    parser.add_argument('--model', type=str, required=False, default='RAFT',
                        help='model name')
    parser.add_argument('--root_path', type=str, required=False, default='./dataset/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, required=False, default='weather.csv',
                        help='data file name')
    parser.add_argument('--features', type=str, required=False, default='S',
                        help='forecasting task, options: [S, M, MS]')
    parser.add_argument('--target', type=str, required=False, default='OT',
                        help='target variable name in dataset')
    parser.add_argument('--freq', type=str, required=False, default='h',
                        help='frequency for time encoding')

    # --------------------------------------------------
    # Data window lengths
    # --------------------------------------------------
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=24,
                        help='prediction sequence length')
        # Model / retrieval hyperparameters
    parser.add_argument('--n_period', type=int, default=24,
                        help='period length for retrieval (e.g. 24 for 24h daily seasonality)')
        # Model / retrieval hyperparameters
    parser.add_argument('--n_period', type=int, default=24,
                        help='period length for retrieval (e.g. 24 for 24h daily seasonality)')
    parser.add_argument('--topm', type=int, default=5,
                        help='number of top similar patches to retrieve')

        # Model input/output sizes (will be auto-set from data if not given)
    parser.add_argument('--enc_in', type=int, default=None,
                        help='encoder input size (number of features)')
    parser.add_argument('--dec_in', type=int, default=None,
                        help='decoder input size (number of features)')
    parser.add_argument('--c_out', type=int, default=None,
                        help='output size (number of features)')


    # --------------------------------------------------
    # Optimization settings
    # --------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='dataloader workers')

    # --------------------------------------------------
    # GPU / Hardware
    # --------------------------------------------------
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether to use GPU')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple GPUs')
    parser.add_argument('--device_ids', type=str, default='0',
                        help='GPU device ids')

    # --------------------------------------------------
    # Additional settings
    # --------------------------------------------------
    parser.add_argument('--inverse', action='store_true',
                        help='inverse-transform output')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision')
    parser.add_argument('--use_dtw', action='store_true',
                        help='calculate DTW metrics')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='path to save model checkpoints')

    args = parser.parse_args()

    # --------------------------------------------------
    # GPU setup
    # --------------------------------------------------
    if args.use_gpu:
        args.use_gpu = torch.cuda.is_available()
    if args.use_multi_gpu:
        args.device_ids = [int(i) for i in args.device_ids.split(',')]
    # --------------------------------------------------
    # Infer enc_in / dec_in / c_out from the CSV
    # --------------------------------------------------
    data_full_path = os.path.join(args.root_path, args.data_path)
    df = pd.read_csv(data_full_path)

    # keep only numeric columns (same idea as Dataset_Custom)
    df_num = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])
    num_features = df_num.shape[1]

    if args.enc_in is None:
        args.enc_in = num_features
    if args.dec_in is None:
        args.dec_in = num_features
    if args.c_out is None:
        args.c_out = num_features

    print("Using GPU:", args.use_gpu)

    # --------------------------------------------------
    # Build experiment
    # --------------------------------------------------
    exp = Exp_Long_Term_Forecast(args)

    # --------------------------------------------------
    # Setting name (for saving checkpoints/results)
    # --------------------------------------------------
    setting = f"{args.model}_{args.data_path.split('.')[0]}_" \
              f"sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    print("\n>>> Training starts for setting:", setting)
    exp.train(setting)

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------
    print("\n>>> Testing starts")
    exp.test(setting, test=1)


if __name__ == '__main__':
    main()
