import argparse
import os
import torch
import pandas as pd
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='RAFT Time-Series Forecasting')

    # --------------------------------------------------
    # Basic / task settings
    # --------------------------------------------------
    parser.add_argument('--model', type=str, default='RAFT',
                        help='model name')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name (used by RAFT internally)')
    parser.add_argument('--root_path', type=str, default='./data',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='weather_raft.csv',
                        help='data file name')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: [S, M, MS]')
    parser.add_argument('--target', type=str, default='OT',
                        help='target variable name in dataset')
    parser.add_argument('--freq', type=str, default='h',
                        help='frequency for time encoding')
    parser.add_argument('--timeenc', type=int, default=0,
                        help='time encoding type, 0 = simple, 1 = embedding')

    # --------------------------------------------------
    # Data window lengths
    # --------------------------------------------------
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=24,
                        help='prediction sequence length')

    # --------------------------------------------------
    # Model input/output sizes
    # (we will auto-set from data if left as None)
    # --------------------------------------------------
    parser.add_argument('--enc_in', type=int, default=None,
                        help='encoder input size (number of features)')
    parser.add_argument('--dec_in', type=int, default=None,
                        help='decoder input size (number of features)')
    parser.add_argument('--c_out', type=int, default=None,
                        help='output size (number of features)')

    # --------------------------------------------------
    # RAFT / Transformer-style hyperparameters
    # (defaults are reasonable; RAFT may or may not use all)
    # --------------------------------------------------
    parser.add_argument('--d_model', type=int, default=512,
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='num of attention heads')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='dimension of fcn in model')
    parser.add_argument('--factor', type=int, default=5,
                        help='probSparse factor or similar hyperparam')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='embedding type')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention maps')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='moving avg window, if used')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')

    # RAFT-specific retrieval hyperparameters
    parser.add_argument('--n_period', type=int, default=24,
                        help='period length for retrieval (e.g. 24 for 24h)')
    parser.add_argument('--topm', type=int, default=5,
                        help='number of top similar patches to retrieve')

    # --------------------------------------------------
    # Optimization settings
    # --------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--train_epochs', type=int, default=3,
                        help='number of training epochs')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='dataloader workers')

    # --------------------------------------------------
    # GPU / Hardware
    # --------------------------------------------------
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether to try to use GPU')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id if using single GPU')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple GPUs')
    parser.add_argument('--device_ids', type=str, default='0',
                        help='GPU device ids, comma-separated')

    # --------------------------------------------------
    # Other settings
    # --------------------------------------------------
    parser.add_argument('--inverse', action='store_true',
                        help='inverse-transform output back to original scale')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision')
    parser.add_argument('--use_dtw', action='store_true',
                        help='calculate DTW metric in testing')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='path to save model checkpoints')

    args = parser.parse_args()

    # --------------------------------------------------
    # Infer enc_in / dec_in / c_out from the CSV if needed
    # --------------------------------------------------
    data_full_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_full_path):
        raise FileNotFoundError(f"Data file not found at: {data_full_path}")

    df = pd.read_csv(data_full_path)
    df_num = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])
    num_features = df_num.shape[1]
    if num_features == 0:
        raise ValueError("No numeric columns found in dataset for model input.")

    if args.enc_in is None:
        args.enc_in = num_features
    if args.dec_in is None:
        args.dec_in = num_features
    if args.c_out is None:
        args.c_out = num_features

    # --------------------------------------------------
    # GPU setup (with graceful CPU fallback)
    # --------------------------------------------------
    if args.use_gpu:
        args.use_gpu = torch.cuda.is_available()
    if args.use_multi_gpu:
        args.device_ids = [int(i) for i in args.device_ids.split(',')]

    print("Using GPU:", args.use_gpu)
    if not args.use_gpu:
        print("Use CPU")

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
