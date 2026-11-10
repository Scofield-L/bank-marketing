"""从任意工作目录运行；默认路径以仓库位置为准。"""
import argparse
from bank_preprocessing import ROOT, run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=ROOT / 'data/bank-full.csv')
    parser.add_argument('--output', default=ROOT / 'data/processed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-duration', action='store_true', help='仅用于通话后对照实验')
    args = parser.parse_args()
    run(args.input, args.output, args.seed, args.include_duration)
