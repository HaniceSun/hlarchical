import argparse
from .preprocess import DataPreprocessor
from .dataset import CustomDataset
from .trainer import Trainer
from .utils import *
from .array import Array

def get_parser():
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest='command', required=True)

    p1 = subparsers.add_parser("preprocess", help="preprocess data")
    p2 = subparsers.add_parser("torch-dataset", help="generate torch datasets")

    p3 = subparsers.add_parser("train", help="train model")
    p3.add_argument('--train_file', type=str, default='hla_dataset_train.pt', help='training dataset file')
    p3.add_argument('--val_file', type=str, default='hla_dataset_val.pt', help='validation dataset file')
    p3.add_argument('--test_file', type=str, default='hla_dataset_test.pt', help='test dataset file')
    p3.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
    p3.add_argument('--resume_epoch', type=str, default=None, help='resume training from a specific epoch if specified')

    p4 = subparsers.add_parser("eval", help="evaluate a trained model")
    p4.add_argument('--epoch', type=int, default=99, help='the epoch of the trained model to be used for prediction')
    p4.add_argument('--val_file', type=str, default='hla_dataset_val.pt', help='val dataset file')
    p4.add_argument('--maps_file', type=str, default='maps.txt', help='the maps file from preprocessing step')

    p5 = subparsers.add_parser("test", help="test a trained model")
    p5.add_argument('--epoch', type=int, default=99, help='the epoch of the trained model to be used for prediction')
    p5.add_argument('--test_file', type=str, default='hla_dataset_test.pt', help='test dataset file')
    p5.add_argument('--maps_file', type=str, default='maps.txt', help='the maps file from preprocessing step')

    p6 = subparsers.add_parser("predict", help="predicted using the trained model")
    p6.add_argument('--pred_file', type=str, default='to_predict.txt', help='input dataset file for prediction')
    p6.add_argument('--maps_file', type=str, default='maps.txt', help='the maps file from preprocessing step')
    p6.add_argument('--epoch', type=int, default=99, help='the epoch of the trained model to be used for prediction')
    p6.add_argument('--output', type=str, default='hla_predicted.txt', help='output prediction file')

    p11 = subparsers.add_parser("format-output", help="format the output to allele table")
    p11.add_argument('--input', type=str, default='data/1958BC_Euro.bgl.phased', help='input file')
    p11.add_argument('--output', type=str, default='data/1958BC_Euro_digit4.txt', help='output file')
    p11.add_argument('--digit', type=int, default=4, help='digit level for HLA alleles')
    p11.add_argument('--from_tool', type=str, default='snp2hla', help='the tool that generated the input file')

    p12 = subparsers.add_parser("run-snp2hla", help="run SNP2HLA on array data")
    p12.add_argument('--input', type=str, default='1958BC', help='input file prefix')
    p12.add_argument('--ref', type=str, default='HM_CEU_REF', help='reference panel prefix, can be HM_CEU_REF or Pan-Asian_REF currently')

    p13 = subparsers.add_parser("run-deephla", help="run CNN-based DEEP*HLA, to be implemented")
    p13.add_argument('--mode', type=str, default='train', help='mode: train or impute')
    p13.add_argument('--input', type=str, default='1958BC_Pan-Asian_REF', help='input file prefix')
    p13.add_argument('--ref', type=str, default='Pan-Asian_REF', help='reference panel prefix, can be HM_CEU_REF or Pan-Asian_REF currently')
    p13.add_argument('--subset', type=str, default=None, help='subset the input to the HLA regions according to the reference genome, e.g., chr6:28510120-33480577 on GRCh37')
    p13.add_argument('--model_json', type=str, default='Pan-Asian_REF.model.json', help='the config file of the model')
    p13.add_argument('--model_dir', type=str, default='model', help='the output directory of the trained model')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.command == 'preprocess':
        dp = DataPreprocessor()
        dp.make_features()
        dp.make_labels()
        dp.make_masks()
    elif args.command == 'torch-dataset':
        ds = CustomDataset()
        ds.split_save_dataset()
    elif args.command == 'train':
        trainer = Trainer(train_file=args.train_file, val_file=args.val_file, test_file=args.test_file)
        trainer.count_parameters()
        trainer.run(end_epoch=args.n_epochs, resume_epoch=args.resume_epoch)
    elif args.command == 'eval':
        trainer = Trainer(val_file=args.val_file)
        trainer.eval(epoch=args.epoch, maps_file=args.maps_file)
    elif args.command == 'test':
        trainer = Trainer(test_file=args.test_file)
        trainer.test(epoch=args.epoch, maps_file=args.maps_file)
    elif args.command == 'predict':
        trainer = Trainer()
        trainer.predict(pred_file=args.pred_file, epoch=args.epoch, out_file=args.output, maps_file=args.maps_file)

    elif args.command == 'format-output':
        ar = Array()
        ar.format_output(in_file=args.input, out_file=args.output, digit=args.digit, from_tool=args.from_tool)
    elif args.command == 'run-snp2hla':
        ar = Array()
        ar.run_snp2hla(in_file=args.input, ref_file=args.ref)
    elif args.command == 'run-deephla':
        ar = Array()
        ar.run_deephla(mode=args.mode, in_file=args.input, ref_file=args.ref, subset=args.subset,
                       model_json=args.model_json, model_dir=args.model_dir)

if __name__ == '__main__':
    main()
