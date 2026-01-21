import argparse
from .preprocess import DataPreprocessor
from .dataset import CustomDataset
from .trainer import Trainer
from .utils import *

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

if __name__ == '__main__':
    main()
