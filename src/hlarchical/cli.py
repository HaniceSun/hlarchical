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
    p1.add_argument('--ref_bim', type=str, default='Pan-Asian_REF.bim', help='reference BIM file')
    p1.add_argument('--sample_bim', type=str, default='1958BC.bim', help='sample BIM file')
    p1.add_argument('--ref_phased', type=str, default='Pan-Asian_REF.bgl.phased', help='reference phased file')
    p1.add_argument('--hla_renaming', type=str, default='true', help='whether to rename HLA alleles to HLA-A:01:01 format')
    p1.add_argument('--expert_by', type=str, default='ld', help='expert by gene or by ld')
    p1.add_argument('--expert_flank', type=int, default=500000, help='flanking size for experts')

    p1.add_argument('--id_by', type=str, default='rs', help='whether to make features by rsID or position')
    p1.add_argument('--check_alleles', type=str, default='true', help='whether to check alleles between reference and sample')
    p1.add_argument('--subset_sample_bim', type=str, default=None, help='subset sample BIM file to the HLA region')

    p1.add_argument('--features_file', type=str, default='features.txt', help='output features file')
    p1.add_argument('--labels_file', type=str, default='labels.txt', help='output labels file')
    p1.add_argument('--maps_file', type=str, default='maps.txt', help='output maps file')
    p1.add_argument('--masks_file', type=str, default='masks.txt', help='output masks file')

    p2 = subparsers.add_parser("torch-dataset", help="generate torch datasets")
    p2.add_argument('--features_file', type=str, default='features.txt', help='input features file')
    p2.add_argument('--labels_file', type=str, default='labels.txt', help='input labels file')
    p2.add_argument('--maps_file', type=str, default='maps.txt', help='input maps file')
    p2.add_argument('--split_ratio', type=str, default='0.8,0.1,0.1', help='split ratio for train, val, test datasets')
    p2.add_argument('--n_cv', type=int, default=0, help='number of cross-validation folds')

    p3 = subparsers.add_parser("train", help="train model")
    p3.add_argument('--config_file', type=str, default='config.yaml', help='the config file for model and training parameters')
    p3.add_argument('--model_name', type=str, default='mlp', help='the model name defined in the config file')
    p3.add_argument('--train_file', type=str, default='hla_dataset_train.pt', help='training dataset file')
    p3.add_argument('--val_file', type=str, default='hla_dataset_val.pt', help='validation dataset file')
    p3.add_argument('--test_file', type=str, default='hla_dataset_test.pt', help='test dataset file')
    p3.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
    p3.add_argument('--resume_epoch', type=str, default=None, help='resume training from a specific epoch if specified')

    p4 = subparsers.add_parser("eval", help="evaluate a trained model")
    p4.add_argument('--config_file', type=str, default='config.yaml', help='the config file for model and training parameters')
    p4.add_argument('--model_name', type=str, default='mlp', help='the model name defined in the config file')
    p4.add_argument('--epoch', type=int, default=99, help='the epoch of the trained model to be used for prediction')
    p4.add_argument('--val_file', type=str, default='hla_dataset_val.pt', help='val dataset file')
    p4.add_argument('--maps_file', type=str, default='maps.txt', help='the maps file from preprocessing step')

    p5 = subparsers.add_parser("test", help="test a trained model")
    p5.add_argument('--config_file', type=str, default='config.yaml', help='the config file for model and training parameters')
    p5.add_argument('--model_name', type=str, default='mlp', help='the model name defined in the config file')
    p5.add_argument('--epoch', type=int, default=99, help='the epoch of the trained model to be used for prediction')
    p5.add_argument('--test_file', type=str, default='hla_dataset_test.pt', help='test dataset file')
    p5.add_argument('--maps_file', type=str, default='maps.txt', help='the maps file from preprocessing step')

    p6 = subparsers.add_parser("prepare-to-predict", help="prepare data for prediction")
    p6.add_argument('--features_file', type=str, default='features.txt', help='output features file')
    p6.add_argument('--sample_phased', type=str, default='sample.bgl.phased', help='sample phased on the reference panel')
    p6.add_argument('--output', type=str, default='to_predict.txt', help='output file for prediction')

    p7 = subparsers.add_parser("predict", help="predicted using the trained model")
    p7.add_argument('--config_file', type=str, default='config.yaml', help='the config file for model and training parameters')
    p7.add_argument('--model_name', type=str, default='mlp', help='the model name defined in the config file')
    p7.add_argument('--pred_file', type=str, default='to_predict.txt', help='input dataset file for prediction')
    p7.add_argument('--maps_file', type=str, default='maps.txt', help='the maps file from preprocessing step')
    p7.add_argument('--epoch', type=int, default=99, help='the epoch of the trained model to be used for prediction')
    p7.add_argument('--output', type=str, default='hla_predicted.txt', help='output prediction file')

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
        hla_renaming = args.hla_renaming.lower() in ('true', '1', 'yes')
        check_alleles = args.check_alleles.lower() in ('true', '1', 'yes')
        subset_sample_bim=args.subset_sample_bim
        if subset_sample_bim is not None:
            subset_sample_bim = subset_sample_bim.replace('-', ':').split(':')

        dp = DataPreprocessor(ref_bim=args.ref_bim, sample_bim=args.sample_bim, ref_phased=args.ref_phased,
                              hla_renaming=hla_renaming, expert_by=args.expert_by)
        dp.make_features(id_by=args.id_by, check_alleles=check_alleles, subset_sample_bim=subset_sample_bim)
        dp.make_labels(out_file=args.labels_file, maps_file=args.maps_file)
        dp.make_masks(out_file=args.masks_file, features_file=args.features_file, flank=args.expert_flank)
    elif args.command == 'torch-dataset':
        ds = CustomDataset(features_file=args.features_file, labels_file=args.labels_file, maps_file=args.maps_file)
        split_ratio = [float(x) for x in args.split_ratio.split(',')]
        ds.split_save_dataset(ratio=split_ratio, n_cv=args.n_cv)
    elif args.command == 'train':
        trainer = Trainer(config_file=args.config_file, model_name=args.model_name, train_file=args.train_file, val_file=args.val_file, test_file=args.test_file)
        trainer.count_parameters()
        trainer.run(end_epoch=args.n_epochs, resume_epoch=args.resume_epoch)
    elif args.command == 'eval':
        trainer = Trainer(config_file=args.config_file, model_name=args.model_name, val_file=args.val_file)
        trainer.eval(epoch=args.epoch, maps_file=args.maps_file)
    elif args.command == 'test':
        trainer = Trainer(config_file=args.config_file, model_name=args.model_name, test_file=args.test_file)
        trainer.test(epoch=args.epoch, maps_file=args.maps_file)
    elif args.command == 'prepare-to-predict':
        dp = DataPreprocessor()
        dp.prepare_to_predict(sample_phased=args.sample_phased, features_file=args.features_file, out_file=args.output)
    elif args.command == 'predict':
        trainer = Trainer(config_file=args.config_file, model_name=args.model_name)
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
