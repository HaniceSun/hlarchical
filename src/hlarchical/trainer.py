from .dataset import *
from .models import *
from .utils import *

class Trainer:
    def __init__(self, config_file='config.yaml', model_name='mlp', train_file=None, val_file=None, test_file=None,
                 metrics_file=None, lr_lambda=None, print_every_n_batches=100):
        self.config = self.load_yaml(config_file)

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if train_file and os.path.exists(train_file):
            self.train_dataset = torch.load(train_file, weights_only=False)
        if val_file and os.path.exists(val_file):
            self.val_dataset = torch.load(val_file, weights_only=False)
        if test_file and os.path.exists(test_file):
            self.test_dataset = torch.load(test_file, weights_only=False)

        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'using device: {self.device}', flush=True)
        model_class = eval(self.config['models'][model_name]['class'])
        cfg = Config(self.config['models'][model_name]['params'])
        cfg.device = self.device
        self.model = model_class(cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.loss_fn = CustomLoss(cfg=cfg)
        self.print_every_n_batches = print_every_n_batches


        self.metrics_file = metrics_file
        if not self.metrics_file:
            self.metrics_file = f'{model_name}_metrics.txt'

        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.val_accuracy = []
        self.test_accuracy = []

        self.learning_rates = []
        self.lr_scheduler = None
        if lr_lambda:
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_lambda[epoch])

        self.early_stopping = None
        if cfg.early_stopping:
            self.early_stopping = EarlyStopping()
        self.best_epochs = []

    def train(self, epoch):
        self.model.train()

        loss_total = 0
        for nb, (X, y) in enumerate(self.train_dataset):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_total += loss.detach().item()

            if nb % self.print_every_n_batches == 0:
                print(f'train epoch:{epoch} batch:{nb} loss:{loss.detach().item()}')

        self.save_checkpoint(epoch)
        train_loss = loss_total / len(self.train_dataset)
        self.train_loss.append(train_loss)
        self.epochs.append(epoch)

        if self.lr_scheduler:
            lr = self.lr_scheduler.get_last_lr()[0]
            self.learning_rates.append(lr)
        print(f'train epoch:{epoch} avg loss: {self.train_loss[-1]}')

    def validate(self, epoch, test=False):
        ds = 'val'
        dataset = self.val_dataset
        loss_list = self.val_loss
        if test:
            ds = 'test'
            dataset = self.test_dataset
            loss_list = self.test_loss
            self.epochs.append(epoch)

        self.model.eval()
        loss_total = 0
        with torch.no_grad():
            for nb, (X, y) in enumerate(dataset):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                loss_total += loss.detach().item()

                if nb % self.print_every_n_batches == 0:
                    print(f'{ds} epoch:{epoch} batch:{nb} loss:{loss.detach().item()}', flush=True)
        loss_avg = loss_total / len(dataset)
        loss_list.append(loss_avg)
        print(f'{ds} epoch:{epoch} avg loss: {loss_list[-1]}')

    def eval(self, epoch, maps_file='maps.txt', test=False):
        dataset = self.val_dataset
        accuracy = self.val_accuracy
        if test:
            dataset = self.test_dataset
            accuracy = self.test_accuracy

        self.load_checkpoint(epoch)

        df = pd.read_table(maps_file, header=0, sep='\t')
        digits = {}
        heads = {}
        for n in range(df.shape[0]):
            head = df['head'].iloc[n]
            head_idx = df['head_idx'].iloc[n]
            label = df['label'].iloc[n]
            heads[head] = [head_idx, label]

        for digit in df['digit'].unique():
            df_sub = df[df['digit'] == digit]
            digits[digit] = df_sub['head'].unique().tolist()

        self.metrics = {}
        for head in heads:
            self.metrics.setdefault(head, {})
            self.metrics[head]['accuracy'] = CustomAccuracy(num_classes=int(heads[head][1] + 1))
            self.metrics[head]['auroc'] = MulticlassAUROC(num_classes=int(heads[head][1] + 1), average="macro")
            self.metrics[head]['confusion'] = MulticlassConfusionMatrix(num_classes=int(heads[head][1] + 1))

        self.model.eval()
        with torch.no_grad():
            for nb, (X, y) in enumerate(dataset):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                for head in outputs:
                    y_p = outputs[head]
                    head_idx = heads[head][0]
                    y_t = y[:, :, head_idx]
                    self.metrics[head]['accuracy'].update(y_p, y_t)
                    self.metrics[head]['auroc'].update(y_p, y_t)
                    self.metrics[head]['confusion'].update(y_p, y_t)

        accuracy_dict = {}
        for head in self.metrics:
            acc = self.metrics[head]['accuracy'].compute()
            accuracy_dict[head] = acc
            auroc = self.metrics[head]['auroc'].compute()
            confusion = self.metrics[head]['confusion'].compute()
            print(f'head: {head} accuracy: {acc[0]:.4f} correct: {acc[1]} total: {acc[2]} auroc: {auroc:.4f}')

        accuracy_digit = accuracy_avg_digit(accuracy_dict, digits)
        acc = ', '.join([f'digit{k}:{accuracy_digit[k]:.4f}' for k in accuracy_digit])
        accuracy.append(acc)
        print(f'accuracy per digit: {acc}')
        self.log_metrics()

    def test(self, epoch, maps_file='maps.txt'):
        self.eval(epoch, maps_file=maps_file, test=True)

    def predict(self, epoch=None, pred_file='to_predict.txt', out_file='predicted.txt', maps_file='maps.txt'):
        df = pd.read_table(maps_file, header=0, sep='\t')
        maps = {}
        for n in range(df.shape[0]):
            head = df['head'].iloc[n]
            allele = df['allele'].iloc[n]
            label = df['label'].iloc[n]
            maps.setdefault(head, {})
            maps[head][label] = allele

        df = pd.read_table(pred_file, header=0, sep='\t')
        mat = df.iloc[:, 1:].values
        mat = mat.reshape(mat.shape[0], mat.shape[1]//2, 2)
        X = torch.tensor(mat, dtype=torch.float32).permute(0, 2, 1)
        self.X = X.to(self.device)

        ouFile = open(out_file, 'w')
        ouFile.write('SampleID\tHLA\tAllele1\tAllele2\n')

        self.load_checkpoint(epoch)
        self.model.eval()
        with torch.no_grad():
            for n in range(df.shape[0]):
                outputs = self.model(self.X[n:n+1])
                for head in outputs:
                    pred = torch.softmax(outputs[head], dim=1)
                    clss = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
                    clss = [int(x) for x in clss]
                    allele1 = maps[head].get(clss[0], '.')
                    allele2 = maps[head].get(clss[1], '.')
                    if allele1 != '.' or allele2 != '.':
                        ouFile.write('\t'.join([df.iloc[n, 0], head, allele1, allele2]) + '\n')
        ouFile.close()

    def log_metrics(self):
        cols = ['epoch', 'best_epoch', 'learning_rate',
                      'train_loss', 'val_loss', 'test_loss',
                      'train_accuracy', 'val_accuracy', 'test_accuracy']
        df = pd.DataFrame('.', index=range(len(self.epochs)), columns=cols)

        df['epoch'] = self.epochs
        if self.train_loss:
            df['train_loss'] = self.train_loss
        if self.val_loss:
            df['val_loss'] = self.val_loss
        if self.test_loss:
            df['test_loss'] = self.test_loss

        if self.train_accuracy:
            df['train_accuracy'] = self.train_accuracy
        if self.val_accuracy:
            df['val_accuracy'] = self.val_accuracy
        if self.test_accuracy:
            df['test_accuracy'] = self.test_accuracy

        if self.learning_rates:
            df['learning_rate'] = self.learning_rates
        if self.best_epochs:
            df['best_epoch'] = self.best_epochs

        if os.path.exists(self.metrics_file):
            df.tail(1).to_csv(self.metrics_file, index=False, header=False, sep='\t', mode='a')
        else:
            df.tail(1).to_csv(self.metrics_file, index=False, header=True, sep='\t')

        if self.train_loss and self.val_loss:
            plot_file = self.metrics_file.replace('.txt', '.png')
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(df['epoch'], df['train_loss'], label='train')
            ax.plot(df['epoch'], df['val_loss'], label='val')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()

    def count_parameters(self, with_lazy=True, show_details=False):
        if with_lazy:
            X,y = next(iter(self.train_dataset))
            pred = self.model.forward(X)

        L = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                pnum = parameter.numel()
                L.append([name, pnum])
        df = pd.DataFrame(L)
        df.columns = ['name', 'pnum']
        if show_details:
            print(df)
        print(f'Total Trainable Parameters: {df["pnum"].sum()}')

    def set_learning_rate(self, lr=1e-3):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, epoch):
        checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }
        out_file = f'{self.model_name}_ckpt_{epoch}.pt'
        torch.save(checkpoint, out_file) 

    def load_checkpoint(self, epoch):
        in_file = f'{self.model_name}_ckpt_{epoch}.pt'
        if os.path.exists(in_file):
            print(f'loading checkpoint from {in_file}')
            checkpoint = torch.load(in_file, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print(f'checkpoint file {in_file} not found!')

    def load_yaml(self, config_file):
        if not os.path.exists(config_file):
            config_file = config_dir + '/' + config_file
            print(f'using config {config_file}')
        config = {}
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f'Error loading config file {e}')

    def run(self, resume_epoch=None, start_epoch=0, end_epoch=10, validate=True):
        if resume_epoch is not None:
            self.load_checkpoint(resume_epoch)
            start_epoch = resume_epoch + 1

        for epoch in range(start_epoch, end_epoch):
            self.train(epoch)
            if validate:
                self.validate(epoch)
                if self.early_stopping:
                    self.early_stopping(self.val_loss[-1], epoch)
                    self.best_epochs.append(self.early_stopping.best_epoch)

                    if self.early_stopping.stopped:
                        print(f'Early stopped, beast epoch: {self.early_stopping.best_epoch}')
                        break

            if self.lr_scheduler:
                self.lr_scheduler.step()

            self.log_metrics()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.count_parameters()
    trainer.run()
    trainer.test()
    trainer.predict()
