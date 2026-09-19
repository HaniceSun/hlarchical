"""Training, evaluation, and prediction driver for the hierarchical HLA imputation model."""

from typing import Any

from .dataset import *
from .models import *
from .utils import *


class Trainer:
    """Orchestrates model construction, training, evaluation, and prediction.

    On construction, ``Trainer`` loads a YAML config describing the model
    architecture and hyperparameters, resolves the label maps / expert masks
    files (falling back to the packaged with/without-ancestry data
    directory when not found in the current directory), optionally loads
    pre-saved train/val/test ``DataLoader`` objects, builds the model named
    in the config via ``eval(...)`` on the configured class name, and sets
    up the optimizer, loss function, optional LR scheduler, and optional
    early stopping.

    Attributes:
        config: Parsed YAML config dict (``models.<model_name>.class`` and
            ``models.<model_name>.params`` describe the model to build).
        HLA: The eight HLA gene names handled by ``predict``'s digit-2
            fallback table.
        data_dir: Packaged data directory for the chosen ancestry mode
            (``with_ancestry`` or ``without_ancestry``), used as a fallback
            location for maps/masks/checkpoint files.
        maps_file: Resolved path to the label maps file.
        masks_file: Resolved path to the expert masks file, or ``None``.
        train_dataset: Loaded training ``DataLoader``, or ``None``.
        val_dataset: Loaded validation ``DataLoader``, or ``None``.
        test_dataset: Loaded test ``DataLoader``, or ``None``.
        model_name: Key of the model config used, e.g. ``"mlp"``.
        device: ``torch.device`` selected (CUDA, MPS, or CPU).
        model: The instantiated model (``nn.Module``), moved to ``device``.
        optimizer: ``torch.optim.AdamW`` optimizer over ``model``'s
            parameters.
        loss_fn: ``CustomLoss`` instance used for training/validation.
        print_every_n_batches: Batch-logging interval.
        metrics_file: Path the per-epoch metrics table is appended to.
        epochs: List of epoch indices recorded so far.
        train_loss: Per-epoch average training loss.
        val_loss: Per-epoch average validation loss.
        test_loss: Per-epoch average test loss.
        train_accuracy: Per-epoch training accuracy summaries.
        val_accuracy: Per-epoch validation accuracy summaries.
        test_accuracy: Per-epoch test accuracy summaries.
        learning_rates: Per-epoch learning rate, if an LR scheduler is used.
        lr_scheduler: Optional ``torch.optim.lr_scheduler.LambdaLR``
            instance.
        early_stopping: Optional ``EarlyStopping`` instance.
        best_epochs: Best-epoch-so-far recorded at each early-stopping
            check.
    """

    def __init__(
        self,
        config_file: str = 'config.yaml',
        model_name: str = 'mlp',
        train_file: str | None = None,
        val_file: str | None = None,
        test_file: str | None = None,
        with_ancestry: bool = False,
        maps_file: str = 'maps.txt',
        masks_file: str | None = 'masks.txt',
        metrics_file: str | None = None,
        lr_lambda: list[float] | None = None,
        print_every_n_batches: int = 100,
    ) -> None:
        """Loads config/data and builds the model, optimizer, and loss.

        Args:
            config_file: Path to a YAML file describing model architectures
                and hyperparameters. If not found in the current directory,
                it is looked up under the packaged config directory.
            model_name: Key into ``config['models']`` selecting which model
                architecture/hyperparameters to use.
            train_file: Path to a saved training ``DataLoader`` (via
                ``torch.save``). Skipped if ``None`` or missing.
            val_file: Path to a saved validation ``DataLoader``. Skipped if
                ``None`` or missing.
            test_file: Path to a saved test ``DataLoader``. Skipped if
                ``None`` or missing.
            with_ancestry: Whether the model/data uses ancestry features;
                selects which packaged data subdirectory is used as a
                fallback for maps/masks/checkpoint files.
            maps_file: Path to the label maps file (see ``Processor.make_maps``).
            masks_file: Path to the expert masks file (see
                ``Processor.make_masks``), or ``None`` to proceed without
                masks (requires ``input_length`` in the model config).
            metrics_file: Path to append per-epoch metrics to. Defaults to
                ``f'{model_name}_metrics.txt'``.
            lr_lambda: Per-epoch multiplicative LR factors indexed by epoch,
                passed to ``torch.optim.lr_scheduler.LambdaLR``. Note:
                ``LambdaLR`` is not imported in this module, so passing a
                non-``None`` value here will raise ``NameError`` at
                construction time (pre-existing behavior, unchanged here).
            print_every_n_batches: How often (in batches) to print
                progress during training/validation.

        Raises:
            FileNotFoundError: If ``maps_file`` or ``masks_file`` cannot be
                found either as given or under the packaged data directory.
        """
        self.config = self.load_yaml(config_file)
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        if with_ancestry:
            self.data_dir = data_dir + '/with_ancestry'
        else:
            self.data_dir = data_dir + '/without_ancestry'

        self.maps_file = maps_file
        if not os.path.exists(maps_file):
            self.maps_file = self.data_dir + '/' + maps_file
            if not os.path.exists(self.maps_file):
                raise FileNotFoundError(f'maps file {maps_file} or {self.maps_file} not found!')
        else:
            print(f'using maps file {maps_file}')

        self.masks_file = masks_file
        if masks_file is None:
            print('proceeding without using masks')
        elif not os.path.exists(masks_file):
            self.masks_file = self.data_dir + '/' + masks_file
            if not os.path.exists(self.masks_file):
                raise FileNotFoundError(f'masks file {masks_file} or {self.masks_file} not found!')
        else:
            print(f'using masks file {masks_file}')

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        if train_file and os.path.exists(train_file):
            self.train_dataset = torch.load(train_file, weights_only=False)
        if val_file and os.path.exists(val_file):
            self.val_dataset = torch.load(val_file, weights_only=False)
        if test_file and os.path.exists(test_file):
            self.test_dataset = torch.load(test_file, weights_only=False)

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        print(f'using device: {self.device}', flush=True)
        model_class = eval(self.config['models'][model_name]['class'])
        cfg = Config(self.config['models'][model_name]['params'])
        cfg.device = self.device
        self.model = model_class(cfg, maps_file=self.maps_file, masks_file=self.masks_file).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        self.loss_fn = CustomLoss(cfg=cfg, maps_file=self.maps_file)
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

    def train(self, epoch: int) -> None:
        """Runs one training epoch over ``self.train_dataset``.

        Updates model parameters via backpropagation, logs batch-level
        loss every ``print_every_n_batches`` batches, saves a checkpoint
        for this epoch, appends the epoch's average loss to
        ``self.train_loss``/``self.epochs``, and records the current
        learning rate if an LR scheduler is configured.

        Args:
            epoch: Index of the current epoch (used for logging and
                checkpoint naming).
        """
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

    def validate(self, epoch: int, test: bool = False) -> None:
        """Runs one evaluation pass (no gradient updates) and logs average loss.

        Args:
            epoch: Index of the current epoch (used for logging; also
                appended to ``self.epochs`` when ``test=True``, since
                ``test`` runs are not preceded by a ``train`` call in the
                same epoch).
            test: If ``True``, evaluate ``self.test_dataset`` and append to
                ``self.test_loss`` instead of evaluating
                ``self.val_dataset`` and appending to ``self.val_loss``.
        """
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

    def eval(self, epoch: int, test: bool = False) -> None:
        """Loads a checkpoint and computes per-head accuracy/AUROC/confusion metrics.

        Loads the checkpoint for ``epoch``, computes, for every label head
        defined in ``self.maps_file``, a ``CustomAccuracy``,
        ``MulticlassAUROC``, and ``MulticlassConfusionMatrix`` over the
        chosen dataset, prints per-head results, aggregates accuracy by
        digit level via ``accuracy_avg_digit``, appends the formatted
        digit-level accuracy summary to ``self.val_accuracy`` or
        ``self.test_accuracy``, and calls ``self.log_metrics()``.

        Args:
            epoch: Index of the checkpoint epoch to load and evaluate.
            test: If ``True``, evaluate ``self.test_dataset`` and append to
                ``self.test_accuracy`` instead of ``self.val_dataset`` /
                ``self.val_accuracy``.
        """
        dataset = self.val_dataset
        accuracy = self.val_accuracy
        if test:
            dataset = self.test_dataset
            accuracy = self.test_accuracy

        self.load_checkpoint(epoch)

        df = pd.read_table(self.maps_file, header=0, sep='\t')
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

    def test(self, epoch: int) -> None:
        """Evaluates the test dataset for the given epoch.

        Convenience wrapper around ``self.eval(epoch, test=True)``.

        Args:
            epoch: Index of the checkpoint epoch to load and evaluate.
        """
        self.eval(epoch, test=True)

    def predict(
        self,
        epoch: int | None = None,
        pred_file: str = 'to_predict.txt',
        out_file: str = 'predicted.txt',
        split_by_digit: bool = True,
    ) -> None:
        """Predicts HLA alleles for samples in ``pred_file`` and writes results.

        Loads the checkpoint for ``epoch``, reads the feature file
        ``pred_file`` (same format produced by
        ``Processor.get_sample_features``), runs the model per sample, and
        writes ``SampleID``, ``HLA``, ``Allele1``, ``Allele2`` rows to
        ``out_file`` for every head whose predicted class decodes to a
        real allele (label 0 decodes to ``'.'`` and is skipped).

        If ``split_by_digit`` is set, additionally builds a digit-2-level
        table covering all eight HLA genes (with ``'.'`` placeholders for
        genes absent from the model's heads), and iteratively resolves
        digit-4/6/8 predictions from the previous digit level's output
        file (``out_file`` with ``_digit{digit}`` inserted before the
        extension), writing one file per digit level present in the
        predictions.

        Args:
            epoch: Index of the checkpoint epoch to load.
            pred_file: Path to the tab-separated feature file to predict
                on.
            out_file: Path to write the raw per-head predictions to.
            split_by_digit: Whether to additionally emit digit-level
                cascaded prediction files.
        """
        df = pd.read_table(self.maps_file, header=0, sep='\t')
        maps = {}
        for n in range(df.shape[0]):
            head = df['head'].iloc[n]
            allele = df['allele'].iloc[n]
            label = df['label'].iloc[n]
            maps.setdefault(head, {})
            maps[head][label] = allele

        df = pd.read_table(pred_file, header=0, sep='\t')
        mat = []
        for n in range(1, df.shape[1]):
            mat.append(df.iloc[:, n].str.split('|', expand=True).astype(int).values)
        mat = np.array(mat).transpose(1, 2, 0)
        X = torch.tensor(mat, dtype=torch.float32)
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

        if split_by_digit:
            df = pd.read_table(out_file, header=0, sep='\t')
            D = {}
            digits = []
            for n in range(df.shape[0]):
                sample_id = df['SampleID'].iloc[n]
                hla = df['HLA'].iloc[n]
                digit = len(hla.split(':')) * 2
                if digit not in digits:
                    digits.append(digit)
                allele1 = df['Allele1'].iloc[n]
                allele2 = df['Allele2'].iloc[n]
                D.setdefault(sample_id, {})
                D[sample_id][hla] = (allele1, allele2)

            for digit in sorted(digits):
                L = []
                if digit == 2:
                    for sample_id in D:
                        for hla in self.HLA:
                            allele1, allele2 = D[sample_id].get(hla, ('.', '.'))
                            L.append([sample_id, hla, allele1, allele2])
                else:
                    df = pd.read_table(out_file.replace('.txt', f'_digit{digit - 2}.txt'), header=0, sep='\t')
                    for n in range(df.shape[0]):
                        sample_id = df['SampleID'].iloc[n]
                        hla = df['HLA'].iloc[n]
                        allele1 = df['Allele1'].iloc[n]
                        allele2 = df['Allele2'].iloc[n]
                        allele1_d4 = D[sample_id].get(allele1, ('.', '.'))[0]
                        allele2_d4 = D[sample_id].get(allele2, ('.', '.'))[1]
                        L.append([sample_id, hla, allele1_d4, allele2_d4])
                if len(L) > 0:
                    df_out = pd.DataFrame(L)
                    df_out.columns = ['SampleID', 'HLA', 'Allele1', 'Allele2']
                    df_out.to_csv(out_file.replace('.txt', f'_digit{digit}.txt'), index=False, sep='\t')

    def log_metrics(self) -> None:
        """Appends the latest epoch's metrics to ``self.metrics_file`` and plots loss curves.

        Builds a one-row (well, cumulative-history) table of epoch,
        best_epoch, learning_rate, train/val/test loss, and train/val/test
        accuracy, writing only the last row to ``self.metrics_file``
        (creating it with a header if it doesn't exist, else appending).
        If both training and validation loss have been recorded, also
        (re)plots the train/val loss curves to a PNG alongside the
        metrics file.
        """
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

    def count_parameters(self, with_lazy: bool = True, show_details: bool = False) -> None:
        """Prints the total number of trainable model parameters.

        Args:
            with_lazy: If ``True``, runs one forward pass first so that any
                lazily-initialized layers (e.g. ``nn.LazyLinear``) have
                their parameters materialized before counting.
            show_details: If ``True``, also prints a per-parameter table of
                name and element count.
        """
        if with_lazy:
            X, y = next(iter(self.train_dataset))
            X, y = X.to(self.device), y.to(self.device)
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

    def set_learning_rate(self, lr: float = 1e-3) -> None:
        """Sets the learning rate on every optimizer parameter group.

        Args:
            lr: New learning rate to apply.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, epoch: int) -> None:
        """Saves model and optimizer state to ``f'{model_name}_ckpt_{epoch}.pt'``.

        Args:
            epoch: Epoch index used to construct the checkpoint filename.
        """
        checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }
        out_file = f'{self.model_name}_ckpt_{epoch}.pt'
        torch.save(checkpoint, out_file)

    def load_checkpoint(self, epoch: int) -> None:
        """Loads model and optimizer state from a checkpoint file for the given epoch.

        Looks for ``f'{model_name}_ckpt_{epoch}.pt'`` in the current
        directory, falling back to ``self.data_dir``.

        Args:
            epoch: Epoch index used to construct the checkpoint filename.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found
                either in the current directory or under ``self.data_dir``.
        """
        ckpt_file = f'{self.model_name}_ckpt_{epoch}.pt'
        in_file = ckpt_file
        if not os.path.exists(ckpt_file):
            in_file = self.data_dir + '/' + ckpt_file
            if not os.path.exists(in_file):
                raise FileNotFoundError(f'checkpoint file {ckpt_file} or {in_file} not found!')
        print(f'loading checkpoint from {in_file}')
        checkpoint = torch.load(in_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def load_yaml(self, config_file: str) -> dict[str, Any]:
        """Loads and parses a YAML config file.

        Args:
            config_file: Path to the YAML config. If not found in the
                current directory, it is looked up under the packaged
                config directory (``config_dir``).

        Returns:
            The parsed YAML content as a dict, or an empty dict if loading
            fails (the exception is caught and logged, not re-raised).
        """
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

    def run(
        self,
        resume_epoch: int | None = None,
        start_epoch: int = 0,
        end_epoch: int = 10,
        validate: bool = True,
    ) -> None:
        """Runs the full training loop from ``start_epoch`` to ``end_epoch``.

        For each epoch, trains one pass, optionally validates and checks
        early stopping, steps the LR scheduler if configured, and logs
        metrics. If ``resume_epoch`` is given, loads that checkpoint first
        and continues training from the following epoch.

        Args:
            resume_epoch: If given, load this epoch's checkpoint before
                training and set ``start_epoch`` to ``resume_epoch + 1``.
            start_epoch: First epoch index to train (inclusive), unless
                overridden by ``resume_epoch``.
            end_epoch: Last epoch index to train (exclusive).
            validate: Whether to run validation and early-stopping checks
                after each training epoch.
        """
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
