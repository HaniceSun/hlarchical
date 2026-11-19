import os
import numpy as np
import pandas as pd
import gzip
import subprocess
import yaml
import shutil
import glob
import torch
torch.manual_seed(42)
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import seaborn as sns
from importlib import resources
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassAUROC
from torchmetrics.classification import MulticlassAccuracy

config_dir = f'{resources.files("hlarchical")}/config'
data_dir = f'{resources.files("hlarchical")}/data'


class CustomLoss(torch.nn.Module):
    """Cross-entropy loss over a set of hierarchical HLA prediction heads.

    Sums a standard multi-class cross-entropy loss across every prediction
    head. Optionally supports "ground truth gated" loss, where loss for a
    child head is only accumulated for samples whose parent head's ground
    truth label matches the value required to "activate" that child (as
    defined in the maps file), which mirrors the hierarchical allele digit
    structure (e.g. a 4-digit allele head is only meaningful when the
    2-digit parent head has the corresponding value).

    Attributes:
        loss_fn: The underlying `torch.nn.CrossEntropyLoss` instance.
        ground_truth_gated_loss: Whether gated hierarchical loss is enabled.
        gated_loss_lambda: Scaling factor applied to the gated (child) loss
            term when `ground_truth_gated_loss` is enabled.
        maps: Mapping from head name to `[head_idx, parent, parent_val]`,
            loaded from `maps_file` when `cfg` is provided.
    """

    def __init__(self, cfg: object | None = None, maps_file: str = 'maps.txt') -> None:
        """Initializes the loss and, if configured, loads the head hierarchy.

        Args:
            cfg: Optional config object. If provided, `maps_file` is loaded
                to build the head hierarchy, and `cfg.ground_truth_gated_loss`
                / `cfg.gated_loss_lambda` (if present) control gated loss
                behavior. If `None`, the loss behaves as plain per-head
                cross-entropy and `maps_file` is not read.
            maps_file: Path to a tab-separated file with columns `head`,
                `head_idx`, `parent`, `parent_val`, describing the head
                hierarchy. Only read when `cfg` is truthy.

        Raises:
            FileNotFoundError: If `cfg` is provided and `maps_file` does not
                exist.
        """
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.ground_truth_gated_loss = False
        self.gated_loss_lambda = 1.0
        self.maps = {}
        if cfg:
            if not os.path.exists(maps_file):
                raise FileNotFoundError(f'{maps_file} not found')
            df = pd.read_table(maps_file, header=0, sep='\t')
            for n in range(df.shape[0]):
                head = df['head'].iloc[n]
                head_idx = df['head_idx'].iloc[n]
                parent = df['parent'].iloc[n]
                parent_val = df['parent_val'].iloc[n]
                self.maps[head] = [head_idx, parent, parent_val]

            if hasattr(cfg, 'ground_truth_gated_loss') and cfg.ground_truth_gated_loss:
                self.ground_truth_gated_loss = True
                if hasattr(cfg, 'gated_loss_lambda'):
                    self.gated_loss_lambda = cfg.gated_loss_lambda
                print(f'Using ground truth gated loss with lambda {self.gated_loss_lambda}')

    def forward(
        self,
        y_pred: dict[str, torch.Tensor],
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the (optionally gated) summed cross-entropy loss.

        Args:
            y_pred: Mapping from head name to prediction logits of shape
                `(batch, num_classes, 2)` (one set of logits per allele copy).
            y_true: Ground truth labels of shape `(batch, 2, num_heads)`,
                indexed along the last dimension by each head's `head_idx`.

        Returns:
            A scalar tensor with the total loss across all heads. When
            `ground_truth_gated_loss` is enabled, this is the sum of the
            root-level loss and the lambda-scaled, batch-averaged gated
            child loss.
        """
        loss = torch.tensor(0.0, device=y_true.device)
        if not self.ground_truth_gated_loss:
            for head in y_pred:
                y_p = y_pred[head]
                head_idx = self.maps[head][0]
                y_t = y_true[:, :, head_idx]
                ls = self.loss_fn(y_p, y_t)
                loss += ls
        else:
            loss1 = torch.tensor(0.0, device=y_true.device)
            loss2 = torch.tensor(0.0, device=y_true.device)
            for head in y_pred:
                y_p = y_pred[head]
                head_idx = self.maps[head][0]
                y_t = y_true[:, :, head_idx]
                parent = self.maps[head][1]
                parent_idx = self.maps[parent][0]
                parent_val = self.maps[head][2]
                if parent == '.':
                    ls = self.loss_fn(y_p, y_t)
                    loss1 += ls
                else:
                    for i in range(y_t.shape[0]):
                        for j in range(y_t.shape[1]):
                            if y_true[i, j, parent_idx] == parent_val:
                                ls = self.loss_fn(y_p[i, :, j].unsqueeze(0), y_t[i, j].unsqueeze(0))
                                loss2 += ls
            loss = loss1 + loss2 / y_true.shape[0] * self.gated_loss_lambda
        return loss


class Config:
    """A simple attribute-style wrapper around a config dictionary.

    Converts a plain `dict` (typically loaded from YAML) into an object
    whose keys are accessible as attributes, applying a couple of
    field-specific type coercions along the way.

    Attributes:
        Arbitrary attributes are set dynamically from the keys of the
        `config` dict passed to `__init__`.
    """

    def __init__(self, config: dict[str, object]) -> None:
        """Sets one attribute per key/value pair in `config`.

        Args:
            config: Mapping of config keys to values. Values for the keys
                `learning_rate` and `weight_decay` are cast to `float`.
                The value for the key `NWD` is evaluated with `eval` (e.g.
                to turn a stringified list-of-tuples into a real Python
                object). All other values are set as-is.
        """
        for k, v in config.items():
            if k in ['learning_rate', 'weight_decay']:
                v = float(v)
            elif k in ['NWD']:
                v = eval(v)
            setattr(self, k, v)


class EarlyStopping:
    """Tracks validation loss across epochs and signals when to stop training.

    Stops (sets `stopped = True`) once the validation loss has failed to
    improve by more than `delta` for `patience` consecutive epochs.

    Attributes:
        patience: Number of consecutive non-improving epochs to tolerate
            before stopping.
        delta: Minimum increase over the best-seen loss required to count
            as a non-improving epoch.
        val_loss_min: Best (lowest) validation loss seen so far.
        counter: Number of consecutive non-improving epochs seen so far.
        stopped: Whether the stopping criterion has been triggered.
        best_epoch: Epoch index at which `val_loss_min` was achieved.
    """

    def __init__(self, patience: int = 3, delta: float = 0) -> None:
        """Initializes the early-stopping tracker.

        Args:
            patience: Number of consecutive non-improving epochs to
                tolerate before stopping.
            delta: Minimum increase over the best-seen loss required to
                count as a non-improving epoch.
        """
        self.patience = patience
        self.delta = delta
        self.val_loss_min = np.inf
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> None:
        """Updates tracking state with the latest epoch's validation loss.

        Args:
            val_loss: Validation loss for the current epoch.
            epoch: Index of the current epoch.
        """
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.counter = 0
            self.best_epoch = epoch
        elif val_loss > self.val_loss_min + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True


class CustomAccuracy(MulticlassAccuracy):
    """Multiclass accuracy metric with optional per-allele masking.

    Extends `torchmetrics.classification.MulticlassAccuracy` to ignore
    "missing" labels (encoded as class `0`) either per allele copy or per
    sample, and to report raw correct/total counts alongside the accuracy
    ratio.

    Attributes:
        correct: Running count of correctly predicted (unmasked) entries.
        total: Running count of total (unmasked) entries considered.
        per_allele: Whether masking and accuracy are computed per allele
            copy (`True`) rather than per sample (`False`).
    """

    def __init__(self, num_classes: int, per_allele: bool = True, **kwargs: object) -> None:
        """Initializes the metric and its distributed-reduction state.

        Args:
            num_classes: Number of classes for the underlying multiclass
                accuracy computation.
            per_allele: Whether to mask/aggregate per allele copy (`True`)
                or per sample (`False`).
            **kwargs: Additional keyword arguments forwarded to
                `MulticlassAccuracy`.
        """
        super().__init__(num_classes=num_classes, **kwargs)
        # 'sum' tells TorchMetrics to add values across GPUs in distributed training
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.per_allele = per_allele

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulates correct/total counts for a batch of predictions.

        Args:
            preds: Raw prediction logits of shape `(batch, num_classes, 2)`.
            target: Ground truth class indices of shape `(batch, 2)`.

        Returns:
            None. Updates the internal `correct` and `total` metric state.
        """
        preds = torch.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape

        if self.per_allele:
            preds = preds.view(-1)
            target = target.view(-1)
            mask = target != 0
            preds = preds[mask]
            target = target[mask]
        else:
            mask = target.sum(dim=1) != 0
            preds = preds[mask]
            target = target[mask]

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> list[float | int]:
        """Computes the final accuracy from accumulated state.

        Returns:
            A 3-element list `[accuracy, correct, total]` where `accuracy`
            is a `float` (0.0 if `total` is 0), `correct` is an `int` count
            of correctly predicted entries, and `total` is an `int` count
            of entries considered.
        """
        if self.total == 0:
            accuracy = torch.tensor(0.0)
        else:
            accuracy = self.correct.float() / self.total
        res = [accuracy, self.correct, self.total]
        return [float(res[0]), int(res[1]), int(res[2])]


def accuracy_avg_digit(
    accuracy: dict[str, list[float | int]],
    digits: dict[int, list[str]],
    average: str = 'micro',
) -> dict[int, float]:
    """Averages per-head accuracy into a per-digit-level accuracy.

    Args:
        accuracy: Mapping from head name to a `[accuracy, correct, total]`
            list, as returned by `CustomAccuracy.compute`.
        digits: Mapping from digit level (e.g. 2, 4, 6, 8) to the list of
            head names that belong to that digit level.
        average: Averaging strategy. `'micro'` sums `correct`/`total`
            across heads in a digit level before dividing; `'macro'` takes
            the unweighted mean of per-head accuracies.

    Returns:
        Mapping from digit level to its averaged accuracy.
    """
    A = {}
    for d in digits:
        L = []
        for a in digits[d]:
            L.append(accuracy[a])
        mat = np.array(L)
        if average == 'micro':
            correct = np.sum(mat[:, 1])
            totoal = np.sum(mat[:, 2])
            acc = correct / totoal if totoal > 0 else 0.0
        elif average == 'macro':
            acc = np.mean(mat[:, 0])
        A[d] = acc
    return A


def txt_to_vcf_23andme(in_file: str, genome_build: str = 'GRCh37', chrom_hla: str = '6') -> None:
    """Converts a 23andMe raw genotype export to a single-sample VCF.

    Reads the reference genome FASTA for `genome_build`, extracts the
    sequence for `chrom_hla`, then walks the 23andMe text file restricting
    to variants on `chrom_hla`. For each retained variant the genotype is
    inferred from the reference base at that position: heterozygous calls
    (differing alleles) are always `0/1`; homozygous calls are `0/0` if the
    allele matches the reference base or `1/1` if it matches the (implied)
    alternate base. The resulting VCF is written alongside the input file,
    then bgzipped and tabix-indexed in place.

    Args:
        in_file: Path to the 23andMe `.txt` export. The output VCF is
            written to the same path with `.txt` replaced by `.vcf`
            (subsequently compressed to `.vcf.gz`).
        genome_build: Reference genome build used to fetch the FASTA via
            `Preprocessor.get_genome_reference`.
        chrom_hla: Chromosome name (as it appears in both the FASTA headers
            and the 23andMe file) to restrict the conversion to.

    Returns:
        None. Writes `<in_file stem>.vcf.gz` and its `.tbi` index to disk.
    """
    from .preprocess import Preprocessor
    hla = Preprocessor()
    fa_file = hla.get_genome_reference(genome_build)

    D = {}
    with open(fa_file) as f:
        chrom = ''
        seq = ''
        for line in f:
            if line.startswith('>'):
                if chrom != '':
                    if chrom == chrom_hla:
                        D[chrom] = seq
                chrom = line.strip().split(' ')[0][1:]
                seq = ''
            else:
                seq += line.strip()
        if chrom == chrom_hla:
            D[chrom] = seq
    for k in D:
        print(f'get ref seq of chrom {k} from {fa_file}')

    sample = in_file.split('.txt')[0].split('_')[0]
    out_file = in_file.split('.txt')[0] + '.vcf'
    with open(in_file) as f_in, open(out_file, 'w') as f_out:
        f_out.write('##fileformat=VCFv4.2\n')
        f_out.write('##source=23andMe\n')
        f_out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f_out.write(f'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n')
        for line in f_in:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            chrom = parts[1]
            pos = int(parts[2])
            rsid = parts[0]
            genotype = parts[3]
            if chrom != chrom_hla:
                continue
            if genotype == '--':
                continue
            ref = genotype[0]
            alt = genotype[1]
            fm = '0/0'
            if ref != alt:
                fm = '0/1'
            else:
                if ref == D[chrom][pos - 1]:
                    fm = '0/0'
                elif alt == D[chrom][pos - 1]:
                    fm = '1/1'
            f_out.write(f'{chrom}\t{pos}\t{rsid}\t{ref}\t{alt}\t.\t.\t.\tGT\t{fm}\n')
        print(f'{out_file} written successfully')
    cmd = f'bgzip -f {out_file}; tabix -p vcf -f {out_file}.gz'
    subprocess.run(cmd, shell=True)


def remove_chr_vcf(in_file: str, chr_map: str = 'chr_map.txt') -> None:
    """Strips the ``chr`` prefix from contig names in a VCF via bcftools.

    Derives a chromosome rename map from the VCF header's `##contig`
    lines (mapping e.g. `chr6` -> `6`) and applies it with
    `bcftools annotate --rename-chrs`.

    Args:
        in_file: Path to the input VCF (bgzipped and indexed).
        chr_map: Path to write the intermediate two-column rename map to.

    Returns:
        None. Writes `<in_file stem>_nochr.vcf.gz` and its tabix index.
    """
    out_file = in_file.split('.vcf')[0] + '_nochr.vcf.gz'
    cmd = f"bcftools view -h {in_file} | grep '^##contig=<ID=chr' | sed -e 's/.*ID=//' -e 's/,.*//' | awk " + "'{print $1, substr($1, 4)}'" + f" > {chr_map}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    cmd = f'bcftools annotate --rename-chrs {chr_map} -Oz -o {out_file} {in_file}; tabix -p vcf -f {out_file}'
    print(cmd)
    subprocess.run(cmd, shell=True)
