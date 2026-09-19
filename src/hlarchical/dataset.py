import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from .utils import *


class CustomDataset(Dataset):
    """PyTorch ``Dataset`` wrapping paired feature/label tensors for HLA imputation.

    Features and labels are read from tab-separated text files whose rows are
    samples (matched by the first column) and whose remaining columns encode
    per-marker/per-allele values as ``value1|value2`` pairs. Those pairs are
    parsed into a stacked 3-D array with shape ``(n_samples, 2, n_columns)``
    and exposed as ``self.X`` (float32 features) and ``self.y`` (long labels).

    Attributes:
        out_file: Prefix used when saving split datasets/dataloaders to disk.
        X: Feature tensor of shape ``(n_samples, 2, n_feature_columns)``, dtype ``torch.float32``.
        y: Label tensor of shape ``(n_samples, 2, n_label_columns)``, dtype ``torch.long``.
        random_seed: Random seed used for K-fold splitting.
    """

    def __init__(
        self,
        features_file: str = 'features.txt',
        labels_file: str = 'labels.txt',
        maps_file: str = 'maps.txt',
        out_file: str = 'hla',
    ) -> None:
        """Load features and labels from disk and stack them into tensors.

        Args:
            features_file: Tab-separated file whose first column is the sample
                ID and whose remaining columns contain ``value1|value2``
                encoded feature pairs.
            labels_file: Tab-separated file with the same sample ordering as
                ``features_file``, whose remaining columns contain
                ``value1|value2`` encoded label pairs.
            maps_file: Reserved for future use; currently unused by this
                constructor but accepted for interface consistency with other
                components (e.g. ``CustomLoss``).
            out_file: Output file prefix used by :meth:`split_save_dataset`
                when saving train/val/test datasets and dataloaders.

        Raises:
            ValueError: If the sample column of ``features_file`` and
                ``labels_file`` do not match exactly (same values, same order).
        """
        self.out_file = out_file
        df_features = pd.read_table(features_file, header=0, sep='\t')
        df_labels = pd.read_table(labels_file, header=0, sep='\t')
        if df_features.iloc[:, 0].tolist() != df_labels.iloc[:, 0].tolist():
            raise ValueError('The sample column of features and labels must be the same')

        mat = []
        for n in range(1, df_features.shape[1]):
            mat.append(df_features.iloc[:, n].str.split('|', expand=True).astype(int).values)
        mat = np.array(mat).transpose(1, 2, 0)
        self.X = torch.tensor(mat, dtype=torch.float32)

        mat = []
        for n in range(1, df_labels.shape[1]):
            mat.append(df_labels.iloc[:, n].str.split('|', expand=True).astype(int).values)
        mat = np.array(mat).transpose(1, 2, 0)
        self.y = torch.tensor(mat, dtype=torch.long)
        print(f'X shape: {self.X.shape}')
        print(f'y shape: {self.y.shape}')
        self.random_seed = 42

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            The number of samples, i.e. the size of the first dimension of
            ``self.y``.
        """
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the feature/label tensor pair for a single sample.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple ``(X, y)`` of the feature tensor and label tensor for the
            sample at ``idx``.
        """
        X = self.X[idx]
        y = self.y[idx]
        return X, y

    def split_save_dataset(
        self,
        ratio: list[float] = [0.8, 0.2],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        n_cv: int = 0,
    ) -> None:
        """Split the dataset into train/val/test subsets and save dataloaders to disk.

        With a two-element ``ratio``, the dataset is split into train and test
        subsets. If ``n_cv`` is 0, the resulting train/test dataloaders are
        saved directly; otherwise the train subset is further split into
        ``n_cv`` cross-validation folds via :class:`~sklearn.model_selection.KFold`,
        and one train/val dataloader pair is saved per fold. With a
        three-element ``ratio``, the dataset is split into train/val/test
        subsets and all three dataloaders are saved.

        Args:
            ratio: Split proportions. A 2-element list gives
                ``[train, test]`` proportions; a 3-element list gives
                ``[train, val, test]`` proportions.
            batch_size: Batch size used when constructing each
                :class:`~torch.utils.data.DataLoader`.
            shuffle: Whether each dataloader shuffles samples every epoch.
            num_workers: Number of worker processes used by each dataloader.
            n_cv: Number of cross-validation folds to generate when
                ``len(ratio) == 2``. If 0, no cross-validation folds are
                created and a single train/test split is saved instead.

        Returns:
            None. Dataloaders (and, for the 2-way split without CV, raw
            datasets) are saved to disk as ``.pt`` files derived from
            ``self.out_file``, and the corresponding attributes
            (``self.ds_train``, ``self.dl_train``, etc.) are set on ``self``.
        """
        self.train_file = self.out_file + '_dataset_train.pt'
        self.val_file = self.out_file + '_dataset_val.pt'
        self.test_file = self.out_file + '_dataset_test.pt'

        if len(ratio) == 2:
            self.ds_train, self.ds_test = torch.utils.data.random_split(self, ratio)
            if n_cv == 0:
                self.dl_train = DataLoader(self.ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
                self.dl_test = DataLoader(self.ds_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
                torch.save(self.dl_train, self.train_file)
                torch.save(self.dl_test, self.test_file)
                print(f'datasets saved to {self.train_file} and {self.test_file}')
            else:
                kf = KFold(n_splits=n_cv, shuffle=True, random_state=self.random_seed)
                for fold, (train_index, val_index) in enumerate(kf.split(self.ds_train)):
                    ds_train_cv = torch.utils.data.Subset(self.ds_train, train_index)
                    ds_val_cv = torch.utils.data.Subset(self.ds_train, val_index)

                    dl_train_cv = DataLoader(ds_train_cv, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
                    dl_val_cv = DataLoader(ds_val_cv, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

                    train_file_cv = self.out_file + f'_dataset_train_fold{fold+1}.pt'
                    val_file_cv = self.out_file + f'_dataset_val_fold{fold+1}.pt'

                    torch.save(dl_train_cv, train_file_cv)
                    torch.save(dl_val_cv, val_file_cv)
                    print(f'datasets saved to {train_file_cv} and {val_file_cv}')

        elif len(ratio) == 3:
            self.ds_train, self.ds_val, self.ds_test = torch.utils.data.random_split(self, ratio)

            self.dl_train = DataLoader(self.ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            self.dl_val = DataLoader(self.ds_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            self.dl_test = DataLoader(self.ds_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            torch.save(self.dl_train, self.train_file)
            torch.save(self.dl_val, self.val_file)
            torch.save(self.dl_test, self.test_file)
            print(f'datasets saved to {self.train_file}, {self.val_file}, and {self.test_file}')


if __name__ == '__main__':
    ds = CustomDataset()
    ds.split_save_dataset(ratio=[0.8, 0.2])
