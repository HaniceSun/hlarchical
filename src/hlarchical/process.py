from .utils import *


class Processor:
    """Build model-ready features, labels, and expert masks from a phased reference VCF.

    A ``Processor`` loads a phased reference VCF (samples already imputed/phased,
    e.g. via Beagle) and splits its variant rows into two subsets: HLA allele
    calls used as prediction targets (``ref_phased_target``) and all other
    variants used as model input features (``ref_phased_feature``). It also
    defines the linkage-disequilibrium (LD) blocks and "expert" groupings of
    HLA genes used by the hierarchical/mixture-of-experts model architecture,
    and exposes methods to materialize the on-disk artifacts consumed by
    training (``features.txt``, ``labels.txt``, ``maps.txt``, ``masks.txt``)
    as well as to featurize a new sample VCF for prediction.

    Attributes:
        label_include: Substrings used to select target (label) variant rows
            by matching against the VCF ``ID`` column (e.g. ``['HLA']``).
        feature_exclude: Substrings used to exclude variant rows from the
            feature set by matching against the VCF ``ID`` column.
        expert_by: Strategy used to group HLA genes into "experts", either
            ``'gene'`` (one expert per gene) or ``'ld'`` (one expert per LD
            block, per ``self.expert_groups``).
        with_ancestry: Whether ancestry is used as an additional one-hot
            encoded feature/covariate.
        ancestry_file: Resolved path to the tab-delimited ancestry file
            (sample ID in column 0, ancestry label in column 1). Only set
            when ``with_ancestry`` is True.
        data_dir: Package data directory used as a fallback location to
            resolve relative file paths, namespaced by whether ancestry is
            used (``data_dir/with_ancestry`` or ``data_dir/without_ancestry``).
        ref_phased: Full phased reference VCF loaded as a DataFrame (via
            :meth:`read_vcf`).
        ref_phased_target: Subset of ``ref_phased`` rows whose ``ID`` matches
            ``label_include`` (the HLA allele rows used as labels).
        ref_phased_feature: Subset of ``ref_phased`` rows whose ``ID`` does
            not match ``feature_exclude`` (the non-HLA rows used as features).
        ld_blocks: Mapping from HLA gene name to the list of genes in its LD
            block (used when ``expert_by == 'ld'``).
        expert_groups: Mapping from expert ID (e.g. ``'E0'``) to the list of
            HLA genes assigned to that expert.
    """

    def __init__(
        self,
        ref_phased: str = '1000G_REF_phased.vcf.gz',
        with_ancestry: bool = False,
        ancestry_file: str = 'ancestry.txt',
        label_include: list[str] = ['HLA'],
        feature_exclude: list[str] = ['HLA'],
        expert_by: str = 'ld',
    ) -> None:
        """Load and split the phased reference VCF into feature/target subsets.

        Args:
            ref_phased: Path to the phased reference VCF (``.vcf`` or
                ``.vcf.gz``). If not found relative to the current directory,
                it is resolved under the package's ``data_dir``.
            with_ancestry: Whether to also load an ancestry file for use as
                an additional feature/covariate.
            ancestry_file: Path to the tab-delimited ancestry file, used only
                when ``with_ancestry`` is True. Resolved under
                ``data_dir/with_ancestry`` if not found as given.
            label_include: Substrings matched against the VCF ``ID`` column
                to select rows used as prediction targets.
            feature_exclude: Substrings matched against the VCF ``ID`` column
                to exclude rows from the feature set.
            expert_by: Grouping strategy for HLA genes into experts, either
                ``'gene'`` or ``'ld'``.

        Raises:
            FileNotFoundError: If ``ref_phased`` cannot be found either as
                given or under ``data_dir``.
        """
        self.label_include: list[str] = label_include
        self.feature_exclude: list[str] = feature_exclude
        self.expert_by: str = expert_by

        self.with_ancestry: bool = with_ancestry
        if with_ancestry:
            self.data_dir: str = data_dir + '/with_ancestry'
        else:
            self.data_dir = data_dir + '/without_ancestry'
        if with_ancestry:
            self.ancestry_file: str = ancestry_file
            if not os.path.exists(ancestry_file):
                self.ancestry_file = self.data_dir + '/' + ancestry_file
                if not os.path.exists(self.ancestry_file):
                    print(f'ancestry file {ancestry_file} not found')
                else:
                    print(f'using ancestry file {self.ancestry_file}')
            else:
                print(f'using ancestry file {ancestry_file}')

        if not os.path.exists(ref_phased):
            ref_phased = data_dir + '/' + ref_phased
            if not os.path.exists(ref_phased):
                raise FileNotFoundError(
                    f'reference phased VCF file {ref_phased} not found current directory or {data_dir}'
                )
            else:
                print(f'using reference phased VCF file {ref_phased}')

        self.ref_phased: pd.DataFrame = self.read_vcf(ref_phased)

        # Extract HLA for targets
        wh = []
        for n in range(self.ref_phased.shape[0]):
            ID = str(self.ref_phased['ID'].iloc[n])
            flag = False
            for flt in label_include:
                if ID.find(flt) != -1:
                    flag = True
                    break
            wh.append(flag)
        self.ref_phased_target: pd.DataFrame = self.ref_phased[wh].copy()

        # Extract non-HLA for features
        wh = []
        for n in range(self.ref_phased.shape[0]):
            ID = str(self.ref_phased['ID'].iloc[n])
            flag = True
            for flt in feature_exclude:
                if ID.find(flt) != -1:
                    flag = False
                    break
            wh.append(flag)
        self.ref_phased_feature: pd.DataFrame = self.ref_phased[wh].copy()

        self.ld_blocks: dict[str, list[str]] = {}
        self.ld_blocks['HLA-A'] = ['HLA-A']
        self.ld_blocks['HLA-B'] = ['HLA-B', 'HLA-C']
        self.ld_blocks['HLA-C'] = ['HLA-B', 'HLA-C']
        self.ld_blocks['HLA-DPA1'] = ['HLA-DPA1', 'HLA-DPB1']
        self.ld_blocks['HLA-DPB1'] = ['HLA-DPA1', 'HLA-DPB1']
        self.ld_blocks['HLA-DQA1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.ld_blocks['HLA-DQB1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.ld_blocks['HLA-DRB1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.expert_groups: dict[str, list[str]] = {}
        self.expert_groups['E0'] = ['HLA-A']
        self.expert_groups['E1'] = ['HLA-B', 'HLA-C']
        self.expert_groups['E2'] = ['HLA-DPA1', 'HLA-DPB1']
        self.expert_groups['E3'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def make_features(self, out_file: str = 'features.txt') -> None:
        """Write the model input feature matrix built from non-HLA variants.

        Transposes ``ref_phased_feature`` so that rows are samples and
        columns are ``{ID}_{POS}`` variant identifiers, with each cell holding
        the raw phased genotype string (e.g. ``'0|1'``). When ``with_ancestry``
        is set, one-hot encoded ancestry columns (each duplicated as
        ``'{value}|{value}'`` to match the genotype-string format) are merged
        in by sample.

        Args:
            out_file: Path to write the tab-delimited feature matrix to. A
                companion file with the ``_list`` suffix (e.g.
                ``features_list.txt``) is also written, containing one
                feature column name per line.

        Raises:
            ValueError: If sample names differ between the feature matrix and
                the ancestry file after merging.
        """
        df = self.ref_phased_feature.iloc[:, 9:].T
        df.columns = self.ref_phased_feature['ID'] + '_' + self.ref_phased_feature['POS'].astype(str)
        df.reset_index(inplace=True, names=['sample'])

        if self.with_ancestry:
            df_anc = pd.read_table(self.ancestry_file, header=0, sep='\t', low_memory=False)
            df_anc_encoded = pd.get_dummies(df_anc.iloc[:, 1], prefix='ancestry').astype(int).astype(str)
            df_anc_encoded = df_anc_encoded + '|' + df_anc_encoded
            df_anc_encoded['sample'] = df_anc.iloc[:, 0]
            df_merged = pd.merge(df, df_anc_encoded, on='sample')
            if df_merged['sample'].tolist() != df['sample'].tolist():
                raise ValueError('Mismatch in sample names between features and ancestry data.')
            df = df_merged

        df.to_csv(out_file, sep='\t', index=False, header=True)
        out_file_list = out_file.split('.txt')[0] + '_list.txt'
        df.columns[1:].to_series().to_csv(out_file_list, index=False, header=False)

    def make_maps(self, out_file: str = 'maps.txt') -> None:
        """Build and write the hierarchical label map for HLA allele targets.

        For each distinct HLA allele "head" (the allele ID truncated by one
        ``:``-separated field, e.g. ``HLA-A:01:01`` from ``HLA-A:01:01:01``),
        assigns an integer ``label`` to every allele under that head, records
        the allele's digit resolution (``digit``), a dense ``head_idx`` used
        to order model output heads by digit level, the head's ``parent``
        head and ``parent_val`` (the parent's label, for allele IDs above the
        coarsest 2-digit level), and an ``expert`` assignment (either the
        gene name when ``expert_by == 'gene'``, or an LD-block expert ID from
        ``self.expert_groups`` when ``expert_by == 'ld'``).

        Args:
            out_file: Path to write the resulting tab-delimited maps table to,
                with columns ``digit``, ``allele``, ``label``, ``head``,
                ``head_idx``, ``parent``, ``parent_val``, ``expert``.
        """
        D = {}
        for n in range(self.ref_phased_target.shape[0]):
            ID = self.ref_phased_target['ID'].iloc[n]
            fields = ID.split(':')
            head = ':'.join(fields[0:-1])
            D.setdefault(head, [])
            if ID not in D[head]:
                D[head].append(ID)

        H = {}
        for k in sorted(D):
            H[k] = sorted(D[k])

        maps = []
        for head in H:
            for allele in H[head]:
                digit = len(allele.split(':')[1:]) * 2
                maps.append([digit, allele, H[head].index(allele) + 1, head])
        maps = pd.DataFrame(maps, columns=['digit', 'allele', 'label', 'head'])
        maps.sort_values(by=['digit', 'head', 'label'], inplace=True)

        heads = []
        for digit in sorted(maps['digit'].unique()):
            df_sub = maps[maps['digit'] == digit]
            for n in range(df_sub.shape[0]):
                head = df_sub['head'].iloc[n]
                if head not in heads:
                    heads.append(head)
        maps['head_idx'] = [heads.index(x) for x in maps['head']]

        parent = []
        parent_value = []
        expert = []
        for n in range(maps.shape[0]):
            digit = maps['digit'].iloc[n]
            head = maps['head'].iloc[n]
            if digit == 2:
                p = '.'
                p_val = -1
                e = maps['head'].iloc[n]
            else:
                p = ':'.join(head.split(':')[0:-1])
                p_val = H[p].index(head) + 1
                e = head.split(':')[0]
            parent.append(p)
            parent_value.append(p_val)
            expert.append(e)
        maps['parent'] = parent
        maps['parent_val'] = parent_value

        if self.expert_by == 'gene':
            maps['expert'] = expert
        elif self.expert_by == 'ld':
            E = []
            for x in expert:
                expert_id = '.'
                for k in self.expert_groups:
                    if x in self.expert_groups[k]:
                        expert_id = k
                        break
                if expert_id == '.':
                    expert_id = self.expert_groups.keys()[0]
                    print(f'{x} not found in LD groups, assigned to {expert_id}')
                E.append(expert_id)
            maps['expert'] = E

        maps.to_csv(out_file, sep='\t', index=False, header=True)
        print('processed label maps:')
        print(maps)
        print(f'maps data saved to {out_file}')

    def make_labels(self, out_file: str = 'labels.txt', maps_file: str = 'maps.txt') -> None:
        """Build and write per-sample ground-truth labels for each model head.

        For every sample and every head (as defined in ``maps_file``),
        determines whether each of the sample's two phased haplotypes carries
        an allele under that head, and writes the corresponding integer label
        (0 if absent) for each haplotype, joined as ``'{label1}|{label2}'``.

        Args:
            out_file: Path to write the tab-delimited labels table to, indexed
                by ``sample`` with one column per head.
            maps_file: Path to the maps table produced by :meth:`make_maps`,
                used to resolve allele IDs to integer labels and heads.
        """
        maps = pd.read_table(maps_file, header=0, sep='\t', low_memory=False)
        heads = maps['head'].unique().tolist()
        n_heads = len(heads)
        samples = self.ref_phased_target.columns[9:]
        D = dict(zip(maps['allele'], maps['label']))
        df = pd.DataFrame('0', index=samples, columns=heads)
        df1 = pd.DataFrame('0', index=samples, columns=heads)
        df2 = pd.DataFrame('0', index=samples, columns=heads)
        for n in range(self.ref_phased_target.shape[0]):
            ID = self.ref_phased_target['ID'].iloc[n]
            fields = ID.split(':')
            head = ':'.join(fields[0:-1])
            head_idx = heads.index(head)
            for m in range(len(samples)):
                allele = self.ref_phased_target.iloc[n, 9 + m].split('|')
                if len(allele) == 2:
                    if allele[0] == '1':
                        df1.iloc[m, head_idx] = D[ID]
                    if allele[1] == '1':
                        df2.iloc[m, head_idx] = D[ID]
        for n in range(df.shape[0]):
            for m in range(df.shape[1]):
                df.iloc[n, m] = str(df1.iloc[n, m]) + '|' + str(df2.iloc[n, m])
        df.index.name = 'sample'
        df.to_csv(out_file, sep='\t', index=True, header=True)

    def make_masks(
        self,
        out_file: str = 'masks.txt',
        features_file: str = 'features.txt',
        flank: int = 500000,
    ) -> None:
        """Build and write a per-expert boolean feature mask.

        For each expert (gene or LD block, per ``expert_by``), computes the
        genomic span covering its target variants (plus ``flank`` bases on
        either side) and marks which feature columns in ``features_file``
        fall within that span (ancestry feature columns are always included).

        Args:
            out_file: Path to write the tab-delimited masks table to, with an
                ``expert`` column plus one 0/1 column per feature.
            features_file: Path to the feature matrix produced by
                :meth:`make_features`, used to determine feature column names
                and positions.
            flank: Number of bases to extend the expert's variant span by on
                each side when selecting feature columns.
        """
        H = {}
        for n in range(self.ref_phased_target.shape[0]):
            gene = self.ref_phased_target['ID'].iloc[n].split(':')[0]
            pos = self.ref_phased_target['POS'].iloc[n]
            H.setdefault(gene, [])
            H[gene].append(pos)

        start_end_dict = {}
        for gene in H:
            if self.expert_by == 'gene':
                positions = H[gene]
                start_end_dict[gene] = (min(positions), max(positions))
            elif self.expert_by == 'ld':
                positions = []
                if gene in self.ld_blocks:
                    for g in self.ld_blocks[gene]:
                        if g in H:
                            positions += H[g]
                start_end_dict[gene] = (min(positions), max(positions))

        features = pd.read_table(features_file, header=0, sep='\t', low_memory=False)
        L = []
        E = []
        for gene in start_end_dict:
            if self.expert_by == 'gene':
                expert = gene
            elif self.expert_by == 'ld':
                expert = '.'
                for k in self.expert_groups:
                    if gene in self.expert_groups[k]:
                        expert = k
                        break
                if expert == '.':
                    expert = self.expert_groups.keys()[0]
                    print(f'{gene} not found in LD groups, assigned to {expert}')

            if expert not in E:
                E.append(expert)
                pos_min, pos_max = start_end_dict[gene]
                m = []
                for n in range(1, features.shape[1]):
                    fields = features.columns[n].split('_')
                    name = '_'.join(fields[0:-1])
                    if name.find('ancestry') != -1:
                        m.append(1)
                    else:
                        pos = int(fields[-1])
                        if pos >= pos_min - flank and pos <= pos_max + flank:
                            m.append(1)
                        else:
                            m.append(0)
                L.append([expert] + m)
        df = pd.DataFrame(L)
        df.columns = ['expert'] + features.columns[1:].tolist()
        df.to_csv(out_file, sep='\t', index=False, header=True)
        print(f'processed masks data: {out_file}')

    def get_sample_features(
        self,
        sample_vcf: str,
        features_file: str = 'features_list.txt',
        out_file: str = 'to_predict.txt',
    ) -> None:
        """Featurize a new sample VCF into the format expected for prediction.

        Reads ``sample_vcf`` and, for each feature listed in ``features_file``
        (in the same order used at training time), extracts the genotype
        string for every sample, defaulting to ``'0|0'`` for variants absent
        from the sample VCF. If ``self.with_ancestry`` is set, also merges in
        one-hot encoded ancestry columns (matching the categories implied by
        the ``ancestry_*`` entries in ``features_file``) from
        ``self.ancestry_file``.

        Args:
            sample_vcf: Path to the (typically phased) sample VCF to featurize.
            features_file: Path to a single-column file listing feature names
                in training order (as produced by :meth:`make_features`'s
                ``_list`` output). Resolved under ``self.data_dir`` if not
                found as given.
            out_file: Path to write the resulting tab-delimited feature table
                to, in the same column format as the training feature matrix.

        Raises:
            FileNotFoundError: If ``features_file`` cannot be found either as
                given or under ``self.data_dir``.
            ValueError: If ``features_file`` has more than one column, if the
                ancestry file's categories are not a subset of the ancestry
                categories implied by ``features_file``, if sample names
                differ between the feature table and the ancestry file after
                merging, or if the final feature column order does not match
                ``features_file``.
        """
        if not os.path.exists(features_file):
            features_list = self.data_dir + '/' + features_file
            if os.path.exists(features_list):
                features_file = features_list
            else:
                raise FileNotFoundError(f'Features file {features_file} or {features_list} not found.')

        df = self.read_vcf(sample_vcf)
        samples = df.columns[9:].tolist()
        df_features = pd.read_table(features_file, header=None, sep='\t', low_memory=False)
        if df_features.shape[1] > 1:
            raise ValueError(
                f'Expected features file to have only one column, but found {df_features.shape[1]} columns.'
            )
        features = [x for x in df_features.iloc[:, 0] if not x.startswith('ancestry_')]
        ancestry = [x.split('ancestry_')[-1] for x in df_features.iloc[:, 0] if x.startswith('ancestry_')]
        if len(ancestry):
            print(f'Ancestry features identified: {ancestry}')

        D = {}
        for n in range(df.shape[0]):
            k = str(df['ID'].iloc[n]) + '_' + str(df['POS'].iloc[n])
            D[k] = [x.split(':')[0] for x in df.iloc[n, 9:]]
        L = []
        for f in features:
            if f in D:
                L.append(D[f])
            else:
                L.append(['0|0'] * len(samples))
        df = pd.DataFrame(L).T
        df.index = samples
        df.columns = features
        df.reset_index(inplace=True, names=['sample'])

        if self.with_ancestry:
            df_anc = pd.read_table(self.ancestry_file, header=0, sep='\t', low_memory=False)
            # ancestry encoding must match the one used in the training data
            if not set(df_anc.iloc[:, 1]).issubset(set(ancestry)):
                raise ValueError('Ancestry categories in the ancestry file do not match those in the features list.')
            df_anc_encoded = pd.get_dummies(
                pd.Categorical(df_anc.iloc[:, 1], categories=ancestry), prefix='ancestry'
            ).astype(int).astype(str)
            df_anc_encoded = df_anc_encoded + '|' + df_anc_encoded
            df_anc_encoded['sample'] = df_anc.iloc[:, 0]
            df_merged = pd.merge(df, df_anc_encoded, on='sample')
            if df_merged['sample'].tolist() != df['sample'].tolist():
                raise ValueError('Mismatch in sample names between features and ancestry data.')
            df = df_merged
        else:
            print(f'Proceeding without ancestry features.')

        # double check that the feature columns in the sample data match the features list
        if df.columns[1:].to_list() != df_features.iloc[:, 0].to_list():
            raise ValueError('Mismatch in feature names between sample data and features list.')

        df.to_csv(out_file, sep='\t', index=False, header=True)
        print(f'sample features saved to {out_file}')

    def read_vcf(self, in_file: str) -> pd.DataFrame:
        """Read a plain or gzip-compressed VCF into a DataFrame.

        Skips all ``##`` meta-information lines and loads the ``#CHROM``
        header row and variant rows via ``pandas.read_table``, renaming the
        ``#CHROM`` column to ``CHROM``.

        Args:
            in_file: Path to the VCF file, must end with ``.vcf`` or
                ``.vcf.gz``.

        Returns:
            The VCF contents as a DataFrame, with standard VCF columns
            (``CHROM``, ``POS``, ``ID``, ``REF``, ``ALT``, ``QUAL``,
            ``FILTER``, ``INFO``, ``FORMAT``) followed by one column per
            sample.

        Raises:
            ValueError: If ``in_file`` does not end with ``.vcf`` or
                ``.vcf.gz``.
        """
        if in_file.endswith('vcf.gz'):
            with gzip.open(in_file, 'rt') as f:
                lines = [l for l in f if l.startswith('##')]
                n_header = len(lines)
        elif in_file.endswith('vcf'):
            with open(in_file, 'r') as f:
                lines = [l for l in f if l.startswith('##')]
                n_header = len(lines)
        else:
            raise ValueError('Input file must VCF format end with .vcf or .vcf.gz')

        df = pd.read_table(in_file, sep='\t', skiprows=n_header, low_memory=False)
        df.rename(columns={'#CHROM': 'CHROM'}, inplace=True)
        return df


if __name__ == '__main__':
    pass
