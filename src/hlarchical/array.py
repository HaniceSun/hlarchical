from .utils import *


class Array():
    """Wrapper around array-based HLA imputation tools (SNP2HLA, HIBAG, DEEP*HLA).

    Each method shells out to a third-party tool (via ``subprocess``) to impute
    HLA alleles from SNP array genotype data, using either a bundled vendor
    installation or a Conda environment.

    Attributes:
        HLA: The eight classical HLA genes handled throughout this package.
    """

    def __init__(self) -> None:
        self.HLA: list[str] = [
            'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1',
        ]

    def run_snp2hla(
        self,
        in_file: str = '1958BC',
        ref_file: str = 'HM_CEU_REF',
        out_file: str = '1958BC_European_SNP2HLA',
        snp2hla_dir: str | None = None,
        heap_size: int = 2000,
        window_size: int = 1000,
    ) -> None:
        """Impute HLA alleles from array data using SNP2HLA.

        Copies the vendored SNP2HLA installation into a temporary working
        directory (to avoid clobbering it across concurrent runs), then invokes
        ``SNP2HLA.csh`` there and removes the working directory afterward.

        Args:
            in_file: Prefix of the input PLINK bed/bim/fam fileset to impute.
                Must exist as ``{in_file}.bed``.
            ref_file: Prefix of the reference panel PLINK bed/bim/fam fileset.
                Must exist as ``{ref_file}.bed``.
            out_file: Prefix for SNP2HLA's output files.
            snp2hla_dir: Path to the vendored SNP2HLA installation directory.
                If None, defaults to ``vendor/SNP2HLA/home`` relative to the
                installed ``hlarchical`` package.
            heap_size: Java heap size (in MB) passed to SNP2HLA.
            window_size: Sliding window size passed to SNP2HLA.

        Raises:
            FileNotFoundError: If ``{in_file}.bed`` or ``{ref_file}.bed`` does
                not exist.
        """
        if not snp2hla_dir:
            snp2hla_dir = f'{resources.files("hlarchical").parent.parent}/vendor/SNP2HLA/home'

        out_file = os.path.abspath(out_file)

        if os.path.exists(f'{in_file}.bed'):
            in_file = os.path.abspath(in_file)
        else:
            raise FileNotFoundError(f'Input file {in_file}.bed not found')

        if os.path.exists(f'{ref_file}.bed'):
            ref_file = os.path.abspath(ref_file)
        else:
            raise FileNotFoundError(f'Reference file {ref_file}.bed not found')

        # make a temporary working directory, not using the installation directory to avoid potential overwritten issues of multiple runs at the same time
        working_dir = f'{out_file}_working'
        shutil.copytree(snp2hla_dir, working_dir)
        os.chdir(working_dir)
        print(f'running SNP2HLA in {working_dir}', flush=True)

        cmd = f'tcsh SNP2HLA.csh {in_file} {ref_file} {out_file} plink {heap_size} {window_size}'
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
        shutil.rmtree(working_dir)

    def run_hibag(
        self,
        in_file: str = '1958BC',
        ref: str = 'European',
        out_file: str = '1958BC_European_HIBAG',
        Renv: str = 'R4.5',
    ) -> None:
        """Impute HLA alleles from array data using HIBAG.

        Downloads the pre-trained HIBAG reference model for the requested
        ancestry if not already present, then runs the bundled ``hibag.R``
        script inside the given Conda R environment.

        Args:
            in_file: Prefix of the input PLINK bed/bim/fam fileset to impute.
            ref: Ancestry reference model to use (e.g. ``European``, ``Asian``,
                ``African``, ``Hispanic``). Determines which HIBAG ``.RData``
                model file is downloaded/used.
            out_file: Prefix for HIBAG's output files.
            Renv: Name of the Conda environment with R and the HIBAG package
                installed.
        """
        hibag_script = f'{resources.files("hlarchical").parent.parent}/vendor/HIBAG/hibag.R'
        ref_file = f'{ref}-HLA4-hg19.RData'
        if not os.path.exists(ref_file):
            cmd = f'wget https://hibag.s3.amazonaws.com/download/hlares_param/{ref_file}'
            subprocess.run(cmd, shell=True, check=True)
        cmd = f'conda run -n {Renv} Rscript {hibag_script} {in_file} {ref_file} {out_file}'
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

    def run_deephla(
        self,
        mode: str = 'train',
        in_file: str = 'OMNI_Pan-Asian_REF',
        ref_file: str = 'Pan-Asian_REF',
        model_json: str = 'Pan-Asian_REF.model.json',
        hla_json: str = 'Pan-Asian_REF.hla.json',
        model_dir: str = 'model_Pan-Asian_REF',
        deephla_dir: str | None = None,
        out_file: str = 'OMNI_Pan-Asian',
        subset: list[str] | str = [],
    ) -> None:
        """Train or run inference with the vendored CNN-based DEEP*HLA model.

        In ``'train'`` mode, optionally restricts the phased Beagle input to a
        genomic subregion, generates the HLA-info JSON if missing, then
        trains a DEEP*HLA model. In ``'impute'`` mode, runs imputation with an
        already-trained model.

        Args:
            mode: Either ``'train'`` to train a new model or ``'impute'`` to
                run imputation with an existing one.
            in_file: Prefix of the input phased Beagle (``.bgl.phased``)
                sample fileset.
            ref_file: Prefix of the phased Beagle reference panel fileset,
                used in ``'train'`` mode.
            model_json: Path to the DEEP*HLA model config JSON file (must end
                in ``.model.json``).
            hla_json: Path to the HLA-info JSON file (must end in
                ``.hla.json``); generated automatically in ``'train'`` mode if
                it does not already exist.
            model_dir: Output directory for the trained model.
            deephla_dir: Path to the vendored DEEP-HLA installation directory.
                If None, defaults to ``vendor/DEEP-HLA`` relative to the
                installed ``hlarchical`` package.
            out_file: Output file prefix, used in ``'impute'`` mode.
            subset: A ``'chrom:start-end'``-style region string (or an empty
                list, the default, to disable subsetting) used to restrict the
                training input to a genomic window before training.

        Raises:
            FileNotFoundError: If ``model_json`` does not exist, or if
                (in ``'train'`` mode) the required phased ``.bgl.phased``
                inputs are missing.
        """
        if not deephla_dir:
            deephla_dir = f'{resources.files("hlarchical").parent.parent}/vendor/DEEP-HLA'

        if mode == 'train':
            if not os.path.exists(f'{in_file}.bgl.phased') or not os.path.exists(f'{ref_file}.bgl.phased'):
                print('Use hlarchical run-snp2hla first to get the phased bgl files')
                return

            if subset:
                region = subset.replace('-', ':').split(':')
                df = pd.read_table(f'{in_file}.bgl.phased', sep=' ', header=None)
                wh = []
                for n in range(df.shape[0]):
                    chrom = str(df.iloc[n, 0])
                    pos = int(df.iloc[n, 1])
                    if chrom == region[0] and pos >= int(region[1]) and pos <= int(region[2]):
                        wh.append(True)
                    else:
                        wh.append(False)
                df = df.loc[wh, ]
                df.to_csv(f'{in_file}.bgl.phased', header=False, index=False, sep=' ')

            if not os.path.exists(model_json):
                raise FileNotFoundError(f'{model_json} not found')

            if not os.path.exists(hla_json):
                print('Generating HLA info JSON file...')
                cmd = f'conda run -n DEEP-HLA python {deephla_dir}/make_hlainfo.py --ref {ref_file} --out {hla_json}'
                print(cmd)
                subprocess.run(cmd, shell=True)

            if os.path.exists(model_json) and os.path.exists(hla_json):
                model_json = model_json.split('.model.json')[0]
                hla_json = hla_json.split('.hla.json')[0]
                cmd = f'conda run -n DEEP-HLA python {deephla_dir}/train.py --ref {ref_file} --sample {in_file} --model {model_json} --hla {hla_json} --model-dir {model_dir}'
                print(cmd)
                subprocess.run(cmd, shell=True)
            else:
                raise FileNotFoundError(f'{model_json} not found')

        elif mode == 'impute':
            model_json = model_json.split('.model.json')[0]
            hla_json = hla_json.split('.hla.json')[0]
            cmd = f'conda run -n DEEP-HLA python {deephla_dir}/impute.py --sample {in_file} --model {model_json} --hla  {hla_json} --model-dir {model_dir} --out {out_file}'
            print(cmd)
            subprocess.run(cmd, shell=True)
