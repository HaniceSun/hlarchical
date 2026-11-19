from .utils import *


class Seq:
    """Helpers that generate shell scripts for WGS-based HLA typing pipelines.

    Each method writes a shell script (one or more commands per sample) to
    an output file rather than executing the commands directly; the caller
    is expected to run the generated script (e.g. via ``bash`` or a job
    scheduler).
    """

    def __init__(self) -> None:
        """Initialize the Seq helper (no state is held)."""
        pass

    def subset_cram_to_fastq(
        self,
        cram_dir: str = 'wgs_cram',
        fasta_file: str = 'Homo_sapiens_assembly38.fasta',
        bed_file: str = 'cram_subset_hla_assembly38.bed',
        out_file: str = 'subset_cram_to_fastq.sh',
        n_threads: int = 4,
    ) -> None:
        """Write a script that subsets CRAM files to HLA regions and converts them to FASTQ.

        For every ``.cram`` file found in ``cram_dir``, writes two commands to
        ``out_file``: one that uses ``samtools view`` to subset the CRAM to the
        regions listed in ``bed_file``, and one that pipes ``samtools collate``
        into ``samtools fastq`` to produce paired-end FASTQ files.

        ``bed_file`` should include chr6 plus other chroms other than
        chr1-chr5, chr7-chr22, chrX, chrY, chrMT, such as
        chr1_KI270706v1_random and chrUn_KI270302v1 etc.

        Args:
            cram_dir: Directory containing the input ``.cram`` files.
            fasta_file: Reference FASTA used by ``samtools`` for CRAM decoding.
            bed_file: BED file of regions (HLA plus related contigs) to subset to.
            out_file: Path of the shell script to write.
            n_threads: Number of threads passed to ``samtools fastq -@``.

        Returns:
            None. The commands are written to ``out_file``; the subset CRAMs
            and FASTQs are written under ``{cram_dir}_subset`` when the
            generated script is executed.

        Raises:
            FileNotFoundError: If ``fasta_file`` or ``bed_file`` does not exist.
        """
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f'{fasta_file} not found')
        if not os.path.exists(bed_file):
            raise FileNotFoundError(f'{bed_file} not found')

        out_dir = cram_dir + '_subset'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        fs = sorted([x for x in os.listdir(cram_dir) if x.endswith('.cram')])
        with open(out_file, 'w') as outfile:
            for f in fs:
                subset_file = os.path.basename(f).replace('.cram', '_subset.cram')
                fq1 = subset_file.replace('.cram', '.R1.fastq.gz')
                fq2 = subset_file.replace('.cram', '.R2.fastq.gz')
                cmd = f'samtools view -T {fasta_file} -C -L {bed_file} {cram_dir}/{f} > {out_dir}/{subset_file}'
                cmd2 = f'samtools collate -u -O {out_dir}/{subset_file} | samtools fastq --reference {fasta_file} -@ {n_threads} -1 {out_dir}/{fq1} -2 {out_dir}/{fq2} -0 /dev/null -s /dev/null -n'
                outfile.write(cmd + '\n')
                outfile.write(cmd2 + '\n')

    def run_hlahd(
        self,
        fastq_dir: str = 'wgs_cram_subset',
        freq_data: str = 'freq_data',
        dict_data: str = 'dictionary',
        gene_split_file: str = 'HLA_gene.split.txt',
        out_file: str = 'run_hlahd.sh',
        out_dir: str = 'HLA-HD',
        n_threads: int = 4,
        min_read_length: int = 100,
        trim_rate: float = 0.95,
    ) -> None:
        """Write a script that runs HLA-HD on paired-end FASTQ files.

        For every ``*.R1.fastq.gz`` file found in ``fastq_dir``, writes one
        ``hlahd.sh`` command to ``out_file``, pairing it with the matching
        ``*.R2.fastq.gz`` file and deriving the sample ID from the filename.

        Args:
            fastq_dir: Directory containing paired ``*.R1.fastq.gz`` /
                ``*.R2.fastq.gz`` files.
            freq_data: Path to the HLA-HD frequency data directory.
            dict_data: Path to the HLA-HD dictionary directory.
            gene_split_file: Path to the HLA-HD gene split definition file.
            out_file: Path of the shell script to write.
            out_dir: Output directory passed to ``hlahd.sh`` for results.
            n_threads: Number of threads passed to ``hlahd.sh -t``.
            min_read_length: Minimum read length passed to ``hlahd.sh -m``.
            trim_rate: Trim rate passed to ``hlahd.sh -c``.

        Returns:
            None. The commands are written to ``out_file``.

        Raises:
            FileNotFoundError: If ``freq_data``, ``dict_data``, or
                ``gene_split_file`` does not exist.
        """
        if not os.path.exists(freq_data):
            raise FileNotFoundError(f'{freq_data} not found')
        if not os.path.exists(dict_data):
            raise FileNotFoundError(f'{dict_data} not found')
        if not os.path.exists(gene_split_file):
            raise FileNotFoundError(f'{gene_split_file} not found')

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fq1_files = sorted([os.path.join(fastq_dir, x) for x in os.listdir(fastq_dir) if x.endswith('.R1.fastq.gz')])
        with open(out_file, 'w') as outfile:
            for fq1 in fq1_files:
                fq2 = fq1.replace('.R1.fastq.gz', '.R2.fastq.gz')
                sample_id = os.path.basename(fq1).split('_subset')[0].split('.R1')[0].split('.R2')[0]
                cmd = f'hlahd.sh -t {n_threads} -m {min_read_length} -c {trim_rate} -f {freq_data} {fq1} {fq2} {gene_split_file} {dict_data} {sample_id} {out_dir}'
                outfile.write(cmd + '\n')

    def run_xhla(
        self,
        fastq_dir: str = 'wgs_cram_subset',
        sif_file: str = 'xhla.sif',
        genome_index: str = 'hg38',
        out_file: str = 'run_xhla.sh',
        n_threads: int = 4,
        out_dir: str = 'xHLA',
    ) -> None:
        """Write a script that aligns reads and runs xHLA via Singularity.

        For every ``*.R1.fastq.gz`` file found in ``fastq_dir``, writes a
        ``bwa mem`` + ``samtools sort``/``index`` command followed by a
        ``singularity exec`` invocation of xHLA's ``run.py``.

        Note:
            The genome reference must be hg38 rather than GRCh38, because the
            HLA region has been hard coded with a ``chr`` prefix in xHLA.

        Args:
            fastq_dir: Directory containing paired ``*.R1.fastq.gz`` /
                ``*.R2.fastq.gz`` files.
            sif_file: Path to the xHLA Singularity image file.
            genome_index: BWA genome index prefix (must be hg38-based).
            out_file: Path of the shell script to write.
            n_threads: Number of threads passed to ``bwa mem -t`` and
                ``samtools sort -@``.
            out_dir: Output directory for BAM files and xHLA results.

        Returns:
            None. The commands are written to ``out_file``.

        Raises:
            FileNotFoundError: If ``sif_file`` does not exist.
        """
        if not os.path.exists(sif_file):
            raise FileNotFoundError(f'{sif_file} not found')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fq1_files = sorted([os.path.join(fastq_dir, x) for x in os.listdir(fastq_dir) if x.endswith('.R1.fastq.gz')])
        with open(out_file, 'w') as outfile:
            for fq1 in fq1_files:
                fq2 = fq1.replace('.R1.fastq.gz', '.R2.fastq.gz')
                sample_id = os.path.basename(fq1).split('_subset')[0]
                bam_file = f'{out_dir}/{sample_id}.bam'
                cmd = f'bwa mem -t {n_threads} {genome_index} {fq1} {fq2} | samtools sort -@ {n_threads} -o {bam_file}; samtools index {bam_file}'
                outfile.write(cmd + '\n')
                cmd = f'singularity exec -B `pwd`:`pwd` --pwd `pwd` {sif_file} run.py --sample_id {sample_id} --input_bam_path {bam_file} --output_path {out_dir}/{sample_id}'
                outfile.write(cmd + '\n')

    def run_optitype(
        self,
        fastq_dir: str = 'wgs_cram_subset',
        sif_file: str = 'OptiType.sif',
        out_file: str = 'run_optitype.sh',
        n_threads: int = 4,
        out_dir: str = 'OptiType',
        seq_type: str = 'dna',
    ) -> None:
        """Write a script that runs OptiType via Singularity on paired-end FASTQ files.

        For every ``*.R1.fastq.gz`` file found in ``fastq_dir``, writes one
        ``singularity exec ... OptiTypePipeline.py`` command to ``out_file``.

        Args:
            fastq_dir: Directory containing paired ``*.R1.fastq.gz`` /
                ``*.R2.fastq.gz`` files.
            sif_file: Path to the OptiType Singularity image file.
            out_file: Path of the shell script to write.
            n_threads: Unused by the generated command; kept for interface
                consistency with the other ``run_*`` methods.
            out_dir: Output directory passed to ``OptiTypePipeline.py -o``.
            seq_type: Sequencing type flag for OptiType, e.g. ``'dna'`` or
                ``'rna'`` (passed as ``--{seq_type}``).

        Returns:
            None. The commands are written to ``out_file``.

        Raises:
            FileNotFoundError: If ``sif_file`` does not exist.
        """
        if not os.path.exists(sif_file):
            raise FileNotFoundError(f'{sif_file} not found')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fq1_files = sorted([os.path.join(fastq_dir, x) for x in os.listdir(fastq_dir) if x.endswith('.R1.fastq.gz')])
        with open(out_file, 'w') as outfile:
            for fq1 in fq1_files:
                fq2 = fq1.replace('.R1.fastq.gz', '.R2.fastq.gz')
                sample_id = os.path.basename(fq1).split('_subset')[0]
                cmd = f'singularity exec -B `pwd`:`pwd` --pwd `pwd` {sif_file} OptiTypePipeline.py --{seq_type} -i {fq1} {fq2} -o {out_dir} -p {sample_id} -v'
                outfile.write(cmd + '\n')
