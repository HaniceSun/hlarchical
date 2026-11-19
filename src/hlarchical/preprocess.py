from .utils import *


class Preprocessor:
    """VCF and reference-panel preprocessing utilities for HLA imputation.

    Wraps common ``bcftools``/``beagle`` shell pipelines used to prepare
    array/WGS sample VCFs and HLA reference panels for imputation: building
    a pseudo-VCF of HLA alleles, subsetting and strand-fixing sample
    variants, building a merged/phased reference panel, and phasing a
    sample against that reference panel.

    Attributes:
        HLA: The eight classical HLA genes handled throughout this class.
    """

    def __init__(self) -> None:
        """Initialize the preprocessor with the list of classical HLA genes."""
        self.HLA: list[str] = [
            'HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1',
        ]

    def hlarchical_table_to_vcf(
        self,
        in_file: str = '1000G_WGS_HLA-HD.txt',
        genome_build: str = 'GRCh37',
        hla_pos_file: str = 'HLA_gene_position_GRCh37.txt',
    ) -> None:
        """Convert a long-format hlarchical allele table into a pseudo-VCF.

        Reads a table with one row per (SampleID, HLA, Allele1, Allele2),
        derives 2-digit and 4-digit allele resolutions, and emits one VCF
        record per distinct allele (treated as a presence/absence "variant")
        with a 0/0, 0/1, or 1/1 genotype per sample depending on whether
        that allele is present in Allele1/Allele2. The resulting VCF is
        written, sample-sorted, coordinate-sorted, bgzipped, and indexed
        via ``bcftools``.

        Args:
            in_file: Input table path (tab-separated, with header
                ``SampleID, HLA, Allele1, Allele2``).
            genome_build: Genome build used to look up HLA gene positions
                if ``hla_pos_file`` does not already exist.
            hla_pos_file: Path to a tab-separated, headerless file of
                ``gene, chrom, pos`` used to place each allele "variant" on
                the genome. If missing, it is looked up under the package
                data directory or generated via :meth:`get_hla_position`.

        Raises:
            FileNotFoundError: If ``hla_pos_file`` cannot be found or
                generated.
        """
        if not os.path.exists(hla_pos_file):
            if os.path.exists(data_dir + '/' + hla_pos_file):
                hla_pos_file = data_dir + '/' + hla_pos_file
            else:
                self.get_hla_position(out_file=hla_pos_file, genome_build=genome_build)
        if not os.path.exists(hla_pos_file):
            raise FileNotFoundError(f'Error: HLA position file {hla_pos_file} not found')
        df_pos = pd.read_table(hla_pos_file, header=None, sep='\t', dtype=str)
        pos_dict = {}
        for n in range(df_pos.shape[0]):
            gene = df_pos.iloc[n, 0]
            chrom = df_pos.iloc[n, 1]
            pos = df_pos.iloc[n, 2]
            pos_dict[gene] = (chrom, pos)

        df = pd.read_table(in_file, header=0, sep='\t', dtype=str)

        D = {}
        A = []
        S = []
        for k, g in df.groupby('SampleID'):
            if k not in S:
                S.append(k)
            D.setdefault(k, {})
            for gene in self.HLA:
                for a in ['Allele1', 'Allele2']:
                    allele2d = '.'
                    allele4d = '.'
                    allele = g.loc[g['HLA'] == gene, a].values[0]
                    D[k].setdefault(a, [])
                    if allele not in ['.', './.', '0', 'NA']:
                        alleles = allele.split(':')
                        allele2d = f'{alleles[0]}:{alleles[1]}'
                        allele4d = f'{alleles[0]}:{alleles[1]}:{alleles[2]}'
                        D[k][a].append(allele2d)
                        D[k][a].append(allele4d)
                        A.append(allele2d)
                        A.append(allele4d)

        L = []
        for allele in sorted(set(A)):
            gene = allele.split(':')[0]
            if gene in pos_dict:
                chrom, pos = pos_dict[gene]
                ref = 'A'
                alt = 'P'
                row = [chrom, pos, allele, ref, alt, '.', 'PASS', '.', 'GT']
                for sample in S:
                    if allele in D[sample]['Allele1'] and allele in D[sample]['Allele2']:
                        gt = '1/1'
                    elif allele in D[sample]['Allele1'] or allele in D[sample]['Allele2']:
                        gt = '0/1'
                    else:
                        gt = '0/0'
                    row.append(gt)
                L.append(row)

        bfile = in_file.split('.txt')[0]
        df = pd.DataFrame(L)
        df.columns = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'] + S
        out_file = f'{bfile}.vcf'
        with open(out_file, 'w') as outfile:
            outfile.write('##fileformat=VCFv4.2\n')
            outfile.write('##source=VCFPhaser\n')
            outfile.write('##contig=<ID=6>\n')
            outfile.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        df.to_csv(out_file, sep='\t', index=False, mode='a')

        samples_sorted = f'{bfile}_samples_sorted.txt'
        cmd = f'bcftools query -l {out_file} | sort > {samples_sorted}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = f'bcftools view -S {samples_sorted} {out_file} -Oz -o {bfile}_sampleSorted.vcf.gz; rm {out_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = (
            f'bcftools sort {bfile}_sampleSorted.vcf.gz -Oz -o {bfile}.vcf.gz; '
            f'rm {bfile}_sampleSorted.vcf.gz; bcftools index {bfile}.vcf.gz'
        )
        print(cmd)
        subprocess.run(cmd, shell=True)

    def subset_samples_vcf(self, vcf_file: str, sample_list: str, out_file: str, n_threads: int = 4) -> None:
        """Subset a VCF to a given list of samples via ``bcftools view -S``.

        Args:
            vcf_file: Input VCF/BCF path.
            sample_list: Path to a file listing sample IDs to keep, one per line.
            out_file: Output bgzipped VCF path; also indexed with ``tabix``.
            n_threads: Number of threads passed to ``bcftools view``.
        """
        cmd = f'bcftools view --threads {n_threads} -S {sample_list} {vcf_file} -Oz -o {out_file}; tabix -p vcf {out_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def subset_variants_vcf(
        self,
        vcf_file: str,
        genome_build: str = 'GRCh37',
        hla_pos_file: str = 'HLA_gene_position_GRCh37.txt',
        flank: float = 1e6,
        n_threads: int = 4,
    ) -> None:
        """Subset a VCF to the HLA region (plus flanking bases) via ``bcftools view -r``.

        The HLA region boundaries are derived from ``hla_pos_file`` as the
        min/max gene start position across all HLA genes, expanded by
        ``flank`` bases on each side.

        Args:
            vcf_file: Input bgzipped VCF path; output is written next to it
                as ``<prefix>_variantSubset.vcf.gz`` and indexed with ``tabix``.
            genome_build: Genome build used to look up HLA gene positions if
                ``hla_pos_file`` does not already exist.
            hla_pos_file: Path to a tab-separated, headerless file of
                ``gene, chrom, pos``. If missing, it is looked up under the
                package data directory or generated via :meth:`get_hla_position`.
            flank: Number of bases to pad on each side of the HLA region.
            n_threads: Number of threads passed to ``bcftools view``.

        Raises:
            FileNotFoundError: If ``hla_pos_file`` cannot be found or generated.
        """
        if not os.path.exists(hla_pos_file):
            if os.path.exists(data_dir + '/' + hla_pos_file):
                hla_pos_file = data_dir + '/' + hla_pos_file
            else:
                self.get_hla_position(out_file=hla_pos_file, genome_build=genome_build)
        if not os.path.exists(hla_pos_file):
            raise FileNotFoundError(f'Error: HLA position file {hla_pos_file} not found')

        out_file = vcf_file.split('.vcf.gz')[0] + '_variantSubset.vcf.gz'
        df = pd.read_table(hla_pos_file, header=None, sep='\t')
        chrom = df.iloc[0, 1].astype(str)
        start = df.iloc[:, 2].min() - int(flank)
        end = df.iloc[:, 2].max() + int(flank)
        cmd = f'bcftools view --threads {n_threads} -r {chrom}:{start}-{end} {vcf_file} -Oz -o {out_file}; tabix -p vcf {out_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def fixref_vcf(self, vcf_file: str, genome_build: str = 'GRCh37') -> None:
        """Flip/swap REF-strand mismatches in a VCF using ``bcftools +fixref``.

        Writes ``<prefix>_fixref.vcf.gz`` and prints a ``bcftools +fixref``
        stats summary comparing the fixed VCF against the reference FASTA.

        Args:
            vcf_file: Input VCF path.
            genome_build: Genome build used to obtain the reference FASTA
                (see :meth:`get_genome_reference`).
        """
        out_file = vcf_file.split('.vcf')[0] + '_fixref.vcf.gz'
        fasta_file = self.get_genome_reference(genome_build=genome_build)
        # flip/swap using fixref
        cmd = f'bcftools +fixref {vcf_file} -Oz -o {out_file} -- -d -f {fasta_file} -m flip'
        print(cmd)
        subprocess.run(cmd, shell=True)
        # stats after fixref
        cmd = f'bcftools +fixref {out_file} -- -f {fasta_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def make_reference(
        self,
        ref_variant_vcf: str = '1000G_array_broad_overlappedWGS_variantSubset_fixref.vcf.gz',
        ref_hla_vcf: str = '1000G_WGS_HLA-HD_overlappedBroad.vcf.gz',
        out_file: str = '1000G_REF_phased.vcf.gz',
        marker_file: str | None = None,
        burnin: int = 10,
        iterations: int = 15,
    ) -> None:
        """Build a phased reference panel by merging array-variant and HLA-allele VCFs.

        Verifies that ``ref_variant_vcf`` and ``ref_hla_vcf`` contain the
        same samples in the same order, concatenates them, deduplicates
        overlapping positions (see :meth:`unique_vcf_pos`), and phases the
        merged callset with ``beagle``.

        Args:
            ref_variant_vcf: Fixref'd, HLA-region-subset array/WGS variant
                VCF (see :meth:`subset_variants_vcf` and :meth:`fixref_vcf`).
            ref_hla_vcf: Pseudo-VCF of HLA alleles for the same samples
                (see :meth:`hlarchical_table_to_vcf`).
            out_file: Output path prefix for the phased reference panel
                (``.vcf.gz`` suffix stripped when passed to ``beagle``).
            marker_file: Currently unused; accepted for future use.
            burnin: Number of ``beagle`` burn-in iterations.
            iterations: Number of ``beagle`` phasing iterations.

        Returns:
            None. Prints an error and returns early if sample names in the
            two input VCFs do not match, instead of raising.
        """
        cmd = f'bcftools query -l {ref_variant_vcf}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ref_variant_samples = result.stdout.strip().split('\n')
        print(f'Number of samples in variant VCF: {len(ref_variant_samples)}')
        cmd = f'bcftools query -l {ref_hla_vcf}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ref_hla_samples = result.stdout.strip().split('\n')
        print(f'Number of samples in HLA VCF: {len(ref_hla_samples)}')
        if ref_variant_samples != ref_hla_samples:
            print('Error: Sample names in variant and HLA VCF files do not match.')
            return

        concated_file = out_file.split('_phased')[0] + '_concated.vcf.gz'
        bfile = concated_file.split('.vcf.gz')[0]
        pos_uniq_file = bfile + '_posUniq.vcf.gz'
        cmd = f'bcftools concat {" ".join([ref_variant_vcf, ref_hla_vcf])} -Oz -o {concated_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        self.unique_vcf_pos(concated_file)
        cmd = f'bcftools sort {pos_uniq_file} -Oz -o {concated_file}; bcftools index {concated_file}; rm {pos_uniq_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = f'beagle gt={concated_file} out={out_file.split(".vcf")[0]} burnin={burnin} iterations={iterations}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        os.remove(concated_file)
        os.remove(f'{concated_file}.csi')

    def phase_sample_on_reference(
        self,
        sample_vcf: str = 'GDA.vcf.gz',
        ref_vcf: str = '1000G_REF_phased.vcf.gz',
        genome_build: str = 'GRCh37',
        subset_variants: bool = True,
        fix_ref: bool = True,
        flank: float = 1e6,
    ) -> None:
        """Phase/impute a sample VCF against a phased reference panel using ``beagle``.

        Optionally subsets the sample VCF to the HLA region
        (:meth:`subset_variants_vcf`) and fixes strand/REF mismatches
        (:meth:`fixref_vcf`) before running ``beagle`` with ``ref_vcf`` as
        the reference panel.

        Args:
            sample_vcf: Input, tabix/csi-indexed sample VCF.
            ref_vcf: Phased reference panel VCF (see :meth:`make_reference`).
                If not found in the current directory, it is looked up
                under the package data directory.
            genome_build: Genome build of ``sample_vcf``, used when
                subsetting/fixing reference.
            subset_variants: Whether to subset ``sample_vcf`` to the HLA
                region plus ``flank`` before phasing.
            fix_ref: Whether to run ``bcftools +fixref`` on ``sample_vcf``
                before phasing.
            flank: Bases of padding used when subsetting to the HLA region.

        Raises:
            FileNotFoundError: If ``sample_vcf``, its index, or ``ref_vcf``
                cannot be found.
        """
        out_file = sample_vcf.split('.vcf')[0] + '_phased_on_' + ref_vcf
        if not os.path.exists(sample_vcf):
            raise FileNotFoundError(f'Error: Sample VCF {sample_vcf} not found')
        elif not os.path.exists(sample_vcf + '.tbi') and not os.path.exists(sample_vcf + '.csi'):
            raise FileNotFoundError(f'Error: Sample VCF index {sample_vcf}.tbi or {sample_vcf}.csi not found')

        if not os.path.exists(ref_vcf):
            ref_vcf = data_dir + '/' + ref_vcf
            if not os.path.exists(ref_vcf):
                raise FileNotFoundError(f'Error: Reference VCF {ref_vcf} not found')
            else:
                print(f'using reference: {ref_vcf}')

        if subset_variants:
            print('subsetting sample variants to HLA region...')
            self.subset_variants_vcf(sample_vcf, genome_build=genome_build, flank=flank)
            sample_vcf = sample_vcf.split('.vcf')[0] + '_variantSubset.vcf.gz'
        if fix_ref:
            print('performing fixref on sample VCF...')
            self.fixref_vcf(sample_vcf, genome_build=genome_build)
            sample_vcf = sample_vcf.split('.vcf')[0] + '_fixref.vcf.gz'
        cmd = f'beagle gt={sample_vcf} ref={ref_vcf} out={out_file.split(".vcf")[0]}'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def get_genome_reference(self, genome_build: str = 'GRCh37') -> str:
        """Download (if needed) and return the path to a reference FASTA.

        Fetches the Ensembl primary-assembly FASTA for the requested
        genome build into the current directory, decompresses it, and
        returns its filename. If the decompressed file already exists,
        skips downloading.

        Args:
            genome_build: Either ``'GRCh37'`` (Ensembl release 75) or
                ``'GRCh38'`` (Ensembl release 101). Any other value leaves
                ``fasta_url`` unset and raises ``UnboundLocalError``.

        Returns:
            The local filename of the decompressed reference FASTA.
        """
        if genome_build in ['GRCh37']:
            fasta_url = 'ftp://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz'
        elif genome_build in ['GRCh38']:
            fasta_url = 'ftp://ftp.ensembl.org/pub/release-101/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz'
        fasta_file = fasta_url.split('/')[-1].split('.gz')[0]
        files = os.listdir('.')
        if fasta_file not in files:
            subprocess.run(f'wget {fasta_url}; gunzip {fasta_file}.gz', shell=True)
        return fasta_file

    def get_hla_position(
        self,
        out_file: str,
        genome_build: str = 'GRCh37',
        hla_chroms: list[str] = ['6', 'chr6'],
    ) -> None:
        """Locate the genomic start position of each HLA gene via an Ensembl GTF.

        Downloads (if needed) the Ensembl GTF for ``genome_build``, parses
        ``gene`` feature records on ``hla_chroms``, and for each gene in
        :attr:`HLA` writes its chromosome and earliest start position (by
        start coordinate) to ``out_file``.

        Args:
            out_file: Output path for a tab-separated, headerless
                ``gene, chrom, pos`` file.
            genome_build: Either ``'GRCh37'`` (Ensembl release 75) or
                ``'GRCh38'`` (Ensembl release 101).
            hla_chroms: Chromosome name(s) (with/without ``chr`` prefix) to
                scan for HLA gene records in the GTF.
        """
        D = {}
        if genome_build in ['GRCh37']:
            gtf_url = 'ftp://ftp.ensembl.org/pub/release-75/gtf/homo_sapiens/Homo_sapiens.GRCh37.75.gtf.gz'
        elif genome_build in ['GRCh38']:
            gtf_url = 'ftp://ftp.ensembl.org/pub/release-101/gtf/homo_sapiens/Homo_sapiens.GRCh38.101.gtf.gz'

        if gtf_url:
            gtf_file = gtf_url.split('/')[-1].split('.gz')[0]
            if not os.path.exists(gtf_file):
                subprocess.run(f'wget {gtf_url}; gunzip {gtf_file}.gz', shell=True)
            print(f'downloaded {gtf_file}')

            with open(gtf_file) as infile:
                for line in infile:
                    line = line.strip()
                    fields = line.split('\t')
                    if len(fields) > 8:
                        chrom = fields[0]
                        typ = fields[2]
                        start = int(fields[3])
                        end = int(fields[4])
                        strand = fields[6]
                        info = fields[8]
                        if typ == 'gene':
                            gene = '.'
                            if info.find('gene_name') != -1:
                                for attr in info.split(';'):
                                    attr = attr.strip()
                                    if attr.startswith('gene_name'):
                                        gene = attr.split(' ')[1].replace('"', '')
                            if chrom in hla_chroms and gene in self.HLA:
                                D.setdefault(gene, [])
                                D[gene].append((chrom, start, end, strand))
            os.remove(gtf_file)
        L = []
        for gene in self.HLA:
            if gene in D:
                item = sorted(D[gene], key=lambda x: x[1])
                L.append([gene, item[0][0], item[0][1]])
        df = pd.DataFrame(L)
        df.to_csv(out_file, sep='\t', header=False, index=False)

    def unique_vcf_pos(self, in_file: str) -> None:
        """Rewrite a VCF so that duplicate positions become consecutive unique positions.

        For each run of records sharing the same (CHROM, POS), the k-th
        duplicate (0-indexed) has its POS incremented by k. This lets
        downstream tools (e.g. ``beagle``) that require unique positions
        process concatenated VCFs (such as variant + pseudo-HLA-allele
        records) without collisions. Header lines are copied through
        unchanged.

        Args:
            in_file: Input VCF path, plain text (``.vcf``) or bgzipped
                (``.vcf.gz``). Output is written to
                ``<in_file with 'vcf' replaced by 'posUniq.vcf'>``, preserving
                the ``.gz`` suffix/compression when present.
        """
        seen = {}
        out_file = in_file.replace('.vcf', '_posUniq.vcf')
        if in_file.endswith('.vcf.gz'):
            infile = gzip.open(in_file, 'rt')
            outfile = gzip.open(out_file, 'wt')
        else:
            infile = open(in_file, 'r')
            outfile = open(out_file, 'w')

        for line in infile:
            line = line.strip()
            if line.startswith('#'):
                outfile.write(line + '\n')
            else:
                fields = line.split("\t")
                chrom = fields[0]
                pos = int(fields[1])
                key = (chrom, pos)
                count = seen.get(key, 0)
                seen[key] = count + 1
                new_pos = pos + count
                fields[1] = str(new_pos)
                outfile.write('\t'.join(fields) + '\n')
        infile.close()
        outfile.close()


if __name__ == '__main__':
    pass
