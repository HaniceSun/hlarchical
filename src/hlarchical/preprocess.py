import os
import pandas as pd
import subprocess
import gzip

class Preprocessor:
    def __init__(self):
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.hla_chrom = ['6', 'chr6']
        self.ped_cols = ['FID', 'IID', 'PID', 'MID', 'SEX', 'PHENOTYPE'] + [f'{x}_{a}' for x in self.HLA for a in ['A1', 'A2']]

    def bed_to_vcf(self, in_file='HAPMAP_CEU.bed', input_genome_build='hg18', output_genome_build='GRCh37'):
        bfile = in_file.split('.bed')[0]
        cmd = f'plink2 --bfile {bfile} --recode vcf bgz --out {bfile}'
        print(cmd)
        subprocess.run(cmd, shell=True)

        samples_sorted = f'{bfile}_samples_sorted.txt'
        cmd = f'bcftools query -l {bfile}.vcf.gz | sort > {samples_sorted}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = f'bcftools view -S {samples_sorted} {bfile}.vcf.gz -Oz -o {bfile}_sampleSorted.vcf.gz; rm {bfile}.vcf.gz'
        print(cmd)
        subprocess.run(cmd, shell=True)
        fasta_source = self.get_genome_reference(genome_build=input_genome_build)
        fasta_dest = self.get_genome_reference(genome_build=output_genome_build)
        cmd = f'bcftools norm -m-any --check-ref s -f {fasta_source} {bfile}_sampleSorted.vcf.gz -Oz -o {bfile}_norm.vcf.gz'
        print(cmd)
        subprocess.run(cmd, shell=True)
        if input_genome_build != output_genome_build:
            self.vcf_liftover(f'{bfile}_sampleSorted.vcf.gz', out_file=f'{bfile}_lifted.vcf.gz', input_genome_build=input_genome_build, output_genome_build=output_genome_build)
            cmd = f'bcftools sort {bfile}_lifted.vcf.gz -Oz -o {bfile}_posSorted.vcf.gz; bcftools index {bfile}_posSorted.vcf.gz'
            print(cmd)
            subprocess.run(cmd, shell=True)
            os.remove(f'{bfile}_lifted.vcf.gz')
        else:
            cmd = f'bcftools sort {bfile}_sampleSorted.vcf.gz -Oz -o {bfile}_posSorted.vcf.gz; bcftools index {bfile}_posSorted.vcf.gz'
            print(cmd)
            subprocess.run(cmd, shell=True)
        cmd = f'bcftools norm -m-any --check-ref s -f {fasta_dest} {bfile}_posSorted.vcf.gz -Oz -o {bfile}.vcf.gz'
        print(cmd)
        subprocess.run(cmd, shell=True)
        os.remove(f'{bfile}_sampleSorted.vcf.gz')
        os.remove(f'{bfile}_posSorted.vcf.gz')
        os.remove(f'{bfile}_norm.vcf.gz')

    def vcf_liftover(self, in_file, out_file, input_genome_build='hg18', output_genome_build='GRCh37'):
        if input_genome_build in ['hg18', 'NCBI36'] and output_genome_build in ['GRCh37', 'hg19']:
            chain_url = 'http://ftp.ensembl.org/pub/assembly_mapping/homo_sapiens/NCBI36_to_GRCh37.chain.gz'
            chain_file = chain_url.split('/')[-1]
            if os.path.exists(chain_file) == False:
                subprocess.run(f'wget {chain_url}', shell=True)
            fasta_source = self.get_genome_reference(genome_build=input_genome_build)
            fasta_dest = self.get_genome_reference(genome_build=output_genome_build)
            cmd = f'bcftools +liftover -Ou {in_file} -- -s {fasta_source} -f {fasta_dest} -c {chain_file} | bcftools sort -Oz -o {out_file}'
            print(cmd)
            subprocess.run(cmd, shell=True)

    def ped_to_vcf(self, in_file='HAPMAP_CEU_HLA.ped', genome_build='GRCh37', hla_pos_file='HLA_gene_position.txt'):
        hla_pos_file = hla_pos_file.split('.txt')[0] + f'_{genome_build}.txt'
        if os.path.exists(hla_pos_file) == False:
            self.get_hla_position(out_file=hla_pos_file, genome_build=genome_build)
        df_pos = pd.read_table(hla_pos_file, header=None, sep='\t', dtype=str)
        pos_dict = {}
        for n in range(df_pos.shape[0]):
            gene = df_pos.iloc[n, 0]
            chrom = df_pos.iloc[n, 1]
            pos = df_pos.iloc[n, 2]
            pos_dict[gene] = (chrom, pos)

        df = pd.read_table(in_file, header=None, sep='\t', dtype=str)
        if df.shape[1] == len(self.ped_cols):
            df.columns = self.ped_cols
        else:
            raise ValueError('The number of columns in the PED file does not match the expected number of columns.')

        D = {}
        A = []
        S = []
        for n in range(df.shape[0]):
            row = df.iloc[n]
            fid = row['FID']
            iid = row['IID']
            k = '_'.join([fid, iid])
            if k not in S:
                S.append(k)
            D.setdefault(k, {})
            for gene in self.HLA:
                for a in ['A1', 'A2']:
                    allele = row[f'{gene}_{a}']
                    if allele == '0':
                        allele2d = '.'
                        allele4d = '.'
                    elif len(allele) < 4:
                        allele2d = f'{gene}:{allele}'
                        allele4d = '.'
                    elif len(allele) == 4:
                        allele2d = f'{gene}:{allele[0:-2]}'
                        allele4d = f'{gene}:{allele[0:-2]}:{allele[-2:]}'
                    elif len(allele) == 5:
                        allele2d = f'{gene}:{allele[0:-3]}'
                        allele4d = f'{gene}:{allele[0:-3]}:{allele[-3:]}'
                    D[k].setdefault(a, [])
                    D[k][a].append(allele2d)
                    D[k][a].append(allele4d)
                    if allele2d != '.':
                        A.append(allele2d)
                    if allele4d != '.':
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
                    if allele in D[sample]['A1'] and allele in D[sample]['A2']:
                        gt = '1/1'
                    elif allele in D[sample]['A1'] or allele in D[sample]['A2']:
                        gt = '0/1'
                    else:
                        gt = '0/0'
                    row.append(gt)
                L.append(row)

        bfile = in_file.split('.ped')[0]
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
        cmd = f'bcftools sort {bfile}_sampleSorted.vcf.gz -Oz -o {bfile}.vcf.gz; rm {bfile}_sampleSorted.vcf.gz; bcftools index {bfile}.vcf.gz'
        print(cmd)
        subprocess.run(cmd, shell=True)

    def make_reference(self, in_file=['HAPMAP_CEU.vcf.gz', 'HAPMAP_CEU_HLA.vcf.gz'], maker_file=None, out_file='HAPMAP_CEU_REF_phased.vcf.gz', burnin=10, iterations=15):
        concated_file = out_file.split('_phased')[0] + '_concated.vcf.gz'
        cmd = f'bcftools concat {" ".join(in_file)} -Oz -o {concated_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        self.vcf_pos_unique(concated_file)	
        bfile = concated_file.split('.vcf.gz')[0]
        cmd = f'bcftools sort {bfile}_PosUniq.vcf.gz -Oz -o {concated_file}; rm {bfile}_PosUniq.vcf.gz; bcftools index {concated_file}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        cmd = f'beagle gt={concated_file} out={out_file.split(".vcf")[0]} burnin={burnin} iterations={iterations}'
        print(cmd)
        subprocess.run(cmd, shell=True)
        os.remove(concated_file)
        os.remove(f'{concated_file}.csi')

    def phase_sample(self, sample_file='GDA.vcf.gz', ref_file='HAPMAP_CEU_REF_phased.vcf.gz', out_file='GDA_phased_HAPMAP_CEU_REF.vcf.gz', sample_build='GRCh37', ref_build='GRCh37'):
        bfile = sample_file.split('.vcf')[0].split('.bed')[0]
        if sample_file.find('.vcf') == -1:
            cmd = f'plink2 --bfile {bfile} --chr 6 --recode vcf bgz --out {bfile}'
            print(cmd)
            subprocess.run(cmd, shell=True)
        if sample_build == ref_build:
            fasta_file = self.get_genome_reference(genome_build=sample_build)
            cmd = f'bcftools norm -m-any --check-ref s -f {fasta_file} {bfile}.vcf.gz -Oz -o {bfile}_norm.vcf.gz'
            print(cmd)
            subprocess.run(cmd, shell=True)
            cmd = f'beagle gt={bfile}_norm.vcf.gz ref={ref_file} out={out_file.split(".vcf")[0]}'
            print(cmd)
            subprocess.run(cmd, shell=True)
            #os.remove(f'{bfile}_norm.vcf.gz')

    def get_genome_reference(self, genome_build='GRCh37'):
        if genome_build in ['GRCh38', 'hg38']:
            fasta_url = 'ftp://ftp.ensembl.org/pub/release-101/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz'
        elif genome_build in ['GRCh37', 'hg19']:
            fasta_url = 'ftp://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz'
        elif genome_build in ['GRCh36', 'hg18', 'NCBI36']:
            fasta_url = 'ftp://ftp.ensembl.org/pub/release-54/fasta/homo_sapiens/dna/Homo_sapiens.NCBI36.54.dna.toplevel.fa.gz'
        fasta_file = fasta_url.split('/')[-1].split('.gz')[0]
        files = os.listdir('.')
        if fasta_file not in files:
            subprocess.run(f'wget {fasta_url}; gunzip {fasta_file}.gz', shell=True)
        return fasta_file

    def get_hla_position(self, out_file, genome_build='GRCh38'):
        D = {}
        if genome_build in ['GRCh38', 'hg38']:
            gtf_url = 'ftp://ftp.ensembl.org/pub/release-101/gtf/homo_sapiens/Homo_sapiens.GRCh38.101.gtf.gz'

        elif genome_build in ['GRCh37', 'hg19']:
            gtf_url = 'ftp://ftp.ensembl.org/pub/release-75/gtf/homo_sapiens/Homo_sapiens.GRCh37.75.gtf.gz'

        elif genome_build in ['hg18', 'NCBI36', 'GRCh36']:
            #gtf_url = 'ftp://ftp.ensembl.org/pub/release-54/gtf/homo_sapiens/Homo_sapiens.NCBI36.54.gtf.gz'
            gtf_url = None
            # from DEEP-HLA
            D['HLA-A'] = [('6', 30019970)]
            D['HLA-B'] = [('6', 31431272)]
            D['HLA-C'] = [('6', 31346171)]
            D['HLA-DPA1'] = [('6', 33145064)]
            D['HLA-DPB1'] = [('6', 33157346)]
            D['HLA-DQA1'] = [('6', 32716284)]
            D['HLA-DQB1'] = [('6', 32739039)]
            D['HLA-DRB1'] = [('6', 32660042)]

        if gtf_url:
            files = os.listdir('.')
            self.gtf_file = gtf_url.split('/')[-1].split('.gz')[0]
            if self.gtf_file not in files:
                subprocess.run(f'wget {gtf_url}; gunzip {self.gtf_file}.gz', shell=True)
            print(self.gtf_file)

            with open(self.gtf_file) as infile:
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
                        if typ  == 'gene':
                            gene = '.'
                            if info.find('gene_name') != -1:
                                for attr in info.split(';'):
                                    attr = attr.strip()
                                    if attr.startswith('gene_name'):
                                        gene = attr.split(' ')[1].replace('"', '')
                            if chrom in self.hla_chrom and gene in self.HLA:
                                D.setdefault(gene, [])
                                D[gene].append((chrom, start, end, strand))
        L = []
        for gene in self.HLA:
            if gene in D:
                item = sorted(D[gene], key=lambda x: x[1])
                L.append([gene, item[0][0], item[0][1]])
        df = pd.DataFrame(L)
        df.to_csv(out_file, sep='\t', header=False, index=False)

    def vcf_pos_unique(self, in_file):
        seen = {}
        out_file = in_file.replace('.vcf', '_PosUniq.vcf')
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
    pp = Preprocessor()
    pp.bed_to_vcf()
    pp.ped_to_vcf()
    pp.make_reference()
    pp.phase_sample(sample_file='GDA.bed')
