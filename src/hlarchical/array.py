import os
import gzip
import pandas as pd
import subprocess
import torch
from importlib import resources

class Array():
    def __init__(self):
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def run_snp2hla(self, in_file='1958BC', ref_file='HM_CEU_REF', snp2hla_dir=None, heap_size=2000, window_size=1000):
        if not snp2hla_dir:
            snp2hla_dir = f'{resources.files("hlarchical").parent.parent}/vendor/SNP2HLA/home'

        out_file = os.path.abspath(f'{in_file}_{ref_file}')

        if os.path.exists(f'{in_file}.bed'):
            in_file = os.path.abspath(in_file)
        else:
            raise FileNotFoundError(f'Input file {in_file}.bed not found')

        if os.path.exists(f'{ref_file}.bed'):
            ref_file = os.path.abspath(ref_file)
        else:
            raise FileNotFoundError(f'Reference file {ref_file}.bed not found')

        os.chdir(snp2hla_dir)
        cmd = f'tcsh SNP2HLA.csh {in_file} {ref_file} {out_file} plink {heap_size} {window_size}'
        subprocess.run(cmd, shell=True, check=True)

    def run_deephla(self, mode='train', in_file='1958BC_Pan-Asian_REF', ref_file='Pan-Asian_REF', subset=[], model_json=None, model_dir='model', deephla_dir=None):
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

            hla_json = f'{ref_file}.hla.json'
            if not os.path.exists(hla_json):
                print('Generating HLA info JSON file...')
                cmd = f'conda run -n DEEP-HLA python {deephla_dir}/make_hlainfo.py --ref {ref_file} --out {ref_file}.hla.json'
                subprocess.run(cmd, shell=True, check=True)

            if not model_json:
                model_json = f'{ref_file}.model.json'
            if os.path.exists(model_json):
                model_json = model_json.split('.model.json')[0]
                hla_json = hla_json.split('.hla.json')[0]
                cmd = f'conda run -n DEEP-HLA python {deephla_dir}/train.py --ref {ref_file} --sample {in_file} --model {model_json} --hla {hla_json} --model-dir {model_dir}'
                subprocess.run(cmd, shell=True, check=True)
            else:
                raise FileNotFoundError(f'{model_json} not found')

        elif mode == 'impute':
            model_json = model_json.split('.model.json')[0]
            hla_json = ref_file
            cmd = f'conda run -n DEEP-HLA python {deephla_dir}/impute.py --sample {in_file} --model {model_json} --hla  {hla_json} --model-dir {model_dir} --out {in_file}'
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)

    def format_output(self, in_file='1958BC_Euro.bgl.phased', fam='1958BC_Euro.fam', out_file='1958BC_Euro_digit4.txt', digit=4, from_tool='snp2hla'):
        if from_tool == 'snp2hla':
            sep = ' '
            skiprows = 1
            col = 1
            in_header = 0
        elif from_tool == 'deephla':
            sep = '\t'
            skiprows = 0
            col = 0
            in_header = None
            samples = pd.read_table(in_file.replace('.deephla.phased', '.fam'), sep=' ', header=None).iloc[:, 1].tolist()

        if in_file.endswith('.phased'):
            df = pd.read_table(in_file, sep=sep, skiprows=skiprows, header=in_header)
            df = df.loc[df.iloc[:, col].str.startswith('HLA'), ]

            header = ['SampleID', 'HLA', 'Allele1', 'Allele2']
            wh = [len(x.split('_')[2]) == digit for x in df.iloc[:, col]]
            df = df.loc[wh, ]

            L = []
            for n in range(col + 1, df.shape[1], 2):
                if from_tool == 'snp2hla':
                    sample_id = df.columns[n]
                elif from_tool == 'deephla':
                    sample_id = samples[int((n-col)/2)]
                allele1 = {}
                allele2 = {}
                for m in range(df.shape[0]):
                    allele = df.iloc[m, col]
                    if digit == 4:
                        allele = f'{allele[0:-2]}:{allele[-2:]}'
                    fields = allele.split('_')
                    k = '-'.join(fields[0:2])
                    if df.iloc[m, n] == 'P':
                        allele1.setdefault(k, [])
                        allele1[k].append(':'.join([k] + fields[2:]))
                    if df.iloc[m, n + 1] == 'P':
                        allele2.setdefault(k, [])
                        allele2[k].append(':'.join([k] + fields[2:]))

                for hla in self.HLA:
                    L.append([sample_id, hla, ','.join(allele1.get(hla, 'X')), ','.join(allele2.get(hla, 'X'))])

            df = pd.DataFrame(L)
            df.columns = header
            df.to_csv(out_file, header=True, index=False, sep='\t')
            print('Formatted output saved to', out_file)

    def make_ped_map_hla_ref(self, hla_ped='HAPMAP_CEU_HLA.ped', hla_pos='HLA_pos.txt', out_file='HLA_REF'):
        D = {}
        df = pd.read_table(hla_ped, header=None, sep='\t', dtype=str)
        for n in range(len(self.HLA)):
            for m in range(df.shape[0]):
                a1 = df.iloc[m, 6 + n * 2]
                a2 = df.iloc[m, 6 + n * 2 + 1]
                for a in [a1, a2]:
                    if len(a) >= 2 and len(a) < 4:
                        a = ':'.join([self.HLA[n], a])
                    elif len(a) >= 4 and len(a) < 6:
                        a = ':'.join([self.HLA[n], a[0:-2], a[-2:]])
                    elif len(a) >= 6 and  len(a) < 8:
                        a = ':'.join([self.HLA[n], a[0:-4], a[-4:-2], a[-2:]])
                    else:
                        continue
                    D[a] = True
                    if len(a.split(':')) > 2:
                        D[':'.join(a.split(':')[0:-1])] = True
        L = sorted(D.keys())
        M = np.zeros((df.shape[0], len(L) * 2), dtype=int)
        for m in range(df.shape[0]):
            for n in range(len(self.HLA)):
                a1 = df.iloc[m, 6 + n * 2]
                a2 = df.iloc[m, 6 + n * 2 + 1]
                for k, a in enumerate([a1, a2]):
                    if len(a) >= 2 and len(a) < 4:
                        a = ':'.join([self.HLA[n], a])
                    elif len(a) >= 4 and len(a) < 6:
                        a = ':'.join([self.HLA[n], a[0:-2], a[-2:]])
                    elif len(a) >= 6 and  len(a) < 8:
                        a = ':'.join([self.HLA[n], a[0:-4], a[-4:-2], a[-2:]])
                    try:
                        j = L.index(a)
                        M[m, j * 2 + k] = 1
                    except ValueError:
                        pass
        df_ped = pd.DataFrame(M, dtype=str)
        df_ped.columns = [f'{y}_{x}' for x in L for y in ('A0', 'A1')]
        df_ped[df_ped == '0'] = 'A'
        df_ped[df_ped == '1'] = 'P'
        df_ped.index = df.index
        df_merged = pd.merge(df.iloc[:, 0:6], df_ped, left_index=True, right_index=True)
        df_merged.to_csv(f'{out_file}.ped', sep='\t', header=False, index=False)

        D = {}
        df = pd.read_table(hla_pos, header=None, sep='\t', dtype=str)
        for n in range(df.shape[0]):
            allele = df.iloc[n, 0]
            ch = df.iloc[n, 1]
            pos = int(df.iloc[n, 2])
            D[allele] = [ch, pos]
        M = []
        A = {}
        shift = {}
        for allele in L:
            a = allele.split(':')[0]
            shift.setdefault(a, 0)
            ch, pos = D.get(a, ['0', 0])
            M.append([ch, allele, '0', pos + shift[a]])
            shift[a] += 1
        df_map = pd.DataFrame(M)
        df_map.to_csv(f'{out_file}.map', sep='\t', header=False, index=False)

    def phase_using_beagle(self, ref_bed='HAPMAP_CEU', ref_ped='HLA_REF', sample_bed='GDA', chrom='6', n_burnin=10, n_iter=50):
        '''
        not tested yet
        '''

        ref_hla = 'HAPMAP_CEU_HLA_REF'
        sample_phased = f'{sample_bed}_phased'
        L = []
        cmd = f'plink2 --bfile {ref_bed} --export vcf bgz --out {ref_bed}'
        L.append(cmd)
        cmd = f'bcftools index {ref_bed}.vcf.gz'
        L.append(cmd)
        cmd = f'bcftools query -l {ref_bed}.vcf.gz > _ref_samples.txt'
        L.append(cmd)

        cmd = f'plink2 --pedmap {ref_ped} --export vcf bgz --out {ref_ped}'
        L.append(cmd)
        cmd = f'bcftools sort {ref_bed}.vcf.gz -Oz -o {ref_bed}.vcf.gz'
        L.append(cmd)
        cmd = f'bcftools view -S ref_samples.txt {ref_ped}.vcf.gz -Oz -o {ref_ped}.vcf.gz'
        L.append(cmd)

        cmd = f'bcftools index {ref_ped}.vcf.gz'
        L.append(cmd)
        cmd = f'bcftools concat {ref_bed}.vcf.gz {ref_ped}.vcf.gz -Oz -o {ref_hla}.vcf.gz'
        L.append(cmd)
        cmd = f'bcftools sort {ref_hla}.vcf.gz -Oz -o {ref_hla}.vcf.gz'
        L.append(cmd)
        cmd = f'bcftools index {ref_hla}.vcf.gz'
        L.append(cmd)
        cmd = f'plink2 --bfile {sample_bed} -chr {chrom} --export vcf bgz --out {sample_bed}'
        L.append(cmd)
        cmd = f'beagle gt={ref_hla}.vcf.gz out={ref_hla}_phased burnin={n_burnin} iterations={n_iter}'
        L.append(cmd)
        cmd = f'beagle gt={sample_bed}.vcf.gz ref={ref_hla}_phased.vcf.gz out={sample_phased}'
        L.append(cmd)

        for cmd in L:
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    ar = Array()
    ar.run_snp2hla()
    ar.run_deephla()
