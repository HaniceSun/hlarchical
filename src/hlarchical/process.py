import numpy as np
import pandas as pd
import gzip

class Processor:
    def __init__(self, ref_phased='EuropeanREF_phased.vcf.gz', label_include=['HLA'], feature_exclude=['HLA', 'AA_'], hla_renaming=True, expert_by='ld'):
        self.label_include = label_include
        self.feature_exclude = feature_exclude
        self.expert_by = expert_by

        self.ref_phased = self.read_vcf(ref_phased)

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
        self.ref_phased_target = self.ref_phased[wh].copy()

        self.hla_renaming = hla_renaming
        if hla_renaming:
            self.ref_phased_target = self.renaming_hla(self.ref_phased_target)
        self.ref_phased_target.to_csv('t1.txt', sep='\t', index=False, header=True)

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
        self.ref_phased_feature = self.ref_phased[wh].copy()

        self.ld_blocks = {}
        self.ld_blocks['HLA-A'] = ['HLA-A']
        self.ld_blocks['HLA-B'] = ['HLA-B', 'HLA-C']
        self.ld_blocks['HLA-C'] = ['HLA-B', 'HLA-C']
        self.ld_blocks['HLA-DPA1'] = ['HLA-DPA1', 'HLA-DPB1']
        self.ld_blocks['HLA-DPB1'] = ['HLA-DPA1', 'HLA-DPB1']
        self.ld_blocks['HLA-DQA1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.ld_blocks['HLA-DQB1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.ld_blocks['HLA-DRB1'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']
        self.expert_groups = {}
        self.expert_groups['E0'] = ['HLA-A']
        self.expert_groups['E1'] = ['HLA-B', 'HLA-C']
        self.expert_groups['E2'] = ['HLA-DPA1', 'HLA-DPB1']
        self.expert_groups['E3'] = ['HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def make_features(self, subset=[], out_file='features.txt'):
        if subset:
            wh1 = self.ref_phased_feature['CHROM'] == subset[0]
            wh2 = self.ref_phased_feature['POS'] >= subset[1]
            wh3 = self.ref_phased_feature['POS'] <= subset[2]
            wh = wh1 & wh2 & wh3
            df = self.ref_phased_feature[wh].copy()

        df = self.ref_phased_feature.iloc[:, 9:].T
        df.columns = self.ref_phased_feature['ID'] + '_' + self.ref_phased_feature['POS'].astype(str)
        df.index.name = 'sample'
        df.to_csv(out_file, sep='\t', index=True, header=True)

    def make_labels(self, out_file='labels.txt', maps_file='maps.txt'):
        self.maps = pd.read_table(maps_file, header=0, sep='\t', low_memory=False)
        heads = self.maps['head'].unique().tolist()
        n_heads = len(heads)
        samples = self.ref_phased.columns[9:]
        n_samples = len(samples)
        D = dict(zip(self.maps['allele'], self.maps['label']))
        df = pd.DataFrame(index=samples, columns=heads)
        for n in range(self.ref_phased_target.shape[0]):
            ID = self.ref_phased_target['ID'].iloc[n]
            fields = ID.split(':')
            head = ':'.join(fields[0:-1])
            head_idx = heads.index(head)

            for m in range(n_samples):
                A = [0, 0]
                allele = self.ref_phased.iloc[n, 9 + m].split('|')
                if len(allele) == 2:
                    if allele[0] == '1':
                        A[0] = D[ID]
                    if allele[1] == '1':
                        A[1] = D[ID]
                df.iloc[m, head_idx] = '|'.join([str(x) for x in A])
        df.index.name = 'sample'
        df.to_csv(out_file, sep='\t', index=True, header=True)

    def make_maps(self, out_file='maps.txt'):
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

        self.maps = maps
        self.maps.to_csv(out_file, sep='\t', index=False, header=True)
        print('processed label maps:')
        print(self.maps)
        print(f'maps data saved to {out_file}')

    def make_masks(self, out_file='masks.txt', features_file='features.txt', flank=500000):
        print(self.ref_phased_target)
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

    def renaming_hla(self, df):
        new_ids = []
        for n in range(df.shape[0]):
            ID = df['ID'].iloc[n].replace(':', '')
            fields = ID.split('_')
            gene = '-'.join(fields[0:2])
            allele = fields[-1]
            if len(allele) < 4:
                new_id = f"{gene}:{allele}"
            elif len(allele) == 4:
                new_id = f"{gene}:{allele[0:-2]}:{allele[-2:]}"
            elif len(allele) == 5:
                new_id = f"{gene}:{allele[0:-3]}:{allele[-3:]}"
            elif len(allele) == 6:
                new_id = f"{gene}:{allele[0:-4]}:{allele[-4:-2]}:{allele[-2:]}"
            else:
                new_id = '.'
                print(f"Warning: unexpected allele: {allele} excluded")
            new_ids.append(new_id)
        df['ID'] = new_ids
        df = df[df['ID'] != '.'].copy()
        return df

    def prepare_to_predict(self, features_file='features.txt', out_file='to_predict.txt'):
        df_features = pd.read_table(features_file, header=0, sep='\t', low_memory=False)
        features = []
        for n in range(1, df_features.shape[1], 2):
            fields = df_features.columns[n].split('_')
            k = fields[1]
            if k not in features:
                features.append(k)

        df = pd.merge(self.ref_bim, self.sample_phased, left_on='id_ref', right_on='id_sample')
        wh = df['id_ref'].isin(features)
        df = df[wh]
        if df.shape[0] != len(features):
            print('Warning: some features are missing in the sample phased data')

        M = np.zeros((len(self.sample_ids), len(features) * 2), dtype=int)
        H = []
        idx_start = 8
        for n in range(idx_start, df.shape[1], 2):
            for j in range(2):
                wh = (df.iloc[:, n + j] == df['A1_ref']).values.astype(int)
                for m in range(len(wh)):
                    M[(n - idx_start) // 2, m * 2 + j] = wh[m]
                    if j == 0 and n == idx_start:
                        id_ref = df['id_ref'].iloc[m]
                        H.append(f'A1_{id_ref}')
                        H.append(f'A2_{id_ref}')

        df = pd.DataFrame(M)
        df.index = self.sample_ids
        df.index.name = 'sample'
        df.columns = H
        df.to_csv(out_file, sep='\t', index=True, header=True)
        print(f'features data to predict saved to {out_file}')

    def read_vcf(self, in_file):
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

        df = pd.read_table(in_file, sep='\t', skiprows=n_header)
        df.rename(columns={'#CHROM': 'CHROM'}, inplace=True)
        return df



if __name__ == '__main__':
    pc = Processor(ref_phased='')
    pc.make_features()
    pc.make_maps()
    pc.make_labels()
    pc.make_masks()
