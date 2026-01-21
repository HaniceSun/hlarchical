import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self, ref_bim='Pan-Asian_REF.bim', sample_bim='1958BC.bim', ref_phased='Pan-Asian_REF.bgl.phased', hla_filter=['HLA'], non_hla_filter=['HLA'], hla_renaming=True, expert_by_gene=False, expert_by_ld=True):
        self.ref_bim = pd.read_table(ref_bim, header=None, sep='\t')
        self.ref_phased = pd.read_table(ref_phased, header=None, sep=' ')
        self.sample_bim = pd.read_table(sample_bim, header=None, sep='\t')

        bim_cols = ['chrom', 'id', 'cm', 'pos', 'A1', 'A2']
        self.ref_bim.columns = [f'{x}_ref' for x in bim_cols]
        self.sample_bim.columns = [f'{x}_sample' for x in bim_cols]

        cols = list(self.ref_phased.columns)
        cols[0] = 'I'
        cols[1] = 'id_ref'
        for i in range(2, self.ref_phased.shape[1], 2):
            cols[i] = f'A1_s{i//2}'
            cols[i + 1] = f'A2_s{i//2}'
        self.ref_phased.columns = cols

        # Extract HLA for labels
        wh = []
        for n in range(self.ref_phased.shape[0]):
            id_ref = self.ref_phased['id_ref'].iloc[n]
            flag = False
            for hf in hla_filter:
                if str(id_ref).find(hf) != -1:
                    flag = True
                    break
            wh.append(flag)
        self.ref_phased_hla = self.ref_phased[wh].copy()

        self.hla_filter = hla_filter
        self.non_hla_filter = non_hla_filter
        self.expert_by_gene = expert_by_gene
        self.expert_by_ld = expert_by_ld

        self.hla_renaming = hla_renaming
        if hla_renaming:
            self.ref_phased_hla = self.renaming_hla(self.ref_phased_hla)

        # Extract non-HLA for features
        wh = []
        for n in range(self.ref_phased.shape[0]):
            id_ref = self.ref_phased['id_ref'].iloc[n]
            flag = True
            for nhf in non_hla_filter:
                if str(id_ref).find(nhf) != -1:
                    flag = False
                    break
            wh.append(flag)
        self.ref_phased_non_hla = self.ref_phased[wh].copy()

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

    def make_features(self, by_id=True, by_pos=False, check_alleles=True, subset_sample_bim=None, out_file='features.txt'):
        # Subset sample bim to the HLA region
        if subset_sample_bim is not None:
            wh = []
            for n in range(self.sample_bim.shape[0]):
                flag = False
                ch = self.sample_bim['chrom_sample'].iloc[n]
                pos = self.sample_bim['pos_sample'].iloc[n]
                if str(ch) == str(subset_sample_bim[0]):
                    if pos < subset_sample_bim[2] and pos > subset_sample_bim[1]:
                        flag = True
                        break
                wh.append(flag)
            self.sample_bim = self.sample_bim[wh]
            print(f"Subsetted sample bim to the region: {subset_sample_bim}")

        # Get shared variants between reference and sample
        if by_id:
            df = pd.merge(self.ref_bim, self.sample_bim, left_on='id_ref', right_on='id_sample')
        elif by_pos:
            df = pd.merge(self.ref_bim, self.sample_bim, left_on=['chrom_ref', 'pos_ref'], right_on=['chrom_sample', 'pos_sample'])
        else:
            raise ValueError("Either by_id or by_pos must be True.")

        if check_alleles:
            wh = (df['A1_ref'] == df['A1_sample']) & (df['A2_ref'] == df['A2_sample'])
            df = df[wh]

        # Merge with phased data
        df = pd.merge(df, self.ref_phased_non_hla, on='id_ref') 
        L = []
        L.append(df[['id_ref', 'pos_ref']])
        for col in df.columns:
            col2 = str(col)
            if (col2.startswith('A1_') or col2.startswith('A2_')) and col2.find('_ref') == -1 and col2.find('_sample') == -1:
                d = {col:(df[col]==df['A1_ref']).astype(int)}
                L.append( pd.DataFrame(d))
        df = pd.concat(L, axis=1)
        df = df.sort_values(by='pos_ref').reset_index(drop=True)
        print('processed feature data before transpose:')
        print(df)
        print(f'features data saved to {out_file}')

        M = np.zeros((int(df.shape[1]/2 - 1), df.shape[0] * 2), dtype=int)
        H = []
        S = []
        for n in range(2, df.shape[1], 2):
            S.append(df.columns[n].split('_')[-1])
            for m in range(df.shape[0]):
                id_ref = df['id_ref'].iloc[m]
                pos_ref = df['pos_ref'].iloc[m]
                for j in range(2):
                    M[n // 2 - 1, m * 2 + j] = df.iloc[m, n + j]
                if n == 2:
                    H.append(f'A1_{id_ref}_{pos_ref}')
                    H.append(f'A2_{id_ref}_{pos_ref}')
        df = pd.DataFrame(M)
        df.index = S
        df.index.name = 'sample'
        df.columns = H
        df.to_csv(out_file, sep='\t', index=True, header=True)

    def make_labels(self, out_file='labels.txt', maps_file='maps.txt'):
        self.get_label_maps(maps_file)
        print('processed label data before transpose:')
        print(self.ref_phased_hla)

        n_heads = (self.maps['head_idx'].max() + 1) * 2
        heads = self.maps['head'].unique()
        S = []
        H = []
        for h in heads:
            H.append(f"A1_{h}")
            H.append(f"A2_{h}")
        M = np.zeros((int(self.ref_phased_hla.shape[1]/2 - 1), n_heads), dtype=int)
        for n in range(2, self.ref_phased_hla.shape[1], 2):
            S.append(self.ref_phased_hla.columns[n].split('_')[-1])
            for j in range(2):
                variants = self.ref_phased_hla.loc[self.ref_phased_hla.iloc[:, n + j] == 'P', 'id_ref'].values
                df_sub = self.maps[self.maps['allele'].isin(variants)]
                for m in range(df_sub.shape[0]):
                    head_idx = df_sub['head_idx'].iloc[m]
                    label = df_sub['label'].iloc[m]
                    M[n // 2 - 1, head_idx * 2 + j] = label
        df = pd.DataFrame(M)
        df.index = S
        df.index.name = 'sample'
        df.columns = H
        df.to_csv(out_file, sep='\t', index=True, header=True)
        print(f'labels data saved to {out_file}')

    def get_label_maps(self, out_file='maps.txt'):
        D = {}
        for n in range(self.ref_phased_hla.shape[0]):
            id_ref = self.ref_phased_hla['id_ref'].iloc[n]
            fields = id_ref.split(':')
            head = ':'.join(fields[0:-1])
            D.setdefault(head, [])
            if id_ref not in D[head]:
                D[head].append(id_ref)

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

        if self.expert_by_gene:
            maps['expert'] = expert
        else:
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
        # Extract HLA
        wh = []
        for n in range(self.ref_bim.shape[0]):
            id_ref = self.ref_bim['id_ref'].iloc[n]
            flag = False
            for hf in self.hla_filter:
                if str(id_ref).find(hf) != -1:
                    flag = True
                    break
            wh.append(flag)
        ref_bim_hla = self.ref_bim[wh].copy()
        if self.hla_renaming:
            ref_bim_hla = self.renaming_hla(ref_bim_hla)

        H = {}
        for n in range(ref_bim_hla.shape[0]):
            gene = ref_bim_hla['id_ref'].iloc[n].split(':')[0]
            pos = ref_bim_hla['pos_ref'].iloc[n]
            H.setdefault(gene, [])
            H[gene].append(pos)

        start_end_dict = {}
        for gene in H:
            if self.expert_by_gene:
                positions = H[gene]
                start_end_dict[gene] = (min(positions), max(positions))
            elif self.expert_by_ld:
                positions = []
                if gene in self.ld_blocks:
                    for g in self.ld_blocks[gene]:
                        if g in H:
                            positions += H[g]
                start_end_dict[gene] = (min(positions), max(positions))

        features = pd.read_table(features_file, header=0, sep='\t')
        L = []
        E = []
        for gene in start_end_dict:
            if self.expert_by_gene:
                expert = gene
            else:
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
                for n in range(1, features.shape[1], 2):
                    fields = features.columns[n].split('_')
                    pos = int(fields[-1])
                    if pos >= pos_min - flank and pos <= pos_max + flank:
                        m.append(1)
                    else:
                        m.append(0)
                L.append([expert] + m)
        df = pd.DataFrame(L)
        df.columns = ['expert'] + ['_'.join(features.columns[n].split('_')[1:]) for n in range(1, features.shape[1], 2)]
        df.to_csv(out_file, sep='\t', index=False, header=True)
        print(f'processed masks data: {out_file}')

    def renaming_hla(self, df):
        new_ids = []
        for n in range(df.shape[0]):
            id_ref = df['id_ref'].iloc[n]
            fields = id_ref.split('_')
            gene = '-'.join(fields[0:2])
            allele = fields[-1]
            if len(allele) < 4:
                new_id = f"{gene}:{allele}"
            elif len(allele) < 6:
                new_id = f"{gene}:{allele[0:-2]}:{allele[-2:]}"
            elif len(allele) < 8:
                new_id = f"{gene}:{allele[0:-4]}:{alele[-4:-2]}:{allele[-2:]}"
            else:
                new_id = '.'
                print(f"Warning: unexpected allele: {allele} excluded")
            new_ids.append(new_id)
        df['id_ref'] = new_ids
        df = df[df['id_ref'] != '.'].copy()
        return df

if __name__ == '__main__':
    dp = DataPreprocessor()
    #dp = DataPreprocessor(ref_bim='HM_CEU_REF.bim', sample_bim='1958BC.bim', ref_phased='HM_CEU_REF.bgl.phased')
    #dp.make_features()
    #dp.make_labels()
    #dp.make_masks()
