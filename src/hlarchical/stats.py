import os
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from forestplot import forestplot
from .utils import *

class AssociationDiseaseHLA():
    def __init__(self):
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def preprocess(self, in_file, digit=2):
        df = pd.read_table(in_file, header=0, sep='\t')
        for hla in self.HLA:
            df2 = df[df['HLA'] == hla].copy()
            df2 = self.allele_binary_encoding(df2, digit=digit)
            out_file = in_file.replace('.txt', f'_{hla}_digit{digit}.txt')
            df2.to_csv(out_file, index=False, sep='\t')

    def allele_binary_encoding(self, df, digit=2):
        alleles = []
        A1 = []
        A2 = []
        for n in range(df.shape[0]):
            allele1 = df.iloc[n]['Allele1']
            allele2 = df.iloc[n]['Allele2']
            a1 = '.'
            a2 = '.'
            if allele1 not in ['.', 'X']:
                a1 = ':'.join(allele1.split(':')[0:int(digit/2) + 1])
                if a1.find('--') == -1:
                    if a1 not in alleles:
                        alleles.append(a1)
                else:
                    a1 = '.'
            if allele2 not in ['.', 'X']:
                a2 = ':'.join(allele2.split(':')[0:int(digit/2) + 1])
                if a2.find('--') == -1:
                    if a2 not in alleles:
                        alleles.append(a2)
                else:
                    a2 = '.'
            A1.append(a1)
            A2.append(a2)
        df['A1'] = A1
        df['A2'] = A2
        for allele in sorted(alleles):
            L = []
            for n in range(df.shape[0]):
                nA = 0
                if df.iloc[n]['A1'] == allele:
                    nA += 1
                if df.iloc[n]['A2'] == allele:
                    nA += 1
                L.append(nA)
            df[allele] = L
        return df

    def association_test(self, in_file, formula='Condition ~ HLA + Ancestry', out_dir='stats'):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        df0 = pd.read_table(in_file, header=0, sep='\t')
        idx = df0.columns.get_loc('A2')
        for n in range(idx + 1, df0.shape[1]):
            hla = df0.columns[n]
            out_file = out_dir + '/' + in_file.replace('.txt', f'_{hla}_stats.txt')
            out_file_covs = out_dir + '/' + in_file.replace('.txt', f'_{hla}_covs.txt')
            df = pd.DataFrame()
            df['HLA'] = df0[hla].astype(int)
            df['Condition'] = [0 if x == 'control' else 1 for x in df0['Condition']]
            df['Ancestry'] = df0['Ancestry'].astype('category')

            if formula.find('Ancestry') != -1:
                self._set_cat_reference(df, 'Ancestry', 'EUR') 

            model = smf.logit(formula, data=df)
            try:
                result = model.fit()
                summary = result.summary()
                df_covs = pd.DataFrame(result.cov_params())
                with open(out_file, 'w') as f:
                    f.write(hla + '\n')
                    f.write(summary.as_text())
                df_covs.to_csv(out_file_covs, sep='\t')
            except Exception as e:
                print(e)

    def beta_sort_by_pvalue(self, in_dir, out_file, on='HLA'):
        fs = sorted([f for f in os.listdir(in_dir) if f.endswith('_stats.txt')])
        L = []
        for f in fs:
            with open(in_dir + '/' + f, 'r') as infile:
                hla = infile.readline().strip()
                for line in infile:
                    values = line.strip().split()
                    if values[0] == on:
                        L.append([hla] + values[1:])
        df = pd.DataFrame(L, columns=['param', 'coef', 'std_err', 'z', 'pvalue', 'ci_low', 'ci_high'])
        df.sort_values(by='pvalue', inplace=True)
        df.to_csv(out_file, index=False, sep='\t')

    def beta_to_odds_ratio(self, in_dir, out_file='HLAtyping_T1D_1KG_interaction_stats_HLA_Ancestry.txt', cols=['param', 'coef', 'std_err', 'z', 'pvalue', 'ci_low', 'ci_high'], critical_value=1.96, main_effects=['HLA'], cols_out=['HLA', 'param', 'OR', 'ci_low', 'ci_high', 'pvalue'], sig_threshold=0.05):
        fs = sorted([f for f in os.listdir(in_dir) if f.endswith('_stats.txt')])
        ORL = []
        for f in fs:
            L = []
            hla = f.split('_')[-2]
            with open(in_dir + '/' + f) as fi:
                flag = False
                for line in fi:
                    line = line.strip()
                    if line.find('coef') != -1:
                        flag = True
                        continue
                    if flag and line.find('---') == -1 and line.find('===') == -1:
                        fields = line.split()
                        L.append(fields)
            df_stats = pd.DataFrame(L)
            df_stats.columns = cols
            df_stats.set_index('param', inplace=True)
            covs_file = f.replace('_stats.txt', '_covs.txt')
            df_covs = pd.read_csv(in_dir + '/' + covs_file, sep='\t', index_col=0)
    
            for param in df_stats.index:
                if param.find(':') != -1:
                    pvalue = float(df_stats.loc[param, 'pvalue'])
                    fields = param.split(':')
                    p1 = fields[0]
                    p2 = fields[1]
                    b1 = float(df_stats.loc[p1, 'coef'])
                    b2 = float(df_stats.loc[param, 'coef'])
                    b_comb = b1 + b2
                    var_comb = float(df_covs.loc[p1, p1]) +  float(df_covs.loc[param, param]) + 2 * float(df_covs.loc[p1, param])
                    se_comb = np.sqrt(var_comb)
                    beta = [hla, param, b_comb, b_comb - critical_value * se_comb, b_comb + critical_value * se_comb, pvalue]
                    odds_ratio = [beta[0], beta[1], np.exp(beta[2]), np.exp(beta[3]), np.exp(beta[4]), beta[5]]
                    ORL.append(odds_ratio)
                elif param in main_effects:
                    pvalue = float(df_stats.loc[param, 'pvalue'])
                    b = float(df_stats.loc[param, 'coef'])
                    se = float(df_stats.loc[param, 'std_err'])
                    beta = [hla, param, b, b - critical_value * se, b + critical_value * se, pvalue]
                    odds_ratio = [beta[0], beta[1], np.exp(beta[2]), np.exp(beta[3]), np.exp(beta[4]), beta[5]]
                    ORL.append(odds_ratio)
        df = pd.DataFrame(ORL)
        df.columns = cols_out
        df.to_csv(out_file, sep='\t', index=False)

        out_file_sig = out_file.replace('.txt', '_sig.txt')
        D = {}
        for gi, g in df.groupby('HLA'):
            p_min = g['pvalue'].min()
            OR_min = g['OR'].min() 
            if p_min < sig_threshold:
                D[gi] = [g, p_min, OR_min]
        D_sorted = sorted(D.items(), key=lambda x: [x[1][1], x[1][2]])
        df_sig = pd.concat([x[1][0] for x in D_sorted])
        df_sig.to_csv(out_file_sig, sep='\t', index=False)

    def _set_cat_reference(self, df, column, reference):
        df[column] = df[column].cat.reorder_categories(
                [reference] + [x for x in df[column].cat.categories if x != reference], ordered=True)

    def llr_test(self, ll_full, ll_reduced, df_full, df_reduced):
        llr_stat = 2 * (ll_full - ll_reduced)
        df_diff = df_full - df_reduced
        p_value = chi2.sf(llr_stat, df_diff)
        print(f"LLR Statistic: {llr_stat}, p-value: {p_value}")
        return llr_stat, p_value

    def forest_plot(self, df, estimate='OR', xlabel='Odds Ratio', y_ticklabels='HLA', logscale=True, s=40, palette='Set2', hue=None, hue_order=None, show_grid=True, out_file='forest_plot.pdf', title=None, figsize=(4, 4), fontsize_params={'xlabel': 16, 'ylabel': 10, 'yticklabels':8, 'title': 16}, line_params={'color':'C0', 'lw':1}, grid_params={'ls':'--', 'alpha':0.7}, legend_params={'fontsize': 8, 'loc': 'upper right', 'bbox_to_anchor': (1, 0)}):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        ax.set_ylim(0.5, df.shape[0] + 0.5)
        df['y'] = range(df.shape[0], 0, -1)
        ax.set_yticks(df['y'], df[y_ticklabels], fontsize=fontsize_params['yticklabels'])

        if hue is not None:
            sns.scatterplot(data=df, x=estimate, y='y', ax=ax, marker='s', hue=hue, hue_order=hue_order, palette=palette, edgecolor=None, s=s)
            plt.legend(**legend_params)
        else:
            sns.scatterplot(data=df, x=estimate, y='y', ax=ax, marker='s', color='C0', edgecolor=None, s=s)

        for n in range(df.shape[0]):
            y = df.iloc[n]['y']
            ci_low = df.iloc[n]['ci_low']
            ci_high = df.iloc[n]['ci_high']
            ax.plot([ci_low, ci_high], [y, y], color=line_params['color'], lw=line_params['lw'])

        ax.set_xlabel(xlabel, fontsize=fontsize_params['xlabel'])
        ax.set_ylabel('')
        if logscale:
            plt.xscale('log')
        if show_grid:
            ax.grid(axis='y', linestyle=grid_params['ls'], alpha=grid_params['alpha'])
        if title is not None:
            ax.set_title(title, fontsize=fontsize_params['title'])
        ax.axvline(x=1, color='orange', linestyle='--')
        plt.tight_layout()
        plt.savefig(out_file)

if __name__ == '__main__':

	value_clip = [0.1, 10]
	hla = AssociationDiseaseHLA()

	in_file = 'HLAtyping_T1D_1KG_additive_stats_HLA_Ancestry_sig.txt'
	out_file = in_file.replace('.txt', '_forest.pdf')
	df = pd.read_table(in_file, header=0, sep='\t')
	df['gene'] = [x.split(':')[0] for x in df['HLA']]
	df['allele'] = [int(x.split(':')[-1]) for x in df['HLA']]
	df.sort_values(by=['gene', 'allele', 'param', 'pvalue'], inplace=True)
	df['OR'] = df['OR'].clip(upper=value_clip[1], lower=value_clip[0])
	df['ci_low'] = df['ci_low'].clip(upper=value_clip[1], lower=value_clip[0])
	df['ci_high'] = df['ci_high'].clip(upper=value_clip[1], lower=value_clip[0])
	hla.forest_plot(df=df, out_file=out_file, title='T1D ~ HLA + Ancestry', figsize=[4, 6])

	in_file = 'HLAtyping_T1D_1KG_interaction_stats_HLA_Ancestry_sig.txt'
	out_file = in_file.replace('.txt', '_forest.pdf')
	df = pd.read_table(in_file, header=0, sep='\t')
	wh = df['param'] != 'HLA:Ancestry[T.AMR]'
	df = df[wh]

	df['OR'] = df['OR'].clip(upper=value_clip[1], lower=value_clip[0])
	df['ci_low'] = df['ci_low'].clip(upper=value_clip[1], lower=value_clip[0])
	df['ci_high'] = df['ci_high'].clip(upper=value_clip[1], lower=value_clip[0])

	D = {}
	D['HLA'] = 'EUR'
	D['HLA:Ancestry[T.EAS]'] = 'EAS'
	D['HLA:Ancestry[T.SAS]'] = 'SAS'
	df['Ancestry'] = df['param'].map(D)

	df['gene'] = [x.split(':')[0] for x in df['HLA']]
	df['allele'] = [int(x.split(':')[-1]) for x in df['HLA']]
	df.sort_values(by=['gene', 'allele', 'param', 'pvalue'], inplace=True)
	hla.forest_plot(df=df, out_file=out_file, title='T1D ~ HLA * Ancestry', hue='Ancestry', hue_order=['EUR', 'EAS', 'SAS'], figsize=[4, 8])
