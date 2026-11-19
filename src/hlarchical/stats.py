import os

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2
from forestplot import forestplot

from .utils import *


class AssociationDiseaseHLA():
    """HLA-disease association testing and forest-plot visualization.

    Provides utilities to binary-encode HLA alleles as dosage covariates,
    run per-allele logistic regression association tests against a
    disease/condition outcome, aggregate and rank the resulting
    coefficients (including combining main-effect and interaction-term
    coefficients into odds ratios with confidence intervals), and render
    forest plots of the results.

    Attributes:
        HLA: The eight classical HLA genes considered throughout this
            class, in a fixed canonical order used for grouping/sorting.
    """

    def __init__(self) -> None:
        """Initialize the list of HLA genes handled by this class."""
        self.HLA = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DRB1']

    def preprocess(self, in_file: str, digit: int = 2, encoding='binary') -> None:
        """Split an hlarchical table by HLA gene and binary-encode alleles.

        Reads a long-format hlarchical table (columns including ``HLA``,
        ``Allele1``, ``Allele2``), splits it into one subset per HLA gene,
        binary-encodes the alleles at the requested digit resolution via
        `allele_binary_encoding`, and writes one output file per gene.

        Args:
            in_file: Path to the input tab-separated hlarchical table.
            digit: Allele resolution (in "digits", e.g. 2 or 4) at which to
                truncate/group alleles before encoding.
            encoding: Encoding method to use, currently only supports ``'binary'`` and ``'binomial'``.
        """
        df = pd.read_table(in_file, header=0, sep='\t')
        for hla in self.HLA:
            df2 = df[df['HLA'] == hla].copy()
            if encoding == 'binary':
                df2 = self.allele_binary_encoding(df2, digit=digit)
                out_file = in_file.replace('.txt', f'_{hla}_digit{digit}.txt')
                df2.to_csv(out_file, index=False, sep='\t')
            elif encoding == 'binomial':
                df2 = self.allele_binomial_encoding(df2, digit=digit)
                out_file = in_file.replace('.txt', f'_{hla}_digit{digit}_binomial.txt')
                df2.to_csv(out_file, index=False, sep='\t')

    def allele_binary_encoding(self, df: pd.DataFrame, digit: int = 2) -> pd.DataFrame:
        """Encode alleles in ``Allele1``/``Allele2`` as per-allele dosage columns.

        For each row, truncates ``Allele1``/``Allele2`` to the requested
        digit resolution (storing the truncated values in new ``A1``/``A2``
        columns), collects the set of distinct alleles observed, then adds
        one integer dosage column per allele (0, 1, or 2 copies) counting
        how many of ``A1``/``A2`` match that allele.

        Args:
            df: Input DataFrame with ``Allele1`` and ``Allele2`` columns.
            digit: Allele resolution (in "digits") to truncate alleles to;
                alleles are truncated to keep the first ``digit // 2 + 1``
                colon-separated fields.

        Returns:
            The input DataFrame with ``A1``, ``A2`` columns added, plus one
            additional integer dosage column per distinct allele observed.
        """
        alleles = []
        A1 = []
        A2 = []
        for n in range(df.shape[0]):
            allele1 = df.iloc[n]['Allele1']
            allele2 = df.iloc[n]['Allele2']
            a1 = '.'
            a2 = '.'
            if allele1 not in ['.', 'X']:
                a1 = ':'.join(allele1.split(':')[0:int(digit / 2) + 1])
                if a1.find('--') == -1:
                    if a1 not in alleles:
                        alleles.append(a1)
                else:
                    a1 = '.'
            if allele2 not in ['.', 'X']:
                a2 = ':'.join(allele2.split(':')[0:int(digit / 2) + 1])
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

    def allele_binomial_encoding(self, df: pd.DataFrame, digit: int = 2) -> pd.DataFrame:
        D = {}
        A = []
        for n in range(df.shape[0]):
            donor = df.iloc[n]['DonorID']
            ancestry = df.iloc[n]['Ancestry']
            allele1 = df.iloc[n]['Allele1']
            allele2 = df.iloc[n]['Allele2']
            a1 = '.'
            a2 = '.'
            if allele1 not in ['.', 'X']:
                a1 = ':'.join(allele1.split(':')[0:int(digit / 2) + 1])
                if a1.find('--') != -1:
                    a1 = '.'
            if allele2 not in ['.', 'X']:
                a2 = ':'.join(allele2.split(':')[0:int(digit / 2) + 1])
                if a2.find('--') != -1:
                    a2 = '.'
            D.setdefault(ancestry, {})
            D[ancestry].setdefault('donor', set())
            D[ancestry]['donor'].add(donor)
            if a1 != '.':
                D[ancestry].setdefault(a1, 0)
                D[ancestry][a1] += 1
                A.append(a1)
            if a2 != '.':
                D[ancestry].setdefault(a2, 0)
                D[ancestry][a2] += 1
                A.append(a2)
        L = []
        for allele in sorted(set(A)):
            for ancestry in sorted([x for x in D.keys() if x != 'donor']):
                nA = D[ancestry].get(allele, 0)
                nD = len(D[ancestry]['donor']) * 2
                L.append([allele, ancestry, nA, nD])
        df = pd.DataFrame(L, columns=['Allele', 'Ancestry', 'nA', 'nD'])
        return df

    def association_test(
        self,
        in_file: str,
        formula: str = 'Condition ~ HLA + Ancestry',
        out_dir: str = 'stats',
    ) -> None:
        """Run per-allele logistic regression association tests.

        Reads a table pre-encoded by `allele_binary_encoding` (one dosage
        column per allele following the ``A2`` column), and for each allele
        column fits a logistic regression (`statsmodels.formula.api.logit`)
        of disease ``Condition`` (recoded 0=control, 1=case) on ``HLA``
        dosage and, if present in the formula, ``Ancestry`` (with ``EUR``
        set as the reference category). Writes a text summary and a
        covariance-matrix file per allele.

        Args:
            in_file: Path to the input tab-separated, allele-encoded table.
                Must contain ``A2``, ``Condition``, and (if used in
                ``formula``) ``Ancestry`` columns, plus one dosage column
                per allele after ``A2``.
            formula: Patsy-style regression formula passed to
                `statsmodels.formula.api.logit`, referencing ``HLA``,
                ``Condition``, and optionally ``Ancestry``.
            out_dir: Directory to write per-allele stats/covariance files
                to; created if it does not exist.
        """
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

    def beta_sort_by_pvalue(self, in_dir: str, out_file: str, on: str = 'HLA') -> None:
        """Aggregate per-allele association stats and sort by p-value.

        Scans ``in_dir`` for files produced by `association_test`
        (``*_stats.txt``), extracts the regression coefficient row whose
        parameter name equals ``on``, and writes a combined table sorted by
        ascending p-value.

        Args:
            in_dir: Directory containing ``*_stats.txt`` files written by
                `association_test`.
            out_file: Path to write the combined, sorted table to.
            on: Name of the regression parameter/covariate row to extract
                from each stats file (e.g. ``'HLA'``).
        """
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

    def beta_to_odds_ratio(
        self,
        in_dir: str,
        out_file: str = 'HLAtyping_T1D_1KG_interaction_stats_HLA_Ancestry.txt',
        cols: list[str] = ['param', 'coef', 'std_err', 'z', 'pvalue', 'ci_low', 'ci_high'],
        critical_value: float = 1.96,
        main_effects: list[str] = ['HLA'],
        cols_out: list[str] = ['HLA', 'param', 'OR', 'ci_low', 'ci_high', 'pvalue'],
        sig_threshold: float = 0.05,
    ) -> None:
        """Convert regression coefficients to odds ratios with CIs.

        Scans ``in_dir`` for ``*_stats.txt``/``*_covs.txt`` file pairs
        written by `association_test`, and for each allele:

        - For interaction terms (parameter names containing ``':'``),
          combines the main-effect and interaction coefficients
          (``beta = b1 + b2``) and their combined variance (using the
          covariance matrix) into a combined beta, CI, and p-value.
        - For parameters listed in ``main_effects``, uses the coefficient,
          standard error, and p-value directly.

        In both cases, exponentiates beta and its CI bounds to obtain the
        odds ratio and its confidence interval. Writes the full odds-ratio
        table to ``out_file``, and a second file (``out_file`` with
        ``_sig`` inserted before the extension) containing only genes
        where the minimum p-value across parameters is below
        ``sig_threshold``, sorted by that minimum p-value and OR.

        Args:
            in_dir: Directory containing ``*_stats.txt`` and matching
                ``*_covs.txt`` files written by `association_test`.
            out_file: Path to write the full odds-ratio table to.
            cols: Column names of the per-allele stats table as written by
                `association_test` (used to parse ``*_stats.txt``).
            critical_value: Z critical value used to build confidence
                intervals (default corresponds to a two-sided 95% CI).
            main_effects: Parameter names treated as simple main effects
                (as opposed to interaction terms containing ``':'``).
            cols_out: Column names for the output odds-ratio table.
            sig_threshold: P-value threshold below which a gene's results
                are included in the significant-results output file.
        """
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
                    var_comb = float(df_covs.loc[p1, p1]) + float(df_covs.loc[param, param]) + 2 * float(df_covs.loc[p1, param])
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

    def _set_cat_reference(self, df: pd.DataFrame, column: str, reference: str) -> None:
        """Reorder a categorical column in-place so ``reference`` is first.

        Reordering the categories so the desired level comes first makes
        it the reference (baseline) category in downstream patsy-formula
        regressions (e.g. via `statsmodels.formula.api.logit`).

        Args:
            df: DataFrame containing the categorical column to reorder.
                Modified in place.
            column: Name of the categorical column to reorder.
            reference: Category value to move to the front (the new
                reference level).
        """
        df[column] = df[column].cat.reorder_categories(
            [reference] + [x for x in df[column].cat.categories if x != reference], ordered=True)

    def llr_test(
        self,
        ll_full: float,
        ll_reduced: float,
        df_full: int,
        df_reduced: int,
    ) -> tuple[float, float]:
        """Perform a likelihood-ratio test between two nested models.

        Args:
            ll_full: Log-likelihood of the full (unrestricted) model.
            ll_reduced: Log-likelihood of the reduced (nested) model.
            df_full: Degrees of freedom (number of parameters) of the full
                model.
            df_reduced: Degrees of freedom of the reduced model.

        Returns:
            A tuple ``(llr_stat, p_value)`` where ``llr_stat`` is the
            likelihood-ratio test statistic and ``p_value`` is the
            corresponding chi-squared upper-tail p-value.
        """
        llr_stat = 2 * (ll_full - ll_reduced)
        df_diff = df_full - df_reduced
        p_value = chi2.sf(llr_stat, df_diff)
        print(f"LLR Statistic: {llr_stat}, p-value: {p_value}")
        return llr_stat, p_value

    def forest_plot(
        self,
        df: pd.DataFrame,
        estimate: str = 'OR',
        xlabel: str = 'Odds Ratio',
        y_ticklabels: str = 'HLA',
        logscale: bool = True,
        s: int = 40,
        palette: str | list[str] = 'Set2',
        hue: str | None = None,
        hue_order: list[str] | None = None,
        show_grid: bool = True,
        out_file: str = 'forest_plot.pdf',
        title: str | None = None,
        figsize: tuple[float, float] = (4, 4),
        fontsize_params: dict[str, int] = {'xlabel': 16, 'ylabel': 10, 'yticklabels': 8, 'title': 16},
        line_params: dict[str, object] = {'color': 'C0', 'lw': 1},
        grid_params: dict[str, object] = {'ls': '--', 'alpha': 0.7},
        legend_params: dict[str, object] = {'fontsize': 8, 'loc': 'upper right', 'bbox_to_anchor': (1, 0)},
        cmap_gene: str = 'Dark2',
        y_ticklabels_mask: list[int] = [],
    ) -> None:
        """Render and save a forest plot of effect estimates with CIs.

        Plots one point per row of ``df`` at ``estimate`` with a
        horizontal confidence-interval line from ``ci_low`` to
        ``ci_high``, y-axis tick labels taken from ``y_ticklabels`` and
        colored by HLA gene (parsed from ``df['gene']`` and mapped via
        `self.HLA` colors from ``cmap_gene``), optionally colored/grouped
        by ``hue``, and a reference vertical line at OR/estimate = 1.

        Args:
            df: DataFrame with (at least) columns ``gene``, ``ci_low``,
                ``ci_high``, the column named by ``estimate``, and the
                column named by ``y_ticklabels``.
            estimate: Name of the column holding the point estimate to
                plot on the x-axis (e.g. odds ratio).
            xlabel: Label for the x-axis.
            y_ticklabels: Name of the column used for y-axis tick labels.
            logscale: If True, use a log scale for the x-axis.
            s: Marker size for the scatter points.
            palette: Color palette passed to `seaborn.scatterplot` when
                ``hue`` is set.
            hue: Optional column name to color points by.
            hue_order: Optional explicit ordering of ``hue`` categories;
                also used to determine tick-label masking periodicity.
            show_grid: If True, draw horizontal gridlines.
            out_file: Path to save the resulting figure to.
            title: Optional plot title.
            figsize: Figure size in inches, as ``(width, height)``.
            fontsize_params: Font sizes for ``'xlabel'``, ``'ylabel'``,
                ``'yticklabels'``, and ``'title'``.
            line_params: Keyword-style dict controlling the CI line
                appearance, with keys ``'color'`` and ``'lw'``.
            grid_params: Keyword-style dict controlling gridline
                appearance, with keys ``'ls'`` and ``'alpha'``.
            legend_params: Keyword arguments forwarded to
                `matplotlib.pyplot.legend` when ``hue`` is set.
            cmap_gene: Colormap name used to color y-tick labels by gene.
            y_ticklabels_mask: Indices (modulo ``len(hue_order)``) of
                y-tick labels to hide, e.g. to avoid duplicate labels
                across hue groups.

        Raises:
            TypeError: If ``hue_order`` is ``None`` while tick labels are
                being masked (``n % len(hue_order)`` requires a sized
                ``hue_order``).
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        ax.set_ylim(0.5, df.shape[0] + 0.5)
        df['y'] = range(df.shape[0], 0, -1)
        df['y_color'] = df['gene'].map(dict(zip(self.HLA, sns.color_palette(cmap_gene, n_colors=len(self.HLA)))))
        ax.set_yticks(df['y'], df[y_ticklabels], fontsize=fontsize_params['yticklabels'])

        n = -1
        for ticklabel, color in zip(ax.get_yticklabels(), df['y_color']):
            n += 1
            ticklabel.set_color(color)
            if n % len(hue_order) in y_ticklabels_mask:
                ticklabel.set_visible(False)

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
    # Example usage: build forest plots for additive and interaction
    # HLA-Ancestry association results.
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
    df['class'] = df['OR'].apply(lambda x: 'risk' if x > 1 else 'protective')
    hla.forest_plot(df=df, out_file=out_file, title='T1D ~ HLA + Ancestry', hue='class', hue_order=['protective', 'risk'], figsize=[4, 6], legend_params={'fontsize': 8, 'loc': 'lower right', 'bbox_to_anchor': (1, 0)}, palette=['C2', 'C3'])

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
    hla.forest_plot(df=df, out_file=out_file, title='T1D ~ HLA * Ancestry', hue='Ancestry', hue_order=['EUR', 'EAS', 'SAS'], figsize=[4, 8], legend_params={'fontsize': 8, 'loc': 'lower right', 'bbox_to_anchor': (1, 0.6)}, y_ticklabels_mask=[0, 2], fontsize_params={'xlabel': 16, 'ylabel': 10, 'yticklabels': 12, 'title': 16})
