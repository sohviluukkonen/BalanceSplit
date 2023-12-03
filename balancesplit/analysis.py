import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from typing import Callable

def min_Tanimoto_distance(fp, fps):
    return 1 - np.max(DataStructs.BulkTanimotoSimilarity(fp, fps))

class AnalyzeSplit():

    def __init__(
            self, 
            df: pd.DataFrame, 
            smiles_col: str = 'SMILES', 
            split_col: str = 'Split',
            task_cols: list[str] | None = None,
            ignore_cols: list[str] | None = None,
            output_path: str | None= None,
            fractions: list[float] | None = None,
            fp_calculator: Callable = GetMorganGenerator(radius=3, fpSize=2048),
            distance_metric: Callable = min_Tanimoto_distance,):
        
        self.df = df
        self.smiles_col = smiles_col
        self.split_col = split_col
        self.output_path = output_path
        self.subsets = sorted(df[split_col].unique())
        self._get_task_cols(task_cols, ignore_cols)
        self._get_ideal_fractions(fractions)

        # Balance score
        self._get_balance_stats()
        self._balance_score()

        # Fingerprints
        self.featurizer = fp_calculator
        self._compute_fingerprints()
        self.distance_metric = distance_metric
        self._compute_min_interset_distances()
        self._dissimilarity_score()

    def _get_task_cols(self, task_cols, ignore_cols):
        """ Get task columns from dataframe. """
        if not task_cols:
            self.task_cols = [col for col in self.df.columns if col not in [self.smiles_col, self.split_col]]
        if ignore_cols:
            self.task_cols = [col for col in self.task_cols if col not in ignore_cols]

    def _get_ideal_fractions(self, fractions):
        """ Set idels fractions depending on the split column. """
        if self.split_col.startswith('Fold'):
            n_folds = df.split_cols.nunique()
            self.ideal_fractions = [1/n_folds]*n_folds
        elif self.split_col.startswith('Split'):
            self.ideal_fractions = fractions
            assert len(self.ideal_fractions) == len(self.subsets), f"Expected {len(self.subsets)} fractions, got {len(self.ideal_fractions)} instead."

    def _get_balance_stats(self):

        if not hasattr(self, 'df_balance'):
        
            df_balance = pd.DataFrame(columns=['Task', 'Subset', 'Count', 'Fraction', 'Balance'])
            df_counts = self.df.groupby(self.split_col).count().reset_index()

            for task in self.task_cols:
                for subset in self.subsets:
                    df_balance = pd.concat([df_balance, pd.DataFrame({
                        'Task': task,
                        'Subset': subset,
                        'Count': df_counts.loc[df_counts[self.split_col] == subset, task],
                        'Fraction': df_counts.loc[df_counts[self.split_col] == subset, task]/df_counts[task].sum(),
                        'Balance': df_counts.loc[df_counts[self.split_col] == subset, task]/df_counts[task].sum() - self.ideal_fractions[self.subsets.index(subset)]
                    })], ignore_index=True)

            self.df_balance = df_balance 

    def _balance_score(self):

        harmonic_weights_denominator = sum([1/f for f in self.ideal_fractions])
        harmonic_weights = [ (1/f) / harmonic_weights_denominator for f in self.ideal_fractions ]
        bs = 0
        for i, subset in enumerate(self.subsets):
            diffs = self.df_balance.loc[self.df_balance['Subset'] == subset, 'Balance'].tolist()
            bs += harmonic_weights[i] * sum([abs(d) for d in diffs])

        self.balance_score = bs

    def _compute_fingerprints(self):
        self.df['Fingerprint'] = self.df[self.smiles_col].apply(
            lambda x: self.featurizer.GetFingerprint(Chem.MolFromSmiles(x))
        )

    def _compute_min_interset_distances(self):

        # Check df contains "MinIntersetDistance" column
        if 'MinIntersetDistance' not in self.df.columns:
            
            for subset in self.subsets:
                other_fps = self.df[self.df[self.split_col] != subset]['Fingerprint'].tolist()
                self.df.loc[self.df[self.split_col] == subset, 'MinIntersetDistance'] = self.df[self.df[self.split_col] == subset]['Fingerprint'].apply(
                    lambda x: self.distance_metric(x, other_fps)
                )

    def _dissimilarity_score(self):
        self.dissimilarity_score = self.df.groupby(self.split_col)['MinIntersetDistance'].median().min()

    def plot_dissimilarity_score_distribution_per_subset(self, figsize=None, fontsize=12, title=None, legend=True, plot_function: Callable = sns.boxplot, **kwargs):

        if not figsize:
            figsize = (len(self.subsets), 4)
        fig, ax = plt.subplots(figsize=figsize)
        plot_function(data=self.df, x=self.split_col, y='MinIntersetDistance', ax=ax, **kwargs)

        ax.set_xlabel('Subset', fontsize=fontsize)
        ax.set_ylabel('Min. Interset Distance', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        
        if not legend:
            ax.get_legend().remove()
        if self.output_path:
            plt.savefig(self.output_path, dpi=300)

        return fig, ax  

    def plot_chemical_space(self, dim_reduction=PCA(n_components=2), figsize : tuple[float, float] =(4,4), fontsize=12, title=None, legend=True, **kwargs):

        fps = np.array(self.df['Fingerprint'].tolist())
        fps_red = dim_reduction.fit_transform(fps)
        assert len(fps_red[0]) == 2, f"Expected 2 dimensions, got {fps_red.shape[1]} instead."

        df_red = pd.DataFrame(fps_red, columns=['D1', 'D2'])
        df_red[self.split_col] = self.df[self.split_col]

        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(data=df_red, x='D1', y='D2', hue=self.split_col, ax=ax, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        if not legend:
            ax.get_legend().remove()
        if self.output_path:
            plt.savefig(self.output_path, dpi=300)

        return fig, ax
    
    def get_balance_stats(self):

        df_pivot = self.df_balance.pivot(index='Task', columns='Subset', values=['Count', 'Fraction', 'Balance'])

    def plot_fractions_per_task(self, figsize=None, fontsize=12, title=None, legend=True):


        if not figsize:
            figsize = (len(self.task_cols), 4)
        fig, ax = plt.subplots(figsize=figsize)
        df_pivot = self.df_balance.pivot(index='Task', columns='Subset', values='Fraction')
        df_pivot.plot.bar(ax=ax, stacked=True)

        yline = 0
        for f in self.ideal_fractions[:-1]:
            yline += f
            ax.axhline(yline, color='black', linestyle='--', linewidth=1)
        
        ax.set_ylim(0, 1)
        
        ax.set_xlabel('Task', fontsize=fontsize)
        ax.set_ylabel('Fraction', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        
        if not legend:
            ax.get_legend().remove()
        if self.output_path:
            plt.savefig(self.output_path, dpi=300)

        return fig, ax

    def plot_balance_score_distribution_per_subset(self, figsize=None, fontsize=12, title=None, legend=True, plot_function: Callable = sns.boxplot, **kwargs):

        
        if not figsize:
            figsize = (len(self.subsets), 4)
        fig, ax = plt.subplots(figsize=figsize)
        plot_function(data=self.df_balance, x='Subset', y='Balance', ax=ax, **kwargs)

        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        ax.set_xlabel('Subset', fontsize=fontsize)
        ax.set_ylabel('Balance Score', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        
        if not legend:
            ax.get_legend().remove()
        if self.output_path:
            plt.savefig(self.output_path, dpi=300)

        return fig, ax
    