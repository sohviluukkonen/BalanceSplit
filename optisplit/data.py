"""
Module for splitting data in a dataframe with pre- and post-processing.

The module contains the following classes:

Author: Sohvi Luukkonen
"""
import numpy as np
import pandas as pd

import logging

from typing import Callable, Literal

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .logs import logger
# from .splitters import OptiSplitBase, OptiSplitSplit, OptiSplitRepeatedSplit, OptiSplitKFold, OptiSplitRepeatedKFold
from .splitter import OptiSplit

def split_dataset(
        data : pd.DataFrame | None = None,
        data_path : str | None = None,
        splitter : OptiSplit = OptiSplit(sizes=[0.8,0.2]),
        smiles_col : str = 'SMILES',
        task_cols : list | None = None,
        ignore_cols : list | None = None,
        output_path : str | None = None,
        compute_stats : bool = True,
        preassigned_smiles : list | None = None,
        dissimilarity_fp_calculator : Callable = GetMorganGenerator(radius=3, fpSize=2048),
        keep_stratified : bool = False,
        verbose : bool = False,
        ) -> pd.DataFrame:

        # If verbose, set logger level to debug
        if verbose:
            # Create a console handler
            ch = logging.StreamHandler()
            # Set the level of the console handler to INFO
            ch.setLevel(logging.INFO)
            # Add the console handler to the logger
            logger.addHandler(ch)
        
        # Get data
        if data is None and data_path is None:
            raise ValueError('Both data and data_path cannot be defined.')
        elif data is not None and data_path is not None:
            raise ValueError('Neither data nor data_path is defined.')
        elif data_path:
            try:
                data = pd.read_csv(data_path)
            except :
                try:
                    data = pd.read_csv(data_path, sep='\t')
                except:
                    raise ValueError('Could not read data from data_path.')

        # Header : 80 characters : text padded with '=' on both sides
        # TODO : give lots of info about the split

        # Get task columns
        if not task_cols:
            task_cols = [col for col in data.columns if col != smiles_col]
        if ignore_cols:
            task_cols = [col for col in task_cols if col not in ignore_cols]

        # Splitter arguments
        smiles_list = data[smiles_col].tolist()
        y = data[task_cols].to_numpy()
        X = np.zeros((y.shape[0], 1)) # Dummy X required by sklearn splitters

        # Split data
        split = splitter.split(
            X, 
            y, 
            smiles_list, 
            task_cols,
            preassigned_smiles=preassigned_smiles,
        )

        # Append split indices to dataframe
        if splitter.n_repeats == 1:
            if splitter.n_splits == 1:
                for i, subset in enumerate(next(split)):
                    data.loc[subset, "Split"] = i
            else:
                for i in range(splitter.n_splits):
                    _, test = next(split)
                    data.loc[test, f"Folds"] = i
        else:
            if splitter.n_splits == 1:
                for r in range(splitter.n_repeats):
                    for i, subset in enumerate(next(split)):
                        data.loc[subset, f"Split_{r}"] = i
            else:
                for r in range(splitter.n_repeats):
                    for i in range(splitter.n_splits):
                        _, test = next(split)
                        data.loc[test, f"Folds_{r}"] = i

        # Recreate stratified task columns
        if splitter.stratify and keep_stratified:
                df_stratified_y = pd.DataFrame(splitter.y, columns=splitter.task_names)
                data = pd.concat([data, df_stratified_y], axis=1).reset_index(drop=True)
                # If column names are not unique, keep only the first occurence
                data = data.loc[:,~data.columns.duplicated()]



        # Compute statistics
        if compute_stats:
            compute_balance_stats(data, splitter.sizes, task_cols)
            compute_dissimilarity_stats(data, smiles_col, dissimilarity_fp_calculator)
            logger.info('-' * 80)

        # Remove stratified task columns
        if splitter.stratify and not keep_stratified:
            stratified_cols = [col for col in data.columns for task in task_cols if col.startswith(task) if col != task] 
            data = data.drop(columns=stratified_cols)
            

        # Save data
        if output_path:
            data.to_csv(output_path, index=False)

        return data



def compute_balance_stats(data : pd.DataFrame, sizes : list[float], task_cols : list[str]):
    """
    Compute split balance statistics for a dataframe.

    Args:
        data (pd.DataFrame): Dataframe containing split indices.
        sizes (list[float]): Relative sizes of the splits.
        task_cols (list[str]): Names of the task columns.
        startified_task_cols (list[str], optional): Names of the stratified task columns. Defaults to None.
    """

    split_cols = [col for col in data.columns if col.startswith('Split') or col.startswith('Folds')]
    if not split_cols:
        raise ValueError('No split columns found in dataframe.')
        
    # Compute balance stats for each split
    for split_col in split_cols:

        # Header : 80 characters : text padded with '=' on both sides
        text = f" Balance statistics for {split_col} "
        pad = int((80 - len(text)) /2)
        header = '-' * pad + text + '-' * pad
        logger.info(header)


        padding_witdth = max(data.columns.str.len()) + 2

        # If stratified, compute balance stats for each stratified task
        for task in task_cols:
            stratified_cols = [col for col in data.columns if col.startswith(task) and col != task]
            if len(stratified_cols) > 0:
                for stask in stratified_cols:
                    counts = data[[stask, split_col]].groupby(split_col).count()
                    n = counts[stask].sum()
                    txt = ''
                    for subset in sorted(data[split_col].unique()):
                        n_subset = counts.loc[subset, stask]
                        txt += f"{int(subset)}: {n_subset/n:.3f} [{n_subset}]\t"
                    logger.info(f"{stask:<{padding_witdth}}\t {txt}")
        logger.info('')
        
        # Compute balance stats for all original tasks
        for task in task_cols:
            padding_witdth = padding_witdth if padding_witdth else max([len(task) for task in task_cols]) + 2
            counts = data[[task, split_col]].groupby(split_col).count()
            n = counts[task].sum()
            txt = ''
            for subset in sorted(data[split_col].unique()):
                n_subset = counts.loc[subset, task]
                txt += f"{int(subset)}: {n_subset/n:.3f} [{n_subset}]\t"
            logger.info(f"{task:<{padding_witdth}}\t {txt}")
        logger.info('')

        # Compute balance stats for all tasks
        balance_score = 0
        harmonic_weights = [ (1/f) / sum([1/f for f in sizes]) for f in sizes ]
        txt = f"{'Overall':<{padding_witdth}}\t "
        n = data.shape[0]
        for i, subset in enumerate(sorted(data[split_col].unique())):
            n_subset = data[data[split_col] == subset].shape[0]
            txt += f"{int(subset)}: {n_subset/n:.3f} [{n_subset}]\t"
            balance_score += np.abs(n_subset/n - sizes[i]) * harmonic_weights[i]
        
        logger.info(f"{txt}")
        
        
        logger.info(f"Balance score: {balance_score:.3f}")      

def compute_dissimilarity_stats(
        data : pd.DataFrame, 
        smiles_col : str, 
        dissimilarity_fp_calculator : Callable = GetMorganGenerator(2, 1024)):

    """
    Compute chemical dissimilarity between subsets of a split.
    """

    split_cols = [col for col in data.columns if col.startswith('Split') or col.startswith('Folds')]
    if not split_cols:
        raise ValueError('No split columns found in dataframe.')
    
    # Compute molecular fingerprints
    mols = [Chem.MolFromSmiles(smiles) for smiles in data[smiles_col].tolist()]
    fps = [dissimilarity_fp_calculator.GetFingerprint(mol) for mol in mols]

    # Compute dissimilarity for each split
    for split_col in split_cols:

        # Header : 80 characters : text padded with '=' on both sides
        text = f" Dissimilarity statistics for {split_col} "
        pad = int((80 - len(text)) /2)
        header = '-' * pad + text + '-' * pad
        logger.info(header)


        # Compute dissimilarity for each subset
        medians = []
        for subset in sorted(data[split_col].unique()):
            min_interset_distances = []
            subset_fps = [fps[i] for i in data[data[split_col] == subset].index]
            other_fps = [fps[i] for i in data[data[split_col] != subset].index]
            for fp in subset_fps:
                sim = DataStructs.BulkTanimotoSimilarity(fp, other_fps)
                min_interset_distances.append(1 - np.max(sim))
            mean, std, median = np.mean(min_interset_distances), np.std(min_interset_distances), np.median(min_interset_distances)
            medians.append(median)
            logger.info(f"Subset {int(subset)}: {mean:.3f} +/- {std:.3f} (median: {median:.3f})")
        logger.info(f"Chemical dissimilarity score: {np.min(medians):.3f}")