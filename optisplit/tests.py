# Unit test for gbmt-splits
import os
from time import clock_settime
import pandas as pd
import numpy as np
from unittest import TestCase
from parameterized import parameterized
from scipy import cluster

from .clustering import (
    RandomClustering, 
    MaxMinClustering, 
    LeaderPickerClustering, 
    MurckoScaffoldClustering
)
from .splitter import OptiSplit
from .data import split_dataset

preassigned_smiles = {
    'Brc1cccc(Nc2nc3c(N4CCCC4)ncnc3s2)c1' : 0,
    'C#CCn1c(=O)c2c(nc3n2CCCN3C2CCC2)n(C)c1=O' : 1,
}

import logging
logging.basicConfig(level=logging.DEBUG)

class TestSplits(TestCase):

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    seed = 2022
    time_limit = 10

    df = pd.read_csv(test_data_path)
    smiles_list = df['SMILES'].tolist()
    task_list = df.drop(columns=['SMILES']).columns.tolist()
    y = df.drop(columns=['SMILES']).to_numpy()
    X = np.zeros((y.shape[0], 1))

    @parameterized.expand([
        (None, {'clustering_method' : RandomClustering(), 'n_splits' : 3 , 'n_repeats' : 3}, ),
        (None, {'clustering_method' : MaxMinClustering(), 'n_splits' : 3},),
        (preassigned_smiles, {'clustering_method' : LeaderPickerClustering(), 'sizes' : [0.8, 0.1, 0.1],}, ),
        (None, {'clustering_method' : MurckoScaffoldClustering(), 'sizes' : [0.8, 0.1, 0.1]},),
        (None, {'clustering_method' : 'predefined_clusters', 'sizes' : [0.8, 0.1, 0.1]},),
    ])

    def test_OptiSplit(self, preassigned_smiles, kwargs):
        kwargs.update({'time_limit_seconds' : self.time_limit, })

        if kwargs['clustering_method'] == 'predefined_clusters':
            # Create predifined clusters
            mol_idx = np.arange(len(self.smiles_list))
            clusters = np.array_split(mol_idx, 20)
            kwargs['clustering_method'] = {i : [] for i in range(20)}
            for i, cluster in enumerate(clusters):
                for idx in cluster:
                    kwargs['clustering_method'][i].append(self.smiles_list[idx])

        splitter = OptiSplit(**kwargs)
        split_generator = splitter.split(
            self.X, 
            self.y, 
            self.smiles_list, 
            self.task_list,
            preassigned_smiles=preassigned_smiles,
            )
        
        for i, split in enumerate(split_generator):
            if splitter.n_splits > 1:
                assert len(split) == 2
            else:
                assert len(split) == len(kwargs['sizes'])
            
            if preassigned_smiles:
                for asg_smiles, asg_subset in preassigned_smiles.items():
                    smiles_idx = self.smiles_list.index(asg_smiles)
                    indices = split[asg_subset]
                    assert smiles_idx in indices
        
        assert i == splitter.n_splits * splitter.n_repeats - 1
        assert splitter.n_tasks == self.y.shape[1]
    


class TestDatasetSplit(TestCase):
    
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    seed = 2022
    time_limit = 10

    df = pd.read_csv(test_data_path)
    smiles_list = df['SMILES'].tolist()
    y = df.drop(columns=['SMILES']).to_numpy()
    X = np.zeros((y.shape[0], 1))

    @parameterized.expand([
        (OptiSplit(sizes=[0.8,0.2]), {}),
        (OptiSplit(sizes=[0.8,0.2], n_repeats=2), {}),
        (OptiSplit(n_splits=3, n_repeats=1), {}),
        (OptiSplit(n_splits=3, n_repeats=2), {}),
    ])

    def test_split_dataset(self, splitter, kwargs):

        data = split_dataset(
            data=self.df.copy(),
            splitter=splitter,
            smiles_col='SMILES',
            **kwargs,
        )

        if splitter.n_splits == 1:
            split_cols = [col for col in data.columns if col.startswith('Split')]
            assert len(split_cols) == splitter.n_repeats
            for col in split_cols:
                assert len(data[col].unique()) == 2
        else:
            fold_cols = [col for col in data.columns if col.startswith('Folds')]
            assert len(fold_cols) == splitter.n_repeats
            for col in fold_cols:
                assert len(data[col].unique()) == splitter.n_splits

class TestClusteringMethods(TestCase):
    
    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.csv')
    seed = 2022
    time_limit = 10

    df = pd.read_csv(test_data_path)
    smiles_list = df['SMILES'].tolist()
    y = df.drop(columns=['SMILES']).to_numpy()
    X = np.zeros((y.shape[0], 1))

    @parameterized.expand([
        (None),
        (10),
    ])
    def test_random_clustering(self, n_clusters):
        clustering = RandomClustering(n_clusters=n_clusters)
        clusters = clustering(self.smiles_list)
        if n_clusters:
            assert len(clusters) == n_clusters

    def test_murcko_scaffold_clustering(self):
        clustering = MurckoScaffoldClustering()
        clusters = clustering(self.smiles_list)
        
    @parameterized.expand([
        (100, None),
        (None, (50, 200)),
    ])
    def test_maxmin_clustering(self, n_clusters, cluster_range):
        clustering = MaxMinClustering(n_clusters=n_clusters, cluster_optimization_range=cluster_range)
        clusters = clustering(self.smiles_list)
        if n_clusters:
            assert len(clusters) == n_clusters
        elif cluster_range:
            assert len(clusters) >= cluster_range[0]
            assert len(clusters) <= cluster_range[1]

    @parameterized.expand([
        (0.8, None),
        (None, (0.5, 0.9)),
    ])
    def test_leaderpicker_clustering(self, similarity_threshold, sim_range):
        clustering = LeaderPickerClustering(similarity_threshold=similarity_threshold, cluster_optimization_range=sim_range)
        clusters = clustering(self.smiles_list)
