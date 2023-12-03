import optuna
import logging
import numpy as np


from functools import reduce
from operator import or_
from abc import ABC, abstractmethod
from typing import Callable

from rdkit import Chem, DataStructs
# from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from .logs import logger
from .scaffolds import Scaffold, Murcko

optuna_logger = logging.getLogger('optuna')
optuna_logger.addHandler(logger)

class ClusteringMethod(ABC):

    """
    Abstract base class for clustering methods.
    """

    @abstractmethod
    def __call__(self, smiles_list : list[str]) -> dict:
        pass

    def get_name(self) -> str:
        return self.__class__.__name__

    def _set_n_clusters(self, N : int) -> None:
        self.n_clusters = self.n_clusters if self.n_clusters is not None else N // 20 



class RandomClustering(ClusteringMethod):

    """
    Randomly cluster a list of SMILES strings into n_clusters clusters.
    
    Attributes
    ----------
    n_clusters : int | tuple[int], optional
        Number of initial clusters. If None, n_clusters = len(smiles_list) // 10.
    seed : int, optional
        Random seed.
    """

    def __init__(self, n_clusters : int| None = None, seed : int = 42, *args, **kwargs) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.seed = seed

    def __call__(self, smiles_list : list[str]) -> dict:
        """
        Randomly cluster a list of SMILES strings into n_clusters clusters.
        
        Parameters
        ----------
        smiles_list : list[str]
            list of SMILES strings to cluster.
        
        Returns
        -------
        clusters : dict
            Dictionary of clusters, where keys are cluster indices and values are indices of SMILES strings.
        """

        self._set_n_clusters(len(smiles_list))

        logger.info("Randomly clustering molecules into %d clusters.", self.n_clusters)

        # Initialize clusters
        clusters = { i: [] for i in range(self.n_clusters) } # type: ignore

        # Randomly assign each molecule to a cluster
        indices = np.random.RandomState(seed=self.seed).permutation(len(smiles_list))
        for i, index in enumerate(indices):
            clusters[i % self.n_clusters].append(index) # type: ignore

        return clusters

        
class ScaffoldClustering(ClusteringMethod):

    """
    Cluster a list of SMILES strings based on Murcko scaffolds.
    """

    def __init__(self, scaffold : Scaffold = Murcko()) -> None:
        super().__init__()
        self.scaffold = scaffold

    def __call__(self, smiles_list : list[str]) -> dict:

        """
        Cluster a list of SMILES strings based on Murcko scaffolds.

        Parameters
        ----------
        smiles_list : list[str]
            list of SMILES strings to cluster.
        """

        logger.info("Clustering molecules based on Murcko scaffolds.")
            
        # Generate scaffolds for each molecule
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        scaffolds = [ self.scaffold(mol) for mol in mols ]

        # Get unique scaffolds and initialize clusters
        unique_scaffolds = list(set(scaffolds))
        clusters = { i: [] for i in range(len(unique_scaffolds)) }

        # Cluster molecules based on scaffolds
        for i, scaffold in enumerate(scaffolds):
            clusters[unique_scaffolds.index(scaffold)].append(i)

        return clusters
    
class DissimilarityClustering(ClusteringMethod):
    
        """
        Abstract base class for clustering methods based on molecular dissimilarity.
        """
    
        def __init__(self, fp_calculator : Callable = GetMorganGenerator(radius=3, fpSize=2048), seed : int = 42, cluster_optimization_range = None ) -> None:
            super().__init__()
            self.fp_calculator = fp_calculator
            self.seed = seed
            self.cluster_optimization_range = cluster_optimization_range

        def __call__(self, smiles_list : list[str]) -> dict:

            """
            Cluster a list of SMILES strings based on molecular dissimilarity.
            
            Parameters
            ----------
            smiles_list : list[str]
                list of SMILES strings to cluster.
            
            Returns
            -------
            clusters : dict
                Dictionary of clusters, where keys are cluster indices and values are indices of SMILES strings.
            """

            logger.info(f"Clustering molecules based on molecular dissimilarity using {self.get_name()}.")

            fps = [self.fp_calculator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]

            if self.cluster_optimization_range is not None:
                self._optimize_clusters(fps)

            clusters = self._get_clusters(fps)

            return clusters
        
        def _get_clusters(self, fps : list) -> dict:
            """
            Get clusters.
            """
            # Get cluster centroids and initialize clusters
            centroid_indices = self._get_centroids(fps)
            clusters = { i: [] for i in range(len(centroid_indices)) }

            # Cluster molecules based on centroids
            for i, fp in enumerate(fps):
                similarities = [DataStructs.FingerprintSimilarity(fp, fps[j]) for j in centroid_indices]
                clusters[np.argmax(similarities)].append(i)

            return clusters
        
        def _optimize_clusters(self, fps : list):
            """
            Optimize number of clusters (in MaxMin) or dissimilarity threshold (in Leadpicker) to maximize
            the dissimilarity between clusters.

            Parameters
            ----------
            fps : list
                list of molecular fingerprints.

            Returns
            -------
            n_clusters or similarity_threshold : int | float
                Optimized number of clusters (in MaxMin) or dissimilarity threshold (in Leadpicker).
            """

            logger.info("Optimizing number of clusters or similarity threshold to maximize dissimilarity between clusters.")

            # Use Optuna to optimize number of clusters or similarity threshold
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda trial: self._maximize_cluster_dissimilarity(trial, fps), n_trials=10, show_progress_bar=True)
            # Get best number of clusters or similarity threshold
            if self.get_name() == "MaxMinClustering":
                logger.info(f"Optimal number of clusters: {study.best_params['n_clusters']}")
                self.n_clusters = study.best_params["n_clusters"]
            else:
                logger.info(f"Optimal similarity threshold: {study.best_params['similarity_threshold']}")
                self.similarity_threshold = study.best_params["similarity_threshold"]
    
        
        def _maximize_cluster_dissimilarity(self, trial : optuna.Trial, fps : list) -> float:
            """
            Optimize number of clusters (in MaxMin) or dissimilarity threshold (in Leadpicker) to maximize
            the dissimilarity between clusters.

            Parameters
            ----------
            trial : optuna.Trial
                Optuna trial.
            
            Returns
            -------
            dissimilarity : float
                Dissimilarity between clusters.
            """

            # Get number of clusters or similarity threshold
            if self.get_name() == "MaxMinClustering":
                self.n_clusters = trial.suggest_int("n_clusters", self.cluster_optimization_range[0], self.cluster_optimization_range[1])
            elif self.get_name() == "LeaderPickerClustering":
                self.similarity_threshold = trial.suggest_float("similarity_threshold", self.cluster_optimization_range[0], self.cluster_optimization_range[1])

            # Get clusters
            clusters = self._get_clusters(fps)

            
            # For each cluster, compute a global fingerprint : sum of fingerprints of all molecules in the cluster
            global_cluster_fps = []
            for cluster in clusters.values():
                cluster_fps = [fps[i] for i in cluster]
                if len(cluster_fps) > 1:
                    global_cluster_fps.append( reduce(or_, cluster_fps))
                else:
                    global_cluster_fps.append(cluster_fps[0])
            # Compute median pairwise dissimilarity between global fingerprints
            dissimilarity = np.median([1 - DataStructs.FingerprintSimilarity(fp1, fp2) for i, fp1 in enumerate(global_cluster_fps) for j, fp2 in enumerate(global_cluster_fps) if i != j])

            return dissimilarity

        
        @abstractmethod
        def _get_centroids(self, fps : list) -> list:
            pass

class MaxMinClustering(DissimilarityClustering):

    """
    Cluster a list of SMILES strings based on molecular dissimilarity using the MaxMin algorithm.

    Attributes
    ----------
    fp_calculator : Callable, optional. 
        Function to compute molecular fingerprints.
    n_clusters : int | tuple(None), optional
        Number of clusters. If None, n_clusters = len(smiles_list) // 10. 
    cluster_optimization_range : tuple[int], optional
        Range of number of clusters to optimize to maximize dissimilarity between clusters.
        If None, n_clusters will not be optimized.
    seed : int, optional
        Random seed.
    """

    def __init__(
            self, 
            fp_calculator : Callable = GetMorganGenerator(radius=3, fpSize=2048),
            n_clusters : int | None = None,
            cluster_optimization_range : tuple[int,int] | None = None,
            seed : int = 42,
        ) -> None:
        super().__init__(fp_calculator, seed, cluster_optimization_range)
        self.n_clusters = n_clusters
        # self.seed = seed
        # self.cluster_optimization_range = cluster_optimization_range

    def _get_centroids(self, fps : list) -> list:

        """
        Get cluster centroids using the MaxMin algorithm.
        
        Parameters
        ----------
        fps : list
            list of molecular fingerprints.
        
        Returns
        -------
        centroid_indices : list
            list of indices of cluster centroids.
        """

        self._set_n_clusters(len(fps))

        picker = rdSimDivPickers.MaxMinPicker()
        centroid_indices = picker.LazyBitVectorPick(fps, len(fps), self.n_clusters, seed=self.seed)

        return centroid_indices
    
class LeaderPickerClustering(DissimilarityClustering):

    """
    Cluster a list of SMILES strings based on molecular dissimilarity using LeadPicker to select centroids.

    Attributes
    ----------
    fp_calculator : Callable, optional.
        Function to compute molecular fingerprints.
    similarity_threshold : float, optional
        Similarity threshold for clustering. 
    cluster_optimization_range : tuple[float], optional
        Range of similarity threshold to optimize to maximize dissimilarity between clusters.
        If None, similarity_threshold will not be optimized.
    """

    def __init__(
            self, 
            fp_calculator: Callable = GetMorganGenerator(radius=3, fpSize=2048),
            similarity_threshold : float | None = 0.7,
            cluster_optimization_range : tuple[float, float] | None = None,
            seed : int = 42,
     ) -> None:
        super().__init__(fp_calculator, seed, cluster_optimization_range)
        self.similarity_threshold = similarity_threshold
        # self.cluster_optimization_range = cluster_optimization_range
        if cluster_optimization_range is not None:
            assert cluster_optimization_range[0] > 0 and cluster_optimization_range[1] < 1, \
                "cluster_optimization_range must be a tuple of floats between 0 and 1."

    def _get_centroids(self, fps : list) -> list:

        """
        Get cluster centroids using LeadPicker.

        Parameters
        ----------
        fps : list
            list of molecular fingerprints.
        
        Returns
        -------
        centroid_indices : list
            list of indices of cluster centroids.
        """

        picker = rdSimDivPickers.LeaderPicker()
        centroid_indices = picker.LazyBitVectorPick(fps, len(fps), self.similarity_threshold)

        return centroid_indices
