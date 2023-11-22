"""
Module for splitting data into subsets for globally balanced multi-task learning.

Authors: Sohvi Luukkonen & Giovanni Tricarico
"""

from heapq import merge
from pickletools import int4
from turtle import st
from typing import Generator
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from pulp import *
from sklearn.cluster import cluster_optics_dbscan

from .clustering import ClusteringMethod, MaxMinClustering, RandomClustering
from .logs import logger

class OptiSplit():
    """
    Base class for OptiSplit splitters

    Attributes
    ----------
    n_splits : int
        The number of splits to be generated. If sizes, n_splits is ignored.
    n_repeats : int
        The number of repeats to be generated
    sizes : list[float] | None
        The relative sizes of the subsets. If none, n_splits is used to generate equal sized subsets for k-fold splitting.
    stratify : bool | list[str]
        If True, stratify the data in each task. If False, do not stratify the data.
        If list of task names, stratify only the tasks in the list.
    stratify_reg_nbins : int
        The number of bins for stratification of numerical values
    clustering_method : ClusteringMethod | dict
        The clustering method or dictionary of precomputed clusters
    equal_weight_perc_compounds_as_tasks : bool
        If True, matching the % records will have the same weight as matching the % self.df of individual tasks.
        If False, matching the % records will have a weight X times larger than the X tasks.
    custom_weights : list[float] | None
        The list of custom weights for the records (1st element) and tasks (rest of the elements).
        If None, the weights are calculated automatically, based on value of equal_weight_perc_compounds_as_tasks.
    absolute_gap : float
        The absolute gap between the absolute optimal objective and the current one at which the solver
        stops and returns a solution. Can be very useful for cases where the exact solution requires
        far too long to be found to be of any practical use.
    time_limit_seconds : int
        The time limit in seconds for the solver (by default depends on the number of datapoints and tasks)
    n_jobs : int
        The maximal number of threads to be used by the solver.

    Methods
    -------
    split(X : np.ndarray, y : np.ndarray, smiles_list : list[str] | None = None, task_names : list[str] | None = None, preassigned_smiles : dict[str, int] | None = None)
        Split the data into subsets in sklearn style
    get_n_splits(X : np.ndarray, y : np.ndarray = None)
        Get the number of splits
    get_meta_routing()
        Get the meta routing
    _split(X : np.ndarray, y : np.ndarray, smiles_list : list[str] | None = None, task_names : list[str] | None = None, preassigned_smiles : dict[str, int] | None = None)
        Split the data into subsets
    _get_predefined_clusters(smiles_list : list[str])
        Get predefined clusters.
    _check_input_consistency(X : np.ndarray, y : np.ndarray, smiles_list : list[str] | None = None, task_names : list[str] | None = None, preassigned_smiles : dict[str, int] | None = None)
        Check that the input data is consistent
    _stratify(y : np.ndarray, task_names : list[str] | None = None)
        Stratify the data in each task
    _get_preassigned_clusters(smiles_list : list[str], preassigned_smiles : dict[str, int], clusters : dict)
        Preassign clusters to subset/folds based on preassigned smiles
    _one_hot_encode(y : np.ndarray)
        One hot encode the target values
    _get_default_time_limit_seconds(nmols : int, ntasks : int)
        Compute the default time limit for linear programming
    _get_data_summary_per_cluster(y : np.ndarray, clusters : dict)
        Compute the number of datapoints per task for each cluster
    _merge_clusters_with_balancing_mapping(tasks_vs_clusters_array : np.ndarray, sizes : list[float] = [0.9, 0.1, 0.1], equal_weight_perc_compounds_as_tasks : bool = False, absolute_gap : float = 1e-3, time_limit_seconds : int = 60*60, max_N_threads : int = 1, preassigned_clusters : dict[int, int] | None = None)
        Linear programming function needed to balance the self.df while merging clusters
    """
    def __init__(
        self,
        n_splits : int | None = None,
        n_repeats : int = 1,
        sizes : list[float] | None = None,
        clustering_method : ClusteringMethod | dict = MaxMinClustering(),
        equal_weight_perc_compounds_as_tasks : bool = False,
        custom_weights : list[float] | None = None,
        absolute_gap : float = 1e-3,
        time_limit_seconds : int | None = None,
        n_jobs : int = 1,
        stratify : bool | list[str] = True,
        stratify_reg_nbins : int = 5,  
    ):
        assert any([n_splits, sizes]), "Either n_splits or sizes must be provided"
        assert not (n_splits and sizes), "Only one of n_splits and sizes must be provided"

        if sizes:
            self.n_splits = 1
            self.sizes = sizes
        else:
            self.n_splits = n_splits
            self.sizes = [1 / n_splits] * n_splits
        self.n_subsets = len(self.sizes)
        
        self.n_repeats = n_repeats
        self.clustering_method = clustering_method
        self.equal_weight_perc_compounds_as_tasks = equal_weight_perc_compounds_as_tasks
        self.custom_weights = custom_weights
        self.absolute_gap = absolute_gap
        self.time_limit_seconds = time_limit_seconds
        self.n_jobs = n_jobs
        self.stratify = stratify
        self.stratify_reg_nbins = stratify_reg_nbins

        if self.n_repeats > 1:
            # Check if self.clustering_method is MaxMinClustering or RandomClustering
            if not (isinstance(self.clustering_method, MaxMinClustering) or isinstance(self.clustering_method, RandomClustering)):
                raise ValueError("n_repeats only supports MaxMinClustering and RandomClustering")  

        logger.info(f"OptiSplit splitter initialized with {self.n_splits} subsets of sizes {self.sizes}")  
            
    def split(
            self, 
            X : np.ndarray,
            y : np.ndarray, 
            smiles_list : list[str] | None = None, 
            task_names : list[str] | None = None,
            preassigned_smiles : dict[str, int] | None = None,
            *args, **kwargs) -> Generator:
        
        """
        Split the data into subsets.

        Parameters
        ----------
        X : np.ndarray
            The feature matrix
        y : np.ndarray
            The target value matrix
        smiles_list : list[str] | None
            The list of SMILES strings
        task_names : list[str] | None
            The list of task names
        preassigned_smiles : dict[str, int] | None
            The dictionary of preassigned smiles. The keys are the smiles and
            the values are the subset/fold indices.

        Returns
        -------
        Generator
            The generator of tuples of molecule indices for each subset
        """       
        # Check that the input data is consistent
        self._check_input_consistency(X, y, smiles_list, task_names, preassigned_smiles)

        # Get initial objective weights (modified later if stratify is True)
        self._get_initial_objective_weights()
        
        # Stratify the data
        if self.stratify:
            self._stratify()

        # One hot encode the target value matrix
        self.y = self._one_hot_encode(self.y)     

        for i in range(self.n_repeats):

            # Cluster the data
            if isinstance(self.clustering_method, dict):
                clusters = self._get_predefined_clusters(self.smiles_list)
            else:
                clusters = self.clustering_method.__call__(self.smiles_list)

            # Preassign clusters to subset/folds based on preassigned smiles
            if self.preassigned_smiles:
                preassigned_clusters = self._get_preassigned_clusters(self.smiles_list, self.preassigned_smiles, clusters)
            else:
                preassigned_clusters = None

            # Compute the number of datapoints per task for each cluster
            task_vs_clusters = self._get_data_summary_per_cluster(self.y, clusters)

            # Set time limit for linear programming
            if self.time_limit_seconds is None:
                self.time_limit_seconds = self._get_default_time_limit_seconds(self.y.shape[0], self.y.shape[1])

            # Merge the clusters with a linear programming method to create the subsets
            merged_clusters_mapping = self._merge_clusters_with_balancing_mapping(
                task_vs_clusters, 
                preassigned_clusters)  
            
            # Subset : smiles -mapping:
            subset_mols_mapping = {}
            for subset in range(self.n_subsets):
                cluster_indices = [i for i, x in enumerate(merged_clusters_mapping) if x == subset]
                smiles_indices = [x for i, cluster in clusters.items() if i in cluster_indices for x in cluster]
                subset_mols_mapping[subset] = sorted(smiles_indices)

            if preassigned_clusters:
                for cluster_idx, subset in preassigned_clusters.items():
                    assert merged_clusters_mapping[cluster_idx] == subset, "The preassigned clusters are not assigned to the correct subsets"

            # Create folds
            if self.n_splits == 1:
                fold = [ subset_mols_mapping[subset] for subset in range(self.n_subsets) ]
                yield fold
            else:
                for i in range(self.n_subsets):
                    train = [ subset_mols_mapping[subset] for subset in range(self.n_subsets) if subset != i ]
                    train = sorted([ item for sublist in train for item in sublist ])
                    test = sorted(subset_mols_mapping[i])
                    yield train, test

            # Update clustering seed
            if self.n_repeats > 1:
                self.clustering_method.seed += 1
                    

    def get_n_splits(self, X: np.ndarray, y: np.ndarray | None = None, *args, **kwargs):
        return self.n_splits * self.n_repeats

    def _get_initial_objective_weights(self):
        """
        Get the weights for the records and (non-stratified) tasks.
        """

        # Create WT = obj_weights
        if self.custom_weights is not None:
            self.obj_weights = np.array(self.custom_weights)
        elif ((self.n_tasks > 1) & (self.equal_weight_perc_compounds_as_tasks == False)):
            self.obj_weights = np.array([self.n_tasks] + [1] * (self.n_tasks))
        else:
            self.obj_weights = np.array([1] * (self.n_tasks+1) )


        assert len(self.obj_weights) == self.n_tasks + 1, "The number of custom weights must be equal to the number of tasks + 1"
        
        self.obj_weights = self.obj_weights / np.sum(self.obj_weights)

    def _get_predefined_clusters(self, smiles_list : list[str]):
        """
        Get predefined clusters, check that the SMILES strings in the clustering dictionary
        match the SMILES strings in the SMILES list, and transform SMILES strings to indices.

        Parameters
        ----------
        smiles_list : list[str]
            The list of SMILES strings

        Returns
        -------
        dict
            The dictionary of clusters. The keys are cluster indices and values are indices of SMILES strings.
        """

        smiles_from_clusters = [smiles for cluster in self.clustering_method.values() for smiles in cluster]
        if set(smiles_list) != set(smiles_from_clusters):
            raise ValueError("The SMILES strings in the clustering dictionary must match the SMILES strings in the SMILES list")
        # Clusters : cluster index -> list of smiles indices
        clusters = {i : [j for j, smiles in enumerate(smiles_list) if smiles in smiles_from_clusters] for i in self.clustering_method.keys()}

        return clusters
    
    def _check_input_consistency(
            self, 
            X : np.ndarray, 
            y : np.ndarray, 
            smiles_list : list[str] | None = None, 
            task_names : list[str] | None = None,
            preassigned_smiles : dict[str, int] | None = None):
        
        """
        Check that the input data is consistent and raise errors if not.
        If smiles_list and task_names are not provided, they are created.
        If all checks pass, create smiles_list, task_names and preassigned_smiles attributes.
        """

        # Check that X and y have the same number of rows and that smiles_list and task_names have the same number of columns
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"
        self.y = y

        if smiles_list is not None:
            assert X.shape[0] == len(smiles_list), "X and smiles_list must have the same number of rows"
            self.smiles_list = smiles_list
        if task_names is not None:
            assert y.shape[1] == len(task_names), "y and task_names must have the same number of columns"
            self.task_names = task_names

        # Create task_names if not provided
        if not task_names:
            self.task_names = [f"task_{i}" for i in range(y.shape[1])]
        self.n_tasks = len(self.task_names)

        # Check that if preassigned_smiles is provided, (0) n_splits not > 1 (1) smiles_list must be provided, (2) preassigned_smiles must be a dictionary, 
        # (3) preassigned_smiles must be a subset of smiles_list, (4) preassigned_smiles must be a subset of range(len(sizes))
        if preassigned_smiles:
            if self.n_splits != 1:
                raise ValueError("preassigned_smiles can only be used with n_splits = None and sizes has to be provided")
            if smiles_list is None:
                raise ValueError("smiles_list must be provided when preassigned_smiles is provided")
            if not isinstance(preassigned_smiles, dict):
                raise ValueError("preassigned_smiles must be a dictionary")
            if not set(preassigned_smiles.keys()).issubset(set(smiles_list)):
                raise ValueError("preassigned_smiles must be a subset of smiles_list")
            if not set(preassigned_smiles.values()).issubset(set(range(len(self.sizes)))):
                raise ValueError("preassigned_smiles must be a subset of range(len(sizes))")
        self.preassigned_smiles = preassigned_smiles if preassigned_smiles else None

        # If smiles_list is not provided, check that clustering_method is RandomClustering or a dictionary, and create smiles_list
        if not smiles_list:
            if not isinstance(self.clustering_method, RandomClustering) or \
                not isinstance(self.clustering_method, dict):
                raise ValueError("smiles_list must be provided when clustering_method is not RandomClustering")
            else:
                self.smiles_list = [f"smiles_{i}" for i in range(X.shape[0])]   

        # Check that sizes is a list of floats and that the sum of sizes is 1
        if self.custom_weights:
            assert self.n_tasks + 1 == len(self.custom_weights), "The number of custom weights must be equal to the number of tasks + 1"

        # return smiles_list, task_names
    
    def _stratify(self,):
        """
        Stratify the data in each task. If the values are floats, the data is stratified
        into bins, and each bin is one-hot encoded to new columns. If values are integers
        or strings, the data is stratified into one-hot encoded to new columns. 
        """        
        def is_numeric(x):
            try:
                float(x)
                return True
            except:
                return False
        
        df = pd.DataFrame(self.y, columns=self.task_names)
        stratified_task_names = []
        stratified_objective_weights = [self.obj_weights[0]]
        
        for task_name in self.task_names:
            task_weight = self.obj_weights[self.task_names.index(task_name)+1]
            
            # If stratify is a list of task names, don't stratify other tasks
            if isinstance(self.stratify, list) and task_name not in self.stratify:
                df = df[[c for c in df if c not in [task_name]] + [task_name]] # Move task to the end
                stratified_task_names.append(task_name)
                stratified_objective_weights.append(task_weight)
                continue

            task_values = df[task_name].dropna().unique()
            
            # Check if values are numerical
            task_values_numerical = np.array([is_numeric(task_value) for task_value in task_values])
            all_values_numerical = np.all(task_values_numerical)
            any_values_numerical = np.any(task_values_numerical)
            
            # If values both numerical and non-numerical, raise error
            if not all_values_numerical and any_values_numerical:
                raise ValueError(f"Task {task_name} has both numerical and non-numerical values, which is not supported for stratification")
            
            # If only non-numerical values or only integers, stratify into one-hot encoded columns
            elif not any_values_numerical or all( value % 1 == 0 for value in task_values):
                for value in task_values:
                    stratified_task_names.append(f"{task_name}_{value}")
                    df[f"{task_name}_{value}"] = df[task_name].apply(lambda x: 1 if x == value else np.nan)
            # If values are floats, stratify into bins
            else:
                sorted_task_values = np.sort(task_values)
                bins = np.array_split(sorted_task_values, self.stratify_reg_nbins)
                for i, bin in enumerate(bins):
                    key = f"{task_name}_{bin[0]:.2f}_{bin[-1]:.2f}"
                    stratified_task_names.append(key)
                    df[key] = df[task_name].apply(lambda x: 1 if x in bin else np.nan)
            df = df.drop(columns=[task_name])

            number_of_subtasks = df.columns.str.startswith(task_name).sum()
            logger.info(f"Task '{task_name}' stratified into {number_of_subtasks} subtasks for balancing.")

            stratified_objective_weights += [task_weight / number_of_subtasks] * number_of_subtasks

        self.y = df.to_numpy()
        self.task_names = stratified_task_names
        self.obj_weights = stratified_objective_weights

    def _get_preassigned_clusters(
            self, 
            smiles_list : list[str],
            preassigned_smiles : dict[str, int], 
            clusters : dict) -> dict:
        """
        Preassign clusters to subset/folds based on preassigned smiles

        Parameters
        ----------
        smiles_list : list[str]
            The list of SMILES strings
        preassigned_smiles : dict[str, int]
            The dictionary of preassigned smiles. The keys are the smiles and
            the values are the subset/fold indices.
        clusters : dict
            The dictionary of clusters. The keys are cluster indices and 
            the values are indices of SMILES strings.

        Returns
        -------
        dict
            The dictionary of preassigned clusters. The keys are the cluster indices and
            the values are the subset/fold indices.
        """

        preassigned_clusters = {}
        for smiles, subset in preassigned_smiles.items():
            if smiles not in smiles_list:
                raise ValueError(f"Preassigned SMILES string {smiles} not found in smiles_list")
            else:
                smiles_idx = smiles_list.index(smiles)
            for cluster_idx, cluster in clusters.items():
                if smiles_idx in cluster:
                    preassigned_clusters[cluster_idx] = subset
                    logger.info(f"Preassigned cluster {cluster_idx} (containing {smiles}) to subset {subset}")

        return preassigned_clusters

    def _one_hot_encode(self, y : np.ndarray) -> np.ndarray:
        """
        One hot encode the target values. All non-NaN values are encoded as 1,
        and all NaN values are encoded as NaN.
        """
        y_ = np.empty(y.shape)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if not isinstance(y[i, j], float) or not np.isnan(y[i, j]):
                    y_[i, j] = 1
                else:
                    y_[i, j] = y[i, j]
        return y_

    def _get_default_time_limit_seconds(self, nmols : int, ntasks : int) -> int:
        """
        Compute the default time limit for linear programming based on 
        number of datapoints and number of tasks.
        
        Parameters
        ----------
        nmols : int
            Number of datapoints
        ntasks : int
            Number of tasks
        
        Returns
        -------
        int
            The default time limit in seconds
        """
        tmol = nmols ** (1/3)
        ttarget = np.sqrt(ntasks)
        tmin = 10
        tmax = 60 * 60
        tlim = int(min(tmax, max(tmin, tmol * ttarget)))
        logger.info(f'Time limit for LP: {tlim}s')
        return tlim

    def _get_data_summary_per_cluster(
            self,
            y : np.ndarray,
            clusters : dict ) -> np.ndarray:
        
        """
        Compute the number of datapoints per task for each cluster.

        Parameters
        ----------
        y : np.ndarray
            The target values
        clusters : dict
            The dictionary of clusters. The keys are cluster indices and values are indices of SMILES strings.

        Returns
        -------
        np.ndarray of shape (len(tasks)+1, len(clusters))
            Array with each columns correspoding to a cluster and each row to a task
            plus the 1st row for the number of molecules per cluster    
        """

        ntasks = y.shape[1]
        task_vs_clusters = np.zeros((ntasks+1, len(clusters)))

        # 1st row is the number of molecules per cluster
        task_vs_clusters[0, :] = [len(cluster) for cluster in clusters.values()]

        # Compute the number of datapoints per task for each cluster
        for i in range(ntasks):
            for j, cluster in clusters.items():
                task_vs_clusters[i+1, j] = np.count_nonzero(y[cluster, i])
        
        return task_vs_clusters

    def _merge_clusters_with_balancing_mapping(
            self, 
            tasks_vs_clusters_array : np.ndarray,
            preassigned_clusters : dict[int, int] | None = None) -> list[list[int]]:
            """
            Linear programming function needed to balance the self.df while merging clusters.

            Paper: Tricarico et al., Construction of balanced, chemically dissimilar training, validation 
            and test sets for machine learning on molecular self.dfsets, 2022, 
            DOI: https://doi.org/10.26434/chemrxiv-2022-m8l33-v2

            Parameters
            ----------
            tasks_vs_clusters_array : 2D np.ndarray
                - the cross-tabulation of the number of self.df points per cluster, per task.
                - columns represent unique clusters.
                - rows represent tasks, except the first row, which represents the number of records (or compounds).
                - Optionally, instead of the number of self.df points, the provided array may contain the *percentages*
                    of self.df points _for the task across all clusters_ (i.e. each *row*, NOT column, may sum to 1).
                IMPORTANT: make sure the array has 2 dimensions, even if only balancing the number of self.df records,
                    so there is only 1 row. This can be achieved by setting ndmin = 2 in the np.ndarray function.
            preassigned_clusters : dict
                - a dictionary of the form {cluster_index: ML_subset_index} to force the clusters to be assigned
                    to the ML subsets as specified by the user.
            
            Returns
            ------
            list (of length equal to the number of columns of tasks_vs_clusters_array) of final cluster identifiers
                (integers, numbered from 1 to len(sizes)), mapping each unique initial cluster to its final cluster.
            Example: if sizes == [20, 10, 70], the output will be a list like [3, 3, 1, 2, 1, 3...], where
                '1' represents the final cluster of relative size 20, '2' the one of relative size 10, and '3' the 
                one of relative size 70.
            """

            # Calculate the fractions from sizes

            fractional_sizes = self.sizes / np.sum(self.sizes)

            S = len(self.sizes)

            # Normalise the self.df matrix
            tasks_vs_clusters_array = tasks_vs_clusters_array / tasks_vs_clusters_array.sum(axis = 1, keepdims = True)

            # Find the number of tasks + compounds (M) and the number of initial clusters (N)
            M, N = tasks_vs_clusters_array.shape
            if (S > N):
                errormessage = 'The requested number of new clusters to make ('+ str(S) + ') cannot be larger than the initial number of clusters (' + str(N) + '). Please review.'
                raise ValueError(errormessage)

            # Given matrix A (M x N) of fraction of self.df per cluster, assign each cluster to one of S final ML subsets,
            # so that the fraction of self.df per ML subset is closest to the corresponding fraction_size.
            # The weights on each ML subset (WML, S x 1) are calculated from fractional_sizes harmonic-mean-like.
            # The weights on each task (WT, M x 1) are calculated as requested by the user.
            # In the end: argmin SUM(ABS((A.X-T).WML).WT)
            # where X is the (N x S) binary solution matrix
            # where T is the (M x S) matrix of task fraction sizes (repeat of fractional_sizes)
            # constraint: assign one cluster to one and only one final ML subset
            # i.e. each row of X must sum to 1

            A = np.copy(tasks_vs_clusters_array)

            # Create WML
            sk_harmonic = (1 / fractional_sizes) / np.sum(1 / fractional_sizes)

            # Round all values to have only 3 decimals > reduce computational time
            A = np.round(A, 3)
            fractional_sizes = np.round(fractional_sizes, 3)
            obj_weights = np.round(self.obj_weights, 3)
            sk_harmonic = np.round(sk_harmonic, 3)     

            # Create the pulp model
            prob = LpProblem("Data_balancing", LpMinimize)

            # Create the pulp variables
            # x_names represent clusters, ML_subsets, and are binary variables
            x_names = ['x_'+str(i) for i in range(N * S)]
            x = [LpVariable(x_names[i], lowBound = 0, upBound = 1, cat = 'Integer') for i in range(N * S)]
            # X_names represent tasks, ML_subsets, and are continuous positive variables
            X_names = ['X_'+str(i) for i in range(M * S)]
            X = [LpVariable(X_names[i], lowBound = 0, cat = 'Continuous') for i in range(M * S)]

            # Add the objective to the model

            obj = []
            coeff = []
            for m in range(S):
                for t in range(M):
                    obj.append(X[m*M+t])
                    coeff.append(sk_harmonic[m] * obj_weights[t])

            prob += LpAffineExpression([(obj[i],coeff[i]) for i in range(len(obj)) ])

            # Add the constraints to the model

            # Constraints forcing each cluster to be in one and only one ML_subset
            for c in range(N):
                prob += LpAffineExpression([(x[c+m*N],+1) for m in range(S)]) == 1

            # If preassigned_clusters is provided, add the constraints to the model to force the clusters
            # to be assigned to the ML subset preassigned_clusters[t]
            if preassigned_clusters:
                for c, subset in preassigned_clusters.items():
                    # prob += LpAffineExpression(x[c+(subset)*N]) == 1
                    prob += x[c+(subset)*N] == 1

            # Constraints related to the ABS values handling, part 1 and 2
            for m in range(S):
                for t in range(M):
                    cs = [c for c in range(N) if A[t,c] != 0]
                    prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) - X[t] <= fractional_sizes[m]
                    prob += LpAffineExpression([(x[c+m*N],A[t,c]) for c in cs]) + X[t] >= fractional_sizes[m]

            # Solve the model
            prob.solve(PULP_CBC_CMD(gapAbs = self.absolute_gap, timeLimit = self.time_limit_seconds, threads = self.n_jobs, msg=False))

            # Extract the solution
            list_binary_solution = [value(x[i]) for i in range(N * S)]
            list_initial_cluster_indices = [(list(range(N)) * S)[i] for i,l in enumerate(list_binary_solution) if l == 1]
            list_final_ML_subsets = [(list((1 + np.repeat(range(S), N)).astype('int64')))[i] for i,l in enumerate(list_binary_solution) if l == 1]
            mapping = [x-1 for _, x in sorted(zip(list_initial_cluster_indices, list_final_ML_subsets))]

            return mapping 