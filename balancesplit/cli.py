import os
import argparse
from timeit import default_timer as timer
from .logs.config import enable_file_logger
from .splitter import BalanceSplit
from .clustering import RandomClustering, LeaderPickerClustering, MaxMinClustering, MurckoScaffoldClustering
from .data import split_dataset

def cli():

    # Parse command line arguments ###############################
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input file with the data in a pivoted csv/tsv format. \
                            A column with the SMILES must be provided and each target must be in a separate column.')
    parser.add_argument('-sc','--smiles_column', type=str, default='SMILES',
                        help='Name of the column with the SMILES')
    parser.add_argument('-tc','--target_columns', type=str, nargs='+', default=None,
                        help="Name of the columns with the targets. If not provided, \
                            all columns except the SMILES and --ignore_columns' columns will be used")
    parser.add_argument('-ic','--ignore_columns', type=str, nargs='+', default=None,
                        help='Name of the columns to ignore')
    parser.add_argument('-st', '--stratify', nargs='+', default='all', 
                        help='Columns to use for stratification. \
                            Options: all, none or list of column names')

    parser.add_argument('-ns', '--n_splits', type=int, default=None,
                         help='The k-folds to generate. If sizes, n_splits is ignored.')
    parser.add_argument('-nr', '--n_repeats', type=int, default=None,
                         help='Number of times to repeat the splitting process')
    parser.add_argument('-s', '--sizes', type=float, nargs='+', default=None,
                        help='Sizes of the subsets. Overrides --n_splits and --n_repeats')  
    parser.add_argument('-c','--clustering', type=str, default='dissimilarity_leader',
                        help='Clustering algorithm to use. \
                            Options: random, dissimilarity_leader, dissimilarity_maxmin or murcko')
    parser.add_argument('-ct','--cluster_threshold', type=float, default=0.7,
                        help='Minimum distance between cluster centers. Only used for dissimilarity_leader clustering.')
    parser.add_argument('-rs','--random_seed', type=int, default=42,
                        help='Seed for the random and dissimilarity_maxmin clustering')
    parser.add_argument('-cor', '--cluster_optimization_range', nargs=2, default=None,
                        help='Range of values to optimize the cluster size. Only used for \
                            dissimilarity_leader clustering (float) and dissimilarity_maxmin (int).')    
    
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file with the data with additional columns \
                            the assigned subset (and Minimum interset Tanimoto distance).')
    parser.add_argument('-t','--time_limit', type=int, default=60,
                        help='Time limit for linear combination of clusters in seconds')
    
    # Start the timer
    start_time = timer()
    
    # Parse arguments 
    args = parser.parse_args()

    if args.cluster_optimization_range is not None:
        args.cluster_optimization_range = tuple(map(float, args.cluster_optimization_range))

    if args.stratify[0] == 'all':
        args.stratify = True
    elif args.stratify[0] == 'none':
        args.stratify = False

    # Set output file ############################################
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + '_split.csv'


    # Enable logging #############################################
    logSettings = enable_file_logger(
        os.path.dirname(args.output),
        "optisplit.log",
        False,
        __name__,
        vars(args),
        disable_existing_loggers=False,
    )
    log = logSettings.log

    # Set clustering method #######################################
    if args.clustering == 'random':
        clustering = RandomClustering(seed=args.random_seed, )
    elif args.clustering == 'dissimilarity_leader':
        clustering = LeaderPickerClustering(similarity_threshold=args.cluster_threshold, cluster_optimization_range=args.cluster_optimization_range)
    elif args.clustering == 'dissimilarity_maxmin':
        clustering = MaxMinClustering(seed=args.random_seed, cluster_optimization_range=args.cluster_optimization_range )
    elif args.clustering == 'murcko':
        clustering = MurckoScaffoldClustering()
    else:
        raise ValueError('Clustering algorithm not recognized')

    # Set splitter ################################################
    splitter_kwargs = {
        'n_splits': args.n_splits,
        'n_repeats': args.n_repeats if args.n_repeats else 1,
        'clustering_method': clustering,
        'time_limit_seconds': args.time_limit,
        'stratify': args.stratify,
        'sizes': args.sizes,
    }
    splitter = BalanceSplit(**splitter_kwargs)
    
    # Split data #################################################
    split_dataset(
        data_path = args.input,
        smiles_col = args.smiles_column,
        task_cols = args.target_columns,
        ignore_cols = args.ignore_columns,
        splitter = splitter,
        output_path = args.output,
        compute_stats = True,
    )
    
    # Print elapsed time #########################################
    elapsed_time = timer() - start_time
    log.info('Elapsed time: {:.2f} seconds'.format(elapsed_time))

if __name__ == '__main__':
    cli()