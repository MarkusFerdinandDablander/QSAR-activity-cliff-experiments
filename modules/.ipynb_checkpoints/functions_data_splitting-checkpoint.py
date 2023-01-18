# import packages

# general tools
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import chain
import random

# RDkit
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold







# data splitting functions and data processing functions

# single molecule prediction

# functions to create all molecular pairs X_smiles_pairs and associated target variable y_pairs which indicates whether a pair has the same class

def create_smiles_pairs(x_smiles, y, shuffle_random_seed = 42):

    # define parameters and dictionaries
    n_smiles = len(x_smiles)
    y_smiles_to_activity_dict = dict(list(zip(x_smiles, y)))

    # preallocate data structures
    X_smiles_pairs = list(np.zeros(int(n_smiles*(n_smiles-1)/2)))

    # create all smiles pairs
    for i in range(n_smiles):
        for j in range(i+1, n_smiles):

            index = int((-i**2)/2 + i*(n_smiles - 3/2) +j - 1)

            X_smiles_pairs[index] = [x_smiles[i], x_smiles[j]]

    # label created smiles pairs
    y_pairs = np.array([int(y_smiles_to_activity_dict[smiles_1]!=y_smiles_to_activity_dict[smiles_2]) for [smiles_1, smiles_2] in X_smiles_pairs])

    # convert to numpy array
    X_smiles_pairs = np.array(X_smiles_pairs)

    # randomly shuffle pairs and labels
    np.random.seed(shuffle_random_seed)
    random_indices = np.arange(int(n_smiles*(n_smiles-1)/2))
    np.random.shuffle(random_indices)
    X_smiles_pairs = X_smiles_pairs[random_indices]
    y_pairs = y_pairs[random_indices]

    return (X_smiles_pairs, y_pairs)


# functions for k-fold cross validation data splitting (general)

def k_fold_cross_validation_stratified_random_split(x,
                                                    y,
                                                    k_splits = 5,
                                                    shuffle = True,
                                                    random_state_shuffling = 42,
                                                    disjoint_molecular_spaces = False):

    """random stratified k-fold cross validation"""

    # create index set pairs
    SKF = StratifiedKFold(n_splits = k_splits, shuffle = shuffle, random_state = random_state_shuffling)
    k_fold_cross_validation_index_set_pairs = []

    for (ind_train, ind_test) in SKF.split(x, y):
        k_fold_cross_validation_index_set_pairs.append((ind_train, ind_test))

    if disjoint_molecular_spaces == True:

        for (i, (ind_train, ind_test)) in enumerate(k_fold_cross_validation_index_set_pairs):

            train_space = set(x[ind_train].flatten())
            ind_test_delete = []

            for j in ind_test:

                smiles_1 = x[j][0]
                smiles_2 = x[j][1]

                if len(set([smiles_1, smiles_2]).intersection(train_space)) > 0:
                    ind_test_delete.append(j)

            ind_test_disjoint = [j for j in ind_test if j not in ind_test_delete]

            k_fold_cross_validation_index_set_pairs[i] = (ind_train, ind_test_disjoint)

    return k_fold_cross_validation_index_set_pairs


# functions for train/val/test data splitting (general)

def train_val_test_stratified_random_split(x,
                                           y,
                                           splitting_ratios = (0.8, 0.1, 0.1),
                                           shuffle = True,
                                           random_state_shuffling = 42,
                                           disjoint_molecular_spaces = False):

    """Split data into train/val/test sets in a random, stratified way."""

    x_indices = np.arange(0, len(x))
    (frac_train, frac_val, frac_test) = splitting_ratios
    (frac_val_norm, frac_test_norm) = (frac_val/(frac_val + frac_test), frac_test/(frac_val + frac_test))
    (ind_train, ind_val_and_test, ind_val, ind_test) = ([],[],[],[])

    (ind_train, ind_val_and_test, y_train, y_val_and_test) = train_test_split(x_indices,
                                                                              y,
                                                                              train_size = frac_train,
                                                                              test_size = frac_val + frac_test,
                                                                              shuffle = True,
                                                                              random_state = random_state_shuffling,
                                                                              stratify = y)

    if frac_val > 0:

        (ind_val, ind_test, y_val, y_test) = train_test_split(ind_val_and_test,
                                                              y_val_and_test,
                                                              train_size = frac_val_norm,
                                                              test_size = frac_test_norm,
                                                              shuffle = True,
                                                              random_state = random_state_shuffling,
                                                              stratify = y_val_and_test)

    else:

        ind_test = ind_val_and_test

    if disjoint_molecular_spaces == True:

        train_space = set(x[ind_train].flatten())

        ind_val_delete = []

        for j in ind_val:

            smiles_1 = x[j][0]
            smiles_2 = x[j][1]

            if len(set([smiles_1, smiles_2]).intersection(train_space)) > 0:
                ind_val_delete.append(j)

        ind_val = [j for j in ind_val if j not in ind_val_delete]

        ind_test_delete = []

        for j in ind_test:

            smiles_1 = x[j][0]
            smiles_2 = x[j][1]

            if len(set([smiles_1, smiles_2]).intersection(train_space)) > 0:
                ind_test_delete.append(j)

        ind_test = [j for j in ind_test if j not in ind_test_delete]

    return (ind_train, ind_val, ind_test)


# functions for k-fold cross validation data splitting (single molecule prediction)

def get_ordered_scaffold_sets(x_smiles, scaffold_func = "Bemis_Murcko_generic"):

    """ This function was taken from https://lifesci.dgl.ai/_modules/dgllife/utils/splitters.html
    and then modified by Markus Ferdinand Dablander, DPhil student at University of Oxford.

    Group molecules based on their Bemis-Murcko scaffolds and
    order these groups based on their sizes.

    The order is decided by comparing the size of groups, where groups with a larger size
    are placed before the ones with a smaller size.

    Parameters
    ----------
    x_smiles : list or 1d np.array of of smiles strings corresponding to molecules which
        will be converted to rdkit mol objects.
    scaffold_func : str
        The function to use for computing Bemis-Murcko scaffolds. If scaffold_func = "Bemis_Murcko_generic",
        then we use first rdkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol and then apply
        rdkit.Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric to the result. The result is a
        scaffold which ignores atom types and bond orders.
        If scaffold_func = "Bemis_Murcko_atom_bond_sensitive", we only use
        dkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol and get scaffolds
        which are sensitive to atom types and bond orders.

    Returns
    -------
    scaffold_sets : list
        Each element of the list is a list of int,
        representing the indices of compounds with a same scaffold.
    """

    assert scaffold_func in ['Bemis_Murcko_generic', 'Bemis_Murcko_atom_bond_sensitive'], \
        "Expect scaffold_func to be 'Bemis_Murcko_generic' or 'Bemis_Murcko_atom_bond_sensitive', " \
        "got '{}'".format(scaffold_func)

    x_smiles = list(x_smiles)
    molecules = [Chem.MolFromSmiles(smiles) for smiles in x_smiles]
    scaffolds = defaultdict(list)

    for i, mol in enumerate(molecules):

        # for mols that have not been sanitized, we need to compute their ring information
        try:
            Chem.rdmolops.FastFindRings(mol)
            if scaffold_func == 'Bemis_Murcko_generic':
                mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                mol_scaffold_generic = MurckoScaffold.MakeScaffoldGeneric(mol_scaffold)
                smiles_scaffold = Chem.CanonSmiles(Chem.MolToSmiles(mol_scaffold_generic))
            if scaffold_func == 'Bemis_Murcko_atom_bond_sensitive':
                mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                smiles_scaffold = Chem.CanonSmiles(Chem.MolToSmiles(mol_scaffold))
            # Group molecules that have the same scaffold
            scaffolds[smiles_scaffold].append(i)
        except:
            print('Failed to compute the scaffold for molecule {:d} '
                  'and it will be excluded.'.format(i + 1))

    # order groups of molecules by first comparing the size of groups
    # and then the index of the first compound in the group.
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

    return scaffold_sets


def number_of_scaffolds(x_smiles, scaffold_func = "Bemis_Murcko_generic"):

    """Get number of distinct scaffolds in the molecular data set x_smiles"""

    scaffold_sets = get_ordered_scaffold_sets(x_smiles = x_smiles, scaffold_func = scaffold_func)
    n_scaffolds = len(scaffold_sets)

    return n_scaffolds


def k_fold_cross_validation_scaffold_split(x_smiles,
                                           k_splits = 5,
                                           scaffold_func = 'Bemis_Murcko_generic'):

    """This function was taken from https://lifesci.dgl.ai/_modules/dgllife/utils/splitters.html
    and then modified by Markus Ferdinand Dablander, DPhil student at University of Oxford.

    Group molecules based on their scaffolds and sort groups based on their sizes.
    The groups are then split for k-fold cross validation.

    Same as usual k-fold splitting methods, each molecule will appear only once
    in the validation set among all folds. In addition, this method ensures that
    molecules with a same scaffold will be collectively in either the training
    set or the validation set for each fold.

    Note that the folds can be highly imbalanced depending on the
    scaffold distribution in the dataset.

    Parameters
    ----------
    x_smiles
        List of smiles strings for molecules which are to be transformed to rdkit mol objects.
    k_splits : int
        Number of folds to use and should be no smaller than 2. Default to be 5.
    scaffold_func : str
        The function to use for computing Bemis-Murcko scaffolds. If scaffold_func = "Bemis_Murcko_generic",
        then we use first rdkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol and then apply
        rdkit.Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric to the result. The result is a
        scaffold which ignores atom types and bond orders.
        If scaffold_func = "Bemis_Murcko_atom_bond_sensitive", we only use
        dkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol and get scaffolds
        which are sensitive to atom types and bond orders.

    Returns
    -------
    list of 2-tuples
        Each element of the list represents a fold and is a 2-tuple (ind_train, ind_test) which represent indices
        for training/testing for each fold.
    """

    assert k_splits >= 2, 'Expect the number of folds to be no smaller than 2, got {:d}'.format(k_splits)
    x_smiles = list(x_smiles)

    scaffold_sets = get_ordered_scaffold_sets(x_smiles, scaffold_func = scaffold_func)

    # k_splits buckets (i.e. chemical compound clusters) that form a relatively balanced partition of the dataset
    index_buckets = [[] for _ in range(k_splits)]
    for group_indices in scaffold_sets:
        bucket_chosen = int(np.argmin([len(bucket) for bucket in index_buckets]))
        index_buckets[bucket_chosen].extend(group_indices)

    k_fold_cross_validation_index_set_pairs = []
    for i in range(k_splits):
        ind_train = list(chain.from_iterable(index_buckets[:i] + index_buckets[i + 1:]))
        ind_test = index_buckets[i]
        k_fold_cross_validation_index_set_pairs.append((ind_train, ind_test))

    return k_fold_cross_validation_index_set_pairs


def k_fold_cross_validation_split_contents(x_smiles,
                                           y,
                                           k_fold_cross_validation_index_set_pairs,
                                           scaffold_contents = True,
                                           scaffold_func = "Bemis_Murcko_generic"):

    """See how large each train- and test set is and how many negatives, positives and scaffolds it contains."""

    k_splits = len(k_fold_cross_validation_index_set_pairs)

    if scaffold_contents == True:

        columns = ["Elements",
                   "Scaffolds",
                   "Training: Elements",
                   "Training: Negatives",
                   "Training: Positives",
                   "Training: Scaffolds",
                   "Testing: Elements",
                   "Testing: Negatives",
                   "Testing: Positives",
                   "Testing: Scaffolds"]
    else:

        columns = ["Elements",
                   "Training: Elements",
                   "Training: Negatives",
                   "Training: Positives",
                   "Testing: Elements",
                   "Testing: Negatives",
                   "Testing: Positives"]


    splits_data = np.zeros((k_splits, len(columns)), dtype = np.int)

    for j in range(k_splits):

        if scaffold_contents == True:

            (ind_train, ind_test)= k_fold_cross_validation_index_set_pairs[j]

            x_smiles_train = x_smiles[ind_train]
            x_smiles_test = x_smiles[ind_test]

            y_train = y[ind_train]
            y_test = y[ind_test]

            splits_data[j,0] = len(y)
            splits_data[j,1] = number_of_scaffolds(x_smiles, scaffold_func = scaffold_func)

            splits_data[j,2] = len(y_train)
            splits_data[j,3] = list(y_train).count(0)
            splits_data[j,4] = list(y_train).count(1)
            splits_data[j,5] = number_of_scaffolds(x_smiles_train, scaffold_func = scaffold_func)

            splits_data[j,6] = len(y_test)
            splits_data[j,7] = list(y_test).count(0)
            splits_data[j,8] = list(y_test).count(1)
            splits_data[j,9] = number_of_scaffolds(x_smiles_test, scaffold_func = scaffold_func)

        else:

            (ind_train, ind_test)= k_fold_cross_validation_index_set_pairs[j]

            y_train = y[ind_train]
            y_test = y[ind_test]

            splits_data[j,0] = len(y)

            splits_data[j,1] = len(y_train)
            splits_data[j,2] = list(y_train).count(0)
            splits_data[j,3] = list(y_train).count(1)

            splits_data[j,4] = len(y_test)
            splits_data[j,5] = list(y_test).count(0)
            splits_data[j,6] = list(y_test).count(1)

    splits_df = pd.DataFrame(data = splits_data, index = list(range(1, k_splits + 1)), columns = columns)

    return splits_df



# functions for train/val/test data splitting (single molecule prediction)

def train_val_test_scaffold_split(x_smiles,
                                  splitting_ratios = (0.8, 0.1, 0.1),
                                  scaffold_func = "Bemis_Murcko_generic"):

    """Split data into train/val/test sets according to Bemis-Murcko scaffolds."""

    n_molecules = len(x_smiles)

    scaffold_sets = get_ordered_scaffold_sets(x_smiles, scaffold_func = scaffold_func)

    frac_train, frac_val, frac_test = splitting_ratios
    (ind_train, ind_val, ind_test) = ([], [], [])

    train_cutoff = int(frac_train * n_molecules)
    val_cutoff = int((frac_train + frac_val) * n_molecules)

    for group_indices in scaffold_sets:

        if len(ind_train) + len(group_indices) > train_cutoff:

            if len(ind_train) + len(ind_val) + len(group_indices) > val_cutoff:
                ind_test.extend(group_indices)
            else:
                ind_val.extend(group_indices)

        else:
            ind_train.extend(group_indices)


    return (ind_train, ind_val, ind_test)



def train_val_test_split_contents(x_smiles,
                                  y,
                                  ind_train,
                                  ind_val,
                                  ind_test,
                                  scaffold_contents = True,
                                  scaffold_func = "Bemis_Murcko_generic"):

    """See how large train/val/test sets are and how many negatives, positives and scaffolds they contain."""

    if scaffold_contents == True:
        columns = ["Elements", "Scaffolds", "Negatives", "Positives"]
    else:
        columns = ["Elements", "Negatives", "Positives"]


    index = ["All", "Train", "Val", "Test"]

    splits_data = np.zeros((4, len(columns)), dtype = np.int)

    if scaffold_contents == True:

        x_smiles_train = x_smiles[ind_train]
        x_smiles_val = x_smiles[ind_val]
        x_smiles_test = x_smiles[ind_test]

        y_train = y[ind_train]
        y_val = y[ind_val]
        y_test = y[ind_test]

        # data for whole data set
        splits_data[0,0] = len(y)
        splits_data[0,1] = number_of_scaffolds(x_smiles, scaffold_func = scaffold_func)
        splits_data[0,2] = list(y).count(0)
        splits_data[0,3] = list(y).count(1)

        # data for training set
        splits_data[1,0] = len(y_train)
        splits_data[1,1] = number_of_scaffolds(x_smiles_train, scaffold_func = scaffold_func)
        splits_data[1,2] = list(y_train).count(0)
        splits_data[1,3] = list(y_train).count(1)

        # data for validation set
        splits_data[2,0] = len(y_val)
        splits_data[2,1] = number_of_scaffolds(x_smiles_val, scaffold_func = scaffold_func)
        splits_data[2,2] = list(y_val).count(0)
        splits_data[2,3] = list(y_val).count(1)

        # data for test set
        splits_data[3,0] = len(y_test)
        splits_data[3,1] = number_of_scaffolds(x_smiles_test, scaffold_func = scaffold_func)
        splits_data[3,2] = list(y_test).count(0)
        splits_data[3,3] = list(y_test).count(1)

        splits_df = pd.DataFrame(data = splits_data, index = index, columns = columns)

    else:

        y_train = y[ind_train]
        y_val = y[ind_val]
        y_test = y[ind_test]

        # data for whole data set
        splits_data[0,0] = len(y)
        splits_data[0,1] = list(y).count(0)
        splits_data[0,2] = list(y).count(1)

        # data for training set
        splits_data[1,0] = len(y_train)
        splits_data[1,1] = list(y_train).count(0)
        splits_data[1,2] = list(y_train).count(1)

        # data for validation set
        splits_data[2,0] = len(y_val)
        splits_data[2,1] = list(y_val).count(0)
        splits_data[2,2] = list(y_val).count(1)

        # data for test set
        splits_data[3,0] = len(y_test)
        splits_data[3,1] = list(y_test).count(0)
        splits_data[3,2] = list(y_test).count(1)

        splits_df = pd.DataFrame(data = splits_data, index = index, columns = columns)

    return splits_df






















# MMPs

# functions for k-fold cross validation data splitting (mmp prediction)

def get_ordered_mmp_core_index_sets(x_smiles_mmp_cores):

    """
    Group matched molecular pairs (mmps) based on their mmp cores and
    order these groups based on their sizes.

    The order is decided by comparing the size of groups, where groups with a larger size
    are placed before the ones with a smaller size.

    Parameters
    ----------
    x_smiles_mmp_cores : 1d np.array of smiles strings corresponding to mmp cores of all mmps
    in X_smiles_mmps.

    Returns
    -------
    mmp_core_index_sets : list
        Each element of the list is a list of int,
        representing the indices of compounds with a same mmp core.
    """

    x_smiles_mmp_cores = list(x_smiles_mmp_cores)
    mmp_cores = defaultdict(list)

    for (i, mmp_core) in enumerate(x_smiles_mmp_cores):
        # Group molecules that have the same scaffold
        mmp_cores[mmp_core].append(i)

    # Order groups of mmp indices by first comparing the size of groups
    # and then the index of the first mmp in the group.
    mmp_cores = {key: sorted(value) for (key, value) in mmp_cores.items()}
    mmp_core_index_sets = [
        mmp_core_index_set for (mmp_core, mmp_core_index_set) in sorted(
            mmp_cores.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

    return mmp_core_index_sets


def number_of_mmp_cores(x_smiles_mmp_cores):

    """Get number of distinct mmp_cores in the mmp data set X_smiles_mmps"""

    mmp_core_index_sets = get_ordered_mmp_core_index_sets(x_smiles_mmp_cores)
    n_mmp_cores = len(mmp_core_index_sets)

    return n_mmp_cores


def k_fold_cross_validation_mmp_core_split(X_smiles_mmps,
                                           x_smiles_mmp_cores,
                                           k_splits = 5,
                                           disjoint_molecular_spaces = False):

    """
    Group mmps based on their mmp cores and sort groups based on their sizes.
    The groups are then split for k-fold cross validation.

    Same as usual k-fold splitting methods, each mmp will appear only once
    in the test set among all folds. In addition, this method ensures that mmps
    with a same mmp core will be collectively in either the training
    set or the test set for each fold.

    Note that the folds can be highly imbalanced depending on the
    mmp core distribution in the dataset.

    Parameters
    ----------
    x_smiles_mmp_cores
        1d np.array of smiles strings corresponding to mmp cores of all mmps
        in X_smiles_mmps.
    k_splits : int
        Number of folds to use and should be no smaller than 2. Default to be 5.
    disjoint_molecular_spaces
        Decides whether training molecular space and testing molecular space are to be disjoint.

    Returns
    -------
    list of 2-tuples of index sets
        Each element of the list represents a fold and is a 2-tuple (ind_train, ind_test) which represent indices
        for training/testing for each fold.
    """

    assert k_splits >= 2, 'Expect the number of folds to be no smaller than 2, got {:d}'.format(k_splits)
    x_smiles_mmp_cores = list(x_smiles_mmp_cores)

    mmp_core_index_sets = get_ordered_mmp_core_index_sets(x_smiles_mmp_cores)

    # k_splits buckets that form a relatively balanced partition of the dataset
    index_buckets = [[] for _ in range(k_splits)]
    for group_indices in mmp_core_index_sets:
        bucket_chosen = int(np.argmin([len(bucket) for bucket in index_buckets]))
        index_buckets[bucket_chosen].extend(group_indices)

    k_fold_cross_validation_index_set_pairs = []

    for i in range(k_splits):
        ind_train = list(chain.from_iterable(index_buckets[:i] + index_buckets[i + 1:]))
        ind_test = index_buckets[i]
        k_fold_cross_validation_index_set_pairs.append((ind_train, ind_test))

    if disjoint_molecular_spaces == True:

        for (i, (ind_train, ind_test)) in enumerate(k_fold_cross_validation_index_set_pairs):

            train_space = set(X_smiles_mmps[ind_train].flatten())
            ind_test_delete = []

            for j in ind_test:

                smiles_1 = X_smiles_mmps[j][0]
                smiles_2 = X_smiles_mmps[j][1]

                if len(set([smiles_1, smiles_2]).intersection(train_space)) > 0:
                    ind_test_delete.append(j)

            ind_test_disjoint = [j for j in ind_test if j not in ind_test_delete]

            k_fold_cross_validation_index_set_pairs[i] = (ind_train, ind_test_disjoint)

    return k_fold_cross_validation_index_set_pairs

def k_fold_cross_validation_mmp_random_space_split(X_smiles_mmps,
                                                   k_splits = 5,
                                                   shuffle = True,
                                                   random_state_shuffling = 42,
                                                   disjoint_molecular_spaces = True):

    k_fold_cross_validation_index_set_pairs = []
    x_smiles = np.array(list(set(X_smiles_mmps.flatten())))

    k_fold_cross_validation_index_set_pairs_sm = k_fold_cross_validation_stratified_random_split(x = x_smiles,
                                                                                                 y = np.ones(len(x_smiles)),
                                                                                                 k_splits = k_splits,
                                                                                                 shuffle = shuffle,
                                                                                                 random_state_shuffling = random_state_shuffling)
    
    for (ind_train_sm, ind_test_sm) in k_fold_cross_validation_index_set_pairs_sm:

        mol_space_train = set(x_smiles[ind_train_sm])
        mol_space_test = set(x_smiles[ind_test_sm])
        
        ind_train = []
        ind_test = []

        for (j, mmp) in enumerate(X_smiles_mmps):

            if set(mmp).issubset(mol_space_train):
                ind_train.append(j)

            if set(mmp).issubset(mol_space_test):
                ind_test.append(j)

            if disjoint_molecular_spaces == False:

                if len(set(mmp).intersection(mol_space_train)) == 1 and len(set(mmp).intersection(mol_space_test)) == 1:
                    ind_test.append(j)

        k_fold_cross_validation_index_set_pairs.append((ind_train, ind_test))

    return k_fold_cross_validation_index_set_pairs


def k_fold_cross_validation_mmp_scaffold_space_split(X_smiles_mmps,
                                                     k_splits = 5,
                                                     scaffold_func = "Bemis_Murcko_generic",
                                                     disjoint_molecular_spaces = False):

    k_fold_cross_validation_index_set_pairs = []
    x_smiles = np.array(list(set(X_smiles_mmps.flatten())))

    k_fold_cross_validation_index_set_pairs_sm = k_fold_cross_validation_scaffold_split(x_smiles = x_smiles,
                                                                                        k_splits = k_splits,
                                                                                        scaffold_func = scaffold_func)

    for (ind_train_sm, ind_test_sm) in k_fold_cross_validation_index_set_pairs_sm:

        mol_space_train = set(x_smiles[ind_train_sm])
        mol_space_test = set(x_smiles[ind_test_sm])

        ind_train = []
        ind_test = []

        for (j, mmp) in enumerate(X_smiles_mmps):

            if set(mmp).issubset(mol_space_train):
                ind_train.append(j)

            if set(mmp).issubset(mol_space_test):
                ind_test.append(j)

            if disjoint_molecular_spaces == False:

                if len(set(mmp).intersection(mol_space_train)) == 1 and len(set(mmp).intersection(mol_space_test)) == 1:
                    ind_test.append(j)

        k_fold_cross_validation_index_set_pairs.append((ind_train, ind_test))

    return k_fold_cross_validation_index_set_pairs


def k_fold_cross_validation_split_contents_for_mmps(x_smiles_mmp_cores,
                                                    y_mmps,
                                                    k_fold_cross_validation_index_set_pairs):

    """For MMPs: See how large each train- and test set is and how many negatives, positives and mmp cores it contains."""

    k_splits = len(k_fold_cross_validation_index_set_pairs)

    columns = ["Elements",
               "MMP Cores",
               "Training: Elements",
               "Training: Negatives",
               "Training: Positives",
               "Training: MMP Cores",
               "Testing: Elements",
               "Testing: Negatives",
               "Testing: Positives",
               "Testing: MMP Cores"]

    splits_data = np.zeros((k_splits, len(columns)), dtype = np.int)

    for j in range(k_splits):

        (ind_train, ind_test) = k_fold_cross_validation_index_set_pairs[j]

        x_smiles_mmp_cores_train = list(x_smiles_mmp_cores[ind_train])
        x_smiles_mmp_cores_test = list(x_smiles_mmp_cores[ind_test])
        x_smiles_mmp_cores_full = x_smiles_mmp_cores_train + x_smiles_mmp_cores_test

        y_mmps_train = list(y_mmps[ind_train])
        y_mmps_test = list(y_mmps[ind_test])
        y_mmps_full = y_mmps_train + y_mmps_test

        splits_data[j,0] = len(y_mmps_full)
        splits_data[j,1] = number_of_mmp_cores(x_smiles_mmp_cores_full)

        splits_data[j,2] = len(y_mmps_train)
        splits_data[j,3] = y_mmps_train.count(0)
        splits_data[j,4] = y_mmps_train.count(1)
        splits_data[j,5] = number_of_mmp_cores(x_smiles_mmp_cores_train)

        splits_data[j,6] = len(y_mmps_test)
        splits_data[j,7] = y_mmps_test.count(0)
        splits_data[j,8] = y_mmps_test.count(1)
        splits_data[j,9] = number_of_mmp_cores(x_smiles_mmp_cores_test)

    splits_df = pd.DataFrame(data = splits_data, index = list(range(1, k_splits + 1)), columns = columns)

    return splits_df


# functions for train/val/test data splitting (mmp prediction)

def train_val_test_mmp_core_split(X_smiles_mmps,
                                  x_smiles_mmp_cores,
                                  splitting_ratios = (0.8, 0.1, 0.1),
                                  disjoint_molecular_spaces = True):

    """Split data into train/val/test sets according to mmp cores."""

    n_mmps = len(x_smiles_mmp_cores)

    mmp_core_index_sets = get_ordered_mmp_core_index_sets(x_smiles_mmp_cores)

    frac_train, frac_val, frac_test = splitting_ratios
    (ind_train, ind_val, ind_test) = ([], [], [])

    train_cutoff = int(frac_train * n_mmps)
    val_cutoff = int((frac_train + frac_val) * n_mmps)

    for group_indices in mmp_core_index_sets:

        if len(ind_train) + len(group_indices) > train_cutoff:

            if len(ind_train) + len(ind_val) + len(group_indices) > val_cutoff:
                ind_test.extend(group_indices)
            else:
                ind_val.extend(group_indices)

        else:
            ind_train.extend(group_indices)

    if disjoint_molecular_spaces == True:

        train_space = set(X_smiles_mmps[ind_train].flatten())

        ind_val_delete = []

        for j in ind_val:

            smiles_1 = X_smiles_mmps[j][0]
            smiles_2 = X_smiles_mmps[j][1]

            if len(set([smiles_1, smiles_2]).intersection(train_space)) > 0:
                ind_val_delete.append(j)

        ind_val = [j for j in ind_val if j not in ind_val_delete]

        ind_test_delete = []

        for j in ind_test:

            smiles_1 = X_smiles_mmps[j][0]
            smiles_2 = X_smiles_mmps[j][1]

            if len(set([smiles_1, smiles_2]).intersection(train_space)) > 0:
                ind_test_delete.append(j)

        ind_test = [j for j in ind_test if j not in ind_test_delete]

    return (ind_train, ind_val, ind_test)

def train_val_test_mmp_random_space_split(X_smiles_mmps,
                                        mol_space_splitting_ratios = (0.8, 0, 0.2),
                                        shuffle = True,
                                        random_state_shuffling = 42,
                                        disjoint_molecular_spaces = True):

    x_smiles = np.array(list(set(X_smiles_mmps.flatten())))

    (ind_train_sm, ind_val_sm, ind_test_sm) = train_val_test_stratified_random_split(x = x_smiles,
                                                                                     y = np.ones(len(x_smiles)),
                                                                                     splitting_ratios = mol_space_splitting_ratios,
                                                                                     shuffle = shuffle,
                                                                                     random_state_shuffling = random_state_shuffling)
    

    mol_space_train = set(x_smiles[ind_train_sm])
    mol_space_val = set(x_smiles[ind_val_sm])
    mol_space_test = set(x_smiles[ind_test_sm])

    ind_train = []
    ind_val = []
    ind_test = []

    for (j, mmp) in enumerate(X_smiles_mmps):

        if set(mmp).issubset(mol_space_train):
            ind_train.append(j)

        if set(mmp).issubset(mol_space_val):
            ind_val.append(j)

        if set(mmp).issubset(mol_space_test):
            ind_test.append(j)

        if disjoint_molecular_spaces == False:

            if len(set(mmp).intersection(mol_space_train)) == 1 and len(set(mmp).intersection(mol_space_val)) == 1:
                ind_val.append(j)

            if len(set(mmp).intersection(mol_space_train)) == 1 and len(set(mmp).intersection(mol_space_test)) == 1:
                ind_test.append(j)

    return (ind_train, ind_val, ind_test)


def train_val_test_mmp_scaffold_space_split(X_smiles_mmps,
                                            mol_space_splitting_ratios = (0.8, 0, 0.2),
                                            scaffold_func = "Bemis_Murcko_generic",
                                            disjoint_molecular_spaces = False):

    x_smiles = np.array(list(set(X_smiles_mmps.flatten())))

    (ind_train_sm, ind_val_sm, ind_test_sm) = train_val_test_scaffold_split(x_smiles,
                                                                            splitting_ratios = mol_space_splitting_ratios,
                                                                            scaffold_func = scaffold_func)

    mol_space_train = set(x_smiles[ind_train_sm])
    mol_space_val = set(x_smiles[ind_val_sm])
    mol_space_test = set(x_smiles[ind_test_sm])

    ind_train = []
    ind_val = []
    ind_test = []

    for (j, mmp) in enumerate(X_smiles_mmps):

        if set(mmp).issubset(mol_space_train):
            ind_train.append(j)

        if set(mmp).issubset(mol_space_val):
            ind_val.append(j)

        if set(mmp).issubset(mol_space_test):
            ind_test.append(j)

        if disjoint_molecular_spaces == False:

            if len(set(mmp).intersection(mol_space_train)) == 1 and len(set(mmp).intersection(mol_space_val)) == 1:
                ind_val.append(j)

            if len(set(mmp).intersection(mol_space_train)) == 1 and len(set(mmp).intersection(mol_space_test)) == 1:
                ind_test.append(j)

    return (ind_train, ind_val, ind_test)


def train_val_test_split_contents_for_mmps(x_smiles_mmp_cores,
                                           y_mmps,
                                           ind_train,
                                           ind_val,
                                           ind_test):

    """See how large train/val/test sets are and how many negatives, positives and mmp cores they contain."""

    columns = ["Elements", "MMP Cores", "Negatives", "Positives"]

    index = ["All", "Train", "Val", "Test"]

    splits_data = np.zeros((4, len(columns)), dtype = np.int)

    x_smiles_mmp_cores_train = list(x_smiles_mmp_cores[ind_train])
    x_smiles_mmp_cores_val = list(x_smiles_mmp_cores[ind_val])
    x_smiles_mmp_cores_test = list(x_smiles_mmp_cores[ind_test])
    x_smiles_mmp_cores_full = x_smiles_mmp_cores_train + x_smiles_mmp_cores_val + x_smiles_mmp_cores_test

    y_mmps_train = list(y_mmps[ind_train])
    y_mmps_val = list(y_mmps[ind_val])
    y_mmps_test = list(y_mmps[ind_test])
    y_mmps_full = y_mmps_train + y_mmps_val + y_mmps_test

    # data for whole data set
    splits_data[0,0] = len(y_mmps_full)
    splits_data[0,1] = number_of_mmp_cores(x_smiles_mmp_cores_full)
    splits_data[0,2] = y_mmps_full.count(0)
    splits_data[0,3] = y_mmps_full.count(1)

    # data for training set
    splits_data[1,0] = len(ind_train)
    splits_data[1,1] = number_of_mmp_cores(x_smiles_mmp_cores_train)
    splits_data[1,2] = y_mmps_train.count(0)
    splits_data[1,3] = y_mmps_train.count(1)

    # data for validation set
    splits_data[2,0] = len(ind_val)
    splits_data[2,1] = number_of_mmp_cores(x_smiles_mmp_cores_val)
    splits_data[2,2] = y_mmps_val.count(0)
    splits_data[2,3] = y_mmps_val.count(1)

    # data for test set
    splits_data[3,0] = len(ind_test)
    splits_data[3,1] = number_of_mmp_cores(x_smiles_mmp_cores_test)
    splits_data[3,2] = y_mmps_test.count(0)
    splits_data[3,3] = y_mmps_test.count(1)

    splits_df = pd.DataFrame(data = splits_data, index = index, columns = columns)

    return splits_df




# function to assign each mmp its largest core, i.e. to remove all mmp duplicates but only leave the ones with maximal cores

def remove_mmp_duplicates(dataframe_mmps, maximal_cores = True):
    
    inds_sorted = np.array([sorted([a,b]) for (a,b) in list(zip(dataframe_mmps.values[:,2], dataframe_mmps.values[:,3]))])
    dataframe_mmps["ind_1_sorted"] = inds_sorted[:,0]
    dataframe_mmps["ind_2_sorted"] = inds_sorted[:,1]
    
    if maximal_cores == False: # randomly choose an MMP, keep it and delete all its duplicates
        
        # randomly shuffle rows in dataframe
        dataframe_mmps = dataframe_mmps.sample(frac = 1, random_state = 42)

        # delete duplicates (keeps first occurence of a duplicate)
        dataframe_mmps = dataframe_mmps.drop_duplicates(subset = ["ind_1_sorted", "ind_2_sorted"], keep = "first")
        
        dataframe_mmps = dataframe_mmps.reset_index(drop = True)
        
        return dataframe_mmps
    
    else: # delete all mmp duplicates except the one with the maximal core
        
        dict_inds_to_core_size_and_row_number = {}
        
        for row_number, row in dataframe_mmps.iterrows():
            
            ind_1_sorted = row["ind_1_sorted"]
            ind_2_sorted = row["ind_2_sorted"]
            core_smiles = row["constant_part"]
            
            core_mol = Chem.MolFromSmiles(core_smiles)
            core_size = core_mol.GetNumHeavyAtoms()
            
            if (ind_1_sorted, ind_2_sorted) not in dict_inds_to_core_size_and_row_number:
                
                dict_inds_to_core_size_and_row_number[(ind_1_sorted, ind_2_sorted)] = (row_number, core_size)
            
            else:
                
                (old_row_number, old_core_size) = dict_inds_to_core_size_and_row_number[(ind_1_sorted, ind_2_sorted)]
                
                if core_size > old_core_size:
                    
                    dict_inds_to_core_size_and_row_number[(ind_1_sorted, ind_2_sorted)] = (row_number, core_size)
        
                    
        filtered_indices = np.array(list(dict_inds_to_core_size_and_row_number.values()))[:,0].astype(int)
        dataframe_mmps = dataframe_mmps.iloc[filtered_indices]
        dataframe_mmps["constant_part_size"] = np.array([dict_inds_to_core_size_and_row_number[(ind_1_sorted, ind_2_sorted)][1] 
                                                         for [ind_1_sorted, ind_2_sorted] in dataframe_mmps[["ind_1_sorted", "ind_2_sorted"]].values.tolist()])
        
        dataframe_mmps = dataframe_mmps.reset_index(drop = True)
        
        return dataframe_mmps
    
    
    
# define functions to map data splits for molecular spaces to data splits for mmp spaces
    
def k_fold_cross_validation_map_mols_to_mmps(x_smiles,
                                             X_smiles_mmps,
                                             k_fold_cross_validation_index_set_pairs_mols):
        """
        Output: (ind_train_mmps, ind_test_mmps, ind_test_one_out_mmps, ind_test_two_out_mmps)
        """
        
        k_fold_cross_validation_index_set_quadruples_mmps = []
        
        for (ind_train_mols, ind_test_mols) in k_fold_cross_validation_index_set_pairs_mols:
            
            train_space_mols = set(x_smiles[ind_train_mols])
            test_space_mols = set(x_smiles[ind_test_mols])
            
            ind_train_mmps = []
            ind_test_mmps = []
            ind_test_one_out_mmps = []
            ind_test_two_out_mmps = []
            
            for (k, mmp) in enumerate(X_smiles_mmps):
                
                if set(mmp).issubset(train_space_mols):
                    ind_train_mmps.append(k)
                
                elif len(set(mmp).intersection(train_space_mols)) == 1 and len(set(mmp).intersection(test_space_mols)) == 1:
                    ind_test_one_out_mmps.append(k)
                    
                elif set(mmp).issubset(test_space_mols):
                    ind_test_two_out_mmps.append(k)
                    
                else:
                    print("Error: some MMP is neither in train_space_mols, nor in test_space_mols, nor in between.")
            
            ind_test_mmps = ind_test_one_out_mmps + ind_test_two_out_mmps
                    
            k_fold_cross_validation_index_set_quadruples_mmps.append((ind_train_mmps, ind_test_mmps, ind_test_one_out_mmps, ind_test_two_out_mmps))
        
        return k_fold_cross_validation_index_set_quadruples_mmps
    
    
    
    
def train_val_test_validation_map_mols_to_mmps(x_smiles,
                                           X_smiles_mmps,
                                           train_val_test_indices_triple):
    """
    Output: (ind_train_mmps, 
             ind_val_mmps,
             ind_test_mmps,
             ind_val_one_out_mmps,
             ind_val_two_out_mmps,
             ind_test_one_out_mmps, 
             ind_test_two_out_mmps)
    """

    ind_train_mols = train_val_test_indices_triple[0]
    ind_val_mols = train_val_test_indices_triple[1]
    ind_test_mols = train_val_test_indices_triple[2]

    train_space_mols = set(x_smiles[ind_train_mols])
    val_space_mols = set(x_smiles[ind_val_mols])
    test_space_mols = set(x_smiles[ind_test_mols])

    ind_train_mmps = []
    ind_val_mmps = []
    ind_test_mmps = []
    ind_val_one_out_mmps = []
    ind_val_two_out_mmps = []
    ind_test_one_out_mmps = []
    ind_test_two_out_mmps = []

    for (k, mmp) in enumerate(X_smiles_mmps):

        if set(mmp).issubset(train_space_mols):
            ind_train_mmps.append(k)

        elif set(mmp).issubset(val_space_mols):
            ind_val_two_out_mmps.append(k)

        elif set(mmp).issubset(test_space_mols):
            ind_test_two_out_mmps.append(k)

        elif len(set(mmp).intersection(train_space_mols)) == 1 and len(set(mmp).intersection(val_space_mols)) == 1:
            ind_val_one_out_mmps.append(k)

        elif len(set(mmp).intersection(train_space_mols)) == 1 and len(set(mmp).intersection(test_space_mols)) == 1:
            ind_test_one_out_mmps.append(k)

        elif len(set(mmp).intersection(val_space_mols)) == 1 and len(set(mmp).intersection(test_space_mols)) == 1:
            ind_test_one_out_mmps.append(k)

        else:
            print("Error: some MMP is neither in train_space_mols, nor in val_space_mol, nor in test_space_mols, nor in between.")

    ind_val_mmps = ind_val_one_out_mmps + ind_val_two_out_mmps
    ind_test_mmps = ind_test_one_out_mmps + ind_test_two_out_mmps

    return (ind_train_mmps, 
            ind_val_mmps,
            ind_test_mmps,
            ind_val_one_out_mmps,
            ind_val_two_out_mmps,
            ind_test_one_out_mmps, 
            ind_test_two_out_mmps)





# create all pairs training data for siamese network


def all_pairs_training_data(x_smiles_train, 
                            x_smiles_to_fp_dict, 
                            x_smiles_to_y_dict, 
                            added_epochs = 1, 
                            random_seed = 42):
    
    random.seed(random_seed)
    X_fp_1_all_pairs_train = []
    X_fp_2_all_pairs_train = []
    y_mmps_quant_all_pairs_train = []
    y_mmps_all_pairs_train = []
    
    for it in range(added_epochs):
        for smiles in x_smiles_train:
            
            random_smiles = smiles
            
            while random_smiles == smiles:
                random_smiles = random.choice(x_smiles_train) 
            
            X_fp_1_all_pairs_train.append(x_smiles_to_fp_dict[smiles])
            X_fp_2_all_pairs_train.append(x_smiles_to_fp_dict[random_smiles])
            y_mmps_quant_all_pairs_train.append(np.abs(x_smiles_to_y_dict[smiles] - x_smiles_to_y_dict[random_smiles]))
            y_mmps_all_pairs_train.append(0)
            
            
    X_fp_1_all_pairs_train = np.array(X_fp_1_all_pairs_train)
    X_fp_2_all_pairs_train = np.array(X_fp_2_all_pairs_train)
    y_mmps_quant_all_pairs_train = np.array(y_mmps_quant_all_pairs_train)
    y_mmps_all_pairs_train = np.array(y_mmps_all_pairs_train)
    
    
    return (X_fp_1_all_pairs_train, 
            X_fp_2_all_pairs_train, 
            y_mmps_quant_all_pairs_train, 
            y_mmps_all_pairs_train)



# functions for bidirectional cor-var-cor fingerprints for mmps

def flip_first_and_last_part(fp_1d_array, length_of_part = 1024):
    
    first_part = fp_1d_array[0: length_of_part]
    middle_part = fp_1d_array[length_of_part: -length_of_part]
    last_part = fp_1d_array[-length_of_part:]
    
    return np.array(list(last_part) + list(middle_part) + list(first_part))

def make_fingerprints_bidirectional(X_fps, length_of_var_fp = 1024):
    
    X_fps_other_direction = np.array([flip_first_and_last_part(fp, length_of_part = length_of_var_fp) for fp in X_fps])
    
    return np.append(X_fps, X_fps_other_direction, axis = 0)


def create_other_fp_bidirection(X_fps, length_of_var_fp = 1024):
    
    return np.array([flip_first_and_last_part(fp, length_of_part = length_of_var_fp) for fp in X_fps])
    
    
    
# data split dictionary for repeated k-fold cross validation with mols and mmps
    
def create_data_split_dictionary_for_mols_and_mmps(x_smiles,
                                                   X_smiles_mmps,
                                                   x_smiles_mmp_cores,
                                                   k_splits,
                                                   m_reps,
                                                   random_state_cv = 42):
    
    data_split_dictionary = {}
    
    for m in range(m_reps):
        
        for (k, (ind_train_mols, ind_test_mols)) in enumerate(KFold(n_splits = k_splits, 
                                                                    shuffle = True, 
                                                                    random_state = random_state_cv + m).split(x_smiles)):
            # define molecular training space and test space
            train_space_mols = set(x_smiles[ind_train_mols])
            test_space_mols = set(x_smiles[ind_test_mols])
            
            # indices of mmps in relation to how many compounds of an mmp are out of the training set
            ind_zero_out_mmps = []
            ind_one_out_mmps = []
            ind_two_out_mmps = []
            ind_two_out_seen_core_mmps = []
            ind_two_out_unseen_core_mmps = []
            
            # create index sets for mmps depending on how many of their compounds are in the molecular training space
            for (j, mmp) in enumerate(X_smiles_mmps):
                
                if set(mmp).issubset(train_space_mols):
                    ind_zero_out_mmps.append(j)
                
                elif len(set(mmp).intersection(train_space_mols)) == 1 and len(set(mmp).intersection(test_space_mols)) == 1:
                    ind_one_out_mmps.append(j)
                    
                elif set(mmp).issubset(test_space_mols):
                    ind_two_out_mmps.append(j)
                                    
                else:
                    print("Error: some MMP is neither in train_space_mols, nor in test_space_mols, nor in between.")
            
            # split up ind_two_out_mmps depending on whethere the mmp core can also be found in the test set
            zero_out_one_out_mmp_cores = set(x_smiles_mmp_cores[ind_zero_out_mmps + ind_one_out_mmps])
            
            for j in ind_two_out_mmps:
                if x_smiles_mmp_cores[j] in zero_out_one_out_mmp_cores:
                    ind_two_out_seen_core_mmps.append(j)
                else:
                    ind_two_out_unseen_core_mmps.append(j)
            
            # add index data structure to dictionary
            data_split_dictionary[(m, k)] = (ind_train_mols, 
                                             ind_test_mols,
                                             ind_zero_out_mmps,
                                             ind_one_out_mmps,
                                             ind_two_out_mmps,
                                             ind_two_out_seen_core_mmps,
                                             ind_two_out_unseen_core_mmps)
            
    return data_split_dictionary


def inspect_data_split_dictionary(data_split_dictionary, y_mmps):
    
    # preallocate pandas dataframe
    df = pd.DataFrame(columns = ["m", 
                                 "k",
                                 "D_train", 
                                 "D_test", 
                                 "M_train_pos",
                                 "M_train_neg",
                                 "M_inter_pos",
                                 "M_inter_neg",
                                 "M_test_pos",
                                 "M_test_neg",
                                 "M_cores_pos",
                                 "M_cores_neg"])

    for (m, k) in data_split_dictionary.keys():

        # extract indices for this data split
        (ind_train_mols, 
         ind_test_mols,
         ind_zero_out_mmps,
         ind_one_out_mmps,
         ind_two_out_mmps,
         ind_two_out_seen_core_mmps,
         ind_two_out_unseen_core_mmps) = data_split_dictionary[(m,k)]

        # generate label data (mmps)
        y_mmps[ind_zero_out_mmps]
        y_mmps[ind_one_out_mmps]
        y_mmps[ind_two_out_mmps]
        y_mmps[ind_two_out_seen_core_mmps]
        y_mmps[ind_two_out_unseen_core_mmps]

        # fill in data
        df.loc[len(df)] = [m, 
                           k, 
                           len(ind_train_mols), 
                           len(ind_test_mols), 
                           list(y_mmps[ind_zero_out_mmps]).count(1),
                           list(y_mmps[ind_zero_out_mmps]).count(0),
                           list(y_mmps[ind_one_out_mmps]).count(1),
                           list(y_mmps[ind_one_out_mmps]).count(0),
                           list(y_mmps[ind_two_out_mmps]).count(1),
                           list(y_mmps[ind_two_out_mmps]).count(0),
                           list(y_mmps[ind_two_out_unseen_core_mmps]).count(1),
                           list(y_mmps[ind_two_out_unseen_core_mmps]).count(0)]

    # add column with averages
    df.loc[len(df)] = ["*", 
                       "*", 
                       np.mean(df["D_train"].values),
                       np.mean(df["D_test"].values),
                       np.mean(df["M_train_pos"].values),
                       np.mean(df["M_train_neg"].values),
                       np.mean(df["M_inter_pos"].values),
                       np.mean(df["M_inter_neg"].values),
                       np.mean(df["M_test_pos"].values),
                       np.mean(df["M_test_neg"].values),
                       np.mean(df["M_cores_pos"].values),
                       np.mean(df["M_cores_neg"].values)]
    
    # set row names
    df = df.rename(index = dict([(k,"*") for k in range(len(df)-1)] + [(len(df)-1, "Avg")]))
    
    # display dataframe
    display(df)
    
    return df

