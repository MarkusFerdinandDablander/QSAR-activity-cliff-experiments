import numpy as np
import pandas as pd
from sklearn.model_selection import KFold



def create_data_split_dictionary_for_mols_and_mmps(x_smiles,
                                                   X_smiles_mmps,
                                                   x_smiles_mmp_cores,
                                                   k_splits,
                                                   m_reps,
                                                   random_state_cv = 42):
    """
    Splits up data via a k-fold cross validation scheme that is repeated with m random seeds. Each of the k*m splits is represented as a tuple of 7 index sets and can be accessed via 
    
    data_split_dictionary[(m, k)] = (ind_train_mols,
                                     ind_test_mols,
                                     ind_zero_out_mmps,
                                     ind_one_out_mmps,
                                     ind_two_out_mmps,
                                     ind_two_out_seen_core_mmps,
                                     ind_two_out_unseen_core_mmps).
    
    For example, x_smiles[ind_test_mols] returns the test set of individual molecules and X_smiles_mmps[ind_one_out_mmps] returns all MMPs with one compound in the training set and one compound in the test set.
    
    """
    
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



def inspect_data_split_dictionary(data_split_dictionary, 
                                  y_mmps):
    """ 
    Takes as input a data_split_dictionary created by the function create_data_split_dictionary_for_mols_and_mmps and a list y_mmps containing the binary AC-labels for the MMPs in X_smiles_mmps. Gives as output an overview over the average sizes of the sets D_train, D_test, M_train_pos, M_train_neg, M_inter_pos, M_inter_neg, M_test_pos, M_test_neg, M_cores_pos, M_cores_neg.
    """
    
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

