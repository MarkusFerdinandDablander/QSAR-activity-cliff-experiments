# import packages

# general tools
import numpy as np
import pandas as pd
import math
import random
import sys
import collections
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF as statsmodels_ecdf
from collections import defaultdict
from itertools import chain
from chembl_structure_pipeline.standardizer import standardize_mol, get_parent_mol

# tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Add, InputSpec, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints

# Spektral
from spektral.layers import GCNConv, GlobalSumPool

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

# sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# Define Functions

## General Functions

# auxiliary functions for mmp analysis

def extract_matched_molecular_series_of_each_core(X_smiles_mmps, x_smiles_mmp_cores):
    
    x_smiles_mms_dict = defaultdict(list)
    distinct_mmp_cores = list(set(x_smiles_mmp_cores))
    
    for (k, [smiles_1, smiles_2]) in enumerate(X_smiles_mmps):
        
        x_smiles_mms_dict[x_smiles_mmp_cores[k]].extend([smiles_1, smiles_2])
    
    for mmp_core in distinct_mmp_cores:
        x_smiles_mms_dict[mmp_core] = list(set(x_smiles_mms_dict[mmp_core]))
                
    return x_smiles_mms_dict


def extract_matched_molecular_pairs_of_each_molecule(X_smiles_mmps):
    
    x_smiles_mmp_dict = defaultdict(list)
    list_of_all_smiles = list(set(np.ndarray.flatten(X_smiles_mmps)))
    
    for [smiles_1, smiles_2] in X_smiles_mmps:
        
        x_smiles_mmp_dict[smiles_1].append([smiles_1, smiles_2])
        x_smiles_mmp_dict[smiles_2].append([smiles_1, smiles_2])
    
    return x_smiles_mmp_dict


# auxiliary functions for data saving/cleaning/preprocessig

def smiles_standardisation(smiles_list, get_parent_smiles = True):
    
    standardised_smiles_list = list(range(len(smiles_list)))
    
    for (k, smiles) in enumerate(smiles_list):
        
        try:

            # convert smiles to mol object
            mol = Chem.MolFromSmiles(smiles)

            # standardise mol object
            standardised_mol = standardize_mol(mol, check_exclusion = True)
            
            if get_parent_smiles == True:
            
                # systematically remove salts, solvents and isotopic information to get parent mol
                (standardised_mol, exclude) = get_parent_mol(standardised_mol, 
                                                             neutralize = True, 
                                                             check_exclusion=True,
                                                             verbose = False)

            # convert mol object back to smiles
            standardised_smiles = Chem.MolToSmiles(standardised_mol)

            # replace smiles with standardised parent smiles
            standardised_smiles_list[k] = standardised_smiles
        
        except:
            
            # leave smiles unchanged if it generates an exception
            standardised_smiles_list[k] = smiles
            
    return np.array(standardised_smiles_list)


def save_X_smiles_as_csv(X_smiles, location):
    
    indices = np.reshape(np.arange(0, len(X_smiles)), (-1,1))
    data = np.concatenate((X_smiles, indices), axis = 1)
    np.savetxt(location, data, delimiter = ",", fmt = "%s")
    
    
def save_Y_as_csv(Y, location):
    
    indices = np.reshape(np.arange(0, len(Y)), (-1,1))
    data = np.concatenate((Y, indices), axis = 1)
    np.savetxt(location, data, delimiter = ",")

# define naive classifiers

def naive_classifier_fifty_fifty(y_true):
    y_pred_proba_pos = np.random.randint(0,2, size = (len(y_true),))
    return y_pred_proba_pos


def naive_classifier_largest_class(y_true):
    y_true = np.array(y_true)
    if np.float32(np.sum(y_true)) > np.float32(len(y_true)/2):
        y_pred_proba_pos = np.ones((len(y_true),))
    else:
        y_pred_proba_pos = np.zeros((len(y_true),))
    return y_pred_proba_pos


def largest_class(y_true):
    y_true = np.array(y_true)
    if np.float32(np.sum(y_true)) > np.float32(len(y_true)/2):
        return 1
    else:
        return 0

# normalisation/standardisation functions for feature matrices

def normaliser_z_score(A):
    
    def normalisation_function(B):
        return (B - np.mean(A, axis = 0))/np.std(A, axis = 0)
    
    A_norm = normalisation_function(A)
    
    return (A_norm, normalisation_function)


def normaliser_min_max(A):
    
    def normalisation_function(B):
        return (B - np.min(A, axis = 0))/(np.max(A, axis = 0) - np.min(A, axis = 0))
    
    A_norm = normalisation_function(A)
    
    return (A_norm, normalisation_function)


def normaliser_cdf(A):
    
    def normalisation_function(B):
        
        B_norm = np.zeros(B.shape)
        n_features = A.shape[1]
        
        
        for feature in range(n_features):
            
            feature_ecdf = statsmodels_ecdf(A[:,feature])
            B_norm[:,feature] = feature_ecdf(B[:, feature])
        
        return B_norm
    
    A_norm = normalisation_function(A)
    
    return (A_norm, normalisation_function)
            

# task-independent molecular featurisations (which only depend on the smiles string of a molecule)

def rdkit_mol_descriptors_from_smiles(smiles_string, descriptor_list = None):
    
    """Generates a vector of rdkit molecular descriptors. Per default, we use the same 200 descriptors as in the papers 
    "Analyzing Learned Molecular Representations for Property Prediction" and
    "Molecular representation learning with languagemodels and domain-relevant auxiliary tasks". """
    
    if descriptor_list == None:
        
        # 200 descriptors
        descriptor_list = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 
                           'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 
                           'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 
                           'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 
                           'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 
                           'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 
                           'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 
                           'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 
                           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 
                           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 
                           'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 
                           'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 
                           'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 
                           'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 
                           'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 
                           'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 
                           'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 
                           'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 
                           'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 
                           'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 
                           'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 
                           'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 
                           'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 
                           'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 
                           'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 
                           'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 
                           'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 
                           'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 
                           'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 
                           'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 
                           'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 
                           'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 
                           'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 
                           'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
    
    mol_descriptor_calculator = MolecularDescriptorCalculator(descriptor_list)
    
    
    mol = Chem.MolFromSmiles(smiles_string)
    descriptor_vals = mol_descriptor_calculator.CalcDescriptors(mol)
        
    return np.array(descriptor_vals)



def circular_fps_from_smiles(smiles_string, 
                             radius = 2, 
                             bitstring_length = 2**10, 
                             use_features = False,
                             use_chirality = False):
    
    """Extended-connectivity fingerprints (ECFP) and functional-connectivity fingerprints (FCFP)."""
    
    molecule = Chem.MolFromSmiles(smiles_string)
    feature_list = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 
                                                                       radius = radius, 
                                                                       nBits = bitstring_length, 
                                                                       useFeatures = use_features,
                                                                       useChirality = use_chirality)
    
    return np.array(feature_list)


# task-independent mmp featurisations (which only depend on a pair of SMILES strings of two molecules)

def circular_fp_abs_diff_from_mmps(mmp, 
                                   radius = 2, 
                                   bitstring_length = 2**10, 
                                   use_features = False,
                                   use_chirality = False):
    
    """
    Extended-connectivity fingerprints (ECFP) and functional-connectivity fingerprints (FCFP). We featurise
    a pair of molecules by computing the difference of two ECFPs/FCFPs and then taking the absolute value.
    """
    
    smiles_string_1 = mmp[0]
    smiles_string_2 = mmp[1]
    
    molecule_1 = Chem.MolFromSmiles(smiles_string_1) 
    molecule_2 = Chem.MolFromSmiles(smiles_string_2)
    
    feature_list_1 = np.array(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule_1, 
                                                                         radius = radius, 
                                                                         nBits = bitstring_length, 
                                                                         useFeatures = use_features,
                                                                         useChirality = use_chirality))
    
    feature_list_2 = np.array(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule_2, 
                                                                         radius = radius, 
                                                                         nBits = bitstring_length, 
                                                                         useFeatures = use_features,
                                                                         useChirality = use_chirality))
    
    circular_fp_abs_diff =  np.abs(feature_list_1 - feature_list_2)
    
    return circular_fp_abs_diff


def circular_fp_abs_diff_from_mmps_array(X_smiles_mmps, 
                                         radius = 2, 
                                         bitstring_length = 2**10, 
                                         use_features = False,
                                         use_chirality = False):
    """
    Extended-connectivity fingerprints (ECFP) and functional-connectivity fingerprints (FCFP). We featurise
    an array consisting of pairs of molecules by computing the difference of two ECFPs/FCFPs and 
    then taking the absolute value.
    """
    
    dict_smiles_to_fps = {}
    all_unique_smiles = set(X_smiles_mmps.flatten())
    
    n_mmps = len(X_smiles_mmps)
    X_fcfp_abs_diff = np.zeros((n_mmps, bitstring_length))
    
    for smiles in all_unique_smiles:
        
        mol = Chem.MolFromSmiles(smiles)
        dict_smiles_to_fps[smiles] = np.array(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 
                                                                                                  radius = radius, 
                                                                                                  nBits = bitstring_length, 
                                                                                                  useFeatures = use_features,
                                                                                                  useChirality = use_chirality))
        
    for k in range(n_mmps):
        
        smiles_1 = X_smiles_mmps[k][0]
        smiles_2 = X_smiles_mmps[k][1]
        
        fp_1 = dict_smiles_to_fps[smiles_1]
        fp_2 = dict_smiles_to_fps[smiles_2]
        
        X_fcfp_abs_diff[k,:] = np.absolute(fp_1 - fp_2)
        
    return X_fcfp_abs_diff


def rdkit_mol_descriptors_abs_diff_from_mmps_array(X_smiles_mmps, descriptor_list = None):
    """
    RDKit descriptor vectors. We featurise an array consisting of pairs of molecules by computing 
    the difference of two RDKit descriptor vectors and then taking the absolute value.
    """
    
    dict_smiles_to_rdkit_vectors = {}
    all_unique_smiles = set(X_smiles_mmps.flatten())
    n_mmps = len(X_smiles_mmps)
    
    if descriptor_list == None:
        n_features = 200
    else:
        n_features = len(descriptor_list)
    
    X_rdkit_abs_diff = np.zeros((n_mmps, n_features))
    
    for smiles in all_unique_smiles:
        
        dict_smiles_to_rdkit_vectors[smiles] = rdkit_mol_descriptors_from_smiles(smiles, 
                                                                                 descriptor_list = descriptor_list)
        
    for k in range(n_mmps):
        
        smiles_1 = X_smiles_mmps[k][0]
        smiles_2 = X_smiles_mmps[k][1]
        
        rdkit_vector_1 = dict_smiles_to_rdkit_vectors[smiles_1]
        rdkit_vector_2 = dict_smiles_to_rdkit_vectors[smiles_2]
        
        X_rdkit_abs_diff[k,:] = np.absolute(rdkit_vector_1 - rdkit_vector_2)
        
    return X_rdkit_abs_diff
    
    
# scoring functions for binary classification

def transform_probs_to_labels(y_pred_proba_pos, cutoff_value = 0.5):
    
    y_pred_proba_pos = np.array(y_pred_proba_pos)
    y_pred = np.copy(y_pred_proba_pos)
    
    y_pred[y_pred > cutoff_value] = 1
    y_pred[y_pred <= cutoff_value] = 0 # per default, sklearn random forest classifiers map a probability of 0.5 to class 0
    
    return y_pred


def binary_classification_scores(y_true, y_pred_proba_pos, display_results = False):
    
    # prepare variables
    y_true = list(y_true)
    y_pred_proba_pos = list(y_pred_proba_pos)
    y_pred = list(transform_probs_to_labels(y_pred_proba_pos))

    n_test_cases = len(y_true)
    n_test_cases_neg = list(y_true).count(0)
    n_test_cases_pos = list(y_true).count(1)
    
    if y_true.count(y_true[0]) != len(y_true):
        conf_matrix = confusion_matrix(y_true, y_pred)
        tn = conf_matrix[0,0]
        fn = conf_matrix[1,0]
        tp = conf_matrix[1,1]
        fp = conf_matrix[0,1]
        
    elif y_true.count(0) == len(y_true):
        
        tn = y_pred.count(0)
        fn = 0
        tp = 0
        fp = y_pred.count(1)
        
    elif y_true.count(1) == len(y_true):
    
        tn = 0
        fn = y_pred.count(0)
        tp = y_pred.count(1)
        fp = 0

    # compute scores
    
    # roc_auc
    if y_true.count(y_true[0]) != len(y_true):
        roc_auc = roc_auc_score(y_true, y_pred_proba_pos)
    else:
        roc_auc = float("NaN")
    
    # accuracy
    accuracy = (tn + tp)/n_test_cases
    
    # balanced accuracy
    if fn+tp != 0 and fp+tn != 0:
        balanced_accuracy = ((tp/(fn+tp))+(tn/(fp+tn)))/2
    else:
        balanced_accuracy = float("NaN")
    
    # f1 score
    f1 = f1_score(y_true, y_pred)
    
    # mcc score
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # sensitivity
    if fn + tp != 0:
        sensitivity = tp/(fn+tp)
    else:
        sensitivity = float("NaN")
    
    #specificity
    if fp+tn != 0:
        specificity = tn/(fp+tn)
    else:
        specificity = float("NaN")
        
    # positive predictive value
    if tp+fp != 0:
        positive_predictive_value = tp/(tp+fp)
    else:
        positive_predictive_value = float("NaN")
    
    # negative predictive value
    if tn+fn != 0:
        negative_predictive_value = tn/(tn+fn)
    else:
        negative_predictive_value = float("NaN")

    # collect scores
    scores_array = np.array([roc_auc, accuracy, balanced_accuracy, f1, mcc, sensitivity, specificity, positive_predictive_value, negative_predictive_value, n_test_cases, n_test_cases_neg, n_test_cases_pos])
    scores_array_2d = np.reshape(scores_array, (1, len(scores_array)))
    columns = ["ROC-AUC", "Acc.", "Bal. Acc.", "F1", "MCC", "Sens.", "Spec.", "Pos. Pred. Val.", "Neg. Pred. Val.", "Test Cases", "Neg. Test Cases", "Pos. Test Cases"] 
    scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

    # display scores
    if display_results == True:
        display(scores_df)
        
    return scores_df


def summarise_binary_classification_scores(scores_array_2d, display_results = False):
    """
    scores_array_2d.shape = (number_of_trials, number_of_scoring_functions)
    """
    
    avgs = np.nanmean(scores_array_2d, axis = 0)
    stds = np.nanstd(scores_array_2d, axis = 0)
    
    summarized_scores_array = np.array([avgs, stds])
    columns = ["ROC-AUC", "Acc.", "Bal. Acc.", "F1", "MCC", "Sens.", "Spec.", "Pos. Pred. Val.", "Neg. Pred. Val.", "Test Cases", "Neg. Test Cases", "Pos. Test Cases"]
    summarized_scores_df = pd.DataFrame(data = summarized_scores_array, index = ["Avg.", "Std."], columns = columns)
    
    if display_results == True:
        display(summarized_scores_df)
        display(pd.DataFrame(data = scores_array_2d, index = range(1, len(scores_array_2d)+1), columns = columns))
        
    return summarized_scores_df
    

# oversampling and undersampling function for imbalanced binary classification problems

def oversample(X,y):
    """
    X.shape = (n_samples, n_features)
    y.shape = (n_samples,)
    y only contains 0s and 1s, indicating the class membership, y can be a list of a numpy array
    """
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(y)
    n_positive_samples = np.sum(np.array(y))
    n_negative_samples = n_samples - n_positive_samples
    delta = abs(n_positive_samples - n_negative_samples)

    if n_negative_samples <= n_positive_samples:
        minority_class = 0
    else:
        minority_class = 1

    ind_minority_class = np.where(y == minority_class)[0]
    X_minority = X[ind_minority_class]

    np.random.seed(42)
    surplus_ind = np.random.randint(len(X_minority), size = delta)
    X_surplus = X_minority[surplus_ind]
    
    y_surplus = (np.ones(delta)*minority_class).astype(int)

    X_oversampled = np.append(X, X_surplus, axis = 0)
    y_oversampled = np.append(y, y_surplus)

    return (X_oversampled, y_oversampled)



def undersample(X,y):
    """
    X.shape = (n_samples, n_features)
    y.shape = (n_samples,)
    y only contains 0s and 1s, indicating the class membership, y can be a list of a numpy array
    """
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(y)
    n_positive_samples = np.sum(np.array(y))
    n_negative_samples = n_samples - n_positive_samples
    delta = abs(n_positive_samples - n_negative_samples)

    if n_negative_samples <= n_positive_samples:
        majority_class = 1
    else:
        majority_class = 0
    
    ind_majority_class = np.where(y == majority_class)[0]
    
    np.random.seed(42)
    ind_delete = np.random.choice(ind_majority_class, delta, replace = False)
    
    X_undersampled = np.delete(X, ind_delete, axis = 0)
    y_undersampled = np.delete(y, ind_delete)

    return (X_undersampled, y_undersampled)




## Data Splitting Functions and Data Processing Functions

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
        
        # For mols that have not been sanitized, we need to compute their ring information
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

    # Order groups of molecules by first comparing the size of groups
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



## Deep Learning Functions

# layer for trainable rational activation functions

class RationalLayer(Layer):
    """ This class was taken from Nicolas Boulle at 
    https://github.com/NBoulle/RationalNets/blob/master/src/RationalLayer.py
    
    Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are learned array with the same shape as x.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha_initializer: initializer function for the weights of the numerator P.
        beta_initializer: initializer function for the weights of the denominator Q.
        alpha_regularizer: regularizer for the weights of the numerator P.
        beta_regularizer: regularizer for the weights of the denominator Q.
        alpha_constraint: constraint for the weights of the numerator P.
        beta_constraint: constraint for the weights of the denominator Q.
        shared_axes: the axes along which to share learnable
            parameters for the activation function.
            For example, if the incoming feature maps
            are from a 2D convolution
            with output shape `(batch, height, width, channels)`,
            and you wish to share parameters across space
            so that each filter only has one set of parameters,
            set `shared_axes=[1, 2]`.
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """

    def __init__(self, alpha_initializer=[1.1915, 1.5957, 0.5, 0.0218], beta_initializer=[2.383, 0.0, 1.0], 
                 alpha_regularizer=None, beta_regularizer=None, alpha_constraint=None, beta_constraint=None,
                 shared_axes=None, **kwargs):
        super(RationalLayer, self).__init__(**kwargs)
        self.supports_masking = True

        # Degree of rationals
        self.degreeP = len(alpha_initializer) - 1
        self.degreeQ = len(beta_initializer) - 1
        
        # Initializers for P
        self.alpha_initializer = [initializers.Constant(value=alpha_initializer[i]) for i in range(len(alpha_initializer))]
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        
        # Initializers for Q
        self.beta_initializer = [initializers.Constant(value=beta_initializer[i]) for i in range(len(beta_initializer))]
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        
        self.coeffsP = []
        for i in range(self.degreeP+1):
            # Add weight
            alpha_i = self.add_weight(shape=param_shape,
                                     name='alpha_%s'%i,
                                     initializer=self.alpha_initializer[i],
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
            self.coeffsP.append(alpha_i)
            
        # Create coefficients of Q
        self.coeffsQ = []
        for i in range(self.degreeQ+1):
            # Add weight
            beta_i = self.add_weight(shape=param_shape,
                                     name='beta_%s'%i,
                                     initializer=self.beta_initializer[i],
                                     regularizer=self.beta_regularizer,
                                     constraint=self.beta_constraint)
            self.coeffsQ.append(beta_i)
        
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
                    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
                    self.built = True

    def call(self, inputs, mask=None):
        # Evaluation of P
        outP = tf.math.polyval(self.coeffsP, inputs)
        # Evaluation of Q
        outQ = tf.math.polyval(self.coeffsQ, inputs)
        # Compute P/Q
        out = tf.math.divide(outP, outQ)
        return out

    def get_config(self):
        config = {
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(RationalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    

# function to create an MLP model

def create_mlp_model(architecture = (2**10, 100, 100, 1),
                     hidden_activation = tf.keras.activations.relu,
                     output_activation = tf.keras.activations.sigmoid,
                     use_bias = True,
                     use_batch_norm_input_hidden_lasthidden = (False, False, False),
                     dropout_rates_input_hidden = (0.0, 0.0),
                     hidden_rational_layers = False,
                     rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                     rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                     rational_parameters_shared_axes = [1],
                     model_name = None):
    
    """
    Function to create an MLP model, optionally with trainable rational activation function layers.
    rational_parameters_shared_axes = [1] means that weights are shared across each rational layer
    """
    
    # define variables
    n_hidden = len(architecture)-2
    dropout_rate_input = dropout_rates_input_hidden[0]
    dropout_rate_hidden = dropout_rates_input_hidden[1]
    use_batch_norm_input = use_batch_norm_input_hidden_lasthidden[0]
    use_batch_norm_hidden = use_batch_norm_input_hidden_lasthidden[1]
    use_batch_norm_lasthidden = use_batch_norm_input_hidden_lasthidden[2]
    
    # define first input layer
    input_layer = Input(shape = architecture[0], name = "input")
    hidden = input_layer
    
    # if wanted, add batch normalisation layer right after input layer
    if use_batch_norm_input == True:
        hidden = BatchNormalization()(hidden)
    
    # add dropout layer right after input
    hidden = Dropout(rate = dropout_rate_input, name = "dropout_input")(hidden)

    # define hidden layers
    for h in range(1, n_hidden + 1):
        
        hidden = Dense(units = architecture[h], 
                       activation = hidden_activation, 
                       use_bias = use_bias, 
                       name = "hidden_" + str(h))(hidden)
        
        # if wanted, define additional hidden layers with trainable rational activation functions
        if hidden_rational_layers == True:
            
            hidden = RationalLayer(alpha_initializer = rational_parameters_alpha_initializer,
                                   beta_initializer = rational_parameters_beta_initializer,
                                   shared_axes = rational_parameters_shared_axes,
                                   name = "rational_hidden_" + str(h))(hidden)
            
        # if wanted, add batch normalisation layer after all hidden layers but the last hidden layer
        if use_batch_norm_hidden == True and h != n_hidden:
            hidden = BatchNormalization()(hidden)
            
        # if wanted, add batch normalisation also after the last hidden layer
        if use_batch_norm_lasthidden == True and h == n_hidden:
            hidden = BatchNormalization()(hidden)
            
        # add dropout layer
        hidden = Dropout(rate = dropout_rate_hidden, name = "dropout_hidden" + str(h))(hidden)
        
    
    # define final output layer
    
    if n_hidden >= 0:
        
        output_layer = Dense(architecture[n_hidden + 1], 
                             activation = output_activation, 
                             use_bias = use_bias, 
                             name = 'output')(hidden)
    else:
        
        output_layer = hidden
        
    # build model
    mlp_model = Model(inputs = [input_layer], outputs = [output_layer], name = model_name)

    return mlp_model


# auxiliary functions to plot training histories of tf.keras models

def extract_learning_curves(classifier_model_training_histories):
    
    n_models = len(classifier_model_training_histories)
    n_epochs = len(classifier_model_training_histories[0].history["loss"])
    metric_names = list(classifier_model_training_histories[0].history.keys())
    
    assert metric_names[0] == "loss"
    assert metric_names[1][0:3] == "auc"
    assert metric_names[2] == "binary_accuracy"
    assert metric_names[3] == "val_loss"
    assert metric_names[4][0:7] == "val_auc"
    assert metric_names[5] == "val_binary_accuracy"
    

    train_loss_array = np.zeros((n_models, n_epochs))
    train_roc_auc_array = np.zeros((n_models, n_epochs))
    train_acc_array = np.zeros((n_models, n_epochs))
    test_loss_array = np.zeros((n_models, n_epochs))
    test_roc_auc_array = np.zeros((n_models, n_epochs))
    test_acc_array = np.zeros((n_models, n_epochs))
    
    for (k, history) in enumerate(classifier_model_training_histories):
        
        train_loss_array[k,:] = history.history[metric_names[0]]
        train_roc_auc_array[k,:] = history.history[metric_names[1]]
        train_acc_array[k,:] = history.history[metric_names[2]]

        test_loss_array[k,:] = history.history[metric_names[3]]
        test_roc_auc_array[k,:] = history.history[metric_names[4]]
        test_acc_array[k,:] = history.history[metric_names[5]]
        
    train_loss_avg = np.mean(train_loss_array, axis = 0)
    train_roc_auc_avg = np.mean(train_roc_auc_array, axis = 0)
    train_acc_avg = np.mean(train_acc_array, axis = 0)
    test_loss_avg = np.mean(test_loss_array, axis = 0)
    test_roc_auc_avg = np.mean(test_roc_auc_array, axis = 0)
    test_acc_avg = np.mean(test_acc_array, axis = 0)
    
    return (train_loss_avg, train_roc_auc_avg, train_acc_avg, test_loss_avg, test_roc_auc_avg, test_acc_avg)


def extract_learning_curves_from_dual_task_model(classifier_model_training_histories):
    
    n_models = len(classifier_model_training_histories)
    n_epochs = len(classifier_model_training_histories[0].history["loss"])
    metric_names = list(classifier_model_training_histories[0].history.keys())

    assert metric_names[0] == "loss"
    assert metric_names[1] == "ac_mlp_loss"
    assert metric_names[2] == "dir_mlp_loss"
    assert metric_names[3][0:10] == "ac_mlp_auc"
    assert metric_names[4] == "ac_mlp_binary_accuracy"
    assert metric_names[5][0:11] == "dir_mlp_auc"
    assert metric_names[6] == "dir_mlp_binary_accuracy"
    
    assert metric_names[7] == "val_loss"
    assert metric_names[8] == "val_ac_mlp_loss"
    assert metric_names[9] == "val_dir_mlp_loss"
    assert metric_names[10][0:14] == "val_ac_mlp_auc"
    assert metric_names[11] == "val_ac_mlp_binary_accuracy"
    assert metric_names[12][0:15] == "val_dir_mlp_auc"
    assert metric_names[13] == "val_dir_mlp_binary_accuracy"

    train_loss_array = np.zeros((n_models, n_epochs))
    train_ac_mlp_loss_array = np.zeros((n_models, n_epochs))
    train_dir_mlp_loss_array = np.zeros((n_models, n_epochs))
    train_ac_mlp_auc_array = np.zeros((n_models, n_epochs))
    train_ac_mlp_binary_accuracy_array = np.zeros((n_models, n_epochs))
    train_dir_mlp_auc_array = np.zeros((n_models, n_epochs))
    train_dir_mlp_binary_accuracy_array = np.zeros((n_models, n_epochs))
    
    test_loss_array = np.zeros((n_models, n_epochs))
    test_ac_mlp_loss_array = np.zeros((n_models, n_epochs))
    test_dir_mlp_loss_array = np.zeros((n_models, n_epochs))
    test_ac_mlp_auc_array = np.zeros((n_models, n_epochs))
    test_ac_mlp_binary_accuracy_array = np.zeros((n_models, n_epochs))
    test_dir_mlp_auc_array = np.zeros((n_models, n_epochs))
    test_dir_mlp_binary_accuracy_array = np.zeros((n_models, n_epochs))

    for (k, history) in enumerate(classifier_model_training_histories):

        train_loss_array[k,:] = history.history[metric_names[0]]
        train_ac_mlp_loss_array[k,:] = history.history[metric_names[1]]
        train_dir_mlp_loss_array[k,:] = history.history[metric_names[2]]
        train_ac_mlp_auc_array[k,:] = history.history[metric_names[3]]
        train_ac_mlp_binary_accuracy_array[k,:] = history.history[metric_names[4]]
        train_dir_mlp_auc_array[k,:] = history.history[metric_names[5]]
        train_dir_mlp_binary_accuracy_array[k,:] = history.history[metric_names[6]]
        
        test_loss_array[k,:] = history.history[metric_names[7]]
        test_ac_mlp_loss_array[k,:] = history.history[metric_names[8]]
        test_dir_mlp_loss_array[k,:] = history.history[metric_names[9]]
        test_ac_mlp_auc_array[k,:] = history.history[metric_names[10]]
        test_ac_mlp_binary_accuracy_array[k,:] = history.history[metric_names[11]]
        test_dir_mlp_auc_array[k,:] = history.history[metric_names[12]]
        test_dir_mlp_binary_accuracy_array[k,:] = history.history[metric_names[13]]

    train_loss_avg = np.mean(train_loss_array, axis = 0)
    train_ac_mlp_loss_avg = np.mean(train_ac_mlp_loss_array, axis = 0)
    train_dir_mlp_loss_avg = np.mean(train_dir_mlp_loss_array, axis = 0)
    train_ac_mlp_auc_avg = np.mean(train_ac_mlp_auc_array, axis = 0)
    train_ac_mlp_binary_accuracy_avg = np.mean(train_ac_mlp_binary_accuracy_array, axis = 0)
    train_dir_mlp_auc_avg = np.mean(train_dir_mlp_auc_array, axis = 0)
    train_dir_mlp_binary_accuracy_avg = np.mean(train_dir_mlp_binary_accuracy_array, axis = 0)
    
    test_loss_avg = np.mean(test_loss_array, axis = 0)
    test_ac_mlp_loss_avg = np.mean(test_ac_mlp_loss_array, axis = 0)
    test_dir_mlp_loss_avg = np.mean(test_dir_mlp_loss_array, axis = 0)
    test_ac_mlp_auc_avg = np.mean(test_ac_mlp_auc_array, axis = 0)
    test_ac_mlp_binary_accuracy_avg = np.mean(test_ac_mlp_binary_accuracy_array, axis = 0)
    test_dir_mlp_auc_avg = np.mean(test_dir_mlp_auc_array, axis = 0)
    test_dir_mlp_binary_accuracy_avg = np.mean(test_dir_mlp_binary_accuracy_array, axis = 0)
    
    return (train_loss_avg, 
            train_ac_mlp_loss_avg, 
            train_dir_mlp_loss_avg, 
            train_ac_mlp_auc_avg, 
            train_ac_mlp_binary_accuracy_avg, 
            train_dir_mlp_auc_avg, 
            train_dir_mlp_binary_accuracy_avg, 
            test_loss_avg, 
            test_ac_mlp_loss_avg, 
            test_dir_mlp_loss_avg, 
            test_ac_mlp_auc_avg, 
            test_ac_mlp_binary_accuracy_avg, 
            test_dir_mlp_auc_avg, 
            test_dir_mlp_binary_accuracy_avg)

# functions to convert smiles string to molecular graph arrays (node feature matrix, adjacency matrix, edge feature tensor)

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements not in the permitted list to the last element
    of the permitted list.
    """
    
    if x not in permitted_list:
        x = permitted_list[-1]
        
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    
    return binary_encoding


def get_atom_features(atom):
    
    permitted_list_of_atoms = ["C", "N", "O", "F", "Cl", "S", "P", "Br", "B", "I", "Unknown"]
    
    atom_type_enc = one_hot_encoding(atom.GetSymbol(), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    n_hydrogens_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    formal_charge_enc = [int(atom.GetFormalCharge())]
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atom_feature_vector = np.array(atom_type_enc + 
                                   n_heavy_neighbors_enc + 
                                   n_hydrogens_enc + 
                                   formal_charge_enc +
                                   is_in_a_ring_enc +
                                   is_aromatic_enc)
    
    return atom_feature_vector


def get_bond_features(bond):
    
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE,
                                    Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE,
                                    Chem.rdchem.BondType.AROMATIC]
    
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = np.array(bond_type_enc +
                                   bond_is_conj_enc +
                                   bond_is_in_ring_enc)
                                
    return bond_feature_vector


def create_molecular_graph_arrays_from_smiles(smiles):
    """
    Input: SMILES-string of Molecule
    
    Output: a tuple of three numpy arrays (x, a, e)
    
    x is the atom-feature-matrix of the molecular graph with dimensions (n_nodes, n_node_features).
    a is the adjacency matrix of the molecular graph, only contains 0s and 1s, and has the dimensions (n_nodes, n_nodes).
    e is the edge-feature-tensor of the molecular graph and has dimensions (n_nodes, n_nodes, n_edge_features).
    
    """
    
    # convert smiles to rdkit mol object
    mol = Chem.MolFromSmiles(smiles)
    
    # get dimensions
    n_nodes = mol.GetNumAtoms()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
    
    # construct x
    x = np.zeros((n_nodes, n_node_features))
    
    for atom in mol.GetAtoms():
        row_index = atom.GetIdx()
        x[row_index,:] = get_atom_features(atom)
    
    # construct a
    a = GetAdjacencyMatrix(mol)
    assert a.shape == (n_nodes, n_nodes)
    
    # construct e
    e = np.zeros((n_nodes, n_nodes, n_edge_features))
    (rows, cols) = np.nonzero(a)
    
    for (i,j) in zip(rows, cols):
        e[i,j,:] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
    return (x, a, e)


def tensorise_smiles_for_batch_mode(x_smiles, n_nodes_max = None):
    
    # convert smiles to molecular graph arrays x,a,e which can still have different dimensions n_nodes
    x_mol_graph_arrays = []

    for (k, smiles) in enumerate(x_smiles):
    
        (x, a, e) = create_molecular_graph_arrays_from_smiles(smiles)
        x_mol_graph_arrays.append((x, a, e))
    
    # define dimensional parameters
    n_batch = len(x_smiles)
    n_node_features = x_mol_graph_arrays[0][0].shape[1]
    n_edge_features = x_mol_graph_arrays[0][2].shape[2]
    
    if n_nodes_max == None:
        n_nodes_max = max([x.shape[0] for (x, a, e) in x_mol_graph_arrays])
    
    # perform zero padding n_nodes -> n_nodes_max for batch mode
    X_mol_graphs = np.zeros((n_batch, n_nodes_max, n_node_features))
    A_mol_graphs = np.zeros((n_batch, n_nodes_max, n_nodes_max))
    E_mol_graphs = np.zeros((n_batch, n_nodes_max, n_nodes_max, n_edge_features))
    
    for (k, (x, a, e)) in enumerate(x_mol_graph_arrays):
        
        n_nodes = x.shape[0]
        
        X_mol_graphs[k, 0:n_nodes, 0:n_node_features] = x
        A_mol_graphs[k, 0:n_nodes, 0:n_nodes] = a
        E_mol_graphs[k, 0:n_nodes, 0:n_nodes, 0:n_edge_features] = e
    
    return (X_mol_graphs, A_mol_graphs, E_mol_graphs)


def tensorise_mmps_for_batch_mode(X_smiles_mmps, n_nodes_max = None):
    
    x_smiles_1 = X_smiles_mmps[:,0]
    x_smiles_2 = X_smiles_mmps[:,1]
    
    all_smiles = set(list(x_smiles_1) + list(x_smiles_2))
    
    if n_nodes_max == None:
        n_nodes_max = max([Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in all_smiles])
        
    (X_mol_graphs_1, A_mol_graphs_1, E_mol_graphs_1) = tensorise_smiles_for_batch_mode(x_smiles_1, n_nodes_max = n_nodes_max)
    (X_mol_graphs_2, A_mol_graphs_2, E_mol_graphs_2) = tensorise_smiles_for_batch_mode(x_smiles_2, n_nodes_max = n_nodes_max)
    
    return (X_mol_graphs_1, A_mol_graphs_1, E_mol_graphs_1, X_mol_graphs_2, A_mol_graphs_2, E_mol_graphs_2)
    

# create graph neural network models

def create_gcn_model(gcn_n_node_features = 25,
                     gcn_n_hidden = 2,
                     gcn_channels = 25,
                     gcn_activation = tf.keras.activations.relu,
                     gcn_use_bias = True,
                     gcn_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                     gcn_dropout_rates_input_hidden = (0, 0)):
    
    # define parameters
    gcn_dropout_rates_input = gcn_dropout_rates_input_hidden[0]
    gcn_dropout_rates_hidden = gcn_dropout_rates_input_hidden[1]
    gcn_use_batch_norm_input = gcn_use_batch_norm_input_hidden_lasthidden[0]
    gcn_use_batch_norm_hidden = gcn_use_batch_norm_input_hidden_lasthidden[1]
    gcn_use_batch_norm_lasthidden = gcn_use_batch_norm_input_hidden_lasthidden[2]
    
    # define input tensors
    X_mol_graphs = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs = Input(shape=(None, None))
    
    # define GNN convolutional layers
    X = X_mol_graphs
    A = A_mod_mol_graphs
    
    # if wanted, batch norm layer right after input
    if gcn_use_batch_norm_input == True:
        X = BatchNormalization()(X)
    
    # add dropout layer right after input
    X = Dropout(gcn_dropout_rates_input)(X)
    
    # hidden layers
    for h in range(1, gcn_n_hidden + 1):
        X = GCNConv(channels = gcn_channels, activation = gcn_activation, use_bias = gcn_use_bias)([X, A])
        
        if gcn_use_batch_norm_hidden == True and h != gcn_n_hidden:
            X = BatchNormalization()(X)
        
        if gcn_use_batch_norm_lasthidden == True and h == gcn_n_hidden:
            X = BatchNormalization()(X)
        
        X = Dropout(gcn_dropout_rates_hidden)(X)
   
    # define global pooling layer to reduce graph to a single vector via node features
    X = GlobalSumPool()(X)
    
    # define final model
    model = Model(inputs = [X_mol_graphs, A_mod_mol_graphs], outputs = X, name = "gcn_model")
    
    return model
    
    
def create_gcn_mlp_model(gcn_n_node_features = 25,
                         gcn_n_hidden = 2,
                         gcn_channels = 25,
                         gcn_activation = tf.keras.activations.relu,
                         gcn_use_bias = True,
                         gcn_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                         gcn_dropout_rates_input_hidden = (0,0),
                         mlp_architecture = (25, 25, 1),
                         mlp_hidden_activation = tf.keras.activations.relu,
                         mlp_output_activation = tf.keras.activations.sigmoid,
                         mlp_use_bias = True,
                         mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                         mlp_dropout_rates_input_hidden = (0, 0),
                         mlp_hidden_rational_layers = False,
                         mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                         mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                         mlp_rational_parameters_shared_axes = [1]):
    
    # define input tensors
    X_mol_graphs = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs = Input(shape=(None, None))
    
    # define and apply GCN model
    gcn_model = create_gcn_model(gcn_n_node_features = gcn_n_node_features,
                                 gcn_n_hidden = gcn_n_hidden,
                                 gcn_channels = gcn_channels,
                                 gcn_activation = gcn_activation,
                                 gcn_use_bias = gcn_use_bias,
                                 gcn_use_batch_norm_input_hidden_lasthidden = gcn_use_batch_norm_input_hidden_lasthidden,
                                 gcn_dropout_rates_input_hidden = gcn_dropout_rates_input_hidden)
    
    X = gcn_model([X_mol_graphs, A_mod_mol_graphs])
    
    # define and apply MLP model
    mlp_model = create_mlp_model(architecture = mlp_architecture,
                                 hidden_activation = mlp_hidden_activation,
                                 output_activation = mlp_output_activation,
                                 use_bias = mlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = mlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer, 
                                 rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)
    
    X = mlp_model(X)
    
    # define final model
    model = Model(inputs = [X_mol_graphs, A_mod_mol_graphs], outputs = X)
    
    return model

# function to create a siamese MLP MLP model

def create_siamese_mlp_mlp_model(smlp_architecture = (2**10, 100),
                                 smlp_hidden_activation = tf.keras.activations.relu,
                                 smlp_output_activation = tf.keras.activations.sigmoid,
                                 smlp_use_bias = True,
                                 smlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                 smlp_dropout_rates_input_hidden = (0, 0),
                                 smlp_hidden_rational_layers = False,
                                 smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                                 smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                 smlp_rational_parameters_shared_axes = [1],
                                 mlp_architecture = (100, 100, 1),
                                 mlp_hidden_activation = tf.keras.activations.relu,
                                 mlp_output_activation = tf.keras.activations.sigmoid,
                                 mlp_use_bias = True,
                                 mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                 mlp_dropout_rates_input_hidden = (0, 0),
                                 mlp_hidden_rational_layers = False,
                                 mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                                 mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                 mlp_rational_parameters_shared_axes = [1]):
    
    # define input tensors
    X_fcfp_1 = Input(shape=(smlp_architecture[0],))
    X_fcfp_2 = Input(shape=(smlp_architecture[0],))

    
    # define and apply siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer, 
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)
    
    X_emb_1 = smlp_model(X_fcfp_1)
    X_emb_2 = smlp_model(X_fcfp_2)
    
    # combine embeddings to a single vector in a symmetric manner
    X = tf.math.abs(X_emb_1 - X_emb_2)
    
    # define and apply MLP model
    mlp_model = create_mlp_model(architecture = mlp_architecture,
                                 hidden_activation = mlp_hidden_activation,
                                 output_activation = mlp_output_activation,
                                 use_bias = mlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = mlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer, 
                                 rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)
    
    X = mlp_model(X)
    
    # define final model
    model = Model(inputs = [X_fcfp_1, X_fcfp_2], outputs = X)
    
    return (model, smlp_model)

# function to create a siamese MLP cosine similarity model

def create_siamese_mlp_cos_sim_model(smlp_architecture = (2**10, 100),
                                     smlp_hidden_activation = tf.keras.activations.relu,
                                     smlp_output_activation = tf.keras.activations.sigmoid,
                                     smlp_use_bias = True,
                                     smlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                     smlp_dropout_rates_input_hidden = (0, 0),
                                     smlp_hidden_rational_layers = False,
                                     smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                                     smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                     smlp_rational_parameters_shared_axes = [1]):
    
    # define input tensors
    X_fcfp_1 = Input(shape=(smlp_architecture[0],))
    X_fcfp_2 = Input(shape=(smlp_architecture[0],))

    # define and apply siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer, 
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)
    
    X_emb_1 = smlp_model(X_fcfp_1)
    X_emb_2 = smlp_model(X_fcfp_2)
    
    # compute cos(alpha) for angle alpha between both embeddings
    cos_alpha = tf.math.reduce_sum(X_emb_1 * X_emb_2, axis = 1)/(tf.norm(X_emb_1, axis = 1) * tf.norm(X_emb_2, axis = 1))
    
    # define final output via sigmoid function
    sigmoid_cos_alpha = tf.math.sigmoid(cos_alpha)
    
    # define final model
    model = Model(inputs = [X_fcfp_1, X_fcfp_2], outputs = 1 - sigmoid_cos_alpha)
    
    return (model, smlp_model)

# function to create a siamese GCN MLP model

def create_siamese_gcn_mlp_model(gcn_n_node_features = 25,
                                 gcn_n_hidden = 2,
                                 gcn_channels = 100,
                                 gcn_activation = tf.keras.activations.relu,
                                 gcn_use_bias = True,
                                 gcn_use_batch_norm_input_hidden_lasthidden = (False, True, False),
                                 gcn_dropout_rates_input_hidden = (0, 0),
                                 dgcn_architecture = (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                 dgcn_hidden_activation = tf.keras.activations.relu,
                                 dgcn_output_activation = tf.keras.activations.relu,
                                 dgcn_use_bias = True,
                                 dgcn_use_batch_norm_input_hidden_lasthidden = (True, True, True),
                                 dgcn_dropout_rates_input_hidden = (0, 0),
                                 dgcn_hidden_rational_layers = False,
                                 dgcn_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218] ,
                                 dgcn_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                 dgcn_rational_parameters_shared_axes = [1],
                                 mlp_architecture = (100, 100, 100, 1),
                                 mlp_hidden_activation = tf.keras.activations.relu,
                                 mlp_output_activation = tf.keras.activations.sigmoid,
                                 mlp_use_bias = True,
                                 mlp_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                 mlp_dropout_rates_input_hidden = (0, 0),
                                 mlp_hidden_rational_layers = False,
                                 mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218],
                                 mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                 mlp_rational_parameters_shared_axes = [1]):
    
    # define input tensors
    X_mol_graphs_1 = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs_1 = Input(shape=(None, None))
    X_mol_graphs_2 = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs_2 = Input(shape=(None, None))
    
    # define and apply GCN model to vectorise molecular graph arrays
    gcn_model = create_gcn_model(gcn_n_node_features = gcn_n_node_features,
                                 gcn_n_hidden = gcn_n_hidden,
                                 gcn_channels = gcn_channels,
                                 gcn_activation = gcn_activation,
                                 gcn_use_bias = gcn_use_bias,
                                 gcn_use_batch_norm_input_hidden_lasthidden = gcn_use_batch_norm_input_hidden_lasthidden,
                                 gcn_dropout_rates_input_hidden = gcn_dropout_rates_input_hidden)
    
    X_vec_1 = gcn_model([X_mol_graphs_1, A_mod_mol_graphs_1])
    X_vec_2 = gcn_model([X_mol_graphs_2, A_mod_mol_graphs_2])
    
    # define and apply DGCN model (dense layer on top of GCN) to create molecular embeddings
    dgcn_model = create_mlp_model(architecture = dgcn_architecture,
                                  hidden_activation = dgcn_hidden_activation,
                                  output_activation = dgcn_output_activation,
                                  use_bias = dgcn_use_bias,
                                  use_batch_norm_input_hidden_lasthidden = dgcn_use_batch_norm_input_hidden_lasthidden,
                                  dropout_rates_input_hidden = dgcn_dropout_rates_input_hidden,
                                  hidden_rational_layers = dgcn_hidden_rational_layers,
                                  rational_parameters_alpha_initializer = dgcn_rational_parameters_alpha_initializer, 
                                  rational_parameters_beta_initializer = dgcn_rational_parameters_beta_initializer,
                                  rational_parameters_shared_axes = dgcn_rational_parameters_shared_axes)
    
    X_emb_1 = dgcn_model(X_vec_1)
    X_emb_2 = dgcn_model(X_vec_2)
    
    # combine embeddings to a single vector in a symmetric manner
    X = tf.math.abs(X_emb_1 - X_emb_2)
    
    # define and apply MLP model
    mlp_model = create_mlp_model(architecture = mlp_architecture,
                                 hidden_activation = mlp_hidden_activation,
                                 output_activation = mlp_output_activation,
                                 use_bias = mlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = mlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer, 
                                 rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)
    
    X = mlp_model(X)
    
    # define final models
    model = Model(inputs = [X_mol_graphs_1, A_mod_mol_graphs_1, X_mol_graphs_2, A_mod_mol_graphs_2], outputs = X)
    sgcn_model = Model(inputs = [X_mol_graphs_1, A_mod_mol_graphs_1], outputs = X_emb_1)
    
    return (model, sgcn_model)

# function to create a trained siamese MLP MLP model

def create_trained_siamese_mlp_mlp_model(x_smiles_train,
                                         y_train, 
                                         x_smiles_to_fcfp_dict,
                                         X_smiles_mmps,
                                         smlp_architecture = (2**10, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                         smlp_hidden_activation = tf.keras.activations.relu,
                                         smlp_output_activation = tf.keras.activations.linear,
                                         smlp_use_bias = True,
                                         smlp_use_batch_norm_input_hidden_lasthidden = (False, True, True),
                                         smlp_dropout_rates_input_hidden = (0, 0),
                                         smlp_hidden_rational_layers = False,
                                         smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218] ,
                                         smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                         smlp_rational_parameters_shared_axes = [1],
                                         mlp_architecture = (100, 100, 100, 1),
                                         mlp_hidden_activation = tf.keras.activations.relu,
                                         mlp_output_activation = tf.keras.activations.sigmoid,
                                         mlp_use_bias = True,
                                         mlp_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                         mlp_dropout_rates_input_hidden = (0, 0),
                                         mlp_hidden_rational_layers = False,
                                         mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218] ,
                                         mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                         mlp_rational_parameters_shared_axes = [1],
                                         batch_size = 2**9,
                                         epochs = 1,
                                         optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
                                         loss = tf.keras.losses.BinaryCrossentropy(),
                                         performance_metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.BinaryAccuracy()],
                                         verbose = 1):
    
    # get indices for mmps for which both molecules are in training set
    ind_train_mmps = []
    for (k, [smiles_1, smiles_2]) in enumerate(X_smiles_mmps):
        if smiles_1 in x_smiles_train and smiles_2 in x_smiles_train:
            ind_train_mmps.append(k)
    
    # extract mmps which lie in training set
    X_smiles_mmps_train = X_smiles_mmps[ind_train_mmps]
    
    # label the extracted mmps
    y_train_smiles_to_label_dict = dict(list(zip(x_smiles_train, y_train)))
    y_mmps_train = np.array([int(y_train_smiles_to_label_dict[smiles_1]!=y_train_smiles_to_label_dict[smiles_2]) for [smiles_1, smiles_2] in X_smiles_mmps_train])
    
    # construct binary predictor variables X_fcfp_1 and X_fcfp_2 ( = fixed molecular features)
    X_fcfp_1_train = list(range(len(X_smiles_mmps_train)))
    X_fcfp_2_train = list(range(len(X_smiles_mmps_train)))

    for k in range(len(X_smiles_mmps_train)):

        X_fcfp_1_train[k] = x_smiles_to_fcfp_dict[X_smiles_mmps_train[k,0]]
        X_fcfp_2_train[k] = x_smiles_to_fcfp_dict[X_smiles_mmps_train[k,1]]

    X_fcfp_1_train = np.array(X_fcfp_1_train)
    X_fcfp_2_train = np.array(X_fcfp_2_train)
    
    
    # create fresh SMLP model
    (smlp_mlp_model, smlp_model) = create_siamese_mlp_mlp_model(smlp_architecture = smlp_architecture,
                                                                smlp_hidden_activation = smlp_hidden_activation,
                                                                smlp_output_activation = smlp_output_activation,
                                                                smlp_use_bias = smlp_use_bias,
                                                                smlp_use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                                                smlp_dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                                                smlp_hidden_rational_layers = smlp_hidden_rational_layers,
                                                                smlp_rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer, 
                                                                smlp_rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                                                smlp_rational_parameters_shared_axes = smlp_rational_parameters_shared_axes,
                                                                mlp_architecture = mlp_architecture,
                                                                mlp_hidden_activation = mlp_hidden_activation,
                                                                mlp_output_activation = mlp_output_activation,
                                                                mlp_use_bias = mlp_use_bias,
                                                                mlp_use_batch_norm_input_hidden_lasthidden = mlp_use_batch_norm_input_hidden_lasthidden,
                                                                mlp_dropout_rates_input_hidden = mlp_dropout_rates_input_hidden,
                                                                mlp_hidden_rational_layers = mlp_hidden_rational_layers,
                                                                mlp_rational_parameters_alpha_initializer = mlp_rational_parameters_alpha_initializer, 
                                                                mlp_rational_parameters_beta_initializer = mlp_rational_parameters_beta_initializer,
                                                                mlp_rational_parameters_shared_axes = mlp_rational_parameters_shared_axes)

    # compile the model
    smlp_mlp_model.compile(optimizer = optimizer,
                           loss = loss,
                           metrics = performance_metrics)

    # fit the model
    if epochs > 0:
        
        smlp_mlp_model.fit([X_fcfp_1_train, X_fcfp_2_train], 
                            y_mmps_train, 
                            epochs = epochs, 
                            batch_size = batch_size,
                            verbose = verbose)
    
    return (smlp_mlp_model, smlp_model)



# function to create dual task siamese mlp mlp model

def create_dual_task_siamese_mlp_mlp_model(smlp_architecture = (2**10, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                           smlp_hidden_activation = tf.keras.activations.relu,
                                           smlp_output_activation = tf.keras.activations.sigmoid,
                                           smlp_use_bias = True,
                                           smlp_use_batch_norm_input_hidden_lasthidden = (False, True, False),
                                           smlp_dropout_rates_input_hidden = (0, 0),
                                           smlp_hidden_rational_layers = False,
                                           smlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                                           smlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           smlp_rational_parameters_shared_axes = [1],
                                           ac_mlp_architecture = (100, 100, 100, 1),
                                           ac_mlp_hidden_activation = tf.keras.activations.relu,
                                           ac_mlp_output_activation = tf.keras.activations.sigmoid,
                                           ac_mlp_use_bias = True,
                                           ac_mlp_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                           ac_mlp_dropout_rates_input_hidden = (0, 0),
                                           ac_mlp_hidden_rational_layers = False,
                                           ac_mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                                           ac_mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           ac_mlp_rational_parameters_shared_axes = [1],
                                           dir_mlp_architecture = (100, 100, 100, 1),
                                           dir_mlp_hidden_activation = tf.keras.activations.tanh,
                                           dir_mlp_output_activation = tf.keras.activations.sigmoid,
                                           dir_mlp_use_bias = False,
                                           dir_mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                           dir_mlp_dropout_rates_input_hidden = (0, 0),
                                           dir_mlp_hidden_rational_layers = False,
                                           dir_mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                                           dir_mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           dir_mlp_rational_parameters_shared_axes = [1]):
    
    # define input tensors
    X_fcfp_1 = Input(shape=(smlp_architecture[0],))
    X_fcfp_2 = Input(shape=(smlp_architecture[0],))

    # define and apply siamese MLP model to create embeddings for both molecules
    smlp_model = create_mlp_model(architecture = smlp_architecture,
                                 hidden_activation = smlp_hidden_activation,
                                 output_activation = smlp_output_activation,
                                 use_bias = smlp_use_bias,
                                 use_batch_norm_input_hidden_lasthidden = smlp_use_batch_norm_input_hidden_lasthidden,
                                 dropout_rates_input_hidden = smlp_dropout_rates_input_hidden,
                                 hidden_rational_layers = smlp_hidden_rational_layers,
                                 rational_parameters_alpha_initializer = smlp_rational_parameters_alpha_initializer, 
                                 rational_parameters_beta_initializer = smlp_rational_parameters_beta_initializer,
                                 rational_parameters_shared_axes = smlp_rational_parameters_shared_axes)
    
    X_emb_1 = smlp_model(X_fcfp_1)
    X_emb_2 = smlp_model(X_fcfp_2)
    
    # combine embeddings to two new vectors for ac and dir prediction
    X_ac = tf.math.abs(X_emb_1 - X_emb_2)
    X_dir = X_emb_1 - X_emb_2
    
    # define and apply MLP model for ac prediction
    ac_mlp_model = create_mlp_model(architecture = ac_mlp_architecture,
                                    hidden_activation = ac_mlp_hidden_activation,
                                    output_activation = ac_mlp_output_activation,
                                    use_bias = ac_mlp_use_bias,
                                    use_batch_norm_input_hidden_lasthidden = ac_mlp_use_batch_norm_input_hidden_lasthidden,
                                    dropout_rates_input_hidden = ac_mlp_dropout_rates_input_hidden,
                                    hidden_rational_layers = ac_mlp_hidden_rational_layers,
                                    rational_parameters_alpha_initializer = ac_mlp_rational_parameters_alpha_initializer, 
                                    rational_parameters_beta_initializer = ac_mlp_rational_parameters_beta_initializer,
                                    rational_parameters_shared_axes = ac_mlp_rational_parameters_shared_axes,
                                    model_name = "ac_mlp")
    
    X_ac_pred = ac_mlp_model(X_ac)
    
    # define and apply MLP model for dir prediction
    dir_mlp_model = create_mlp_model(architecture = dir_mlp_architecture,
                                     hidden_activation = dir_mlp_hidden_activation,
                                     output_activation = dir_mlp_output_activation,
                                     use_bias = dir_mlp_use_bias,
                                     use_batch_norm_input_hidden_lasthidden = dir_mlp_use_batch_norm_input_hidden_lasthidden,
                                     dropout_rates_input_hidden = dir_mlp_dropout_rates_input_hidden,
                                     hidden_rational_layers = dir_mlp_hidden_rational_layers,
                                     rational_parameters_alpha_initializer = dir_mlp_rational_parameters_alpha_initializer, 
                                     rational_parameters_beta_initializer = dir_mlp_rational_parameters_beta_initializer,
                                     rational_parameters_shared_axes = dir_mlp_rational_parameters_shared_axes,
                                     model_name = "dir_mlp")

    X_dir_pred = dir_mlp_model(X_dir)
    
    # define final model
    model = Model(inputs = [X_fcfp_1, X_fcfp_2], outputs = [X_ac_pred, X_dir_pred])
    
    return (model, smlp_model)

# function to create dual task siamese gcn mlp model

def create_dual_task_siamese_gcn_mlp_model(gcn_n_node_features = 25,
                                           gcn_n_hidden = 2,
                                           gcn_channels = 100,
                                           gcn_activation = tf.keras.activations.relu,
                                           gcn_use_bias = True,
                                           gcn_use_batch_norm_input_hidden_lasthidden = (False, True, False),
                                           gcn_dropout_rates_input_hidden = (0, 0),
                                           dgcn_architecture = (100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100),
                                           dgcn_hidden_activation = tf.keras.activations.relu,
                                           dgcn_output_activation = tf.keras.activations.linear,
                                           dgcn_use_bias = True,
                                           dgcn_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                           dgcn_dropout_rates_input_hidden = (0, 0),
                                           dgcn_hidden_rational_layers = False,
                                           dgcn_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218] ,
                                           dgcn_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           dgcn_rational_parameters_shared_axes = [1],
                                           ac_mlp_architecture = (100, 100, 100, 1),
                                           ac_mlp_hidden_activation = tf.keras.activations.relu,
                                           ac_mlp_output_activation = tf.keras.activations.sigmoid,
                                           ac_mlp_use_bias = True,
                                           ac_mlp_use_batch_norm_input_hidden_lasthidden = (True, True, False),
                                           ac_mlp_dropout_rates_input_hidden = (0, 0),
                                           ac_mlp_hidden_rational_layers = False,
                                           ac_mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                                           ac_mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           ac_mlp_rational_parameters_shared_axes = [1],
                                           dir_mlp_architecture = (100, 100, 100, 1),
                                           dir_mlp_hidden_activation = tf.keras.activations.tanh,
                                           dir_mlp_output_activation = tf.keras.activations.sigmoid,
                                           dir_mlp_use_bias = False,
                                           dir_mlp_use_batch_norm_input_hidden_lasthidden = (False, False, False),
                                           dir_mlp_dropout_rates_input_hidden = (0, 0),
                                           dir_mlp_hidden_rational_layers = False,
                                           dir_mlp_rational_parameters_alpha_initializer = [1.1915, 1.5957, 0.5, 0.0218], 
                                           dir_mlp_rational_parameters_beta_initializer = [2.383, 0.0, 1.0],
                                           dir_mlp_rational_parameters_shared_axes = [1]):
    
    # define input tensors
    X_mol_graphs_1 = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs_1 = Input(shape=(None, None))
    X_mol_graphs_2 = Input(shape=(None, gcn_n_node_features))
    A_mod_mol_graphs_2 = Input(shape=(None, None))
    
    # define and apply GCN model to vectorise molecular graph arrays
    gcn_model = create_gcn_model(gcn_n_node_features = gcn_n_node_features,
                                 gcn_n_hidden = gcn_n_hidden,
                                 gcn_channels = gcn_channels,
                                 gcn_activation = gcn_activation,
                                 gcn_use_bias = gcn_use_bias,
                                 gcn_use_batch_norm_input_hidden_lasthidden = gcn_use_batch_norm_input_hidden_lasthidden,
                                 gcn_dropout_rates_input_hidden = gcn_dropout_rates_input_hidden)
    
    X_vec_1 = gcn_model([X_mol_graphs_1, A_mod_mol_graphs_1])
    X_vec_2 = gcn_model([X_mol_graphs_2, A_mod_mol_graphs_2])
    
    # define and apply DGCN model (dense layer on top of GCN) to create molecular embeddings
    dgcn_model = create_mlp_model(architecture = dgcn_architecture,
                                  hidden_activation = dgcn_hidden_activation,
                                  output_activation = dgcn_output_activation,
                                  use_bias = dgcn_use_bias,
                                  use_batch_norm_input_hidden_lasthidden = dgcn_use_batch_norm_input_hidden_lasthidden,
                                  dropout_rates_input_hidden = dgcn_dropout_rates_input_hidden,
                                  hidden_rational_layers = dgcn_hidden_rational_layers,
                                  rational_parameters_alpha_initializer = dgcn_rational_parameters_alpha_initializer, 
                                  rational_parameters_beta_initializer = dgcn_rational_parameters_beta_initializer,
                                  rational_parameters_shared_axes = dgcn_rational_parameters_shared_axes)
    
    X_emb_1 = dgcn_model(X_vec_1)
    X_emb_2 = dgcn_model(X_vec_2)
    
    # combine embeddings to two new vectors for ac and dir prediction
    X_ac = tf.math.abs(X_emb_1 - X_emb_2)
    X_dir = X_emb_1 - X_emb_2
    
    # define and apply MLP model for ac prediction
    ac_mlp_model = create_mlp_model(architecture = ac_mlp_architecture,
                                    hidden_activation = ac_mlp_hidden_activation,
                                    output_activation = ac_mlp_output_activation,
                                    use_bias = ac_mlp_use_bias,
                                    use_batch_norm_input_hidden_lasthidden = ac_mlp_use_batch_norm_input_hidden_lasthidden,
                                    dropout_rates_input_hidden = ac_mlp_dropout_rates_input_hidden,
                                    hidden_rational_layers = ac_mlp_hidden_rational_layers,
                                    rational_parameters_alpha_initializer = ac_mlp_rational_parameters_alpha_initializer, 
                                    rational_parameters_beta_initializer = ac_mlp_rational_parameters_beta_initializer,
                                    rational_parameters_shared_axes = ac_mlp_rational_parameters_shared_axes,
                                    model_name = "ac_mlp")
    
    X_ac_pred = ac_mlp_model(X_ac)
    
    # define and apply MLP model for dir prediction
    dir_mlp_model = create_mlp_model(architecture = dir_mlp_architecture,
                                     hidden_activation = dir_mlp_hidden_activation,
                                     output_activation = dir_mlp_output_activation,
                                     use_bias = dir_mlp_use_bias,
                                     use_batch_norm_input_hidden_lasthidden = dir_mlp_use_batch_norm_input_hidden_lasthidden,
                                     dropout_rates_input_hidden = dir_mlp_dropout_rates_input_hidden,
                                     hidden_rational_layers = dir_mlp_hidden_rational_layers,
                                     rational_parameters_alpha_initializer = dir_mlp_rational_parameters_alpha_initializer, 
                                     rational_parameters_beta_initializer = dir_mlp_rational_parameters_beta_initializer,
                                     rational_parameters_shared_axes = dir_mlp_rational_parameters_shared_axes,
                                     model_name = "dir_mlp")

    X_dir_pred = dir_mlp_model(X_dir)
    
    # define final model
    model = Model(inputs = [X_mol_graphs_1, A_mod_mol_graphs_1, X_mol_graphs_2, A_mod_mol_graphs_2], outputs = [X_ac_pred, X_dir_pred])
    sgcn_model = Model(inputs = [X_mol_graphs_1, A_mod_mol_graphs_1], outputs = X_emb_1)
    
    return (model, sgcn_model)