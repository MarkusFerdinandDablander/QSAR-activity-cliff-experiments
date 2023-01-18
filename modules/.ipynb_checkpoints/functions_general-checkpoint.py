# import packages

# general tools
import numpy as np
import os
import glob

# my own functions
from .functions_machine_learning import regression_scores, binary_classification_scores, summarise_scores_from_cubic_scores_array






# general functions

def save_X_smiles_as_csv(X_smiles, location):

    indices = np.reshape(np.arange(0, len(X_smiles)), (-1,1))
    data = np.concatenate((X_smiles, indices), axis = 1)
    np.savetxt(location, data, delimiter = ",", fmt = "%s")


def save_Y_as_csv(Y, location):

    indices = np.reshape(np.arange(0, len(Y)), (-1,1))
    data = np.concatenate((Y, indices), axis = 1)
    np.savetxt(location, data, delimiter = ",")

    
def delete_all_files_in_folder(filepath):
    files = glob.glob(filepath + "*")
    for f in files:
        os.remove(f)
        
        
# functions to create neural network architecture tuples

def all_combs_list(l_1, l_2):
    
    all_combs = []
    
    for a in l_1:
        for b in l_2:
            all_combs.append((a,b))
   
    return all_combs


def arch(input_dim = 200, output_dim = 1, hidden_width = 300, hidden_depth = 10):
    
    hidden_layer_list = [hidden_width for h in range(hidden_depth)]
    arch = tuple([input_dim] + hidden_layer_list + [output_dim])
    
    return arch



# functions for experimental evaluation

def create_scores_dict(k_splits, m_reps, len_y, n_regr_metrics = 8, n_class_metrics = 12):
    scores_dict = {}
    scores_dict["y_pred_array"] = np.zeros((m_reps, k_splits, len_y))
    scores_dict["qsar_train"] = np.zeros((m_reps, k_splits, n_regr_metrics)) # we use 8 performance metrics to evaluate the regression tasks
    scores_dict["qsar_test"] = np.zeros((m_reps, k_splits, n_regr_metrics))
    
    for task in ["ac_train", "ac_inter", "ac_test", "ac_cores", "pd_train", "pd_inter", "pd_test", "pd_cores", 
             "pd_ac_pos_train", "pd_ac_pos_inter", "pd_ac_pos_test", "pd_ac_pos_cores"]:
        scores_dict[task] = np.zeros((m_reps, k_splits, n_class_metrics)) # we use 12 performance metrics to evaluate the classification tasks
    
    return scores_dict


def create_and_store_qsar_ac_pd_results(scores_dict,
                                        x_smiles,
                                        X_smiles_mmps,
                                        y,
                                        y_mmps,
                                        y_mmps_pd,
                                        y_pred,
                                        data_split_dictionary,
                                        m,
                                        k,
                                        ac_threshold = 1.5):
    
    # extract indices for this data split
    (ind_train_mols, 
     ind_test_mols,
     ind_train_mmps,
     ind_inter_mmps,
     ind_test_mmps,
     ind_test_seen_cores_mmps,
     ind_cores_mmps) = data_split_dictionary[(m,k)]
    
    # create qsar-, ac-, and pd-predictions
    y_train = y[ind_train_mols]
    y_test = y[ind_test_mols]
    
    x_smiles_to_y_dict = dict(list(zip(x_smiles, y)))
    x_smiles_to_y_pred_dict = dict(list(zip(x_smiles, y_pred)))
    x_smiles_to_y_hybrid_dict = dict([(smiles, x_smiles_to_y_dict[smiles]) if smiles in x_smiles[ind_train_mols]
                                     else (smiles, x_smiles_to_y_pred_dict[smiles]) for smiles in x_smiles])
    
    y_mmps_quant_pred = np.array([x_smiles_to_y_pred_dict[smiles_1] - x_smiles_to_y_pred_dict[smiles_2]
                                      for [smiles_1, smiles_2] in X_smiles_mmps])
    y_mmps_quant_hybrid = np.array([x_smiles_to_y_hybrid_dict[smiles_1] - x_smiles_to_y_hybrid_dict[smiles_2] 
                                      for [smiles_1, smiles_2] in X_smiles_mmps])

    y_mmps_pred = np.array([2*np.arctan(abs_pot_diff/ac_threshold)/np.pi for abs_pot_diff in np.abs(y_mmps_quant_pred)])
    y_mmps_hybrid = np.array([2*np.arctan(abs_pot_diff/ac_threshold)/np.pi for abs_pot_diff in np.abs(y_mmps_quant_hybrid)])
    y_mmps_pd_pred = np.array([1/(1 + np.exp(pot_diff)) for pot_diff in y_mmps_quant_pred])
    y_mmps_pd_hybrid = np.array([1/(1 + np.exp(pot_diff)) for pot_diff in y_mmps_quant_hybrid])
    
    # compute and store performance results in scores_dict
    scores_dict["y_pred_array"][m,k] = y_pred
    scores_dict["qsar_train"][m,k] = regression_scores(y_train, y_pred[ind_train_mols]).values
    scores_dict["qsar_test"][m,k] = regression_scores(y_test, y_pred[ind_test_mols]).values
        
    scores_dict["ac_train"][m,k] = binary_classification_scores(y_mmps[ind_train_mmps], y_mmps_pred[ind_train_mmps]).values
    scores_dict["ac_inter"][m,k] = binary_classification_scores(y_mmps[ind_inter_mmps], y_mmps_hybrid[ind_inter_mmps]).values
    scores_dict["ac_test"][m,k] = binary_classification_scores(y_mmps[ind_test_mmps], y_mmps_pred[ind_test_mmps]).values
    scores_dict["ac_cores"][m,k] = binary_classification_scores(y_mmps[ind_cores_mmps], y_mmps_pred[ind_cores_mmps]).values
    
    scores_dict["pd_train"][m,k] = binary_classification_scores(y_mmps_pd[ind_train_mmps], y_mmps_pd_pred[ind_train_mmps]).values
    scores_dict["pd_inter"][m,k] = binary_classification_scores(y_mmps_pd[ind_inter_mmps], y_mmps_pd_hybrid[ind_inter_mmps]).values
    scores_dict["pd_test"][m,k] = binary_classification_scores(y_mmps_pd[ind_test_mmps], y_mmps_pd_pred[ind_test_mmps]).values
    scores_dict["pd_cores"][m,k] = binary_classification_scores(y_mmps_pd[ind_cores_mmps], y_mmps_pd_pred[ind_cores_mmps]).values
    
    scores_dict["pd_ac_pos_train"][m,k] = binary_classification_scores(y_mmps_pd[ind_train_mmps][y_mmps_pred[ind_train_mmps]>0.5], 
                                                                  y_mmps_pd_pred[ind_train_mmps][y_mmps_pred[ind_train_mmps]>0.5]).values
    scores_dict["pd_ac_pos_inter"][m,k] = binary_classification_scores(y_mmps_pd[ind_inter_mmps][y_mmps_hybrid[ind_inter_mmps]>0.5], 
                                                                 y_mmps_pd_hybrid[ind_inter_mmps][y_mmps_hybrid[ind_inter_mmps]>0.5]).values
    scores_dict["pd_ac_pos_test"][m,k] = binary_classification_scores(y_mmps_pd[ind_test_mmps][y_mmps_pred[ind_test_mmps]>0.5], 
                                                                 y_mmps_pd_pred[ind_test_mmps][y_mmps_pred[ind_test_mmps]>0.5]).values
    scores_dict["pd_ac_pos_cores"][m,k] = binary_classification_scores(y_mmps_pd[ind_cores_mmps][y_mmps_pred[ind_cores_mmps]>0.5], 
                                                                      y_mmps_pd_pred[ind_cores_mmps][y_mmps_pred[ind_cores_mmps]>0.5]).values

    
    
    
def save_qsar_ac_pd_results(filepath, scores_dict):
    
    delete_all_files_in_folder(filepath)

    np.save(filepath + "y_pred_array.npy", scores_dict["y_pred_array"])

    np.save(filepath + "scores_qsar_train.npy", scores_dict["qsar_train"])
    np.save(filepath + "scores_qsar_test.npy", scores_dict["qsar_test"])

    np.save(filepath + "scores_ac_train.npy", scores_dict["ac_train"])
    np.save(filepath + "scores_ac_inter.npy", scores_dict["ac_inter"])
    np.save(filepath + "scores_ac_test.npy", scores_dict["ac_test"])
    np.save(filepath + "scores_ac_cores.npy", scores_dict["ac_cores"])

    np.save(filepath + "scores_pd_train.npy", scores_dict["pd_train"])
    np.save(filepath + "scores_pd_inter.npy", scores_dict["pd_inter"])
    np.save(filepath + "scores_pd_test.npy", scores_dict["pd_test"])
    np.save(filepath + "scores_pd_cores.npy", scores_dict["pd_cores"])

    np.save(filepath + "scores_pd_ac_pos_train.npy", scores_dict["pd_ac_pos_train"])
    np.save(filepath + "scores_pd_ac_pos_inter.npy", scores_dict["pd_ac_pos_inter"])
    np.save(filepath + "scores_pd_ac_pos_test.npy", scores_dict["pd_ac_pos_test"])
    np.save(filepath + "scores_pd_ac_pos_cores.npy", scores_dict["pd_ac_pos_cores"])
    
def save_experimental_settings(filepath, settings_dict):
    
    with open(filepath + 'experimental_settings.txt', 'w') as f:
        f.write("Experimental Settings: \n \n")

        f.write("target_name = " + str(settings_dict["target_name"]) + "\n")
        f.write("n_molecules = " + str(settings_dict["n_molecules"]) + "\n")
        f.write("n_mmps = " + str(settings_dict["n_mmps"]) + "\n")
        f.write("method_name = " + str(settings_dict["method_name"]) + "\n \n")

        f.write("k_splits = " + str(settings_dict["k_splits"]) + "\n")
        f.write("m_reps = " + str(settings_dict["m_reps"]) + "\n")
        f.write("random_state_cv = " + str(settings_dict["random_state_cv"]) + "\n \n")
        
        if str(settings_dict["method_name"]) == "ecfp_rf" or str(settings_dict["method_name"]) == "ecfp_knn":
            
            f.write("radius = " + str(settings_dict["radius"]) + "\n")
            f.write("bitstring_length = " + str(settings_dict["bitstring_length"]) + "\n")
            f.write("use_features = " + str(settings_dict["use_features"]) + "\n")
            f.write("use_chirality = " + str(settings_dict["use_chirality"]) + "\n \n")

            f.write("j_splits = " + str(settings_dict["j_splits"]) + "\n")
            f.write("h_iters = " + str(settings_dict["h_iters"]) + "\n")
            f.write("random_search_scoring = " + str(settings_dict["random_search_scoring"]) + "\n")
            f.write("random_search_verbose = " + str(settings_dict["random_search_verbose"]) + "\n")
            f.write("random_search_random_state = " + str(settings_dict["random_search_random_state"]) + "\n")
            f.write("random_search_n_jobs = " + str(settings_dict["random_search_n_jobs"]) + "\n")
            f.write("hyperparameter_grid = " + str(settings_dict["hyperparameter_grid"]) + "\n \n")

        elif str(settings_dict["method_name"]) == "ecfp_mlp":
            
            f.write("radius = " + str(settings_dict["radius"]) + "\n")
            f.write("bitstring_length = " + str(settings_dict["bitstring_length"]) + "\n")
            f.write("use_features = " + str(settings_dict["use_features"]) + "\n")
            f.write("use_chirality = " + str(settings_dict["use_chirality"]) + "\n \n")

            f.write("optuna_options = " + str(settings_dict["optuna_options"]) + "\n \n")
            f.write("mlp_hyperparameter_grid = " + str(settings_dict["mlp_hyperparameter_grid"]) + "\n \n")
            f.write("train_hyperparameter_grid = " + str(settings_dict["train_hyperparameter_grid"]) + "\n \n")
        
        elif str(settings_dict["method_name"]) == "pdv_rf" or str(settings_dict["method_name"]) == "pdv_knn":
            
            f.write("descriptor_list = " + str(settings_dict["descriptor_list"]) + "\n \n")

            f.write("j_splits = " + str(settings_dict["j_splits"]) + "\n")
            f.write("h_iters = " + str(settings_dict["h_iters"]) + "\n")
            f.write("random_search_scoring = " + str(settings_dict["random_search_scoring"]) + "\n")
            f.write("random_search_verbose = " + str(settings_dict["random_search_verbose"]) + "\n")
            f.write("random_search_random_state = " + str(settings_dict["random_search_random_state"]) + "\n")
            f.write("random_search_n_jobs = " + str(settings_dict["random_search_n_jobs"]) + "\n")
            f.write("hyperparameter_grid = " + str(settings_dict["hyperparameter_grid"]) + "\n \n")
            
        elif str(settings_dict["method_name"]) == "pdv_mlp":
            
            f.write("descriptor_list = " + str(settings_dict["descriptor_list"]) + "\n \n")
    
            f.write("optuna_options = " + str(settings_dict["optuna_options"]) + "\n \n")
            f.write("mlp_hyperparameter_grid = " + str(settings_dict["mlp_hyperparameter_grid"]) + "\n \n")
            f.write("train_hyperparameter_grid = " + str(settings_dict["train_hyperparameter_grid"]) + "\n \n")
            
        elif str(settings_dict["method_name"]) == "gin_rf" or str(settings_dict["method_name"]) == "gin_knn":
            
            f.write("optuna_options = " + str(settings_dict["optuna_options"]) + "\n \n")
            f.write("mlp_hyperparameter_grid = " + str(settings_dict["mlp_hyperparameter_grid"]) + "\n \n")
            f.write("gin_hyperparameter_grid = " + str(settings_dict["gin_hyperparameter_grid"]) + "\n \n")
            f.write("train_hyperparameter_grid = " + str(settings_dict["train_hyperparameter_grid"]) + "\n \n")

            f.write("j_splits = " + str(settings_dict["j_splits"]) + "\n")
            f.write("h_iters = " + str(settings_dict["h_iters"]) + "\n")
            f.write("random_search_scoring = " + str(settings_dict["random_search_scoring"]) + "\n")
            f.write("random_search_verbose = " + str(settings_dict["random_search_verbose"]) + "\n")
            f.write("random_search_random_state = " + str(settings_dict["random_search_random_state"]) + "\n")
            f.write("random_search_n_jobs = " + str(settings_dict["random_search_n_jobs"]) + "\n")
            f.write("hyperparameter_grid = " + str(settings_dict["hyperparameter_grid"]) + "\n \n")
            
        elif str(settings_dict["method_name"]) == "gin_mlp":
            
            f.write("optuna_options = " + str(settings_dict["optuna_options"]) + "\n \n")
            f.write("mlp_hyperparameter_grid = " + str(settings_dict["mlp_hyperparameter_grid"]) + "\n \n")
            f.write("gin_hyperparameter_grid = " + str(settings_dict["gin_hyperparameter_grid"]) + "\n \n")
            f.write("train_hyperparameter_grid = " + str(settings_dict["train_hyperparameter_grid"]) + "\n \n")
        
        f.write("runtime = " + str(settings_dict["runtime"]) + "\n \n")
        
        
def display_experimental_results(filepath, decimals = 2):

    with open(filepath + 'experimental_settings.txt') as f:
        print(f.read())

    for scores in ["scores_qsar_train", "scores_qsar_test", 
                   "scores_ac_train", "scores_ac_inter", "scores_ac_test", "scores_ac_cores", 
                   "scores_pd_train", "scores_pd_inter", "scores_pd_test", "scores_pd_cores",
                   "scores_pd_ac_pos_train", "scores_pd_ac_pos_inter", "scores_pd_ac_pos_test", "scores_pd_ac_pos_cores"]:

        if scores.startswith("scores_qsar"):
            task_type = "regression"
        else:
            task_type = "classification"

        print(scores, "\n")
        summarise_scores_from_cubic_scores_array(np.load(filepath + scores + ".npy"), 
                                                 display_results = True, 
                                                 decimals = decimals, 
                                                 task_type = task_type);
        print("\n \n")