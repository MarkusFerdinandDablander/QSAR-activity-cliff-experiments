import numpy as np
import pandas as pd
import os
import glob
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, mean_absolute_error, mean_squared_error, median_absolute_error, max_error, r2_score



def transform_probs_to_labels(y_pred_proba_pos, cutoff_value = 0.5):
    """
    Transforms an array of probabilities into a binary array of 0s and 1s.
    """

    y_pred_proba_pos = np.array(y_pred_proba_pos)
    y_pred = np.copy(y_pred_proba_pos)

    y_pred[y_pred > cutoff_value] = 1
    y_pred[y_pred <= cutoff_value] = 0 # per default, sklearn random forest classifiers map a probability of 0.5 to class 0

    return y_pred



def binary_classification_scores(y_true, y_pred_proba_pos, display_results = False):
    """
    For a binary classification task with true labels y_true and predicted probabilities y_pred_proba_pos, this function computes the following metrics: "AUROC", "Accuracy", "Balanced Accuracy", "F1-Score", "MCC", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value", "Test Cases", "Negative Test Cases", "Positive Test Cases".
    """
    
    if len(y_true) == 0:
        
        # collect scores
        scores_array = np.array([float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), 0, 0, 0])
        scores_array_2d = np.reshape(scores_array, (1, len(scores_array)))
        columns = ["AUROC", "Accuracy", "Balanced Accuracy", "F1-Score", "MCC", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value", "Test Cases", "Negative Test Cases", "Positive Test Cases"]
        scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

        # display scores
        if display_results == True:
            display(scores_df)

        return scores_df
        
    else:

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
        columns = ["AUROC", "Accuracy", "Balanced Accuracy", "F1-Score", "MCC", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value", "Test Cases", "Negative Test Cases", "Positive Test Cases"]
        scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

        # display scores
        if display_results == True:
            display(scores_df)

        return scores_df



def regression_scores(y_true, y_pred, display_results = False):
    """
    For a regression task with true labels y_true and predicted labels y_pred, this function computes the following metrics: "MAE", "MedAE", "RMSE", "MaxAE", "MSE", "Pearson's r", "R^2", "Test Cases".
    """
    
    if len(y_true) == 0:
        
        # collect scores
        scores_array = np.array([float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN"), 0])
        scores_array_2d = np.reshape(scores_array, (1, len(scores_array)))
        columns = ["MAE", "MedAE", "RMSE", "MaxAE", "MSE", "Pearson's r", "R^2", "Test Cases"]
        scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

        # display scores
        if display_results == True:
            display(scores_df)

        return scores_df
        
    else:

        # prepare variables
        y_true = list(y_true)
        y_pred = list(y_pred)
        n_test_cases = len(y_true)

        # compute scores

        # mean absolute error
        mae = mean_absolute_error(y_true, y_pred)

        # median absolute error
        medae = median_absolute_error(y_true, y_pred)

        # root mean squared error
        rmse = mean_squared_error(y_true, y_pred, squared = False)

        # max error
        maxe = max_error(y_true, y_pred)

        # mean squared error
        mse = mean_squared_error(y_true, y_pred, squared = True)

        # pearson correlation coefficient
        pearson_corr = pearsonr(y_true, y_pred)[0]
        
        # R2 coefficient of determination
        r2_coeff = r2_score(y_true, y_pred)

        # collect scores
        scores_array = np.array([mae, medae, rmse, maxe, mse, pearson_corr, r2_coeff, n_test_cases])
        scores_array_2d = np.reshape(scores_array, (1, len(scores_array)))
        columns = ["MAE", "MedAE", "RMSE", "MaxAE", "MSE", "Pearson's r", "R^2", "Test Cases"]
        scores_df = pd.DataFrame(data = scores_array_2d, index = ["Scores:"], columns = columns)

        # display scores
        if display_results == True:
            display(scores_df)

        return scores_df



def summarise_scores_from_cubic_scores_array(scores_array, 
                                             display_results = False, 
                                             decimals = 2, 
                                             task_type = "regression"):
    """
    Summarise performance metric results for a k-fold cross validation scheme repeated with m random seeds. The results for this experiment are saved in scores_array which has dimensions (m, k, number_of_metrics). The output specifies the average results for each metric over the m*k trials.
    """
    
    avgs = np.around(np.nanmean(np.nanmean(scores_array, axis = 1), axis = 0), decimals = decimals)
    stds = np.around(np.nanmean(np.nanstd(scores_array, axis = 1), axis = 0), decimals = decimals)
    
    summarised_scores_array = np.array([avgs, stds])
    
    if task_type == "regression":
        columns = ["MAE", "MedAE", "RMSE", "MaxAE", "MSE", "Pearson's r", "R^2", "Test Cases"]
    elif task_type == "classification":
        columns = ["AUROC", "Accuracy", "Balanced Accuracy", "F1-Score", "MCC", "Sensitivity", "Specificity", 
               "Precision", "Negative Predictive Value", "Test Cases", "Negative Test Cases", "Positive Test Cases"]
    
    summarised_scores_df = pd.DataFrame(data = summarised_scores_array, index = ["Avg.", "Std."], columns = columns)

    if display_results == True:
        display(summarised_scores_df)

    return summarised_scores_df



def display_experimental_results(filepath, decimals = 2):
    """
    Print out average experimental results over k-fold cross validation with m random seeds for a chosen QSAR model and data set.
    """

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



def create_scores_dict(k_splits, m_reps, len_y, n_regr_metrics = 8, n_class_metrics = 12):
    """
    Create a dictionary to save the performance results for QSAR-, AC- and PD-prediction in a k-fold cross validation scheme repeated with m random seeds.
    """
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
    """
    Store the performance results of the trial with indices (m, k) in scores_dict.
    """
    
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

    
    
def delete_all_files_in_folder(filepath):
    files = glob.glob(filepath + "*")
    for f in files:
        os.remove(f)

    
        
def save_qsar_ac_pd_results(filepath, scores_dict):
    """
    Save cubic arrays of shape (m, k, n_metrics) that contain the performance results over a k-fold cross validation scheme with m random seends.
    """
    
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
    """
    Save experimental settings contaning information about the used molecular representation, regression technique, optimised hyperparameters, data set and evaluation scheme.
    """
    
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
        
        
        









