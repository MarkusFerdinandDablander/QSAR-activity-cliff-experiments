# import packages

# general tools
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF as statsmodels_ecdf
from scipy.stats import pearsonr

# sklearn
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, mean_absolute_error, mean_squared_error, median_absolute_error, max_error, mean_absolute_percentage_error, r2_score





# naive classifiers

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
















# scoring functions for binary classification

def transform_probs_to_labels(y_pred_proba_pos, cutoff_value = 0.5):

    y_pred_proba_pos = np.array(y_pred_proba_pos)
    y_pred = np.copy(y_pred_proba_pos)

    y_pred[y_pred > cutoff_value] = 1
    y_pred[y_pred <= cutoff_value] = 0 # per default, sklearn random forest classifiers map a probability of 0.5 to class 0

    return y_pred


def binary_classification_scores(y_true, y_pred_proba_pos, display_results = False):
    
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



def summarise_binary_classification_scores(scores_array_2d, display_results = False, decimals = 4):
    """
    scores_array_2d.shape = (number_of_trials, number_of_scoring_functions)
    """

    avgs = np.around(np.nanmean(scores_array_2d, axis = 0), decimals = decimals)
    stds = np.around(np.nanstd(scores_array_2d, axis = 0), decimals = decimals)

    summarised_scores_array = np.array([avgs, stds])
    columns = ["AUROC", "Accuracy", "Balanced Accuracy", "F1-Score", "MCC", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value", "Test Cases", "Negative Test Cases", "Positive Test Cases"]
    summarised_scores_df = pd.DataFrame(data = summarised_scores_array, index = ["Avg.", "Std."], columns = columns)

    if display_results == True:
        display(summarised_scores_df)
        
    return summarised_scores_df




def regression_scores(y_true, y_pred, display_results = False):
    
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


def summarise_regression_scores(scores_array_2d, display_results = False, decimals = 2):
    """
    scores_array_2d.shape = (number_of_trials, number_of_scoring_functions)
    """

    avgs = np.around(np.nanmean(scores_array_2d, axis = 0), decimals = decimals)
    stds = np.around(np.nanstd(scores_array_2d, axis = 0), decimals = decimals)

    summarised_scores_array = np.array([avgs, stds])
    columns = ["MAE", "MedAE", "RMSE", "MaxAE", "MSE", "Pearson's r", "R^2", "Test Cases"]
    summarised_scores_df = pd.DataFrame(data = summarised_scores_array, index = ["Avg.", "Std."], columns = columns)

    if display_results == True:
        display(summarised_scores_df)

    return summarised_scores_df



# summarise scores from a cubic array resulting from a repeated cross validation experiment

def summarise_scores_from_cubic_scores_array(scores_array, display_results = False, decimals = 2, task_type = "regression"):
    
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




# oversampling and undersampling function for imbalanced binary classification problems

def oversample(X,y, random_seed = 42):
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

    np.random.seed(random_seed)
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

