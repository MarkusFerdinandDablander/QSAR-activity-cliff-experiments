import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D



def visualise_results(target,
                      task_x,
                      metric_x,
                      task_y,
                      metric_y,
                      decimals_mean = 6,
                      decimals_std = 6,
                      plot_legend = True,
                      legend_loc = "lower left",
                      plot_title = True,
                      plot_x_label = True,
                      plot_y_label = True,
                      plot_x_ticks = True,
                      plot_y_ticks = True,
                      plot_error_bars = True,
                      x_tick_stepsize = "auto",
                      y_tick_stepsize = "auto",
                      xlim = None,
                      ylim = None,
                      size = 12, 
                      linear_regression = False, 
                      filepath_to_save = "scatter.svg"):
    """
    Visualise performance results for all nine tested QSAR models for a chosen target data set in the form of a scatterplot. Each axis can be chosen to correspond to a task and associated performance metric. 
    
    Example inputs (plot AC-classification MCC against QSAR-prediction MAE in ChEMBL dopamine D2 data set): 
    
    target = "chembl_dopamine_d2"
    task_x = "ac_test",
    metric_x = "MCC",
    task_y = "qsar_test",
    metric_y = "MAE",
    """
    
    # create lists and dictionaries with experimental keys
    target_list = ["chembl_dopamine_d2", "chembl_factor_xa", "postera_sars_cov_2_mpro"]
    mol_repr_list = ["ecfp", "pdv", "gin"]
    regr_type_list = ["rf", "knn", "mlp"]
    task_list = ["qsar_train", 
                 "qsar_test", 
                 "ac_train", 
                 "ac_inter", 
                 "ac_test", 
                 "ac_cores",
                 "pd_train", 
                 "pd_inter", 
                 "pd_test", 
                 "pd_cores", 
                 "pd_ac_pos_train", 
                 "pd_ac_pos_inter", 
                 "pd_ac_pos_test", 
                 "pd_ac_pos_cores"]

    metric_list_regr = ["MAE", "MedAE", "RMSE", "MaxAE", "MSE", "Pearson's r", "R^2", "Test Cases"]
    metric_list_class = ["AUROC", "Accuracy", "Balanced Accuracy", "F1-Score", "MCC", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value", "Test Cases", "Negative Test Cases", "Positive Test Cases"]

    task_name_dict = {"qsar_train" : r"$\mathcal{D}_{\rm train}$ (QSAR-Prediction)",
                      "qsar_test"  : r"$\mathcal{D}_{\rm test}$ (QSAR-Prediction)",
                      "ac_train" : r"$\mathcal{M}_{\rm train}$ (AC-Classification)",
                      "ac_inter" : r"$\mathcal{M}_{\rm inter}$ (AC-Classification)",
                      "ac_test" : r"$\mathcal{M}_{\rm test}$ (AC-Classification)",
                      "ac_cores" : r"$\mathcal{M}_{\rm cores}$ (AC-Classification)",
                      "pd_train" : r"$\mathcal{M}_{\rm train}$ (PD-Classification)",
                      "pd_inter" : r"$\mathcal{M}_{\rm inter}$ (PD-Classification)",
                      "pd_test" : r"$\mathcal{M}_{\rm test}$ (PD-Classification)",
                      "pd_cores" : r"$\mathcal{M}_{\rm cores}$ (PD-Classification)",
                      "pd_ac_pos_train" : r"$\mathcal{M}_{\rm train}$ (PD-Classification for Predicted ACs)",
                      "pd_ac_pos_inter" : r"$\mathcal{M}_{\rm inter}$ (PD-Classification for Predicted ACs)",
                      "pd_ac_pos_test" : r"$\mathcal{M}_{\rm test}$ (PD-Classification for Predicted ACs)",
                      "pd_ac_pos_cores" : r"$\mathcal{M}_{\rm cores}$ (PD-Classification for Predicted ACs)"}

    # create data dictionary which maps experimental keys of the form (this_target, mol_repr, regr_type, task, metric) to data matrices
    A_dict = {}

    for this_target in target_list:
        for mol_repr in mol_repr_list:
            for regr_type in regr_type_list:
                for task in task_list:

                    A_3d = np.load("results/" + this_target + "/" + mol_repr + "_" + regr_type + "/" + "scores_" + task + ".npy")

                    if task in ["qsar_train", "qsar_test"]:
                        for (k, metric) in enumerate(metric_list_regr):
                            A_dict[(this_target, mol_repr, regr_type, task, metric)] = A_3d[:,:,k]
                    else:
                        for (k, metric) in enumerate(metric_list_class):
                            A_dict[(this_target, mol_repr, regr_type, task, metric)] = A_3d[:,:,k]
    
    # preallocate pandas dataframe to collect means and STDs of experimental results for all nine QSAR models
    df = pd.DataFrame(columns = ["x_mean", "x_std", "y_mean", "y_std", "mol_repr", "regr_type"])

    # extract means and stds of experimental results
    for mol_repr in mol_repr_list:
        for regr_type in regr_type_list:
            
            A_x = A_dict[(target, mol_repr, regr_type, task_x, metric_x)]
            A_y = A_dict[(target, mol_repr, regr_type, task_y, metric_y)]

            x_mean = np.around(np.nanmean(np.nanmean(A_x, axis = 1), axis = 0), decimals = decimals_mean)
            x_std = np.around(np.nanmean(np.nanstd(A_x, axis = 1), axis = 0), decimals = decimals_std)
            
            y_mean = np.around(np.nanmean(np.nanmean(A_y, axis = 1), axis = 0), decimals = decimals_mean)
            y_std = np.around(np.nanmean(np.nanstd(A_y, axis = 1), axis = 0), decimals = decimals_std)
            
            df.loc[len(df)] = [x_mean, x_std, y_mean, y_std, mol_repr, regr_type]
            
    # drop rows which contain nan values (happens for precision when ecfp + knn has no positive predictions for any of the mk trials)
    df = df.dropna()

    # plot results with seaborn
    sns.set(rc={"xtick.bottom" : True,
                "xtick.major.size": size/3,
                "xtick.major.width": size/12,
                "ytick.left" : True,
                "ytick.major.size": size/3,
                "ytick.major.width": size/12,
                "axes.edgecolor":"black", 
                "axes.linewidth": size/15, 
                "font.family": ["sans-serif"], 
                "grid.linewidth": size/8}, 
            style = "darkgrid")

    plt.figure(figsize=(size*(2/3), size*(2/3)))

    mol_repr_colour_dict = {"ecfp" : "red", "pdv" : "blue", "gin" : "violet"}
    regr_type_marker_dict = {"rf": "s", "knn": "d", "mlp": "o"}
    
    mol_repr_name_dict = {"ecfp" : "ECFP", "pdv" : "MD", "gin" : "GIN"}
    regr_type_name_dict = {"rf": "RF", "knn": "kNN", "mlp": "MLP"}
    
    sns.scatterplot(data = df,
                    x = "x_mean", 
                    y = "y_mean",
                    hue = "mol_repr",
                    palette = mol_repr_colour_dict,
                    style = "regr_type", 
                    markers = regr_type_marker_dict, 
                    s = 1.4*size**2,
                    linewidth = 0,
                    legend = plot_legend)
    
    if plot_legend == True:
        
        custom = []
        symbol_name_list = []
        for mol_repr in mol_repr_list:
            for regr_type in regr_type_list:
                custom.append(Line2D([], [], marker = regr_type_marker_dict[regr_type], color = mol_repr_colour_dict[mol_repr], linestyle='None'))
                symbol_name_list.append(mol_repr_name_dict[mol_repr] + " + " + regr_type_name_dict[regr_type])

        plt.legend(custom, 
                   symbol_name_list, 
                   loc = legend_loc, 
                   markerscale = size/6 - 1/4, 
                   scatterpoints = 1, 
                   fontsize = 1.1*size)
    
    if plot_title == True:
    
        plt.title(r"Pearson's $r$ = " + str(np.round(np.corrcoef(df["x_mean"], df["y_mean"])[0,1], decimals = 2)),
                  fontsize = 1.1*size, 
                  pad = size*(2/3))
    
    if plot_x_label == True:
    
        plt.xlabel(metric_x + ": " + task_name_dict[task_x], 
                   labelpad = size, 
                   fontsize = 1.1*size)
    else:
        plt.xlabel("")
        
    if plot_y_label == True:
    
        plt.ylabel(metric_y + ": " + task_name_dict[task_y], 
                   labelpad = size, 
                   fontsize = 1.1*size)
    else:
        plt.ylabel("")
    
    
    if plot_x_ticks == True:
        if x_tick_stepsize == "auto":
            plt.xticks(fontsize = 1.1*size)
        else:
            plt.xticks(np.arange(0, 1, x_tick_stepsize), fontsize = 1.1*size)
    else:
        if x_tick_stepsize == "auto":
            plt.xticks(fontsize = 0)
        else:
            plt.xticks(np.arange(0, 1, x_tick_stepsize), fontsize = 0)
    
    if plot_y_ticks == True:
        if y_tick_stepsize == "auto":
            plt.yticks(fontsize = 1.1*size)
        else:
            plt.yticks(np.arange(0, 1, y_tick_stepsize), fontsize = 1.1*size)
    else:
        if y_tick_stepsize == "auto":
            plt.yticks(fontsize = 0)
        else:
            plt.yticks(np.arange(0, 1, y_tick_stepsize), fontsize = 0)
    
    if plot_error_bars == True:
        
        plt.errorbar(df["x_mean"], 
                     df["y_mean"], 
                     xerr = df["x_std"], 
                     yerr = df["y_std"], 
                     ls = "none", 
                     ecolor = "black", 
                     lw = size/15, 
                     capsize = size/3, capthick = size/15)
        
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.tight_layout()
    
    plt.savefig(filepath_to_save)
    
    if linear_regression == True:
    
        # create linear least squares polynomial
        line_coeffs = np.polyfit(df["x_mean"], df["y_mean"], deg = 1)
        line_grid = np.linspace(0, 1, 100)
        line_vals = np.polyval(line_coeffs, line_grid)
        plt.plot(line_grid, line_vals)

        print("line_coeffs = (k, d) = ", line_coeffs)
        print("line(mcc = 1) = ", line_coeffs[0] + line_coeffs[1])
    
    plt.show()