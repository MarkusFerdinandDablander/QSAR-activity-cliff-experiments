# Systematically Exploring QSAR Models for Activity-Cliff Prediction

Code to reproduce the experiments from the paper Systematically Exploring QSAR Models for Activity-Cliff Prediction. This repository also contains clean molecule- and MMP data for all three analysed data sets (dopamine receptor D2, factor Xa, SARS-CoV-2 main protease) as well as the original numerical results from the experiments conducted in the paper.

![Graphical abstract](/figures/graphical_abstract.png)

## Data Sets

The data-folder contains three clean chemical data sets of small-molecule inhibitors of dopamine receptor D2, factor Xa, or SARS-CoV-2 main protease respectively. Each data set is represented by two files: molecule_data_clean.csv and MMP_data_clean.csv. The first file contains SMILES strings with associated activity values and the second file contains all matched molecular pairs (MMPs) identified within the first file.

## Reproducing the Experiments

The experiments in the paper can be reproduced by running the code in the Jupyter notebook QSAR_activity_cliff_experiments.ipynb. First, the QSAR-, AC-, and PD-prediction tasks for the chosen data set are formally constructed in a data-preparation section. Then, an appropriate data split is conducted, both at the level of individual molecules and MMPs. Finally, a molecular representation (PDV, ECFP, or GIN) and a regression technique (RF, kNN, MLP) are chosen and the resulting model is trained and evaluated for QSAR-prediction, AC-classification and PD-classification. The computational environment in which the original results were conducted can be found in environment.yml.

![Graphical abstract](/figures/methods_overview_linear_with_ac.png)

## Visually Investigating the Results:

The experimental results can be visually explored using the visualise_results-function at the end of QSAR_activity_cliff_experiments.ipynb. This function produces scatterplots such as the one in the graphical abstract above. The original numerical results from the paper are saved in the resuls-folder; thus the original plots from the paper (and more) can be generated with visualise_results.
