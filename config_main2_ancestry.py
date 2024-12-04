#!/usr/bin/env python3

import numpy as np

### Configuration file for main2_ancestry_post_imputation_analysis.py
UPDATE_SEX_CASE_FAM = True # whether to update sex in case fam file using a reference file with assigned sex
UPDATE_SEX_CTRL_FAM = True # whether to update sex in ctrl fam file using a reference file with assigned sex
UPDATE_SEX_REF_FAM = True # whether to update sex in ref (1kg) fam file using a reference file with assigned sex
UPDATE_SEX_NAF_FAM = False # whether to update sex in North-African fam file using a reference file with assigned sex
UPDATE_PHENO_CASE_FAM = True # Replace -9 phenotype of case fam file with 2 (case phenotype)
REPLACE_FID_FAM_NAF = True # Replace FID column of North African reference file with "North-African"
UPDATE_PHENO_CTRL_FAM = True # Replace phenotype of control fam file with 1 (control)
N_PC = 20 # number of principal components to consider for PCA
N_PC_OUTLIERS = 0 # number of PC on which outliers will be identified (default: 0, recommended: 2 or more)
N_OUTLIERS_ITER = 0 # number of outliers removal process iterations (default! 0, recommended: 1)
OUTLIER_MODE = 2 # outlier mode (2: no oulier removal, 1: outlier removal)

# PCA plot configuration
REF_COLS_DICT = {
    "Africa": "red",
    "East_Asia": "yellow",
    "South_Asia": "green",
    "Europe": "blue",
    "America": "orange",
    "North_Africa": "purple",
}

NON_CLUSTERED_CASE_CTRL_COLORS = {"Control": "lightslategray", "Case": "turquoise"}

ALPHA_REF = 0.2 # transparency level for reference data.

# Kmeans clustering
N_CLUSTERS = 6

# define clusters colors    
# if new run check order of columns to match correct color with correct cluster to align with reference data points.
# CASE_COLORS_CLUSTERS = np.array(["slateblue", "magenta", "indianred", "lime", "mediumvioletred", "#e8f332", "rebeccapurple"]) # FOR ROMAIN
CASE_COLORS_CLUSTERS = np.array(["slateblue", "mediumvioletred",  "magenta", "#e8f332", "indianred", "lime", "rebeccapurple"])
CTRL_COLORS_CLUSTERS = np.array(["lightskyblue", "hotpink", "violet", "#bac412", "lightcoral", "palegreen", "mediumslateblue"])
HOMEMADE_CLUSTER_CENTERS = [[-0.00245, -0.00668]]
# Homemade modifications of Kmeans clustering - try and error to find the adequate barycenters for your study
# HOMEMADE_CLUSTER_CENTERS = [[-0.00285, -0.00668]] # FOR ROMAIN add new clusters by adding new barycenters [[x_coord, y_coord], ...]

CLUSTER_TO_MODIFY = [(4, [0.00325, 0.0042]), (1, [0.0001, 0]), (2, [-0.00035, -0.001])]
# CLUSTER_TO_MODIFY = [(4, [0.00325, 0.0042]), (1, [0.0001, 0]), (2, [-0.00035, -0.001])] # FOR ROMAIN modify barycenters positions (cluster number, [coordinates to add x, coordinates to add y])