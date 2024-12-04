#!/usr/bin/env python3

case_pheno_value = 2 # 2 is affected
ctrl_pheno_value = 1 # 1 is control (unaffected)

# modifications inside data directory
update_sex_case_fam = True # whether to update or not sex in case fam file from sample_techno_QC_file
keep_only_ctrl_individuals_in_ctrl_data = True # whether to filter only controls (phenotype = 1) individuals in ctrl fam file if anyn contamination by 
update_pheno_case_fam = True # whether to update or not phenotype in case fam file from sample_techno_QC_file

# QC parameters
 
QC_case_params = {
    "mind":0.02, 
    "geno":0.02, 
    "maf":0.01, 
    "hwe":1e-10, 
    "min":0.2, 
    "hwe_zoom":1e-7, 
    "dataset_type": 1, 
    "presence_of_non_auto_chr": True, 
    "fstat_female_threshold": 0.4, 
    "fstat_male_threshold": 0.8
}

QC_ctrl_params = {
    "mind":0.02, 
    "geno":0.02, 
    "maf":0.01, 
    "hwe":1e-6, 
    "min":0.2, 
    "hwe_zoom":1e-3, 
    "dataset_type": 0, 
    "presence_of_non_auto_chr": False, 
    "fstat_female_threshold": 0.4, 
    "fstat_male_threshold": 0.8
}

QC_naf_params = {
    "mind":0.02, 
    "geno":0.02, 
    "maf":0.01, 
    "hwe":1e-6, 
    "min":0.2, 
    "hwe_zoom":1e-3, 
    "dataset_type": 2, 
    "presence_of_non_auto_chr": False, 
    "fstat_female_threshold": 0.4, 
    "fstat_male_threshold": 0.8
}