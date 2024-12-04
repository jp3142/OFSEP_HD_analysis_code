#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
from subprocess import Popen, PIPE
from shlex import split as ssplit
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from copy import copy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import glob

def compute_barycentre(pop: pd.DataFrame):
    """
    Compute barycenter of a Kmeans cluster.
    """
    x = sum(pop.iloc[:, 0])/len(pop)
    y = sum(pop.iloc[:, 1])/len(pop)

    return (x, y)

def distance_to_barycenter(x, y, x_barycenter, y_barycenter):
    """
    Compute individuals distance to barycenter 
    """
    return np.sqrt( np.square(y-y_barycenter) + np.square(x-x_barycenter) )

def replace_fid_fam_by(initial_dataset_prefix, replace_with):
    """"""
    with open(initial_dataset_prefix+"2.fam", 'w') as f:
        awk_c = "".join(["awk ", "'{$1=", f"\"{replace_with}\";", "print $0}'", " "+initial_dataset_prefix+".fam"])
        awk_p = Popen(ssplit(awk_c), stdin=PIPE, stderr=PIPE, stdout=f, text=True) 
        awk_p.communicate()

    mv_c = f"mv {initial_dataset_prefix}2.fam {initial_dataset_prefix}.fam"
    mv_p = Popen(ssplit(mv_c), stdout=PIPE, stderr=PIPE, stdin=PIPE)
    mv_p.communicate() 

    return initial_dataset_prefix
                    
def merge_case_ctrl_and_ref_datasets(case_prefix, ctrl_prefix, ref_prefix, naf_prefix, output_dir):
    """
    # /!\ This function has to be updated if working with another version of the dataset !!! 
    It is very specific to OFSEP-HD data.

    plink_c = ["plink", "--merge-list", all_files, "--make-bed", "--out", merged_dataset_prefix]
    "--set-missing-var-ids", "@:#[b37]\$1,\$2",
    Error: 26 variants with 3+ alleles present.
    * If you believe this is due to strand inconsistency, try --flip with
      main2_output_test/merge_case_ctrl_ref_dataset-merge.missnp.
      (Warning: if this seems to work, strand errors involving SNPs with A/T or C/G
      alleles probably remain in your data.  If LD between nearby SNPs is high,
      --flip-scan should detect them.)
    * If you are dealing with genuine multiallelic variants, we recommend exporting
      that subset of the data to VCF (via e.g. '--recode vcf'), merging with
      another tool/script, and then importing the result; PLINK is not yet suited
      to handling them.
    --> https://www.cog-genomics.org/plink/1.9/data#merge3
    plink_p = Popen(plink_c, stdin=PIPE, stderr=PIPE, stdout=PIPE, text=True)
    plink_p.communicate()
    """
    
    # Merging ref and ctrl
    print("Keep in ctrl data the snps that overlaps with ref data")
    ctrl_overlap = os.path.join(output_dir, ctrl_prefix.split('/')[-1]+"_ctrl_overlap")
    plink_c1 = ["plink", "--bfile", ctrl_prefix, "--extract", ref_prefix+".bim", "--snps-only", "--make-bed", "--out", ctrl_overlap]
    plink_p1 = Popen(plink_c1, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p1.communicate()

    print("Keep in ref data the snps that overlaps with ctrl data")
    ref_overlap = os.path.join(output_dir, ref_prefix.split('/')[-1]+"_ref_overlap")
    plink_c2 = ["plink", "--bfile", ref_prefix, "--extract", ctrl_prefix+".bim", "--snps-only", "--make-bed", "--out", ref_overlap]
    plink_p2 = Popen(plink_c2, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p2.communicate()

    print("Merging ctrl and reference data") 
    merge_ctrl_ref_dataset_prefix = os.path.join(output_dir, "merged_ctrl_ref")
    plink_c3 = ["plink", "--bfile", ctrl_overlap, "--bmerge", ref_overlap, "--make-bed", "--out", merge_ctrl_ref_dataset_prefix]
    plink_p3 = Popen(plink_c3, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p3.communicate()
    
    #flipping some reference snps if needed

    if merge_ctrl_ref_dataset_prefix.split('/')[-1]+"-merge.missnp" in os.listdir(output_dir):
        sys.stderr.write("[Error] Some alleles flip / swap / flip - swap are present in the datasets. Please handle them.\n")
        exit(5)

        ###################################################################################
        # In the case you have multiallelic sites, try uncommenting and updating below code
        #### +3alleles error - check variants to flip
        # remove variants when mismatch
        # print("[Warning] Removing specific snps because of mismatch during merging.")
        # plink_c3_2 = ["plink", "--bfile", ref_overlap, "--exclude", merge_ctrl_ref_dataset_prefix+"-merge.missnp", "--make-bed", "--out", ref_overlap+"_excl_pb_snps"] # FLIPPING 1 kg on missnp
        # plink_p3_2 = Popen(plink_c3_2, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
        # plink_p3_2.communicate()
        # ref_overlap = ref_overlap+"_excl_pb_snps"

        # print("[Warning] Removing specific snps because of mismatch during merging.")
        # plink_c3_3 = ["plink", "--bfile", ctrl_overlap, "--exclude", merge_ctrl_ref_dataset_prefix+"-merge.missnp", "--make-bed", "--out", ref_overlap+"_excl_pb_snps"] # FLIPPING 1 kg on missnp
        # plink_p3_3 = Popen(plink_c3_3, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
        # plink_p3_3.communicate()
        # ctrl_overlap = ctrl_overlap+"_excl_pb_snps"

        # plink_c3 = ["plink", "--bfile", ctrl_overlap, "--bmerge", ref_overlap, "--make-bed", "--out", merge_ctrl_ref_dataset_prefix]
        # plink_p3 = Popen(plink_c3, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
        # plink_p3.communicate()

        # flipping ?
        # print("[Warning] Flipping specific snps of 1kg data becasue strand flip in 1kg dataset.")
        # plink_c3_2 = ["plink", "--bfile", ref_overlap, "--flip", merge_ctrl_ref_dataset_prefix+"-merge.missnp", "--make-bed", "--out", ref_overlap+"_flipped"] # FLIPPING 1 kg on missnp
        # plink_p3_2 = Popen(plink_c3_2, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
        # plink_p3_2.communicate()
        # ref_overlap = ref_overlap+"_flipped"

        # print("Merging ctrl and flipped reference data") 
        # merge_ctrl_ref_dataset_prefix = merge_ctrl_ref_dataset_prefix+"_after_flip"
        # plink_c3 = ["plink", "--bfile", ctrl_overlap, "--bmerge", ref_overlap, "--make-bed", "--out", merge_ctrl_ref_dataset_prefix]
        # plink_p3 = Popen(plink_c3, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
        # plink_p3.communicate()
        #######################################################################################

    # Merging case and ctrl-ref
    print("Keeping in case data the snps that overlaps with merged ctrl-ref dataset")
    case_overlap = os.path.join(output_dir, case_prefix.split('/')[-1]+"_case_overlap")
    plink_c4 = ["plink", "--bfile", case_prefix, "--extract", merge_ctrl_ref_dataset_prefix+".bim", "--snps-only", "--make-bed", "--out", case_overlap]
    plink_p4 = Popen(plink_c4, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
    plink_p4.communicate()

    print("Keep in merged_ref_ctrl dataset the snps that overlaps with case dataset")
    merge_ctrl_ref_overlap = os.path.join(output_dir, merge_ctrl_ref_dataset_prefix.split('/')[-1]+"_overlap")
    plink_c5 = ["plink", "--bfile", merge_ctrl_ref_dataset_prefix, "--extract", case_prefix+".bim", "--snps-only", "--make-bed", "--out", merge_ctrl_ref_overlap]
    plink_p5 = Popen(plink_c5, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p5.communicate()

    print("Merging ctrl-ref dataset with case dataset")
    merge_ctrl_ref_case_dataset_prefix = os.path.join(output_dir, "merged_ctrl_ref_case_last")
    plink_c6 = ["plink", "--bfile", case_overlap, "--bmerge", merge_ctrl_ref_overlap, "--make-bed", "--allow-no-sex", "--out", merge_ctrl_ref_case_dataset_prefix] # --allow-no-sex
    plink_p6 = Popen(plink_c6, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p6.communicate()

    # multi_snps_file = os.path.join(output_dir, "multiallelic_snp_to_exclude.txt") 
    # with open(multi_snps_file, 'w') as f:
    #     f.write("11_127025586\n")
    #     f.write("6_13513390")

    # plink_c6_2 = ["plink", "--bfile", merge_ctrl_ref_overlap, "--exclude", multi_snps_file, "--make-bed", "--out", merge_ctrl_ref_overlap+"_tmp"]
    # plink_p6_2 = Popen(plink_c6_2, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    # plink_p6_2.communicate()
    # merged_ctrl_ref_overlap = merged_ctrl_ref_overlap+"_tmp"

    # plink_c6_3 = ["plink", "--bfile", case_overlap, "--exclude", multi_snps_file, "--make-bed", "--out", case_overlap+"_tmp0"]
    # plink_p6_3 = Popen(plink_c6_3, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    # plink_p6_3.communicate()

    # remove one of the duplicated variant with different alleles in case dataset
    # plink_c6_4 = ["plink2", "--bfile", case_overlap+"_tmp0", "--set-all-var-ids", "@_#,\$a,\$r", "--make-bed", "--out", case_overlap+"_tmp1"]
    # plink_p6_4 = Popen(plink_c6_4, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    # plink_p6_4.communicate()
    
    # snp_to_remove_case = os.path.join(output_dir, "snp_to_remove_case.txt") 
    # with open(snp_to_remove_case, 'w') as f:
    #     f.write("5_159767049G,C\n")

    # # exclude the freq 0% duplicated variant
    # plink_c6_5 = ["plink", "--bfile", case_overlap+"_tmp", "--exclude", snp_to_remove_case, "--make-bed", "--out", case_overlap+"_tmp2"]
    # plink_p6_5 = Popen(plink_c6_5, stdin=PIPE, stderr=PIPE, stdout=PIPE, text=True)
    # plink_p6_5.communicate()

    # # reset back ids to chrom_position
    # plink_c6_6 = ["plink2", "--bfile", case_overlap, "--set-all-var-ids", "@_#,\$a,\$r", "--make-bed", "--out", case_overlap+"_tmp1"]
    # plink_p6_6 = Popen(plink_c6_6, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    # plink_p6_6.communicate()

    # Merging case and ctrl-ref after removing / processing multiallelic variants (n=3)
    print("Keep in case data the snps that overlaps with merged ctrl-ref dataset")
    case_overlap_last = case_overlap+"_last"
    plink_c6_7 = ["plink", "--bfile", case_overlap, "--extract", merge_ctrl_ref_dataset_prefix+".bim", "--snps-only", "--make-bed", "--out", case_overlap_last]
    plink_p6_7 = Popen(plink_c6_7, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
    plink_p6_7.communicate()

    print("Keep in merged_ref_ctrl dataset the snps that overlaps with case dataset")
    merge_ctrl_ref_overlap_last = merge_ctrl_ref_dataset_prefix+"_overlap_last"
    plink_c6_8 = ["plink", "--bfile", merge_ctrl_ref_dataset_prefix, "--extract", case_overlap_last+".bim", "--snps-only", "--make-bed", "--out", merge_ctrl_ref_overlap_last]
    plink_p6_8 = Popen(plink_c6_8, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p6_8.communicate()

    print("Merging ctrl-ref dataset with case dataset")
    merge_ctrl_ref_case_dataset_prefix_last = os.path.join(output_dir, "merged_ctrl_ref_case_last")
    plink_c6_9 = ["plink", "--bfile", case_overlap_last, "--bmerge", merge_ctrl_ref_overlap_last, "--make-bed", "--allow-no-sex", "--out", merge_ctrl_ref_case_dataset_prefix_last] # --allow-no-sex
    plink_p6_9 = Popen(plink_c6_9, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p6_9.communicate()

    # flip alleles if needed (in the merged_ctrl_ref dataset)
    if merge_ctrl_ref_case_dataset_prefix_last.split('/')[-1]+"-merge.missnp" in os.listdir(output_dir):
        print("[Error] Some alleles flip / swap / flip - swap are present in the datasets. Please handle them.")
        exit(5)

        ##############################################################################
        # If some misssnps detected by plink.
        # /!\ IF IN THIS CASE MODIFY THE SCRIPT UNDER TO PROCESS MERGE CONFLICT /!\ #

        ## +3alleles error - check variants to flip
        # print("[Warning] Removing specific snps because of mismatch during merging.")
        # plink_c7 = ["plink", "--bfile", case_overlap_last, "--exclude", merge_ctrl_ref_case_dataset_prefix_last+"-merge.missnp", "--make-bed", "--out", case_overlap_last+"_excl_pb_snps"] # FLIPPING 1 kg on missnp
        # plink_p7 = Popen(plink_c7, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
        # plink_p7.communicate()
        # case_overlap_last = case_overlap_last+"_excl_pb_snps"

        # print("[Warning] Removing specific snps because of mismatch during merging.")
        # plink_c8 = ["plink", "--bfile", merge_ctrl_ref_overlap_last, "--exclude", merge_ctrl_ref_case_dataset_prefix_last+"-merge.missnp", "--make-bed", "--out", merge_ctrl_ref_overlap_last+"_excl_pb_snps"] # FLIPPING 1 kg on missnp
        # plink_p8 = Popen(plink_c8, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
        # plink_p8.communicate()
        # merge_ctrl_ref_overlap_last = merge_ctrl_ref_overlap_last+"_excl_pb_snps"

        # plink_c9 = ["plink", "--bfile", case_overlap_last, "--bmerge", merge_ctrl_ref_overlap_last, "--make-bed", "--out", merge_ctrl_ref_case_dataset_prefix_last]
        # plink_p9 = Popen(plink_c3, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
        # plink_p9.communicate()

        # flipping ?
        # print("[Warning] Flipping specific snps of merged_ctrl_ref_dataset because strand flip with case dataset.")
        # plink_c7 = ["plink", "--bfile", merge_ctrl_ref_overlap_last, "--flip", merge_ctrl_ref_case_dataset_prefix_last+"-merge.missnp", "--make-bed", "--out", merge_ctrl_ref_overlap_last+"_flipped"] # FLIPPING 1 kg on missnp
        # plink_p7 = Popen(plink_c7, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
        # plink_p7.communicate()
        # merge_ctrl_ref_overlap_last = merge_ctrl_ref_overlap_last+"_flipped"

        # print("Merging case and merged reference/ctrl data") 
        # merge_ctrl_ref_case_dataset_prefix_last = merge_ctrl_ref_case_dataset_prefix_last+"_after_flip"
        # plink_c8 = ["plink", "--bfile", merge_ctrl_ref_overlap_last, "--bmerge", case_overlap_last, "--make-bed", "--out", merge_ctrl_ref_case_dataset_prefix_last]
        # plink_p8 = Popen(plink_c8, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
        # plink_p8.communicate()
        #########################################################################################

    # # merging case_ctrl_ref dataset with NAF dataset
    print("Keep in merge ctrl_ref_case data the snps that overlaps with NAF dataset")
    merge_ctrl_ref_case_naf_overlap = merge_ctrl_ref_case_dataset_prefix_last+"_naf"
    plink_c9 = ["plink", "--bfile", merge_ctrl_ref_case_dataset_prefix_last, "--extract", naf_prefix+".bim", "--snps-only", "--make-bed", "--out", merge_ctrl_ref_case_naf_overlap]
    plink_p9 = Popen(plink_c9, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
    plink_p9.communicate()

    print("Keep in NAF dataset the snps that overlaps with merge_ctrl_ref_case dataset")
    naf_overlap = os.path.join(output_dir, naf_prefix.split('/')[-1]+"_overlap")
    plink_c10 = ["plink", "--bfile", naf_prefix, "--extract", merge_ctrl_ref_case_dataset_prefix_last+".bim", "--snps-only", "--make-bed", "--out", naf_overlap]
    plink_p10 = Popen(plink_c10, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p10.communicate()

    print("Merging NAF dataset with ctrl_ref_case dataset")
    merge_ctrl_ref_case_naf_dataset_prefix = os.path.join(output_dir, "merged_ctrl_ref_case_naf_last")
    plink_c11 = ["plink", "--bfile", naf_overlap, "--bmerge", merge_ctrl_ref_case_naf_overlap, "--make-bed", "--allow-no-sex", "--out", merge_ctrl_ref_case_naf_dataset_prefix]
    plink_p11 = Popen(plink_c11, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p11.communicate()

    ##### Removing unecessary intermediate file
    print("[Removing] Removing tmp files ...")
    def rm_prefix(prefix: str):
        """
        Remove all files with name prefix, independently from extension.
        """
        files = glob.glob(f"{prefix}.*") # get list of files with prefix
        for file in files:
            rm_c = f"rm {file}"
            rm_p = Popen(ssplit(rm_c), stderr=PIPE, stdout=PIPE, stdin=PIPE, text=True)
            rm_p.communicate()
    
    # # remove intermediary files
    # for prefix in [ctrl_overlap, ref_overlap, merge_ctrl_ref_dataset_prefix, case_overlap, merge_ctrl_ref_overlap, merge_ctrl_ref_case_dataset_prefix, f"{merge_ctrl_ref_overlap}_tmp", f"{case_overlap}_tmp0", f"{case_overlap}_tmp", f"{case_overlap}_tmp2", f"{case_overlap}_tmp3", f"{case_overlap_last}", f"{merge_ctrl_ref_overlap_last}", f"{merge_ctrl_ref_case_dataset_prefix_last}", f"{merge_ctrl_ref_case_naf_overlap}",  f"{naf_overlap}"]:
    #     rm_prefix(prefix)

    return merge_ctrl_ref_case_naf_dataset_prefix 

def keep_indep_snps(output_dir, dataset_prefix, high_ld_regions_file):
    # --range flag is deprecated use --extract range <filename> ???
    indepSNP_prefix = os.path.join(output_dir, "indepSNP")
    plink_c1 = ["plink", "--bfile", dataset_prefix, "--exclude", high_ld_regions_file, "--range", "--indep-pairwise", "50", "5", "0.2", "--out", indepSNP_prefix]
    plink_p1 = Popen(plink_c1, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p1.communicate()
    print("[Pruning] Independant SNPs list created.")

    # plink --bfile HapMap_3_r3_9 --extract indepSNP.prune.in --het --out R_check
    # This file contains your pruned data set.
    pruned_data_prefix = os.path.join(output_dir, "merged_pruned_dataset")
    plink_c2 = ["plink", "--bfile", dataset_prefix, "--extract", f"{indepSNP_prefix}.prune.in", "--make-bed", "--out", pruned_data_prefix]
    plink_p2 = Popen(plink_c2, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p2.communicate()
    print("[Pruning] Independant SNPs dataset created.")

    return pruned_data_prefix

def create_update_ids_list(output_dir, merged_dataset_prefix, reference_data_pop_list_path):
    """
    reference_data_pop_list_path = 1kg_list.txt
    """
    update_ids_ref = os.path.join(output_dir, "update_ids_ref.txt")
    update_ids_ctrl_case = os.path.join(output_dir, "update_ids_ctrl_case.txt")
    # récupérer les old fam id et old id, new id = old id
    # new fam id = superpopulation (colonne $6 de reference_data_pop_list_path)

    with open(update_ids_ref, 'w') as f1:
        awk_c = ["awk", "-F", "\t", '{if(NR>1){print $1,$1,$6,$1}}', reference_data_pop_list_path] #"FS='\t'"
        awk_p = Popen(awk_c, stdout=f1, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate()

    #replace family id of case and control data by Case or Control
    with open(update_ids_ctrl_case, 'w') as f2:
        awk_c2 = ["awk", '{if($6==1){print $1,$2,"Control",$2} else if($6==2){print $1,$2,"Case",$2}}', merged_dataset_prefix+".fam"] 
        awk_p2 = Popen(awk_c2, stdin=PIPE, stdout=f2, stderr=PIPE, text=True)
        awk_p2.communicate()

    # concat both previously created files as one update_ids file
    concat_update_ids = os.path.join(output_dir, "concat_update_ids.txt")
    with open(concat_update_ids, 'w') as f3:
        cat_c = f"cat {update_ids_ref} {update_ids_ctrl_case}"
        cat_p = Popen(ssplit(cat_c), stdout=f3, stderr=PIPE, stdin=PIPE, text=True)
        cat_p.communicate()

    # shorten too long ids by removing CEL extension
    last_update_ids = os.path.join(output_dir, "update_ids.txt")
    with open(last_update_ids, 'w') as f4:
        #awk_c3 = ["awk", '{sub(".CEL", "", $4); sub("bis_", "", $4); print $0}', concat_update_ids] DO NOT REMOVE bis_ !!!!!
        awk_c3 = ["awk", '{sub(".CEL", "", $4); print $0}', concat_update_ids]

        awk_p3 = Popen(awk_c3, stdout=f4, stderr=PIPE, stdin=PIPE, text=True)
        awk_p3.communicate()

    for file in [update_ids_ref, update_ids_ctrl_case, concat_update_ids]:
        rm_c = f"rm {file}"
        rm_p = Popen(ssplit(rm_c), stdout=PIPE, stderr=PIPE, stdin=PIPE)
        rm_p.communicate()
    
    return last_update_ids

def update_ids(dataset_prefix, update_ids_txt):
    """"""
    updated_ids_prefix =  dataset_prefix+"_updated_ids"
    plink_c = ["plink", "--bfile", dataset_prefix, "--update-ids", update_ids_txt, "--make-bed", "--out", updated_ids_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    plink_p.communicate()

    return updated_ids_prefix

def get_number_of_snps(snp_file):
    """"""
    bash_c = ["wc", "-l", snp_file]
    bash_p = Popen(bash_c, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    res, _ = bash_p.communicate()
   
    return int(res.split(' ')[0])

def create_list_fin_indiv(kg_list_subpop: str, output_dir: str):
    """Create list of finnish individuals to remove"""
    
    exclude_fin_indiv_file = os.path.join(output_dir, "finnish_indiv_to_exclude.txt")
    fin_df = pd.read_table(kg_list_subpop, sep='\t', header='infer')[["Pop_symbol", "Sample", "Biogeographic_group"]]
    fin_df = fin_df[fin_df["Pop_symbol"] == "FIN"][["Biogeographic_group", "Sample"]]
    fin_df.to_csv(exclude_fin_indiv_file, sep=' ', header=False, index=False)

    return exclude_fin_indiv_file

def remove_individuals(indiv_to_exclude_file: str, merged_pruned_prefix: str):
    """
    Remove individuals based on a space/tab delimited individuals to exclude file (for --remove option of plink)
    """
    new_prefix = merged_pruned_prefix + "_removedInd"
    plink_c = ["plink", "--bfile", merged_pruned_prefix, "--remove", indiv_to_exclude_file, "--make-bed", "--out", new_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    plink_p.communicate()

    return new_prefix

def pca(output_dir, pruned_dataset):
    """PLINK PCA"""
    print("Computing PCA...")
    pca_output_prefix = os.path.join(output_dir, "pca_output")
    plink_c = ["plink", "--bfile", pruned_dataset, "--pca", "20", "header", "tabs", "--out", pca_output_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p.communicate()

    return pca_output_prefix

def recode_to_eigensoft_format(output_dir: str, merged_pruned_prefix: str):
    """
    Create files needed for  smart pca: .indiv, .snp and .eigenstrat files
    """

    # convert to .ped and .map
    plink_c = ["plink", "--bfile", merged_pruned_prefix, "--recode", "--out", merged_pruned_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p.communicate()

    # avoid funny values in sixth column of ped file causing ignored individuals during the convertf operation
    with open(merged_pruned_prefix+"2.ped", 'w') as f:
        awk_c = ["awk", '{if($6 != 1 && $6 != 2){$6=1}; print $0}', merged_pruned_prefix+".ped"]
        awk_p = Popen(awk_c, stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate()
        mv_c = f"mv {merged_pruned_prefix}2.ped {merged_pruned_prefix}.ped"
        mv_p = Popen(ssplit(mv_c), stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
        mv_p.communicate()
    
    # create parfile for convertf to eigensoft formats
    output_file_prefix = os.path.join(output_dir, "recoded_eigensoft")
    convertf_parfile = os.path.join(output_dir, "convertf_to_eigensoft.parfile")
    with open(convertf_parfile, 'w') as f:
        f.write(f"genotypename: {merged_pruned_prefix+'.ped'}\n")
        f.write(f"snpname: {merged_pruned_prefix+'.map'}\n")
        f.write(f"indivname: {merged_pruned_prefix+'.ped'}\n")
        f.write(f"outputformat: EIGENSTRAT\n")
        f.write(f"genotypeoutname: {output_file_prefix+'.geno'}\n")
        f.write(f"snpoutname: {output_file_prefix+'.snp'}\n")
        f.write(f"indivoutname: {output_file_prefix+'.ind'}\n")
        f.write(f"familynames: YES\n")

    # create eigensoft geno.eigenstrat, snp.eigenstrat indiv.eigenstrat files
    convertf_c = f"convertf -p {convertf_parfile}"
    convertf_p = Popen(ssplit(convertf_c), stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    convertf_p.communicate()

    # modify .ind file to have correct population and not only control and case (inferred based on phenotype column)
    with open(output_file_prefix+"2.ind", 'w') as f:
        awk_c = ["awk", '{split($1, a, ":"); $3 = a[1]; print $0}', output_file_prefix+".ind"]
        awk_p = Popen(awk_c, stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate()

        mv_c = f"mv {output_file_prefix}2.ind {output_file_prefix}.ind"
        mv_p = Popen(ssplit(mv_c), stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)

    # create poplist file for smartpca parameters file, to specify which categories correspond to population and the other to case ctrl (list of uniq superpopulation in fam file, and omit Case and Control)
    fam_file = merged_pruned_prefix + ".ped"
    poplist_file = output_file_prefix+".poplist"

    with open(poplist_file, 'w') as f:
        # AWK command to filter and print the first column
        awk_c1 = ["awk", "-F", " ", '{print $1}', fam_file]
        # Sort command
        sort_c = ["sort"]
        # Uniq command
        uniq_c = ["uniq"]

        # Create Popen instances for each command
        awk_p1 = Popen(awk_c1, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
        sort_p = Popen(sort_c, stdin=awk_p1.stdout, stdout=PIPE, stderr=PIPE, text=True)
        uniq_p = Popen(uniq_c, stdin=sort_p.stdout, stdout=f, stderr=PIPE, text=True)

        # Close unnecessary file handles
        awk_p1.stdout.close()
        sort_p.stdout.close()

        awk_p1.communicate()
        sort_p.communicate()
        uniq_p.communicate()

    return output_file_prefix

def remove_AT_GC_snps(file_prefix: str):
    """"""
    # create list of AT GC snps
    with open("tmp_output.bim", "r") as f:
        awk_c = ["awk", "'\{if( ($5==\"G\" && $6==\"C\") || ($5==\"C\" && $6==\"G\") || ($5==\"A\" && $6==\"T\") || ($5==\"T\" && $6==\"A\") ){print $2}\}'", file_prefix+".bim"]
        awk_p = Popen(awk_c, text=True, stdin=PIPE, stdout=f, stderr=PIPE)
        awk_p.communicate()
    
    # update file - overwriting
    mv_c = ["mv", "tmp_output.bim", file_prefix+".bim"]
    mv_p = Popen(mv_c, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    mv_p.communicate()
    print("[RemoveATGC] AT anf GC variants removed.")
    
    return file_prefix

def eigensoft_smart_pca(output_dir: str, merged_pruned_dataset: str, n_pc=20, n_pc_outliers=2, outlier_mode=2, n_outliers_iter=1):
    """
    outlier_mode: 1 outlier removal, 2 no outlier removal
    Process smartpca.
    -i: Input genotype file
    -a: Input SNP file
    -b: Input individual file
    -k: Number of principal components -> n_pc
    -o: Output prefix
    -p: prefix for parameter file
    -e: Output file for eigenvalues
    -l: Log file
    -m: Number of outlier iterations -> n_outliers_iter
    -t: Number of outlier principal components -> n_pc_outliers (number of PC on which to look for outliers)
    -s: Outlier sigma threshold
    -w: Poplist file
    -y: Poplist for plot
    -z: Bad SNP file
    -q 1 -> how to remove outliers, 1 is outlier removal, 2 is no outliers removal
    """
    print("Computing Eigensoft smartPCA...")

    eigen_file_prefix = recode_to_eigensoft_format(output_dir, merged_pruned_dataset)

    output_file_prefix = os.path.join(output_dir, "smartpca")

    try:
        pca_c = [
            "smartpca",
            "-i", eigen_file_prefix+".geno",
            "-a", eigen_file_prefix+".snp",
            "-b", eigen_file_prefix+".ind",
            "-k", str(n_pc),
            "-o", output_file_prefix,
            "-p", output_file_prefix,
            "-e", output_file_prefix+".eval", 
            "-l", output_file_prefix+".log",
            "-m", str(n_outliers_iter),
            "-t", str(n_pc_outliers),
            "-w", eigen_file_prefix+".poplist",
            "-q", str(outlier_mode) # mode 1 = remove outliers, 2 = no outlier removal
        ]

        pca_p = Popen(pca_c, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
        success=bool("" == pca_p.stderr.readline())
        res, out = pca_p.communicate()
        assert(success)
        
    except AssertionError:
        sys.stderr.write(f"[SyntaxError] Problem with smartpca {res} {out} Trying using .parfile ...\n")
        smartpca_parfile = output_file_prefix+".parfile"
        with open(smartpca_parfile, 'w') as f:
            f.write(f"genotypename: {eigen_file_prefix}.geno\n")
            f.write(f"snpname: {eigen_file_prefix}.snp\n")
            f.write(f"indivname: {eigen_file_prefix}.ind\n")
            f.write(f"evecoutname: {output_file_prefix}.evec\n")
            f.write(f"evaloutname: {output_file_prefix}.eval\n")
            f.write(f"numoutevec: {str(n_pc)}\n")
            f.write(f"poplistname: {eigen_file_prefix}.poplist\n")
            f.write(f"numoutlieriter: {str(n_outliers_iter)}\n")
            f.write(f"noutlierevec: {str(n_pc_outliers)}\n")
            f.write(f"outliermode: {str(outlier_mode)}\n")
            f.write(f"outputformat: PED\n")
            f.write(f"genotypeoutname: {output_file_prefix}.ped\n")
            f.write(f"snpoutname: {output_file_prefix}.map\n")
            f.write(f"indivoutname: {output_file_prefix}.ped\n")

        pca_c = ["smartpca", "-p", smartpca_parfile]
        pca_p = Popen(pca_c, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
        pca_p.communicate()

    # create population column from pop:patientID sample id.
    with open(output_file_prefix+"2.evec", 'w') as f:
        awk_c = ["awk", '{ if(NR!=1){split($1, a, ":"); $1=a[2]; $NF=a[1]; print $0} else if(NR==1){print $0}}', output_file_prefix+".evec"]
        awk_p = Popen(awk_c, stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate()

    mv_c = f"mv {output_file_prefix}2.evec {output_file_prefix}.evec"
    mv_p = Popen(ssplit(mv_c), stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    mv_p.communicate()

    map_file = output_file_prefix+".map"
    if not map_file in os.listdir(output_dir):
        map_file = merged_pruned_dataset+".map"

    return output_file_prefix, map_file

def compute_explained_variance(pca_prefix, output_dir, output_file_name):
    """
    Computes var explained by each principal component
    output_file_name: pca_explained_variance.csv
    """
    # compute the variance explained by each pc
    print("Compute explained variance for each PC...")
    explained_var = pd.read_table(pca_prefix+".evec", dtype=object, delim_whitespace=True).columns.to_series(name="eigenval").iloc[1:].str.split(pat=".").apply(lambda x: str(x[0])+"."+str(x[1])).astype(float) # keep only first line (column headers) of evec file and remove "#eigvals: " first value and in case number like 1.6.1 (2 '.') it just keeps the 2 first number of the list from the float splitted on '.'
    total = explained_var.sum()
    explained_var = explained_var/total*100 # explained variance in %
    explained_var.reset_index(drop=True, inplace=True)#
    explained_var.to_csv(os.path.join(output_dir, output_file_name))

    return explained_var

def get_eigenvec(pca_prefix):
    """"""
    eigenvec = pd.read_table(pca_prefix+".evec", delim_whitespace=True)
    eigenvec.columns = [*[f"PC{i}" for i in range(1, len(eigenvec.columns))], "FID"] # remove headers name (old header = first line of eigenvalues per PC)
    return eigenvec

def plot_pca_2d(output_dir, eigenvec, explained_var, pca_plot_name, number_of_snps, plot_title="2D pca", ref_colors_dict=None, case_ctrl_colors=None, pc_list=[1, 2], alpha=1, patches=None, xlim=None, ylim=None, ctrl_case_only=False, ref_only=False, grid=False, plot_ctrl_above_case=False, plot_naf=False, loc_legend="upper left"):
    """ctrl_case_only to plot only ctrl and case data without ref data
    xlim et ylim = None -> automatic detection of xlim and ylim
    If xlim et ylim != None then set xlim and ylim with given values
    patches = list or matplotlib.patches
    pc_list = tuple de 2 valeurs: PCx, PCy (in integer format) Cannot be 0 ! 
    alpha for transparency to simulate density with scatter plot (argument alpha= dans scatter) value between 0 and 1"""

    if ctrl_case_only:
        ref_only = False
    
    if ref_only:
        ctrl_case_only= False
        
    print("Ploting 2D PCA...")
    
    # split into case, ctrl and ref dataset
    if not ref_only:
        case = eigenvec.loc[eigenvec["FID"] == "Case"]
        ctrl = eigenvec.loc[eigenvec["FID"] == "Control"]

    if not ctrl_case_only:
        try:
            assert(ref_colors_dict is not None)
        except AssertionError:
            sys.stderr.write("Unable to plot reference population data because colors dictionary is None\n")
            exit(4)
        african = eigenvec.loc[eigenvec["FID"] == "Africa"]
        american = eigenvec.loc[eigenvec["FID"] == "America"]
        east_asian = eigenvec.loc[eigenvec["FID"] == "East_Asia"]
        south_asian = eigenvec.loc[eigenvec["FID"] == "South_Asia"]
        european = eigenvec.loc[eigenvec["FID"] == "Europe"]
        if plot_naf:
            naf = eigenvec.loc[eigenvec["FID"] == "North_Africa"]

    #plot pca
    dot_s = 5 # dot size
    ax = plt.subplot()

    # plot reference pop
    if not ctrl_case_only:
        ax.scatter(african[f"PC{pc_list[0]}"], african[f"PC{pc_list[1]}"], c=ref_colors_dict["Africa"], label=f"African n={len(african)}", s=dot_s, alpha=alpha) # scatter plot permet de changer individuellement les couleurs des points pour chaque sous groupe
        ax.scatter(east_asian[f"PC{pc_list[0]}"], east_asian[f"PC{pc_list[1]}"], c=ref_colors_dict["East_Asia"], label=f"East Asian n={len(east_asian)}", s=dot_s, alpha=alpha)
        ax.scatter(south_asian[f"PC{pc_list[0]}"], south_asian[f"PC{pc_list[1]}"], c=ref_colors_dict["South_Asia"], label=f"South Asian n={len(south_asian)}", s=dot_s, alpha=alpha)
        ax.scatter(european[f"PC{pc_list[0]}"], european[f"PC{pc_list[1]}"], c=ref_colors_dict["Europe"], label=f"European n={len(european)}", s=dot_s, alpha=alpha)
        ax.scatter(american[f"PC{pc_list[0]}"], american[f"PC{pc_list[1]}"], c=ref_colors_dict["America"], label=f"American n={len(american)}", s=dot_s, alpha=alpha)
        if plot_naf:
            ax.scatter(naf[f"PC{pc_list[0]}"], naf[f"PC{pc_list[1]}"], c=ref_colors_dict["North_Africa"], label=f"North African n={len(naf)}", s=dot_s, alpha=alpha)
    
    if not ref_only:
        if plot_ctrl_above_case:
                ax.scatter(case[f"PC{pc_list[0]}"], case[f"PC{pc_list[1]}"], c=case_ctrl_colors["Case"], label=f"OFSEP-HD n={len(case)}", s=dot_s)
                ax.scatter(ctrl[f"PC{pc_list[0]}"], ctrl[f"PC{pc_list[1]}"], c=case_ctrl_colors["Control"], label=f"Ctrl n={len(ctrl)}", s=dot_s)
        else:
            ax.scatter(ctrl[f"PC{pc_list[0]}"], ctrl[f"PC{pc_list[1]}"], c=case_ctrl_colors["Control"], label=f"Ctrl n={len(ctrl)}", s=dot_s)
            ax.scatter(case[f"PC{pc_list[0]}"], case[f"PC{pc_list[1]}"], c=case_ctrl_colors["Case"], label=f"OFSEP-HD n={len(case)}", s=dot_s)
    
    if xlim is None and ylim is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    # add grid and subticks
    if grid:
        minor_locator = AutoMinorLocator(10)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        plt.grid(which='both', axis="both", visible=True)

    if patches is not None:
        if isinstance(patches, list):
            for patch in patches:
                new_patch = copy(patch)
                ax.add_patch(new_patch)
        else:
            ax.add_patch(patches)

    plt.title(f"{plot_title} (n SNPs={number_of_snps})", fontsize=7)
    ax.set_xlabel("{} ({:.3f}%)".format(f"PC{pc_list[0]}", explained_var.iloc[pc_list[0]-1]))
    ax.set_ylabel("{} ({:.3f}%)".format(f"PC{pc_list[1]}", explained_var.iloc[pc_list[1]-1]))

    plt.minorticks_on()
    
    leg = plt.legend(loc=loc_legend, fontsize=7.5)
    for lh in leg.legendHandles: # remove transparency from legend
        lh.set_alpha(1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, pca_plot_name), format="pdf")
    plt.close()
    plt.clf()

    print("[Figure] PCA ancestry plot created.")

    return xlim, ylim


def plot_pca_3d(output_dir, eigenvec, explained_var, pca_plot_name, number_of_snps, plot_title="3D PCA", ref_colors_dict=None, case_ctrl_colors=None, alpha=1, xlim=None, ylim=None, zlim=None, ctrl_case_only=False, ref_only=False, plot_ctrl_above_case=False, plot_naf=False, loc_legend="upper left"):
    """ctrl_case_only to plot only ctrl and case data without ref data
    xlim et ylim = None -> automatic detection of xlim and ylim
    If xlim et ylim != None then set xlim and ylim with given values
    patches = list or matplotlib.patches"""

    print("Ploting 3D PCA...")

    if ctrl_case_only:
        ref_only = False
    
    if ref_only:
        ctrl_case_only= False
    
    # split into case, ctrl and ref dataset
    if not ref_only:
        case = eigenvec.loc[eigenvec["FID"] == "Case"]
        ctrl = eigenvec.loc[eigenvec["FID"] == "Control"]

    if not ctrl_case_only:
        try:
            assert(ref_colors_dict is not None)
        except AssertionError:
            sys.stderr.write("Unable to plot reference population data because colors dictionary is None\n")
            exit(4)
        african = eigenvec.loc[eigenvec["FID"] == "Africa"]
        american = eigenvec.loc[eigenvec["FID"] == "America"]
        east_asian = eigenvec.loc[eigenvec["FID"] == "East_Asia"]
        south_asian = eigenvec.loc[eigenvec["FID"] == "South_Asia"]
        european = eigenvec.loc[eigenvec["FID"] == "Europe"]
        if plot_naf:
            naf = eigenvec.loc[eigenvec["FID"] == "North_Africa"]

    #plot reference pop ancestry
    #ax = plt.subplot()
    dot_s = 5 # dot size
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # /!\ essayer PCA 3D car 3ème dimension explique 9% var (pas l'air de fonctionner scatter plot avec 3 arrays -> argument s en double)

    if not ctrl_case_only:
        # or use seaborn: sns.kdeplot(x=african["PC1"], y=african["PC2"], cmap=cold_dict["Africa"], fill=True, thresh=0.02 -> val à modifier pour densité, ax=ax))
        # voir 3D kde plot
        ax.scatter(african["PC1"], african["PC2"], african["PC3"], c=ref_colors_dict["Africa"], label=f"African n={len(african)}", s=dot_s, alpha=alpha) # scatter plot permet de changer individuellement les couleurs des points pour chaque sous groupe
        ax.scatter(east_asian["PC1"], east_asian["PC2"], east_asian["PC3"], c=ref_colors_dict["East_Asia"], label=f"East Asian n={len(east_asian)}", s=dot_s, alpha=alpha)
        ax.scatter(south_asian["PC1"], south_asian["PC2"], south_asian["PC3"], c=ref_colors_dict["South_Asia"], label=f"South Asian n={len(south_asian)}", s=dot_s, alpha=alpha)
        ax.scatter(european["PC1"], european["PC2"], european["PC3"], c=ref_colors_dict["Europe"], label=f"European n={len(european)}", s=dot_s, alpha=alpha)
        ax.scatter(american["PC1"], american["PC2"], american["PC3"], c=ref_colors_dict["America"], label=f"American n={len(american)}", s=dot_s, alpha=alpha)
        if plot_naf:
            ax.scatter(naf[f"PC1"], naf[f"PC2"], naf["PC3"], c=ref_colors_dict["North_Africa"], label=f"North African n={len(naf)}", s=dot_s, alpha=alpha)
    
    if not ref_only:
        if plot_ctrl_above_case:
                ax.scatter(case["PC1"], case["PC2"], case["PC3"], c=case_ctrl_colors["Case"], label=f"OFSEP-HD n={len(case)}", s=dot_s)
                ax.scatter(ctrl["PC1"], ctrl["PC2"], ctrl["PC3"], c=case_ctrl_colors["Control"], label=f"Ctrl n={len(ctrl)}", s=dot_s)
        else:
            ax.scatter(case["PC1"], case["PC2"], case["PC3"], c=case_ctrl_colors["Case"], label=f"OFSEP-HD n={len(case)}", s=dot_s)
            ax.scatter(ctrl["PC1"], ctrl["PC2"], ctrl["PC3"], c=case_ctrl_colors["Control"], label=f"Ctrl n={len(ctrl)}", s=dot_s)
    
    if xlim is None and ylim is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    plt.title(plot_title + f" (n SNPs={number_of_snps})", fontsize=7)
    ax.set_xlabel("PC1 ({:.3f}%)".format(explained_var.iloc[0]))
    ax.set_ylabel("PC2 ({:.3f}%)".format(explained_var.iloc[1]))
    ax.set_zlabel("PC3 ({:.3f}%)".format(explained_var.iloc[2]))
    
    leg = plt.legend(loc=loc_legend, fontsize=7.5)
    for lh in leg.legendHandles: # remove transparency from legend
        lh.set_alpha(1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, pca_plot_name), format="pdf")
    plt.close()
    plt.clf()

    print("[Figure] PCA ancestry plot created.")

    return xlim, ylim, zlim

def find_nb_clusters_elbow_method(eigenvec, output_dir, output_file, pc_list=[1, 2], max_nb_clusters=10):
    """This method is based on the observation that increasing the number of clusters can help in reducing the sum 
    of the within-cluster variance of each cluster. Having more clusters allows one to extract finer groups of data 
    objects that are more similar to each other. For choosing the ‘right’ number of clusters, the turning point of 
    the curve of the sum of within-cluster variances with respect to the number of clusters is used. The first 
    turning point of the curve suggests the right value of ‘k’ for any k > 0. Let us implement the elbow method in Python.
    
    output_file: define_number_of_clusters_elbow_method.pdf"""
    
    print("Computing elbow method to determine number of clusters in data")


    data_2pc = pd.concat((eigenvec.loc[eigenvec["FID"] == "Case"], eigenvec.loc[eigenvec["FID"]=="Control"]), ignore_index=True)

    # keep 2 PC from case ctrl datas
    data_2pc = data_2pc[[f"PC{pc_list[0]}", f"PC{pc_list[1]}"]].values

    # selecting optimal value of k using elbow method

    # wcss - within cluster sum of squared distance
    wcss = {}

    for k in range(2, max_nb_clusters+1):
        model = KMeans(n_clusters=k)
        model.fit(data_2pc)
        wcss[k] = model.inertia_
    
    # plotting the wcss values to find the elbow value

    plt.plot(wcss.keys(), wcss.values(), "gs-")
    plt.xlabel("Values of k")
    plt.ylabel("WCSS")
    plt.title("Within cluster sum of squared distance evolving according to number of clusters k", fontsize=8)
    plt.savefig(os.path.join(output_dir, output_file), format="pdf")
    plt.close()
    plt.clf()


def find_nb_clusters_silhouette_score(eigenvec, output_dir: str, pc_list=[1, 2], max_nb_clusters=10):
    """Nearest value to +1 is the best number of clusters.
    Silhouette score is used to evaluate the quality of clusters created using clustering algorithms such as 
    K-Means in terms of how well data points are clustered with other data points that are similar to each other. 
    This method can be used to find the optimal value of ‘k’. This score is within the range of [-1,1]. 
    The value of ‘k’ having the silhouette score nearer to 1 can be considered as the ‘right’ number of clusters. 
    """

    print("Computing silhouette score to determine number of clusters in data")

    data_2pc = pd.concat((eigenvec.loc[eigenvec["FID"] == "Case"], eigenvec.loc[eigenvec["FID"]=="Control"]), ignore_index=True)

    # keep 2 PC from case ctrl data
    data_2pc = data_2pc[[f"PC{pc_list[0]}", f"PC{pc_list[1]}"]].values

    # determining the maximum number of clusters
    scores = {}

    for k in range(2, max_nb_clusters+1):
        model = KMeans(n_clusters=k)
        model.fit(data_2pc)
        pred = model.predict(data_2pc)
        score = silhouette_score(data_2pc, pred)
        print(f"Silhouette score for k = {k}: {score}")
        scores[k] = score

    output_file = "silhouette_score_plot.pdf"
    plt.plot(scores.keys(), scores.values(), "gs-")
    plt.xlabel("Values of k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette score evolving according to number of clusters k", fontsize=8)
    plt.savefig(os.path.join(output_dir, output_file), format="pdf")
    plt.close()
    plt.clf()

    return scores # renvoie la plus grande valeur de silhouette_score (la valeur la plus proched 1)

def clustering_case_ctrl(eigenvec, n_clusters: int, pc_list=[1, 2], homemade_cluster_barycenters=None, cluster_to_modify=None):
    """
    Clustering data case and ctrl at the same time and plot the results of case and control clustered populations
    output_file: "kmeans_clustering_ofsep_hd.pdf
    homemade_cluster_barycenter: list(list()) - list of list of homemade cluster barycenters to add to the model
    cluster_to_modify: list of tuple [(index_barycenter_to_modify, [+/-x, +/-y])]
    """

    # return labelled_clusters, np.arange(0, n_clusters), cluster_centers # model_labels pour itérer dans la colonne "cluster" (cluster == i), model.cluster_centers = barycenters of each cluster
    data_2pc = pd.concat((eigenvec.loc[eigenvec["FID"] == "Case"], eigenvec.loc[eigenvec["FID"]=="Control"]), ignore_index=False)

    # keep 2 PC from case ctrl data
    data_2pc_arr = data_2pc[[f"PC{pc_list[0]}", f"PC{pc_list[1]}"]].values

    # Fit initial KMeans model to get cluster centers
    model_initial = KMeans(n_clusters=n_clusters, random_state=42)
    model_initial.fit(data_2pc_arr)

    # Combine initial cluster centers with homemade cluster barycenters
    if homemade_cluster_barycenters:
        initial_cluster_centers = model_initial.cluster_centers_
        combined_cluster_centers = np.vstack((initial_cluster_centers, np.array(homemade_cluster_barycenters)))
    else:
        combined_cluster_centers = model_initial.cluster_centers_

    # In case we need to shift a bit barycenters of clusters
    if cluster_to_modify:
        for index, xy in cluster_to_modify:
            combined_cluster_centers[index-1, 0] += xy[0]
            combined_cluster_centers[index-1, 1] += xy[1]

    # Assign each point to the nearest cluster center
    pred = np.argmin(np.linalg.norm(data_2pc_arr[:, np.newaxis] - combined_cluster_centers, axis=2), axis=1)
    labelled_clusters = pd.DataFrame(data_2pc, columns=[f"PC{pc_list[0]}", f"PC{pc_list[1]}", "FID"], index=data_2pc.index)
    labelled_clusters["cluster"] = pred

    return labelled_clusters, np.arange(0, combined_cluster_centers.shape[0]), combined_cluster_centers

def plot_case_ctrl_ref_clustered_data(labelled_clusters, labels, cluster_centers, output_dir, output_file, number_of_snps, explained_var, alpha=1, eigenvec=None, ctrl_colors=None, case_colors=None, ref_colors_dict=None, pc_list=[1, 2], xlim=None, ylim=None, plot_title="KMeans clustering case ctrl", loc_legend="upper left", plot_case=False, plot_ctrl=False, plot_ref=False, plot_naf=False):
    """"""

    try:
        if plot_ctrl:
            assert(len(ctrl_colors) == len(labels))
        if plot_case:
            assert(len(case_colors) == len(labels))
    except AssertionError:
        sys.stderr.write("[AssertionError] Length of colors list doesn't match n_clusters parameters in clustering of ctrl and case data\n")
        exit(2)

    dot_s = 5
    ax = plt.subplot()

    if plot_ref:
        try:
            assert(eigenvec is not None)
        except AssertionError:
            sys.stderr.write("[AssertionError] Unable to plot reference population data because eigenvec=None\n")
            exit(3)
        african = eigenvec.loc[eigenvec["FID"] == "Africa"]
        american = eigenvec.loc[eigenvec["FID"] == "America"]
        east_asian = eigenvec.loc[eigenvec["FID"] == "East_Asia"]
        south_asian = eigenvec.loc[eigenvec["FID"] == "South_Asia"]
        european = eigenvec.loc[eigenvec["FID"] == "Europe"]
            
        ax.scatter(african[f"PC{pc_list[0]}"], african[f"PC{pc_list[1]}"], c=ref_colors_dict["Africa"], label=f"African n={len(african)}", s=dot_s, alpha=alpha)
        ax.scatter(east_asian[f"PC{pc_list[0]}"], east_asian[f"PC{pc_list[1]}"], c=ref_colors_dict["East_Asia"], label=f"East Asian n={len(east_asian)}", s=dot_s, alpha=alpha)
        ax.scatter(south_asian[f"PC{pc_list[0]}"], south_asian[f"PC{pc_list[1]}"], c=ref_colors_dict["South_Asia"], label=f"South Asian n={len(south_asian)}", s=dot_s, alpha=alpha)
        ax.scatter(european[f"PC{pc_list[0]}"], european[f"PC{pc_list[1]}"], c=ref_colors_dict["Europe"], label=f"European n={len(european)}", s=dot_s, alpha=alpha)
        ax.scatter(american[f"PC{pc_list[0]}"], american[f"PC{pc_list[1]}"], c=ref_colors_dict["America"], label=f"American n={len(american)}", s=dot_s, alpha=alpha)
        if plot_naf:
            naf = eigenvec.loc[eigenvec["FID"] == "North_Africa"]
            ax.scatter(naf[f"PC{pc_list[0]}"], naf[f"PC{pc_list[1]}"], c=ref_colors_dict["North_Africa"], label=f"North African n={len(naf)}", s=dot_s, alpha=alpha)

    for i in labels:
        if plot_case:
            x_case = labelled_clusters[(labelled_clusters["FID"] == "Case") & (labelled_clusters["cluster"] == i)][f"PC{pc_list[0]}"]
            y_case = labelled_clusters[(labelled_clusters["FID"] == "Case") & (labelled_clusters["cluster"] == i)][f"PC{pc_list[1]}"]
            ax.scatter(x_case, y_case, c=case_colors[i], s=dot_s, label=f"cluster{i+1} - OFSEP-HD (n={len(x_case)})")
        
        if plot_ctrl:
            x_ctrl = labelled_clusters[(labelled_clusters["FID"] == "Control") & (labelled_clusters["cluster"] == i)][f"PC{pc_list[0]}"]
            y_ctrl = labelled_clusters[(labelled_clusters["FID"] == "Control") & (labelled_clusters["cluster"] == i)][f"PC{pc_list[1]}"]
            ax.scatter(x_ctrl, y_ctrl, c=ctrl_colors[i], s=dot_s, label=f"cluster{i+1} - ctrl (n={len(x_ctrl)})")
        
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=50, c="black", label="barycenter", marker='+')

    plt.title(f"{plot_title} (n SNPs = {number_of_snps})", fontsize=7)
    ax.set_xlabel(f"PC{pc_list[0]} ({explained_var.iloc[0]:.3f}%)")
    ax.set_ylabel(f"PC{pc_list[1]} ({explained_var.iloc[1]:.3f}%)")
    
    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    plt.minorticks_on()

    leg = plt.legend(loc=loc_legend, fontsize=7.5)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file), format="pdf")
    plt.close()
    plt.clf()
    print("[Figure] Kmeans clustering plot created.")

def compute_cluster_size(labelled_clusters, labels):
    """Return a dict giving number of sample per cluster and per type of patients (case, ctrl)
    labels = np.unique(model.labels_)"""

    len_clusters = {}
    for i in labels:
        len_clusters[f"case_{i}"] = len(labelled_clusters[(labelled_clusters["FID"] == "Case") & (labelled_clusters["cluster"] == i)])
        len_clusters[f"ctrl_{i}"] = len(labelled_clusters[(labelled_clusters["FID"] == "Control") & (labelled_clusters["cluster"] == i)])

    return len_clusters


def write_silhouette_scores(scores, output_dir, output_file="silhouette_scores.txt"):
    """"""
    with open(os.path.join(output_dir, output_file), 'w') as f:
        f.write("k_clusters,silhouette_score\n")
        for k, v in scores.items():
            f.write(f"{k},{v}\n")
        
    return output_file