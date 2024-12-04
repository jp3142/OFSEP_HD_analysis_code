#!/usr/bin/env python3

from modules.admixed_ancestry import *
from modules.ancestry2 import *
from config_main3_admixed_ancestry_analysis import *
import sys 

# /!\ ADMIXED ANCESTRY TO DO ON ALL INDIVIDUALS JUST BEFORE SMARTPCA, not only on individuals that were kept after the pc if you chose to use outlier removal option of smartpca tool.

def main():

    ############## VARIABLE ####################
    # deducted subpopulations to keep from subpop_code_to_remove
    subpop_code_to_keep = list(set([code for codes in matching_superpop_subpop.values() for code in codes]) - set(subpop_code_to_remove))
    ###########################################

    try:
        plink_prefix = sys.argv[1]
        kg_list_file = sys.argv[2] # igsr_samples.tsv
        labelled_clusters = sys.argv[3]
        output_dir = sys.argv[4]
    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: main3_admixed_ancestry_analysis.py <plink_prefix> <igsr_samples.tsv> <output_dir>\n")
        exit(1)
         # Execution example: python3 main3_admixed_ancestry_analysis.py main2_output_last/merged_pruned_dataset main3_test_dir data_for_main3/igsr_samples.tsv 
       
    os.makedirs(output_dir, exist_ok=True)

    # ###########################################################
    # # DEVELOPMENT
    # # Faster exec (to avoid executing admixture again)
    # #plink_prefix="main3_output_test21123/merged_pruned_dataset"
    # #plink_prefix = "main3_test_dir/merged_pruned_dataset"
    # #plink_prefix = "main3_automatisation_test/merged_pruned_dataset"
    # #plink_prefix = "main3_output_sortEU151123/merged_pruned_dataset_only_indiv_to_keep"
    # ###########################################################
    
    cv_errors = {} # store k_value: cv_error

    plink_prefix = change_ctrl_case_name(plink_prefix)

    for k in corres_k_superpop.keys():
        # generate list of individuals to keep

        indiv_to_keep_file = generate_list_indiv_to_keep(plink_prefix, output_dir, corres_k_superpop[k]+["CASE", "CTRL"], subpop_code_to_keep, kg_list_file)
        
        # generating dataset with only needed individuals
        plink_prefix = generate_plink_dataset_with_indiv_to_keep(plink_prefix, output_dir, indiv_to_keep_file)

        # generate pop file
        _ = generate_pop_file(plink_prefix) # for --supervised option of admixture software
        # run admixture
        print(f"### ADMIXTURE FOR K={k} ###")
        Q_file, _, log_file = admixture(plink_prefix, k, output_dir)

    #     ###########################################################
    #     # DEVELOPMENT
    #     # Faster exec (to avoid executing admixture again)
    #     #Q_file = "main3_output_test21123/merged_pruned_dataset.3.Q" # Faster exec (to avoid reexecuting admixture)
    #     #log_file = "main3_output_test21123/merged_pruned_dataset_log3.out"
    #     #Q_file = "main3_test_dir/merged_pruned_dataset.6.Q"
    #     #log_file = "main3_test_dir/merged_pruned_dataset_log6.out"
    #     #Q_file = "main3_automatisation_test/merged_pruned_dataset.4.Q"
    #     #log_file = "main3_automatisation_test/merged_pruned_dataset_log4.out"
    #     #Q_file = "main3_output_sortEU151123/merged_pruned_dataset_only_indiv_to_keep.3.Q"
    #     #log_file = "main3_output_sortEU151123/merged_pruned_dataset_only_indiv_to_keep_log3.out"
    #     ###########################################################

        # compute cv error
        cv_errors[k] = get_cross_validation_error(log_file)
        
        Q_df, Q_file = merge_Q_and_fam_file(plink_prefix, Q_file, kg_list_file, output_dir)
        Q_df, Q_file = label_columns(Q_df, Q_file, corres_k_superpop[k])
        plot_admixed_ancestry(Q_df, output_dir, corres_k_superpop[k], custom_colors, sort_by=sort_by)

    plot_elbow_cross_validation_error(cv_errors, output_dir, "plot_cv_errors.pdf")

    _ = plot_agg_ancestry_proportions(Q_df, output_dir, case_only=False)


    # ####### testing - Fast exec - If no need to run admixture again #######
    # Q_df = pd.read_csv("main3_output_for_julien_160924/merged_pruned_dataset_removedInd_removedFreqDiffSnps0_fam_labelled.1_noATGC_only_indiv_to_keep.3", delim_whitespace=True)
    # os.makedirs(output_dir, exist_ok=True)
    # labelled_clusters_df = plot_admixture_proportions_by_pca_cluster(Q_df, labelled_clusters, output_dir)
    # for k in corres_k_superpop.keys():
    #     plot_admixed_ancestry(Q_df, labelled_clusters_df, output_dir, corres_k_superpop[k], custom_colors, case_colors_clusters_plt_names, sort_by="Africa")
    #     plot_admixed_ancestry(Q_df, labelled_clusters_df, output_dir, corres_k_superpop[k], custom_colors, case_colors_clusters_plt_names, sort_by="East_Asia")



if __name__ == "__main__":
    main()