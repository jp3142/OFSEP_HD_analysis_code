#!/usr/bin/env python3

from modules.summary_module import *

# script for summary table

def main():
    """"""

    estimated_clusters_mapping = {
    0: "CloserToEuropeanReferenceCluster",
    1: "InBetweenNorthAfricanAndAfricanAndAfrocaribbeanReferenceClusters",
    2: "CloserToNorthAfricanReferenceCluster",
    3: "CloserToEastAsianReferenceCluster",
    4: "CloserToAfricanAndAfrocaribbeanReferenceCluster",
    5: "CloserToSouthAsianReferenceCluster",
    6: "InBetweenNorthAfricanAndEuropeanReferenceClusters"
    } # map cluster numbers to cluster names

    palette_mapping = {
        "CloserToEuropeanReferenceCluster": "Blues",
        "InBetweenNorthAfricanAndAfricanAndAfrocaribbeanReferenceClusters": "Purples",
        "CloserToNorthAfricanReferenceCluster": "coolwarm",
        "CloserToEastAsianReferenceCluster": "viridis",
        "CloserToAfricanAndAfrocaribbeanReferenceCluster": "Reds",
        "CloserToSouthAsianReferenceCluster": "Greens",
        "CloserToAsianReferenceClusters": "viridis",
        "InBetweenNorthAfricanAndEuropeanReferenceClusters": "cool"
    } # map cluster names to colors

    try:
        parameters_file = sys.argv[1]
        output_dir = sys.argv[2]
    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: main4_summary_table.py <parameters_file> <output_dir>\n")
        exit(1)

    os.makedirs(output_dir, exist_ok=True)

    parameters_dict = read_parameters_file(parameters_file)

    _, all_indiv_df = get_all_individuals(output_dir, parameters_dict["all_indiv_prefix"]+".fam")

    # QC variables
    f_stats = get_fstats(parameters_dict["fstat_file"])
    het = get_het_rate(parameters_dict["het_file"])
    relatedness = get_relatedness(parameters_dict["relatedness_file"], parameters_dict["ind_to_keep_for_frequency_analysis"]) # une ligne pour chaque couple d'individu à mettre à part

    #Ancestry variables
    pc_values = get_eigenvectors(parameters_dict["eigenvec_file"])

    # admixed_ancestry
    admixed_prop = get_admixture(parameters_dict["admixed_proportion_file"])

    # create snp matrix
    _, snp_matrix, variant_names = get_snp_matrix(parameters_dict["after_imputation_dataset_prefix"], parameters_dict["variants_allele_or_list_file"], output_dir)

    # merge tables f_stats, het, pc_values, admixed_prop 
    merged_table = merge_data([f_stats, het, pc_values, admixed_prop, snp_matrix])

    # add exclude column (an individual is considered as excluded if it isn't in the )
    merged_table = pd.merge(merged_table, all_indiv_df, how="outer", right_index=True, left_index=True)

    list_indiv_kept = get_list_indiv_kept(parameters_dict["list_indiv_kept"])
    merged_table.reset_index(inplace=True, drop=False)

    merged_table['excludedQC'] = merged_table["IID"].apply(lambda x: 0 if x in list_indiv_kept else 1)    

    merged_table = merge_ancestry_cluster_and_description_table(merged_table, parameters_dict["labelled_clusters_ancestry"])  

    # Compute and plot prs (MSGB + risk allele count)
    hla_variants_to_remove = get_hla_variant_list(parameters_dict.get("variants_hla_to_remove"))

    # compute prs
    merged_table, msgb_col1, risk_all_count_col1 = compute_prs_by_individual(parameters_dict.get("variant_or_freq_file"), merged_table, list_indiv_kept, hla_variants_to_remove, sep=";", use_freq_as_dose_when_absent=False)
    merged_table, msgb_col2, risk_all_count_col2 = compute_prs_by_individual(parameters_dict.get("variant_or_freq_file"), merged_table, list_indiv_kept, hla_variants_to_remove, sep=";", use_freq_as_dose_when_absent=True)
    # save tables
    merged_table.to_csv(os.path.join(output_dir, "OFSEP_HD_description.csv"), sep=';', header=True, index=True)
    relatedness.to_csv(os.path.join(output_dir, "OFSEP_HD_relatedness.csv"), sep=';', header=True, index=True)

    # plot prs and save statistics to a file
    prs_col_to_plot = [msgb_col1, risk_all_count_col1, msgb_col2, risk_all_count_col2]

    ##### plot PRS scores #####

    prs_col_to_plot = ["MSGB", "risk_alleles_count", "MSGB_freq_as_dose_when_absent", "risk_alleles_count_freq_when_absent"]
    prs_new_col_names = {
        "MSGB": "MSGB",
        "risk_alleles_count": "Risk Alleles Count",
        "MSGB_freq_as_dose_when_absent": "MSGB Frequency as dose when missing",
        "risk_alleles_count_freq_when_absent": "Risk Alleles Count Frequency as dose when missing"
    } # mapping dict to rename columns for plotting

    merged_table = pd.read_csv(os.path.join(output_dir, "OFSEP_HD_description.csv"), sep=';', header="infer")
    merged_table_to_plot = merged_table.loc[merged_table["excludedQC"]==0, :]

    merged_table_to_plot.rename(prs_new_col_names, inplace=True, axis=1)
    prs_col_to_plot = [prs_new_col_names[col] for col in prs_col_to_plot]

    statistics_global = plot_prs(merged_table_to_plot, prs_col_to_plot, list_indiv_kept, output_dir, scaling=1)
    statistics_by_ancestry = plot_prs_by_ancestry(merged_table_to_plot, prs_col_to_plot, list_indiv_kept, estimated_clusters_mapping, palette_mapping, output_dir, merge_asian=True)

    print("[END SUMMARY]")


if __name__ == "__main__":
    main()