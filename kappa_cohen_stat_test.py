#!/usr/bin/env python3
from sklearn.metrics import cohen_kappa_score
import sys
import pandas as pd
import numpy as np
import os
from modules.statistics_module import *

# script to determine if there is concordance between self-estimated origin
# and origin determined with genetic
# Test for all ancestries (clusters in your pca) and origin, origin father + origin mother.

def main():
    """"""
    try:
        declared_origin_geo = sys.argv[1]
        id_crb = sys.argv[2]
        indiv_to_keep_after_qc = sys.argv[3]
        estimated_origin_geo = sys.argv[4]
        eigenvec_file = sys.argv[5] # = labelled_clusters.csv
        admixture_file = sys.argv[6]
        output_dir = sys.argv[7]

    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: python3 kappa_cohen_stat_test.py <> <>")
        exit(1)

    ### Preparing data for statistical analysis
    np.random.seed(SEED)

    # make sample id without for example _EX180_ADNxxx-F07. Keep only 22SUJxxxxx
    with open(indiv_to_keep_after_qc, 'r') as f:
        indiv_to_keep_after_qc = [elm.split('\n')[0] for elm in f.readlines()]

    if "_" in indiv_to_keep_after_qc[0]:
        for i in range(0, len(indiv_to_keep_after_qc)):
            indiv_to_keep_after_qc[i] = indiv_to_keep_after_qc[i].split('_')[0]
            if " " in indiv_to_keep_after_qc[i]:
                indiv_to_keep_after_qc[i] = indiv_to_keep_after_qc[i].split(" ")[1]

    # load datasets
    declared_origins = pd.read_csv(declared_origin_geo, sep=';', header="infer")
    matching_hd_crb = pd.read_csv(id_crb, sep=';', header="infer")

    # convert id_hd to crb ids
    mapping_dict = matching_hd_crb.set_index("id_hd")["code_suj_crb"].to_dict()
    declared_origins["code_suj_crb"] = declared_origins["id_hd"].map(mapping_dict)
    declared_origins.drop("id_hd", inplace=True, axis=1)

    declared_origins.rename(columns={"id_hd": "code_suj_crb"}, inplace=True)
    declared_origins = declared_origins.loc[declared_origins["code_suj_crb"].isin(indiv_to_keep_after_qc)]

    # replace nan values by 8 in declared_origins
    declared_origins.fillna(7, inplace=True)

    # convert declared origins to clusters 
    declared_origins["zone_origine"] = declared_origins["zone_origine"].apply(lambda x: declared_cluster_mapping[x])
    declared_origins["zone_origine_mere"] = declared_origins["zone_origine_mere"].apply(lambda x: declared_cluster_mapping[x])
    declared_origins["zone_origine_pere"] = declared_origins["zone_origine_pere"].apply(lambda x: declared_cluster_mapping[x])

    estimated_origins = pd.read_csv(estimated_origin_geo, sep=",", header="infer")
    estimated_origins.columns = ["code_suj_crb", "PC1", "PC2", "FID", "cluster"]

    if "_" in estimated_origins.loc[0, "code_suj_crb"]:
        estimated_origins["code_suj_crb"] = estimated_origins["code_suj_crb"].apply(lambda x: x.split('_')[0])
    
    estimated_origins = estimated_origins.loc[estimated_origins["code_suj_crb"].isin(indiv_to_keep_after_qc)]

    # merge east asian and south asian clusters into Asian cluster
    estimated_origins.loc[estimated_origins["cluster"]==5, "cluster"] = 8
    estimated_origins.loc[estimated_origins["cluster"]==3, "cluster"] = 8

    # merge european and inBetweenNAFAndEuropean into European cluster
    estimated_origins.loc[estimated_origins["cluster"]==6, "cluster"] = 0

    declared_origins.sort_values(by=["code_suj_crb"], inplace=True)
    estimated_origins.sort_values(by=["code_suj_crb"], inplace=True)

    declared_origins_for_plotting = declared_origins.copy(deep=True)
    estimated_origins_for_plotting = estimated_origins.copy(deep=True)

    # get admixture data and merge dataframes
    admixture_df = pd.read_csv(admixture_file, delim_whitespace=True, header="infer")
    admixture_df = admixture_df.loc[admixture_df["SUPERPOP"]=="CASE"]
    admixture_df["SAMPLE_ID"] = admixture_df["SAMPLE_ID"].apply(lambda x: x.split('_')[0])
    admixture_df.rename({"SAMPLE_ID": "code_suj_crb"}, axis=1, inplace=True)

    # update declared and estimated df with admixture proportions
    declared_origins_admixed = pd.merge(declared_origins, admixture_df, on="code_suj_crb", how="inner")
    estimated_origins_admixed = pd.merge(estimated_origins, admixture_df, on="code_suj_crb", how="inner")

    os.makedirs(output_dir, exist_ok=True)

    ### Question 1: are there significant discrepancies between self-reported father's / mother's / both origins and child's genetically inferred ancestry ?
    # McNemar tests (paired data - no force of independance) for each parent origin (Yes/No) compared to genetically inferred ancestry of the child / self-reported origin of the child (Yes/No).
    # computation of kappa cohen statistics + McNemar test statistic
    # Done for all ancestries. Some ancestries don't have enough individuals to be actually useful.
    # McNemar test with or without Yates correction, using exact or asymptotic (chi²) distribution.
    
    stats = []
    groups = estimated_origins["cluster"].unique() # cluster numbers to consider (no 3 and 5 because use of 8 which is Asian clusters merged (East + South))

    groups.sort()
    for group in groups:

        # count ancestry
        declared_arr = declared_origins["zone_origine"].copy().to_numpy()
        declared_arr[declared_arr != group] = -2
        declared_arr[declared_arr == group] = -1

        estimated_arr = estimated_origins["cluster"].copy().to_numpy()
        estimated_arr[estimated_arr != group] = -2
        estimated_arr[estimated_arr == group] = -1

        if np.any(np.isnan(declared_arr)) or np.any(np.isnan(estimated_arr)):
            raise ValueError("Declared or estimated arrays contain NaN values, cannot compute Cohen's kappa.")
        if np.any(np.isinf(declared_arr)) or np.any(np.isinf(estimated_arr)):
            raise ValueError("Declared or estimated arrays contain infinite values, cannot compute Cohen's kappa.")

        _, k_c, mean_k_c, std_k_c = compute_kappa_cohen_bootstrap(declared_arr, estimated_arr, n_iter=1000)
        mcnemar_pval, stat, exact, correction, contingency_table, expected_n = mc_nemar_test(declared_arr, estimated_arr)
        contingency_table_df = pd.DataFrame(contingency_table, index=[f"estimated_{estimated_clusters_mapping[group]}", f"estimated_not_{estimated_clusters_mapping[group]}"], columns=[f"declared_{estimated_clusters_mapping[group]}", f"declared_not_{estimated_clusters_mapping[group]}"])
        create_contingency_table_figure(contingency_table_df, output_dir=output_dir, cluster=group)

        current_stat = [estimated_clusters_mapping[group], k_c, mean_k_c, std_k_c, mcnemar_pval, stat, exact, bool(not exact), correction]

        chi2_output = chi2_contingency(contingency_table_df, correction=correction)
        current_stat.append(chi2_output.pvalue)
        current_stat.append(chi2_output.statistic)
        if correction:                
            p_val_fisher = fisher_exact_test_mxn_tables(contingency_table_df)
            current_stat.append(p_val_fisher)
        else:
            current_stat.append(None)
        
        stats.append(tuple(current_stat))

        contingency_table_df.to_csv(os.path.join(output_dir, f"{estimated_clusters_mapping[group]}_contingency_table.csv"), sep=',')
        pd.DataFrame(expected_n, index=[f"estimated_{estimated_clusters_mapping[group]}", f"estimated_not_{estimated_clusters_mapping[group]}"], columns=[f"declared_{estimated_clusters_mapping[group]}", f"declared_not_{estimated_clusters_mapping[group]}"]).to_csv(os.path.join(output_dir, f"{estimated_clusters_mapping[group]}_estimated_frequencies.csv"), sep=',')

        barplots_agreement_disagreement_self_estimated_individual_vs_genet(contingency_table_df, os.path.join(output_dir, f"{estimated_clusters_mapping[group]}_indiv_barplot_agreement_vs_disagreement"))

    output_file = os.path.join(output_dir, "kappa_cohen_mcnemar_results_chi2_results.csv")
    with open(output_file, 'w') as f:
        f.write("population,kappa_cohen,kappa_cohen_mean,kappa_cohen_std,McNemar_pval,McNemar_stat,Mcnemar_binomial_test,Mcnemar_asymptotic_test(chi²),Yate's correction,Chi2_pval,Chi2_stat,Fisher_exact_pval\n")
        for item in stats:
            f.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]},{item[5]},{item[6]},{item[7]},{item[8]},{item[9]},{item[10]},{item[11]}\n")
        f.write("# kappa cohen: -1 no agreement, 1 perfect agreement. 0: agreement not better than what would have been obtained randomly.")
    print("[Stats] Kappa cohen statistic test DONE.")

    agreement_vs_disagreement_stratified_res = compare_agreement_vs_disagreement_proportion_selfchild_genet_stratified_analysis_independent(declared_origins, estimated_origins, clusters_list=[0, 1, 2, 4, 8])

    for cluster, res in agreement_vs_disagreement_stratified_res.items():

        # perform normality test to know if we should use mannWhitney or student test.
        res_df = pd.DataFrame(
            data = {
                "p_value_binomial_test": [v.get("p_val_binomial_test") for v in res.values()],
            },
            index = [k for k in res.keys()]
        )
        
        res_df.to_csv(os.path.join(output_dir, f"{estimated_clusters_mapping[cluster]}_indep_proportion_tests_agreement_vs_disagreement_self_vs_genet_indiv.csv"), sep=",")

    ######## PAIRED ANALYSIS PARENTS ##########

    # description declared origins vs parents for whole dataset
    descr_declared_origins, matching_declared_origins = describe_origins_ind_and_parents(declared_origins)
    descr_declared_origins.to_csv(os.path.join(output_dir, "description_matching_origins_parents_ind.csv"))
    matching_declared_origins.to_csv(os.path.join(output_dir, "stats_matching_origins_parents_ind.csv"))

    # description declared origins vs parents for each ancestry separatelyZ
    confusion_matrices = create_paired_confusion_matrix_indivs_parents_by_ancestry(declared_origins, estimated_origins, output_dir=output_dir, clusters_list=[0, 1, 2, 4, 7, 8])

    # keep self estimated confusion matrices
    self_estimated_confusion_matrices = {}
    genetically_inferred_confusion_matrices = {}
    for k, matrix in confusion_matrices.items():
        self_estimated_confusion_matrices[k] = matrix.loc[("Yes_self_estimated", "No_self_estimated"), :]
        genetically_inferred_confusion_matrices[k] = matrix.loc[("Yes_genetically_inferred", "No_genetically_inferred"), :]
        barplots_agreement_disagreement_father_mother(genetically_inferred_confusion_matrices[k], os.path.join(output_dir, f"{estimated_clusters_mapping[k]}_parents_barplot_agreement_vs_disagreement"))

    # remove unknown from genetically inferred ancestry since no genetically inferred unknown ancestry
    del genetically_inferred_confusion_matrices[7]

    results_self_estimated = mc_nemar_test_from_confusion_matrices_father_mother(self_estimated_confusion_matrices)
    results_genetically_inferred = mc_nemar_test_from_confusion_matrices_father_mother(genetically_inferred_confusion_matrices)

    # write results from parents to file (self estimated)
    for parent in ["father", "mother", "both"]:
            output_file = os.path.join(output_dir, f"self_estimated_mcnemar_matching_indiv_{parent}.csv")
            with open(output_file, 'w') as f:
                f.write("population,Kappa_Cohen,McNemar_pval,mcnemar_binomial_test,mcnemar_asymptotic_test(chi²),Yate's correction\n")
                for clust, d in results_self_estimated.items():
                    print(results_self_estimated[clust])
                    if d is not None:
                        pd.DataFrame(results_self_estimated[clust].get("expected_frq"), index=[f"estimated_{estimated_clusters_mapping[clust]}", f"estimated_not_{estimated_clusters_mapping[clust]}"], columns=[f"declared_{estimated_clusters_mapping[clust]}", f"declared_not_{estimated_clusters_mapping[clust]}"]).to_csv(os.path.join(output_dir, f"{estimated_clusters_mapping[clust]}_parent_self_estimated_{parent}_estimated_frequencies.csv"), sep=',')
                        f.write(f'{estimated_clusters_mapping[clust]},{d.get(parent).get("pvalue")},{d.get(parent).get("stat")},{d.get(parent).get("exact")==True},{d.get(parent).get("exact")==False},{d.get(parent).get("correction")}\n')
                
                f.write("\n H0: Observed differences of classification between both methods(estimated vs declared) are not significantly different, and are likely due to hazard. There is no significant disagreement between the 2 methods, there is no significant difference in the proportions.\nH1: Observed differences of classification between both methods (estimated vs declared) are significantly different and are not only due to hazard. There is a significant disagreement between the 2 methods, there is a significant difference between the 2 proportions.(p<0.05)\n")
    
    groups = estimated_origins["cluster"].unique() # cluster numbers to consider (no 3 and 5 because use of 8 which is Asian clusters merged (East + South))
    groups.sort()

    kc_parents = {k:{} for k in groups}
    corresp_fr_en = {"pere": "father", "mere": "mother", "both": "both"}
    for group in groups:
        for parent in ["pere", "mere", "both"]:

            if parent == "both":
                declared_arr = declared_origins[["zone_origine_pere", "zone_origine_mere"]].copy()
                declared_arr["zone_origine_both"] = -3
                declared_arr.loc[(declared_arr["zone_origine_pere"] != group) & (declared_arr["zone_origine_mere"] != group), "zone_origine_both"] = -2
                declared_arr.loc[(declared_arr["zone_origine_pere"] == group) & (declared_arr["zone_origine_mere"] == group), "zone_origine_both"] = -1
                declared_arr = declared_arr["zone_origine_both"].to_numpy()
            else:
                declared_arr = declared_origins[f"zone_origine_{parent}"].copy().to_numpy()
                declared_arr[declared_arr != group] = -2
                declared_arr[declared_arr == group] = -1

            estimated_arr = estimated_origins["cluster"].copy().to_numpy()
            estimated_arr[estimated_arr != group] = -2
            estimated_arr[estimated_arr == group] = -1

            if np.any(np.isnan(declared_arr)) or np.any(np.isnan(estimated_arr)):
                raise ValueError("Declared or estimated arrays contain NaN values, cannot compute Cohen's kappa.")
            if np.any(np.isinf(declared_arr)) or np.any(np.isinf(estimated_arr)):
                raise ValueError("Declared or estimated arrays contain infinite values, cannot compute Cohen's kappa.")

            _, k_c, mean_k_c, std_k_c = compute_kappa_cohen_bootstrap(declared_arr, estimated_arr, n_iter=1000)

            kc_parents[group].update({f"{corresp_fr_en[parent]}_kc": k_c, f"{corresp_fr_en[parent]}_mean_kc": mean_k_c, f"{corresp_fr_en[parent]}_std_kc": std_k_c})

    for parent in ["father", "mother", "both"]:
            output_file = os.path.join(output_dir, f"genetically_inferred_mcnemar_matching_indiv_{parent}.csv")
            with open(output_file, 'w') as f:
                f.write("population,Kappa_cohen,Mean_Kappa_cohen,Std_Kappa_Cohen,McNemar_pval,statistic,mcnemar_binomial_test,mcnemar_asymptotic_test(chi²),Yate's correction\n")
                for clust, d in results_genetically_inferred.items():
                    if d is not None:
                        pd.DataFrame(results_genetically_inferred[clust].get("expected_frq"), index=[f"estimated_{estimated_clusters_mapping[clust]}", f"estimated_not_{estimated_clusters_mapping[clust]}"], columns=[f"declared_{estimated_clusters_mapping[clust]}", f"declared_not_{estimated_clusters_mapping[clust]}"]).to_csv(os.path.join(output_dir, f"{estimated_clusters_mapping[clust]}_genetic_parent_{parent}_estimated_frequencies.csv"), sep=',')
                        f.write(f'{estimated_clusters_mapping[clust]},{kc_parents[clust].get(f"{parent}_kc")},{kc_parents[clust].get(f"{parent}_mean_kc")},{kc_parents[clust].get(f"{parent}_std_kc")},{d.get(parent).get("pvalue")},{d.get(parent).get("stat")},{d.get(parent).get("exact")==True},{d.get(parent).get("exact")==False},{d.get(parent).get("correction")}\n')
                
                f.write("\n H0: Observed differences of classification between both methods(estimated vs declared) are not significantly different, and are likely due to hazard. There is no significant disagreement between the 2 methods, there is no significant difference in the proportions.\nH1: Observed differences of classification between both methods (estimated vs declared) are significantly different and are not only due to hazard. There is a significant disagreement between the 2 methods, there is a significant difference between the 2 proportions.(p<0.05)\n")
    
    print("[Stats] McNemar tests on self-estimated individuals' / genetically inferred ancestry of individuals vs parents origins'. DONE.")

    # plotting declared origins
    plot_declared_origins_on_pca(declared_origins_for_plotting, estimated_origins_for_plotting, eigenvec_file, estimated_clusters_mapping, colors=COLORS, markers=MARKERS, output_dir=output_dir)
    print("[PlotDeclaredOrigins] DONE.")

    prop_of_disagreement_dicts = compute_proportions_of_disagreement(confusion_matrices)
    prop_of_disagreement_df = pd.DataFrame(
        data={
            "prop_disagreement_father_self_estimated": [d.get("prop_disagreement_father_self_estimated") for k, d in prop_of_disagreement_dicts.items()],
            "prop_disagreement_mother_self_estimated": [d.get("prop_disagreement_mother_self_estimated") for k, d in prop_of_disagreement_dicts.items()],
            "prop_disagreement_father_genet": [d.get("prop_disagreement_father_genet") for k, d in prop_of_disagreement_dicts.items()],
            "prop_disagreement_mother_genet": [d.get("prop_disagreement_mother_genet") for k, d in prop_of_disagreement_dicts.items()],
        },
        index=[estimated_clusters_mapping[k] for k in prop_of_disagreement_dicts.keys()]
    )
    prop_of_disagreement_df.to_csv(os.path.join(output_dir, "proportion_of_disagreement.csv"), sep=",")
    print("[ProportionDisagreement] DONE.")

    agreement_disagreement_analyses_tables = create_mother_vs_father_contingency_tables(declared_origins, estimated_origins, clusters_list=[0, 1, 2, 4, 8])
    
    # write contingency tables to file
    for cluster, d in agreement_disagreement_analyses_tables.items():
        for key, val in d.items():
            val.to_csv(os.path.join(output_dir, f"{estimated_clusters_mapping[cluster]}_{key}.csv"), sep=",")

    agreement_disagreement_results = mc_nemar_test_agreement_disagreement_analyses_father_vs_mother(agreement_disagreement_analyses_tables)

    # write results to file
    agreement_disagreement_results_df = pd.DataFrame(
        data={
            "pvalue": [d2.get("pvalue") for d in agreement_disagreement_results.values() for d2 in d.values()],
            "stat": [d2.get("stat") for d in agreement_disagreement_results.values() for d2 in d.values()],
            "correction": [d2.get("correction") for d in agreement_disagreement_results.values() for d2 in d.values()],
            "exact": [d2.get("exact") for d in agreement_disagreement_results.values() for d2 in d.values()]
        },
        index=[f"{estimated_clusters_mapping[cluster]}_{key}" for cluster, d in agreement_disagreement_results.items() for key in d.keys()]
    )
    agreement_disagreement_results_df.to_csv(os.path.join(output_dir, f"agreement_disagreement_vsparents_mcnemar_tests_results.csv"), sep=",")

    # write expected frq tables to file from agreement / disagreement analysis
    for cluster, d in agreement_disagreement_results.items():
        for key, val in d.items():

            val = val["expected_frq"]
            
            # transform expected frequencies to dataframe, and recover col and row names
            if key.find("_agreement") != -1:
                columns = ["agreement_father", "disagreement_father"]
                index = ["agreement_mother", "disagreement_mother"]

            val = pd.DataFrame(
                data = val,
                columns = columns,
                index = index,
            )
            val.to_csv(os.path.join(output_dir, f"{estimated_clusters_mapping[cluster]}_{key}_estimated_frequencies.csv"), sep=",")

    # paired proportions tests (wilcoxon + z prop test)
    paired_prop_test_results = compare_agreement_disagreement_proportion_father_mother_stratified_analysis(declared_origins, estimated_origins, clusters_list=[0, 1, 2, 4, 8])
    
    for cluster, res in paired_prop_test_results.items():
        res_df = pd.DataFrame(
            data = {
                "zstat": [v.get("zstat") for v in res.values()],
                "pvalue_ztest": [v.get("pvalue_ztest") for v in res.values()],
                "stat_wilcoxon": [v.get("stat_wilcox") for v in res.values()],
                "pvalue_wilcoxon": [v.get("pvalue_wilcox") for v in res.values()]
            },
            index = [k for k in res.keys()]
        )
        res_df.to_csv(os.path.join(output_dir, f"{estimated_clusters_mapping[cluster]}_paired_proportion_tests_parents_vs_genet_ancestry.csv"), sep=",")

    ###### INDEPENDENT ANALYSIS PARENTS #####

    # create independent analysis subdirectory
    output_dir_inde = os.path.join(output_dir, "independent_analysis")
    os.makedirs(output_dir_inde, exist_ok=True)

    inde_confusion_matrices = create_independent_confusion_matrix_indivs_parents_by_ancestry(declared_origins, estimated_origins, output_dir=output_dir_inde, clusters_list=[0, 1, 2, 4, 7, 8])
    
    # keep self estimated confusion matrices
    self_estimated_confusion_matrices_inde = {}
    genetically_inferred_confusion_matrices_inde = {}
    for k, matrix in inde_confusion_matrices.items():
        self_estimated_confusion_matrices_inde[k] = matrix.loc[("Yes_self_estimated", "No_self_estimated"), :]
        genetically_inferred_confusion_matrices_inde[k] = matrix.loc[("Yes_genetically_inferred", "No_genetically_inferred"), :]

    # remove unknown from genetically inferred ancestry since no genetically inferred unknown ancestry
    del genetically_inferred_confusion_matrices_inde[7]
    
    results_self_estimated_inde, expected_frq_self_estimated_inde = association_test_from_confusion_matrices(self_estimated_confusion_matrices_inde)
    results_genetically_inferred_inde, expected_frq_genetically_inferred_inde = association_test_from_confusion_matrices(genetically_inferred_confusion_matrices_inde)

    # create confusion matrices (contingency tables) figures for parents
    for k, matrix in genetically_inferred_confusion_matrices_inde.items(): 
        print("Here", k)
        matrix_to_plot = matrix.loc[:, ("yes_only_father_self_estimated", "yes_only_mother_self_estimated", "yes_both_self_estimated", "no_both_self_estimated")]
        create_larger_contingency_table_figure(matrix_to_plot, output_dir=output_dir_inde, cluster=k)

    # write expected frequencies to file for self_estimated and genetically inferred
    for cluster in expected_frq_self_estimated_inde.keys():
        for expected_frq_key, v in expected_frq_self_estimated_inde[cluster].items():
            pd.DataFrame(v).to_csv(os.path.join(output_dir_inde, f"self_estimated_expected_frequencies_independent_for_association_{cluster}_{expected_frq_key}.csv"), sep=",")

    for cluster in expected_frq_genetically_inferred_inde.keys():
        for expected_frq_key, v in expected_frq_genetically_inferred_inde[cluster].items():
            pd.DataFrame(v).to_csv(os.path.join(output_dir_inde, f"genetically_inferred_expected_frequencies_independent_for_association_{cluster}_{expected_frq_key}.csv"), sep=",")

    # write to files
    _ = write_association_results_to_file(results_self_estimated_inde, output_file=os.path.join(output_dir_inde, "self_estimated_inde_association_results.csv"))
    _ = write_association_results_to_file(results_genetically_inferred_inde, output_file=os.path.join(output_dir_inde, "genetically_inferred_inde_association_results.csv"))

    # tests on admixture proportions for each ancestry - self report indiv vs genet indiv (paired data)
    output_dir_admixture_prop = os.path.join(output_dir, "admixture_prop")
    os.makedirs(output_dir_admixture_prop, exist_ok=True)
    results_admixture_props = get_admixture_proportion_distributions_for_contingency_tables(declared_origins_admixed, estimated_origins_admixed, output_dir=output_dir_admixture_prop)

    # NAF ANALYSES
    # violinplots for interesting NAF %Afr and European admixture proportion to compare

    # A vs B
    admixture_prop_violin_plot(results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_yes_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_yes_genetically_inferred_prop_Africa"], distrib_1_name="Self-reported NAF", distrib_2_name="Not self-reported NAF", output_file=os.path.join(output_dir_admixture_prop, "self_report_vs_non_self_report_NAF_genet_inferred_prop_African(AvsB).pdf"))
    admixture_prop_violin_plot(results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_yes_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_yes_genetically_inferred_prop_Europe"], distrib_1_name="Self-reported NAF", distrib_2_name="Not self-reported NAF", output_file=os.path.join(output_dir_admixture_prop, "self_report_vs_non_self_report_NAF_genet_inferred_prop_European(AvsB).pdf"))
    
    # C vs D
    admixture_prop_violin_plot(results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_no_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_no_genetically_inferred_prop_Africa"], distrib_1_name="Self-reported NAF", distrib_2_name="Not self-reported NAF", output_file=os.path.join(output_dir_admixture_prop, "self_report_vs_non_self_report_non_NAF_genet_inferred_prop_African(CvsD).pdf"))
    admixture_prop_violin_plot(results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_no_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_no_genetically_inferred_prop_Europe"], distrib_1_name="Self-reported NAF", distrib_2_name="Not self-reported NAF", output_file=os.path.join(output_dir_admixture_prop, "self_report_vs_non_self_report_non_NAF_genet_inferred_prop_European(CvsD).pdf"))

    # B vs C
    admixture_prop_violin_plot(results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_yes_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_no_genetically_inferred_prop_Africa"], distrib_1_name="Not self-reported NAF & Genetically inferred NAF", distrib_2_name="Self-reported NAF and Not genetically inferred NAF", output_file=os.path.join(output_dir_admixture_prop, "not_self_report_NAF_and_genetic_NAF_vs_self_report_NAF_and_not_NAF_genet_inferred_prop_African(BvsC).pdf"))
    admixture_prop_violin_plot(results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_yes_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_no_genetically_inferred_prop_Europe"], distrib_1_name="Not self-reported NAF & Genetically inferred NAF", distrib_2_name="Self-reported NAF and Not genetically inferred NAF", output_file=os.path.join(output_dir_admixture_prop, "not_self_report_NAF_and_genetic_NAF_vs_self_report_NAF_and_not_NAF_genet_inferred_prop_European(BvsC).pdf"))
    
    # test normality for each distrib and create qq plots
    distributions_list = [
        results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_yes_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_yes_genetically_inferred_prop_Africa"], 
        results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_yes_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_yes_genetically_inferred_prop_Europe"], 
        results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_no_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_no_genetically_inferred_prop_Africa"], 
        results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_no_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_no_genetically_inferred_prop_Europe"]
    ]

    distribution_names = [
        "yes_self_estimated_&_yes_genetically_inferred_prop_Africa", "no_self_estimated_&_yes_genetically_inferred_prop_Africa",
        "yes_self_estimated_&_yes_genetically_inferred_prop_Europe", "no_self_estimated_&_yes_genetically_inferred_prop_Europe",
        "yes_self_estimated_&_no_genetically_inferred_prop_Africa", "no_self_estimated_&_no_genetically_inferred_prop_Africa",
        "yes_self_estimated_&_no_genetically_inferred_prop_Europe", "no_self_estimated_&_no_genetically_inferred_prop_Europe"
    ]

    print("\n##### Normality test previous to admixture proportions statistical analyses: #####\n")
    normality_results = normality_test(distributions_list, distribution_names)
    print(normality_results)

    print("Creating QQ plots for each distribution compared to a normal distribution...")
    for i in range(0, len(distributions_list)):
        qq_plot_against_normal(distributions_list[i], output_file=os.path.join(output_dir_admixture_prop, f"qq_plot_{distribution_names[i]}_VS_normal_distribution.pdf"))

    # specific statistical test comparing admixture proportions between specific subgroup.
    print("\n##### Comparing admixture proportions - Statistical analyses #####\n")

    # compare prop Afr % between self reported and not self reported among NAF genetically inferred
    t_stat, pval_t, mean1, mean2, std1, std2 = student_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_yes_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_yes_genetically_inferred_prop_Africa"])
    u_stat, pval_mann, mean1, mean2, std1, std2 = mannwhitney_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_yes_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_yes_genetically_inferred_prop_Africa"])
    print("Test self_estimated vs not self estimated for genetically inferred as NAF (test on African admixture %) A vs B: ", "Student: ", t_stat, pval_t, "MannWhitney: ", u_stat, pval_mann, "mean A: ", mean1, "mean B: ", mean2, "std A: ", std1, "std B: ", std2)

    # compare prop euro % between self reported and not self reported among NAF genetically inferred
    t_stat, pval_t, mean1, mean2, std1, std2 = student_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_yes_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_yes_genetically_inferred_prop_Europe"])
    u_stat, pval_mann, mean1, mean2, std1, std2 = mannwhitney_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_yes_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_yes_genetically_inferred_prop_Europe"])
    print("Test self_estimated vs not self estimated for genetically inferred as NAF (test on European admixture %) A vs B: ", "Student: ", t_stat, pval_t, "MannWhitney: ", u_stat, pval_mann, "mean A: ", mean1, "mean B: ", mean2, "std A: ", std1, "std B: ", std2)

    # compare prop Afr % between self reported and not self reported among not NAF genetically inferred
    t_stat, pval_t, mean1, mean2, std1, std2 = student_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_no_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_no_genetically_inferred_prop_Africa"])
    u_stat, pval_mann, mean1, mean2, std1, std2 = mannwhitney_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_no_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_no_genetically_inferred_prop_Africa"])
    print("Test self_estimated vs not self estimated for not genetically inferred as NAF (test on African admixture %) C vs D: ", "Student: ", t_stat, pval_t, "MannWhitney: ", u_stat, pval_mann, "mean C: ", mean1, "mean D: ", mean2, "std C: ", std1, "std D: ", std2)

    # compare prop European % between self reported and not self reported among not NAF genetically inferred
    t_stat, pval_t, mean1, mean2, std1, std2 = student_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_no_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_no_genetically_inferred_prop_Europe"])
    u_stat, pval_mann, mean1, mean2, std1, std2 = mannwhitney_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_no_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_no_genetically_inferred_prop_Europe"])
    print("Test self_estimated vs not self estimated for not genetically inferred as NAF (test on European admixture %) C vs D: ", "Student: ", t_stat, pval_t, "MannWhitney: ", u_stat, pval_mann, "mean C: ", mean1, "mean D: ", mean2, "std C: ", std1, "std D: ", std2)

    # compare prop Afr % between self reported and not NAF geneticlly inferred vs not self reported and NAF genetically inferred
    t_stat, pval_t, mean1, mean2, std1, std2 = student_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_yes_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_no_genetically_inferred_prop_Africa"])
    u_stat, pval_mann, mean1, mean2, std1, std2 = mannwhitney_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_yes_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_no_genetically_inferred_prop_Africa"])
    print("Test not self_estimated and genetically inferred NAF vs self estimated and not genetically inferred as NAF (test on African admixture %): B vs C ","Student: ", t_stat, pval_t, "MannWhitney: ", u_stat, pval_mann, "mean B: ", mean1, "mean C: ", mean2, "std B: ", std1, "std C: ", std2)

    # compare prop Europe % between self reported and not NAF geneticlly inferred vs not self reported and NAF genetically inferred
    t_stat, pval_t, mean1, mean2, std1, std2 = student_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_yes_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_no_genetically_inferred_prop_Europe"])
    u_stat, pval_mann, mean1, mean2, std1, std2 = mannwhitney_test_compare_2_distributions(results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_yes_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_no_genetically_inferred_prop_Europe"])
    print("Test not self_estimated and genetically inferred NAF vs self estimated and not genetically inferred as NAF (test on European admixture %): B vs C", "Student: ", t_stat, pval_t, "MannWhitney: ", u_stat, pval_mann, "mean B: ", mean1, "mean C: ", mean2, "std B: ", std1, "std C: ", std2)

    # prepare data for contingency table figures for admixture proportions - create 3 contingency tables - one for european and one for African and one for Asian genome proportions
    # only for NAF populations
    cross_table_afr_prop_distrib = np.array([
        [distributions_list[0], distributions_list[1]],
        [distributions_list[4], distributions_list[5]]
    ], dtype=np.float32)

    cross_table_eur_prop_distrib = np.array([
        [distributions_list[2], distributions_list[3]],
        [distributions_list[6], distributions_list[7]]
    ], dtype=np.float32)

    cross_table_asia_prop_distrib = np.array([
        [results_admixture_props.get("North-African cluster_East_Asia")["yes_self_estimated_&_yes_genetically_inferred_prop_East_Asia"], results_admixture_props.get("North-African cluster_East_Asia")["no_self_estimated_&_yes_genetically_inferred_prop_East_Asia"]],
        [results_admixture_props.get("North-African cluster_East_Asia")["yes_self_estimated_&_no_genetically_inferred_prop_East_Asia"], results_admixture_props.get("North-African cluster_East_Asia")["no_self_estimated_&_no_genetically_inferred_prop_East_Asia"]]
    ])
    
    results_admixture_props.get("North-African cluster_Africa")["yes_self_estimated_&_yes_genetically_inferred_prop_Africa"], results_admixture_props.get("North-African cluster_Africa")["no_self_estimated_&_yes_genetically_inferred_prop_Africa"], 
    results_admixture_props.get("North-African cluster_Europe")["yes_self_estimated_&_yes_genetically_inferred_prop_Europe"], results_admixture_props.get("North-African cluster_Europe")["no_self_estimated_&_yes_genetically_inferred_prop_Europe"],
    
    create_admixture_table_figure(cross_table_afr_prop_distrib, cross_table_eur_prop_distrib, cross_table_asia_prop_distrib, cluster=2, output_dir=output_dir_admixture_prop)
    
    # pd.DataFrame(
    #     data = {
    #         "yes_self_estimated_North_African": [distributions_list[0].mean(), distributions_list[1].mean()],
    #         "no_self_estimated_North_African": [distributions_list[4].mean(), distributions_list[5].mean()]
    #     },
    #     index = ["yes_North_African_genetic_ancestry", "no_North_African_genetic_ancestry"]
    # )


    # results:
    # 'yes_self_estimated_&_yes_genetically_inferred_prop_Africa' -> comes from normal distribution.
    # 'no_self_estimated_&_yes_genetically_inferred_prop_Africa' -> not normal distribution
    #  'yes_self_estimated_&_yes_genetically_inferred_prop_Europe' -> normal distribution
    # 'no_self_estimated_&_yes_genetically_inferred_prop_Europe' -> pas normal distribution même si kstest dis de très peu (très légèrement > 0.05) normal distribution. --> donc pas normal car les autres tests disent pas normal distrib
    #  'yes_self_estimated_&_no_genetically_inferred_prop_Africa' -> shapiro --> not normal distribution
    # 'no_self_estimated_&_no_genetically_inferred_prop_Africa' -> très très not normal distribution
    # 'yes_self_estimated_&_no_genetically_inferred_prop_Europe' ->  not normal distribution
    # 'no_self_estimated_&_no_genetically_inferred_prop_Europe' --> très très not normal distribution



if __name__ == "__main__":
    main()
