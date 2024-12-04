#!/usr/bin/env python3
from sklearn.metrics import cohen_kappa_score
import sys
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import norm
from scipy.stats import chi2_contingency, kstest, anderson, normaltest, shapiro, fisher_exact
import statsmodels.api as sm

# import R modules for Fisher exact test on mxn contingency tables
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector
import plotly.express as px

# kstest two-sided: The null hypothesis is that the two distributions are identical, F(x)=G(x) -> follow normal distribution (since in one sample test it compares to normal distribution) for all x; the alternative is that they are not identical.
# anderson: Critical values provided are for the following significance levels: normal/exponential 15%, 10%, 5%, 2.5%, 1% --> pour ça qu'on a un array dans critical_values returned by the test. If the returned statistic is larger > than these critical values then for the corresponding significance level, the null hypothesis that the data come from the chosen distribution can be rejected.
# normaltest: statistic float or array --> s^2 + k^2, where s is the z-score returned by skewtest and k is the z-score returned by kurtosistest. This function tests the null hypothesis that a sample comes from a normal distribution. It is based on D’Agostino and Pearson’s [1], [2] test that combines skew and kurtosis to produce an omnibus test of normality.
# shapiro: The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution. (for small dataset - low number of data < 50)

SEED = 42
COLORS = np.array(["slateblue", "mediumvioletred", "magenta" , None, "indianred", None, "rebeccapurple", None, "yellow"])
MARKERS = np.array(["o", "s", "^", None, "d", None, "+", None, "*"])

estimated_clusters_mapping = {
    0: "European cluster",
    1: "Afro-Caribbean cluster",
    2: "North-African cluster",
    3: "East-Asian cluster",
    4: "African cluster",
    5: "South-Asian cluster",
    6: "Admixed European / North-African cluster",
    7: "unknown",
    8: "Asian cluster"
}

declared_cluster_mapping = {
    1: 0,
    2: 1,
    3: 0,
    4: 2,
    5: 4,
    6: 8,
    7: 7,
    8: 7,
}

# import required R packages
base = importr('base')
stats_r = importr('stats')

def compute_kappa_cohen_bootstrap(declared_arr, estimated_arr, n_iter=1000):

    kappa_val = cohen_kappa_score(declared_arr, estimated_arr)
    distrib_kappa = []

    # Perform bootstrap resampling
    for _ in range(n_iter):
        # Resample with replacement
        indices = np.random.randint(0, len(declared_arr), len(estimated_arr))
        sample_a = declared_arr[indices]
        sample_b = estimated_arr[indices]
        
        # Calculate Cohen's Kappa for the bootstrap sample
        kappa_value = cohen_kappa_score(sample_a, sample_b)
        distrib_kappa.append(kappa_value)

    # Calculate the mean and standard deviation of Kappa values
    mean_kappa = np.mean(distrib_kappa)
    std_kappa = np.std(distrib_kappa)

    return distrib_kappa, kappa_val, mean_kappa, std_kappa

def mc_nemar_test(declared, estimated):
    """
    Perform McNemar test for paired nominal data.
    Automatic test to know if better to use exact (binomial) OR not (chi square), yales correction or not

    H0: Observed differences of classification between both methods (estimated vs declared) are not significantly different, and are likely due to hazard. There is no significant disagreement between the 2 methods.
    H1: Observed differences of classification between both methods (estimated vs declared) are significantly different and are not only due to hazard. There is a significant disagreement between the 2 methods.
    """
    ### Contingency table
    ##          declared_-1 declared_-2 
    ## estimated_-1
    ## estimated_-2

    # Define if we need to use exact test and Yale's correction
    exact = False
    correction = False
    n_discordant_pairs = len(np.where(declared != estimated)[0])
    if n_discordant_pairs <= 0:
        raise ValueError("Number of discordant pairs is zero, cannot perform McNemar's test.")
    
    contingency_table = np.array([
        [np.sum((declared == -1) & (estimated == -1)), np.sum((declared == -2) & (estimated == -1))],
        [np.sum((declared == -1) & (estimated == -2)), np.sum((declared == -2) & (estimated == -2))]
    ], dtype=np.int32)

    if np.any(contingency_table < 0):
        raise ValueError("Contingency table contains negative values, cannot perform McNemar's test.")

    # Compute row sums and column sums
    row_sums = np.sum(contingency_table, axis=1)
    col_sums = np.sum(contingency_table, axis=0)

    # Compute expected frequencies (expected number of individuals)
    total_obs = np.sum(row_sums)
    if total_obs == 0:
        raise ValueError("Total observations (total_obs) is zero, cannot compute expected frequencies.")
    expected_frq = np.outer(row_sums, col_sums) / total_obs
    expected_frq_flatten = expected_frq.flatten()

    # check if sample size is (number of discordant pairs) small (<25) or if any cell frequencies >= 0.05
    if np.any(expected_frq_flatten <= 5):
        correction = True

    if n_discordant_pairs <= 10:
        exact = True
    
    result = mcnemar(contingency_table, exact=exact, correction=correction) # exact = False to use chi square distribution
    p_value = result.pvalue
    stat = result.statistic

    return p_value, stat, exact, correction, contingency_table, expected_frq

def compute_proportions_of_disagreement(confusion_matrices: dict):
    """For genetic and self estimated"""
    results = {
        k:None for k in confusion_matrices.keys()
    }

    for cluster, confusion_matrix in confusion_matrices.items():
        confusion_matrix_father = confusion_matrix.loc[:, ("yes_father_self_estimated", "no_father_self_estimated")]
        confusion_matrix_father_self_estimated = confusion_matrix_father.loc[("Yes_self_estimated", "No_self_estimated"), :] # self estimated for individual
        confusion_matrix_father_genet = confusion_matrix_father.loc[("Yes_genetically_inferred", "No_genetically_inferred"), :] # self estimated for individual

        confusion_matrix_mother = confusion_matrix.loc[:, ("yes_mother_self_estimated", "no_mother_self_estimated")]
        confusion_matrix_mother_self_estimated = confusion_matrix_mother.loc[("Yes_self_estimated", "No_self_estimated"), :] # self estimated for individual
        confusion_matrix_mother_genet = confusion_matrix_mother.loc[("Yes_genetically_inferred", "No_genetically_inferred"), :] # self estimated for individual

        # compute proportion of disagreement as ( B+C ) / (A + B + C + D )
        prop_disagreement_father_self_estimated = ( confusion_matrix_father_self_estimated.iloc[0, 1] + confusion_matrix_father_self_estimated.iloc[1, 0] ) / ( confusion_matrix_father_self_estimated.iloc[0, 0] + confusion_matrix_father_self_estimated.iloc[0, 1] + confusion_matrix_father_self_estimated.iloc[1, 0] + confusion_matrix_father_self_estimated.iloc[1, 1] ) 
        prop_disagreement_mother_self_estimated = ( confusion_matrix_mother_self_estimated.iloc[0, 1] + confusion_matrix_mother_self_estimated.iloc[1, 0] ) / ( confusion_matrix_mother_self_estimated.iloc[0, 0] + confusion_matrix_mother_self_estimated.iloc[0, 1] + confusion_matrix_mother_self_estimated.iloc[1, 0] + confusion_matrix_mother_self_estimated.iloc[1, 1] ) 

        prop_disagreement_father_genet = ( confusion_matrix_father_genet.iloc[0, 1] + confusion_matrix_father_genet.iloc[1, 0] ) / ( confusion_matrix_father_genet.iloc[0, 0] + confusion_matrix_father_genet.iloc[0, 1] + confusion_matrix_father_genet.iloc[1, 0] + confusion_matrix_father_genet.iloc[1, 1] ) 
        prop_disagreement_mother_genet = ( confusion_matrix_mother_genet.iloc[0, 1] + confusion_matrix_mother_genet.iloc[1, 0] ) / ( confusion_matrix_mother_genet.iloc[0, 0] + confusion_matrix_mother_genet.iloc[0, 1] + confusion_matrix_mother_genet.iloc[1, 0] + confusion_matrix_mother_genet.iloc[1, 1] ) 
        results[cluster] = {"prop_disagreement_father_self_estimated": prop_disagreement_father_self_estimated, "prop_disagreement_mother_self_estimated": prop_disagreement_mother_self_estimated, "prop_disagreement_father_genet": prop_disagreement_father_genet, "prop_disagreement_mother_genet": prop_disagreement_mother_genet}

    return results

def mc_nemar_test_from_confusion_matrices_father_mother(confusion_matrices: dict):

    results = {
        k:None for k in confusion_matrices.keys()
    }
    
    for cluster, confusion_matrix in confusion_matrices.items():
        exact_father = False
        correction_father = False
        exact_mother = False
        correction_mother = False
        exact_both = False
        correction_both = False

        confusion_matrix_father = confusion_matrix.loc[:, ("yes_father_self_estimated", "no_father_self_estimated")].to_numpy()
        confusion_matrix_mother = confusion_matrix.loc[:, ("yes_mother_self_estimated", "no_mother_self_estimated")].to_numpy()
        confusion_matrix_both = confusion_matrix.loc[:, ("yes_both_self_estimated", "no_both_self_estimated")].to_numpy()

        # Compute row sums and column sums
        row_sums_father = np.sum(confusion_matrix_father, axis=1)
        col_sums_father = np.sum(confusion_matrix_father, axis=0)
        n_discordant_pairs_father = np.sum([confusion_matrix_father[0, 1], confusion_matrix_father[1, 0]])

        row_sums_mother = np.sum(confusion_matrix_mother, axis=1)
        col_sums_mother = np.sum(confusion_matrix_mother, axis=0)
        n_discordant_pairs_mother = np.sum([confusion_matrix_mother[0, 1], confusion_matrix_mother[1, 0]])

        row_sums_both = np.sum(confusion_matrix_both, axis=1)
        col_sums_both = np.sum(confusion_matrix_both, axis=0)
        n_discordant_pairs_both = np.sum([confusion_matrix_both[0, 1], confusion_matrix_both[1, 0]])

        # Compute expected frequencies (expected number of individuals)
        total_obs_father = np.sum(row_sums_father)
        total_obs_mother = np.sum(row_sums_mother)
        total_obs_both = np.sum(row_sums_both)

        # compute for father
        if total_obs_father == 0:
            sys.stderr.write(f"Total observations (total_obs) is zero for father {estimated_clusters_mapping[cluster]}, cannot compute expected frequencies.\n")
            continue

        expected_frq_father = np.outer(row_sums_father, col_sums_father) / total_obs_father
        expected_frq_flatten = expected_frq_father.flatten()

        # check if sample size is (number of discordant pairs) small (<25) or if any cell frequencies >= 0.05
        if np.any(expected_frq_flatten <= 5):
            correction_father = True

        if n_discordant_pairs_father <= 10:
            exact_father = True    

        result_father = mcnemar(confusion_matrix_father, exact=exact_father, correction=correction_father) # exact = False to use chi square distribution
        p_value_father = result_father.pvalue
        stats_father = result_father.statistic

        # compute for mother
        if total_obs_mother == 0:
            sys.stderr.write(f"Total observations (total_obs) is zero for mother {estimated_clusters_mapping[cluster]}, cannot compute expected frequencies.\n")
            continue

        expected_frq_mother = np.outer(row_sums_mother, col_sums_mother) / total_obs_mother
        expected_frq_flatten = expected_frq_mother.flatten()
        # check if sample size is (number of discordant pairs) small (<25) or if any cell frequencies >= 0.05
        if np.any(expected_frq_flatten <= 5):
            correction_mother = True

        if n_discordant_pairs_mother <= 10:
            exact_mother = True    

        result_mother = mcnemar(confusion_matrix_mother, exact=exact_mother, correction=correction_mother) # exact = False to use chi square distribution
        p_value_mother = result_mother.pvalue
        stats_mother = result_mother.statistic

        if total_obs_both == 0:
            sys.stderr.write(f"Total observations (total_obs) is zero for both parents {estimated_clusters_mapping[cluster]}, cannot compute expected frequencies.\n")
            continue
        
        # check if sample size is (number of discordant pairs) small (<25) or if any cell frequencies >= 0.05
        if np.any(expected_frq_flatten <= 5):
            correction_both = True

        if n_discordant_pairs_both <= 10:
            exact_both = True    

        expected_frq_both = np.outer(row_sums_both, col_sums_both) / total_obs_both
        expected_frq_flatten = expected_frq_both.flatten()

        result_both = mcnemar(confusion_matrix_both, exact=exact_both, correction=correction_both) # exact = False to use chi square distribution
        p_value_both = result_both.pvalue
        stats_both = result_both.statistic

        results[cluster] = {
            "father": {"pvalue": p_value_father, "stat": stats_father, "exact": exact_father, "correction": correction_father, "expected_frq": expected_frq_father},
            "mother":  {"pvalue": p_value_mother, "stat": stats_mother, "exact": exact_mother, "correction": correction_mother, "expected_frq": expected_frq_mother},
            "both": {"pvalue": p_value_both, "stat": stats_both, "exact": exact_both, "correction": correction_both, "expected_frq": expected_frq_both}
        }

    return results

def compute_rows_cols_sum_and_total(contingency_table):
    """"""
    row_sums = np.sum(contingency_table, axis=1)
    col_sums = np.sum(contingency_table, axis=0)
    total_obs = np.sum(row_sums)

    return row_sums, col_sums, total_obs

def compute_expected_frequencies(observed):
    """"""
    expected_frq = stats.contingency.expected_freq(observed)

    return np.array(expected_frq, dtype=np.float32)

def association_test_from_confusion_matrices(confusion_matrices: dict):
    """
    confusion_matrix == contingency_table
    """

    results = {
        estimated_clusters_mapping[k]:None for k in confusion_matrices.keys()
    }

    expected_frq_dict = {
        estimated_clusters_mapping[k]:None for k in confusion_matrices.keys()
    }
    
    for cluster, confusion_matrix in confusion_matrices.items():

        confusion_matrices_dict = {
            "father": confusion_matrix.loc[:, ("yes_only_father_self_estimated", "no_only_father_self_estimated")].to_numpy(),
            "mother": confusion_matrix.loc[:, ("yes_only_mother_self_estimated", "no_only_mother_self_estimated")].to_numpy(),
            "both": confusion_matrix.loc[:, ("yes_both_self_estimated", "no_both_self_estimated")].to_numpy(),
            "all_no_dup": confusion_matrix.loc[:, ("yes_only_mother_self_estimated", "yes_only_father_self_estimated", "yes_both_self_estimated", "no_both_self_estimated")].to_numpy(),
            "all": confusion_matrix.to_numpy(),
            "yes_only": confusion_matrix.loc[:, ("yes_only_mother_self_estimated", "yes_only_father_self_estimated", "yes_both_self_estimated")].to_numpy(),
            "yes_only_father_mother": confusion_matrix.loc[:, ("yes_only_mother_self_estimated", "yes_only_father_self_estimated")].to_numpy(),
            "yes_only_father_both": confusion_matrix.loc[:, ("yes_only_father_self_estimated", "yes_both_self_estimated")].to_numpy(),
            "yes_only_mother_both": confusion_matrix.loc[:, ("yes_only_mother_self_estimated", "yes_both_self_estimated")].to_numpy()
        }

        # Compute row sums and column sums and total obs for computing expected frequencies after
        sums = {
            "father": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("father")))},
            "mother": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("mother")))},
            "both": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("both")))},
            "all_no_dup": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("all_no_dup")))},
            "all": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("all")))},
            "yes_only": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("yes_only")))},
            "yes_only_father_mother": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("yes_only_father_mother")))},
            "yes_only_father_both": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("yes_only_father_both")))},
            "yes_only_mother_both": {k:v for k, v in zip(["row_sums", "col_sums", "total_obs"], compute_rows_cols_sum_and_total(confusion_matrices_dict.get("yes_only_mother_both")))}
        }

        # compute expected frequencies
        expected_frq = {}
        for key, d in confusion_matrices_dict.items():
            expected_frq[key] = compute_expected_frequencies(d)

        expected_frq_dict[estimated_clusters_mapping[cluster]] = expected_frq

        # check chi2 of independency assumptions and perform tests for each contingency table in current cluster
        results_current = {k: None for k in confusion_matrices_dict.keys()}
        for (k, expected_v), sums_subdict in zip(expected_frq.items(), sums.items()):
            correction = False
            expected_frq_flatten = expected_v.flatten()
            sums_subdict = sums_subdict[1]
            
            if sums_subdict["total_obs"] == 0:
                sys.stderr.write(f"Total observations (total_obs) is zero for {k} {estimated_clusters_mapping[cluster]}, cannot compute expected frequencies.\n")
                results_current[k] = {"chi2_pvalue": None, "chi2_stat": None, "Yates_correction":None}
                results[estimated_clusters_mapping[cluster]] = results_current
                continue
            
            # check tests assumptions
            if np.any(expected_frq_flatten <= 5):
                if np.any(expected_frq_flatten == 0):
                    sys.stderr.write(f"Total observations (total_obs) in expected frequencies is zero for {k} {estimated_clusters_mapping[cluster]}, cannot perform chi² test..\n")
                    results_current[k] = {"chi2_pvalue": None, "chi2_stat": None, "Yates_correction":None}
                    results[estimated_clusters_mapping[cluster]] = results_current
                    continue
                correction = True

            chi2_output = chi2_contingency(confusion_matrices_dict.get(k), correction=correction)
            results_current[k] = {"chi2_pvalue": chi2_output.pvalue, "chi2_stat": chi2_output.statistic, "Yates_correction":correction}
            if correction:
                p_val_fisher = fisher_exact_test_mxn_tables(confusion_matrices_dict.get(k))
                results_current[k].update({"fisher_pvalue": p_val_fisher})
            else:
                results_current[k].update({"fisher_pvalue": None}) # add empty entries if expected frq > 5

        results[estimated_clusters_mapping[cluster]] = results_current

    return results, expected_frq_dict

def write_association_results_to_file(chi2_results_dict: dict, output_file: str):
    """"""

    with open(output_file, 'w') as f:
        f.write("cluster_table,chi2_p_value,chi2_stat,Yate's_correction,fisher_pvalue\n")
        for cluster, dic in chi2_results_dict.items():
            for table_name, results in dic.items():
                f.write(f"{cluster}_{table_name},{results.get('chi2_pvalue')},{results.get('chi2_stat')},{results.get('Yates_correction'),{results.get('fisher_pvalue')}}\n")

    return output_file

def mc_nemar_test_agreement_disagreement_analyses_father_vs_mother(contingency_tables: dict):
    """
    Example of input contingency_tables dictionary
        # contingency_tables[cluster] = {"self_reported_child_mother_vs_father_agreement": contingency_table_self_estimated_agreement_analysis, 
        #                                "self_reported_child_mother_vs_father_agreement": contingency_table_self_estimated_disagreement_analysis,
        #                                "genet_child_mother_vs_father_agreement_analysis": contingency_table_genet_agreement_analysis,
        #                                "genet_child_mother_vs_father_disagreement_analysis": contingency_table_genet_disagreement_analysis} -> dict of dict
    """

    results = {
        k:{k2:None} for k, v in contingency_tables.items() for k2 in v.keys() 
    }

    for cluster, dic in contingency_tables.items():
        for k, table in dic.items():
            exact = False
            correction = False

            # Compute row sums and column sums
            row_sums = np.sum(table, axis=1)
            col_sums = np.sum(table, axis=0)
            n_discordant_pairs = np.sum([table.iloc[0, 1], table.iloc[1, 0]])

            # Compute expected frequencies (expected number of individuals)
            total_obs = np.sum(row_sums)

            # compute for father
            if total_obs == 0:
                sys.stderr.write(f"Total observations (total_obs) is zero for father {estimated_clusters_mapping[cluster]}, cannot compute expected frequencies.\n")
                continue

            expected_frq = np.outer(row_sums, col_sums) / total_obs
            expected_frq_flatten = expected_frq.flatten()

            # check if sample size is (number of discordant pairs) small (<25) or if any cell frequencies >= 0.05
            if np.any(expected_frq_flatten <= 5):
                correction = True

            if n_discordant_pairs <= 10:
                exact = True    

            result = mcnemar(table, exact=exact, correction=correction) # exact = False to use chi square distribution
            p_value = result.pvalue
            stats = result.statistic

            results[cluster][k] = {"pvalue": p_value, "stat": stats, "exact": exact, "correction": correction, "expected_frq": expected_frq}

    return results


def plot_declared_origins_on_pca(declared_df: pd.DataFrame, estimated_df: pd.DataFrame, eigenvec_file: str, estimated_clusters_mapping: dict, colors: list | np.ndarray, markers: list | np.ndarray, output_dir: str):
    """"""
    
    eigenvec = pd.read_csv(eigenvec_file, sep=',', header=None)
    eigenvec.drop(index=eigenvec.index[0], axis=0, inplace=True)
    eigenvec = eigenvec.iloc[:, 0:3]
    eigenvec.columns = ["code_suj_crb", "PC1_eigen", "PC2_eigen"]

    eigenvec["code_suj_crb"] = eigenvec["code_suj_crb"].apply(lambda x: x.split("_")[0] if x.find("SUJ") != -1 else x)
    eigenvec = eigenvec.loc[eigenvec["code_suj_crb"].isin(estimated_df["code_suj_crb"])]

    eigenvec = eigenvec.sort_values(by="code_suj_crb", ascending=True)

    dot_s = 12
    merged_df = estimated_df.merge(declared_df[["code_suj_crb", "zone_origine"]], how="inner", on="code_suj_crb")
    merged_df = merged_df.sort_values(by="code_suj_crb", ascending=True)
    merged_df = merged_df.merge(eigenvec, how="inner", on="code_suj_crb")


    merged_df.drop("PC1", inplace=True, axis=1)
    merged_df.drop("PC2", inplace=True, axis=1) 
    merged_df.rename({"PC1_eigen": "PC1", "PC2_eigen": "PC2"}, axis=1, inplace=True)

    merged_df = merged_df.sort_values(by="zone_origine", ascending=True)

    merged_df["PC1"] = merged_df["PC1"].astype(np.float32)
    merged_df["PC2"] = merged_df["PC2"].astype(np.float32)

    # plot individuals according to estimated clusters
    for i in merged_df["zone_origine"].unique():
        fig = plt.figure(figsize=(10,6))
        ax = plt.subplot()
        for group in merged_df["cluster"].unique():
            ax.scatter(merged_df.loc[merged_df["cluster"]==group, "PC1"], merged_df.loc[merged_df["cluster"]==group, "PC2"], c=colors[group], marker=markers[group], s=dot_s, alpha=1, label=f"Genetic {estimated_clusters_mapping[group]} - case (n={len(merged_df.loc[merged_df['cluster']==group])})") # scatter plot permet de changer individuellement les couleurs des points pour chaque sous groupe
        
        ax.scatter(merged_df.loc[merged_df["zone_origine"]==i, "PC1"], merged_df.loc[merged_df["zone_origine"]==i, "PC2"], c="black", marker="x", s=dot_s*2, alpha=1, label=f"Self-estimated {estimated_clusters_mapping[i]} - case (n={len(merged_df.loc[merged_df['zone_origine']==i])})")

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "self-estimated_"+estimated_clusters_mapping[i]+".pdf"), format="pdf", dpi=300)
        plt.close()
        plt.clf()
 
  
def describe_origins_ind_and_parents(declared_origins_df: pd.DataFrame):
    """
    Check if individujal declared origins match with parents' declared origins.
    """

    descr_origins_df = declared_origins_df.copy(deep=True)

    descr_origins_df["both_parents_equal_ind"] =descr_origins_df[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == x["zone_origine_pere"] and x["zone_origine"] == x["zone_origine_mere"] else False, axis=1)
    descr_origins_df["father_equal_ind"] = descr_origins_df[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == x["zone_origine_pere"] else False, axis=1)
    descr_origins_df["mother_equal_ind"] = descr_origins_df[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == x["zone_origine_mere"] else False, axis=1)
    descr_origins_df["both_parents_diff_ind"] = descr_origins_df[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != x["zone_origine_pere"] and x["zone_origine"] != x["zone_origine_mere"] else False, axis=1)

    matching_declared_origins = pd.DataFrame()
    # count occurences
    matching_declared_origins["both_parents_diff_ind_count"] = descr_origins_df["both_parents_diff_ind"].value_counts()
    matching_declared_origins["father_equal_ind_count"] = descr_origins_df["father_equal_ind"].value_counts()
    matching_declared_origins["mother_equal_ind_count"] = descr_origins_df["mother_equal_ind"].value_counts()
    matching_declared_origins["mother_equal_ind_count"] = descr_origins_df["mother_equal_ind"].value_counts()

    # get proportions
    matching_declared_origins["both_parents_diff_ind_prop"] = matching_declared_origins["both_parents_diff_ind_count"] / len(descr_origins_df) * 100
    matching_declared_origins["father_equal_ind_prop"] = matching_declared_origins["father_equal_ind_count"] / len(descr_origins_df) * 100
    matching_declared_origins["mother_equal_ind_prop"] = matching_declared_origins["mother_equal_ind_count"] / len(descr_origins_df) * 100
    matching_declared_origins["mother_equal_ind_prop"] = matching_declared_origins["mother_equal_ind_count"] / len(descr_origins_df) * 100

    return descr_origins_df, matching_declared_origins

def create_paired_confusion_matrix_indivs_parents_by_ancestry(declared_origins: pd.DataFrame, estimated_origins: pd.DataFrame, output_dir: str, clusters_list: list[int]):
    """"""
    confusion_matrices = {k:None for k in clusters_list}

    for cluster in clusters_list:
        merged = pd.merge(declared_origins, estimated_origins, on="code_suj_crb", how="inner")
      
        confusion_matrix = pd.DataFrame(
            {"yes_both_self_estimated":  [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(), 
                                merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum()
            ],

             "yes_mother_self_estimated": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                                merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum()
              
            ],

            "no_mother_self_estimated":  [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                                merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum()
            ],

            "yes_father_self_estimated": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_pere"] == cluster else False, axis=1).sum(),
                                merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_pere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_pere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_pere"] == cluster else False, axis=1).sum()
            ],

            "no_father_self_estimated": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_pere"] != cluster else False, axis=1).sum(),
                               merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_pere"] != cluster else False, axis=1).sum(),
                               merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_pere"] != cluster else False, axis=1).sum(),
                               merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_pere"] != cluster else False, axis=1).sum()
            ],

            "no_both_self_estimated": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                               merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                               merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                               merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum()
            ]
             },
            index=["Yes_self_estimated", "No_self_estimated", "Yes_genetically_inferred", "No_genetically_inferred"]
        )
        
        confusion_matrices[cluster] = confusion_matrix
        confusion_matrix.to_csv(os.path.join(output_dir, f"confusion_matrix_paired_ancestry_indivs_parents_{estimated_clusters_mapping[cluster]}.csv"), sep=';')

    return confusion_matrices

def create_independent_confusion_matrix_indivs_parents_by_ancestry(declared_origins: pd.DataFrame, estimated_origins: pd.DataFrame, output_dir: str, clusters_list: list[int]):
    """"""
    confusion_matrices = {k:None for k in clusters_list}

    for cluster in clusters_list:
        merged = pd.merge(declared_origins, estimated_origins, on="code_suj_crb", how="inner")
      
        confusion_matrix = pd.DataFrame(
            {"yes_both_self_estimated":  [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(), 
                                merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum()
            ],

             "yes_only_mother_self_estimated": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_mere"] == cluster and x["zone_origine_pere"] != cluster else False, axis=1).sum(),
                                merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_mere"] == cluster and x["zone_origine_pere"] != cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_mere"] == cluster and x["zone_origine_pere"] != cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_mere"] == cluster and x["zone_origine_pere"] != cluster else False, axis=1).sum()
              
            ],

            "no_only_mother_self_estimated":  [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_mere"] != cluster and x["zone_origine_pere"] == cluster else False, axis=1).sum(),
                                merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_mere"] != cluster and x["zone_origine_pere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_mere"] != cluster and x["zone_origine_pere"] == cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_mere"] != cluster and x["zone_origine_pere"] == cluster else False, axis=1).sum()
            ],

            "yes_only_father_self_estimated": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                                merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                                merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_pere"] == cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum()
            ],

            "no_only_father_self_estimated": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                               merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                               merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum(),
                               merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] == cluster else False, axis=1).sum()
            ],

            "no_both_self_estimated": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                               merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                               merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum(),
                               merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != cluster and x["zone_origine_pere"] != cluster and x["zone_origine_mere"] != cluster else False, axis=1).sum()
            ]
             },
            index=["Yes_self_estimated", "No_self_estimated", "Yes_genetically_inferred", "No_genetically_inferred"]
        )
        
        confusion_matrices[cluster] = confusion_matrix
        confusion_matrix.to_csv(os.path.join(output_dir, f"confusion_matrix_independent_ancestry_indivs_parents_{estimated_clusters_mapping[cluster]}.csv"), sep=';')

    return confusion_matrices

def paired_z_test(differences):
        """
        Statistic comparison of 2 paired proportions.
        """

        # Calculate the mean difference
        mean_diff = np.mean(differences)

        # Calculate the standard deviation of the differences
        std_diff = np.std(differences, ddof=1)

        # Number of paired samples
        n = len(differences)

        # Paired Z-Test: Calculate the z-statistic
        z_stat = mean_diff / (std_diff / np.sqrt(n))

        # Calculate the two-tailed p-value
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

        return z_stat, p_value

def compare_agreement_vs_disagreement_proportion_selfchild_genet_stratified_analysis(declared_origins: pd.DataFrame, estimated_origins: pd.DataFrame, clusters_list: list[int]):
    """
    proportions for ind with genetically inferred ancestry as cluster and ind with not genetically inferred ancestry as cluster. Not for self reported ind as cluster.
    """

    results = {k:None for k in clusters_list}

    for cluster in clusters_list:
        
        merged = pd.merge(declared_origins, estimated_origins, on="code_suj_crb", how="inner")

        ##### 1 - among child of genetically inferred ancestry (a + b) #####
        ind_genet = merged.loc[merged["cluster"] == cluster, ("cluster", "zone_origine")]
        agreement_ind_genet = ind_genet.apply(lambda x: 1 if x['cluster'] == x['zone_origine'] else 0, axis=1).to_numpy()
        disagreement_ind_genet = ind_genet.apply(lambda x: 0 if x['cluster'] == x['zone_origine'] else 1, axis=1).to_numpy()
        diff_ind_genet = agreement_ind_genet - disagreement_ind_genet

        ind_not_genet = merged.loc[merged["cluster"] != cluster, ("cluster", "zone_origine")]
        agreement_ind_not_genet = ind_not_genet.apply(lambda x: 1 if x['zone_origine'] != cluster else 0, axis=1).to_numpy()
        disagreement_ind_not_genet = ind_not_genet.apply(lambda x: 0 if x['zone_origine'] != cluster else 1, axis=1).to_numpy()
        diff_ind_not_genet = agreement_ind_not_genet - disagreement_ind_not_genet

        if len(diff_ind_genet[diff_ind_genet == 1]) > 0:
            ind_genet_stat_ztest, ind_genet_p_ztest = paired_z_test(diff_ind_genet) # parametric paired test
            ind_genet_stat_wilcox, ind_genet_p_wilcox = wilcoxon(diff_ind_genet) # non parametric paired test
    
        if len(diff_ind_not_genet[diff_ind_not_genet == 1]) > 0:
            ind_not_genet_stat_ztest, ind_not_genet_p_ztest = paired_z_test(diff_ind_not_genet) # parametric paired test
            ind_not_genet_stat_wilcox, ind_not_genet_p_wilcox = wilcoxon(diff_ind_not_genet) # non parametric paired test

        
        results[cluster] = {
            "ind_genet_agreement_vs_disagreement": {"zstat": ind_genet_stat_ztest, "pvalue_ztest": ind_genet_p_ztest, "stat_wilcox": ind_genet_stat_wilcox, "pvalue_wilcox": ind_genet_p_wilcox},
            "ind_not_genet_agreement_vs_disagreement": {"zstat": ind_not_genet_stat_ztest, "pvalue_ztest": ind_not_genet_p_ztest, "stat_wilcox": ind_not_genet_stat_wilcox, "pvalue_wilcox": ind_not_genet_p_wilcox}, # non parametric paired test
        }

    return results


def compare_agreement_vs_disagreement_proportion_selfchild_genet_stratified_analysis_independent(declared_origins: pd.DataFrame, estimated_origins: pd.DataFrame, clusters_list: list[int]):
    """
    proportions for ind with genetically inferred ancestry as cluster and ind with not genetically inferred ancestry as cluster. Not for self reported ind as cluster.
    """

    results = {k:None for k in clusters_list}

    for cluster in clusters_list:
        
        merged = pd.merge(declared_origins, estimated_origins, on="code_suj_crb", how="inner")

        ##### 1 - among child of genetically inferred ancestry (a + b) #####
        ind_genet = merged.loc[merged["cluster"] == cluster, ("cluster", "zone_origine")]
        agreement_ind_genet = ind_genet.apply(lambda x: 1 if x['cluster'] == x['zone_origine'] else 0, axis=1).to_numpy()
        disagreement_ind_genet = ind_genet.apply(lambda x: 0 if x['cluster'] == x['zone_origine'] else 1, axis=1).to_numpy()
        diff_ind_genet = agreement_ind_genet - disagreement_ind_genet

        ind_not_genet = merged.loc[merged["cluster"] != cluster, ("cluster", "zone_origine")]
        agreement_ind_not_genet = ind_not_genet.apply(lambda x: 1 if x['zone_origine'] != cluster else 0, axis=1).to_numpy()
        disagreement_ind_not_genet = ind_not_genet.apply(lambda x: 0 if x['zone_origine'] != cluster else 1, axis=1).to_numpy()
        diff_ind_not_genet = agreement_ind_not_genet - disagreement_ind_not_genet

        # perform binomial test
        # Null Hypothesis (H0): The proportion of individuals self-reporting as NAF is equal to some reference value (e.g., 0.5).
        # Alternative Hypothesis (H1): The proportion of individuals self-reporting as NAF is different from the reference value.
        if len(diff_ind_genet[diff_ind_genet == 1]) > 0:
            p_value_ind_genet = stats.binom_test(agreement_ind_genet.sum(), len(agreement_ind_genet), p=0.5, alternative='two-sided')
           
        if len(diff_ind_not_genet[diff_ind_not_genet == 1]) > 0:
            p_value_ind_not_genet = stats.binom_test(agreement_ind_not_genet.sum(), len(agreement_ind_not_genet), p=0.5, alternative="two-sided")
           
        results[cluster] = {
            "ind_genet_agreement_vs_disagreement": {"p_val_binomial_test": p_value_ind_genet},
            "ind_not_genet_agreement_vs_disagreement": {"p_val_binomial_test": p_value_ind_not_genet}, # non parametric paired test
        }

    return results


def compare_agreement_disagreement_proportion_father_mother_stratified_analysis(declared_origins: pd.DataFrame, estimated_origins: pd.DataFrame, clusters_list: list[int]):
    """
    proportions for child with genetically inferred ancestry as cluster and child with not genetically inferred ancestry as cluster. Not for self reported ind as cluster
    """

    results = {k:None for k in clusters_list}

    for cluster in clusters_list:
        
        merged = pd.merge(declared_origins, estimated_origins, on="code_suj_crb", how="inner")

        ##### 1 - among child of genetically inferred ancestry (a + b) #####
        child_genet = merged.loc[merged["cluster"] == cluster, ("cluster", "zone_origine_pere", "zone_origine_mere")]

        # for agreement
        agreement_child_genet_father = child_genet.apply(lambda x: 1 if x['cluster'] == x['zone_origine_pere'] else 0, axis=1).to_numpy()
        agreement_child_genet_mother = child_genet.apply(lambda x: 1 if x['cluster'] == x['zone_origine_mere'] else 0, axis=1).to_numpy()
        diff_agreement_child_genet = agreement_child_genet_father - agreement_child_genet_mother

        ## perform stat tests for agreement
        if len(diff_agreement_child_genet[diff_agreement_child_genet == 1]) > 0: # if no differences, no tests
            agreement_child_genet_stat_ztest, agreement_child_genet_p_ztest = paired_z_test(diff_agreement_child_genet) # parametric paired test
            aggreement_child_genet_stat_wilcox, agreement_child_genet_p_wilcox = wilcoxon(diff_agreement_child_genet) # non parametric paired test

        # for disagreement
        disagreement_child_genet_father = child_genet.apply(lambda x: 0 if x['cluster'] == x['zone_origine_pere'] else 1, axis=1).to_numpy()
        disagreement_child_genet_mother = child_genet.apply(lambda x: 0 if x['cluster'] == x['zone_origine_mere'] else 1, axis=1).to_numpy()
        diff_disagreement_child_genet = disagreement_child_genet_father - disagreement_child_genet_mother

        if len(diff_disagreement_child_genet[diff_disagreement_child_genet == 1]) > 0:
            disagreement_child_genet_stat_ztest, disagreement_child_genet_p_ztest = paired_z_test(diff_disagreement_child_genet) # parametric paired test
            disaggreement_child_genet_stat_wilcox, disagreement_child_genet_p_wilcox = wilcoxon(diff_disagreement_child_genet, alternative="two-sided") # non parametric paired test

        ###### 2 - among child of not genetically inferred ancestry (c + d) #####
        child_not_genet = merged.loc[merged["cluster"] != cluster, ("cluster", "zone_origine_pere", "zone_origine_mere")]

        # for agreement
        agreement_child_not_genet_father = child_not_genet.apply(lambda x: 1 if x['zone_origine_pere'] != cluster else 0, axis=1).to_numpy()
        agreement_child_not_genet_mother = child_not_genet.apply(lambda x: 1 if x['zone_origine_mere'] != cluster else 0, axis=1).to_numpy()
        diff_agreement_child_not_genet = agreement_child_not_genet_father - agreement_child_not_genet_mother

        ## perform stat tests for agreement
        if len(diff_agreement_child_not_genet[diff_agreement_child_not_genet == 1]) > 0:
            agreement_child_not_genet_stat_ztest, agreement_child_not_genet_p_ztest = paired_z_test(diff_agreement_child_not_genet) # parametric paired test
            aggreement_child_not_genet_stat_wilcox, agreement_child_not_genet_p_wilcox = wilcoxon(diff_agreement_child_not_genet, alternative="two-sided") # non parametric paired test

        # for disagreement
        disagreement_child_not_genet_father = child_not_genet.apply(lambda x: 0 if x['zone_origine_pere'] != cluster else 1, axis=1).to_numpy()
        disagreement_child_not_genet_mother = child_not_genet.apply(lambda x: 0 if x['zone_origine_mere'] != cluster else 1, axis=1).to_numpy()
        diff_disagreement_child_not_genet = disagreement_child_not_genet_father - disagreement_child_not_genet_mother

        if len(diff_disagreement_child_not_genet[diff_disagreement_child_not_genet == 1]) > 0:
            disagreement_child_not_genet_stat_ztest, disagreement_child_not_genet_p_ztest = paired_z_test(diff_disagreement_child_not_genet) # parametric paired test
            disaggreement_child_not_genet_stat_wilcox, disagreement_child_not_genet_p_wilcox = wilcoxon(diff_disagreement_child_not_genet, alternative="two-sided") # non parametric paired test

        results[cluster] = {
            "agreement_child_genet": {"zstat": agreement_child_genet_stat_ztest, "pvalue_ztest": agreement_child_genet_p_ztest, "stat_wilcox": aggreement_child_genet_stat_wilcox, "pvalue_wilcox": agreement_child_genet_p_wilcox},
            "disagreement_child_genet": {"zstat": disagreement_child_genet_stat_ztest, "pvalue_ztest": disagreement_child_genet_p_ztest, "stat_wilcox": disaggreement_child_genet_stat_wilcox, "pvalue_wilcox": disagreement_child_genet_p_wilcox},
            "agreement_not_child_genet": {"zstat": agreement_child_not_genet_stat_ztest, "pvalue_ztest": agreement_child_not_genet_p_ztest, "stat_wilcox": aggreement_child_not_genet_stat_wilcox, "pvalue_wilcox": agreement_child_not_genet_p_wilcox},
            "disagreement_not_child_genet:": {"zstat": disagreement_child_not_genet_stat_ztest, "pvalue_ztest": disagreement_child_not_genet_p_ztest, "stat_wilcox": disaggreement_child_not_genet_stat_wilcox, "pvalue_wilcox": disagreement_child_not_genet_p_wilcox}
        }
    
    return results

def confidence_interval(p, n, confidence=0.95):
    """
    n total number of individuals (total used to compute the prop)"
    p proportion computed
    confidence confidence level (default 95%)
    """

    z = norm.ppf(1 - (1 - confidence) / 2)
    error = z * np.sqrt((p * (1 - p)) / n)
    return (p - error, p + error)

def create_mother_vs_father_contingency_tables(declared_origins: pd.DataFrame, estimated_origins: pd.DataFrame, clusters_list: list[int]):
    """"""
    contingency_tables = {k:None for k in clusters_list}

    for cluster in clusters_list:
        merged = pd.merge(declared_origins, estimated_origins, on="code_suj_crb", how="inner")

        merged = merged[merged.apply(
            lambda x: any(x[val] == cluster for val in ["zone_origine", "zone_origine_pere", "zone_origine_mere"]),
            axis=1
        )]

        contingency_table_self_estimated = pd.DataFrame(
            data={
                "agreement_father": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == x["zone_origine_pere"] and x["zone_origine"] == x["zone_origine_mere"] else False, axis=1).sum(), 
                                    merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] == x["zone_origine_pere"] and x["zone_origine"] != x["zone_origine_mere"] else False, axis=1).sum()],
                                    
                "disagreement_father": [merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != x["zone_origine_pere"] and x["zone_origine"] == x["zone_origine_mere"] else False, axis=1).sum(), 
                                    merged[["zone_origine", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["zone_origine"] != x["zone_origine_pere"] and x["zone_origine"] != x["zone_origine_mere"] else False, axis=1).sum()]

            },
            index=["agreement_mother", "disagreement_mother"]
        
        )

        contingency_table_genet = pd.DataFrame(
            data={
                "agreement_father": [merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == x["zone_origine_pere"] and x["cluster"] == x["zone_origine_mere"] else False, axis=1).sum(), 
                                    merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] == x["zone_origine_pere"] and x["cluster"] != x["zone_origine_mere"] else False, axis=1).sum(),],
                                    
                "disagreement_father": [merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != x["zone_origine_pere"] and x["cluster"] == x["zone_origine_mere"] else False, axis=1).sum(), 
                                    merged[["cluster", "zone_origine_pere", "zone_origine_mere"]].apply(lambda x: True if x["cluster"] != x["zone_origine_pere"] and x["cluster"] != x["zone_origine_mere"] else False, axis=1).sum(),]
            },
            index=["agreement_mother", "disagreement_mother"]

        )

        contingency_tables[cluster] = {"self_reported_child_mother_vs_father_agreement": contingency_table_self_estimated, 
                                       "genet_child_mother_vs_father_agreement": contingency_table_genet}
    
    return contingency_tables

def barplots_agreement_disagreement_self_estimated_individual_vs_genet(confusion_matrix: pd.DataFrame, output_filename: str):
    """"""

    total_genet = confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[0, 1]
    total_no_genet = confusion_matrix.iloc[1, 0] + confusion_matrix.iloc[1, 1]
    total_self = confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 0]
    total_not_self = confusion_matrix.iloc[0, 1] + confusion_matrix.iloc[1, 1]

    # self among genet of ancestry
    well_classified_when_of_ancestry = confusion_matrix.iloc[0, 0]
    well_classified_when_of_ancestry_prop = well_classified_when_of_ancestry / total_genet
    bad_classified_when_of_ancestry = confusion_matrix.iloc[0, 1]
    bad_classified_when_of_ancestry_prop = bad_classified_when_of_ancestry / total_genet

    # self among genet not of ancestry
    well_classified_when_not_of_ancestry = confusion_matrix.iloc[1, 1]
    well_classified_when_not_of_ancestry_prop = well_classified_when_not_of_ancestry / total_no_genet
    bad_classified_when_not_of_ancestry = confusion_matrix.iloc[0, 1]
    bad_classified_when_not_of_ancestry_prop = bad_classified_when_not_of_ancestry / total_no_genet

    # genet among self of ancestry
    well_classified_when_self = confusion_matrix.iloc[0, 0]
    well_classified_when_self_prop = well_classified_when_self / total_self
    bad_classified_when_self = confusion_matrix.iloc[1, 0]
    bad_classified_when_self_prop = bad_classified_when_self / total_self

    # genet among self not of ancestry
    well_classified_when_not_self = confusion_matrix.iloc[0, 1]
    well_classified_when_not_self_prop = well_classified_when_not_self / total_not_self
    bad_classified_when_not_self = confusion_matrix.iloc[1, 1]
    bad_classified_when_not_self_prop = bad_classified_when_not_self / total_not_self

    among_of_ancestry = [(well_classified_when_of_ancestry, well_classified_when_of_ancestry_prop), (bad_classified_when_of_ancestry, bad_classified_when_of_ancestry_prop)]
    among_not_of_ancestry = [(well_classified_when_not_of_ancestry, well_classified_when_not_of_ancestry_prop), (bad_classified_when_not_of_ancestry, bad_classified_when_not_of_ancestry_prop)]
    among_self_estimed = [(well_classified_when_self, well_classified_when_self_prop), (bad_classified_when_self, bad_classified_when_self_prop)]
    among_not_self_estimed = [(well_classified_when_not_self, well_classified_when_not_self_prop), (bad_classified_when_not_self, bad_classified_when_not_self_prop)]

    # barplots
    for i, l in enumerate([among_of_ancestry, among_not_of_ancestry, among_self_estimed, among_not_self_estimed]):
        ns = [elm[0] for elm in l]
        props = [elm[1] for elm in l]
        categories = ["Agreement", "Disagreement"]

        # compute confidence interval for each category
        cis = [confidence_interval(props[i], ns[i], confidence=0.95) for i in range(0, len(categories))]

        # Compute error values as the difference between the proportion and the upper/lower CI bound
        errors = [ (props[i] - cis[i][0], cis[i][1] - props[i]) for i in range(0, len(categories)) ]

        # Convert tuple errors to upper and lower bounds
        lower_error = [e[0] for e in errors]
        upper_error = [e[1] for e in errors]
        
        # create bar plots
        fig, ax = plt.subplots(figsize=(14, 10))
        bar_plot = sns.barplot(x=categories, y=props, ax=ax, palette="muted")
         # Retrieve bar positions
        bar_positions = [bar.get_x() + bar.get_width() / 2.0 for bar in bar_plot.patches]

        # Add error bars to the plot using bar positions
        ax.errorbar(x=bar_positions, y=props, yerr=[lower_error, upper_error], fmt='none', capsize=5, color='black')

        for j, (n, prop) in enumerate(zip(ns, props)):
            plt.text(j, prop+upper_error[j]+0.01, f"{n} ({prop*100:.2f}%)", ha="center", va="bottom")

        if i == 0:
            suffix = "_on_total_of_ancestry_ab" # ab for a+b in the contingency table, same principle for below
        elif i == 1:
            suffix = "_on_total_not_of_ancestry_cd"
        elif i == 2:
            suffix = "_on_total_self_estimed_ac"
        elif i == 3:
            suffix = "_on_total_not_self_estimed_bd"

        plt.ylabel("Proportion (%)")
        plt.tight_layout()
        plt.savefig(output_filename+suffix+".pdf", format="pdf", dpi=300)
        plt.clf()
        plt.close()

def barplots_agreement_disagreement_father_mother(confusion_matrix: pd.DataFrame, output_filename: str):
    """
    Using all possible total of contingency tables
    """
    
    # for father
    confusion_matrix_father= confusion_matrix.loc[:, (f"yes_father_self_estimated", f"no_father_self_estimated")]

    total_genet_father = confusion_matrix_father.iloc[0, 0] + confusion_matrix_father.iloc[0, 1] # out of total number of child genet of ancestry for all fathers that answered for ancestry (unknown is an answer)
    total_no_genet_father = confusion_matrix_father.iloc[1, 0] + confusion_matrix_father.iloc[1, 1]
    total_self_father = confusion_matrix_father.iloc[0, 0] + confusion_matrix_father.iloc[1, 0]
    total_no_self_father = confusion_matrix_father.iloc[0, 1] + confusion_matrix_father.iloc[1, 1]

    # father among child of ancestry
    well_classified_father_when_child_of_ancestry = confusion_matrix_father.iloc[0, 0]
    well_classified_father_when_child_of_ancestry_prop = well_classified_father_when_child_of_ancestry / total_genet_father
    bad_classified_father_when_child_of_ancestry = confusion_matrix_father.iloc[0, 1]
    bad_classified_father_when_child_of_ancestry_prop = bad_classified_father_when_child_of_ancestry / total_genet_father

    # father among child not of ancestry
    well_classified_father_when_child_not_of_ancestry = confusion_matrix_father.iloc[1, 1]
    well_classified_father_when_child_not_of_ancestry_prop = well_classified_father_when_child_not_of_ancestry / total_no_genet_father
    bad_classified_father_when_child_not_of_ancestry = confusion_matrix_father.iloc[1, 0]
    bad_classified_father_when_child_not_of_ancestry_prop = bad_classified_father_when_child_not_of_ancestry / total_no_genet_father

    # father when father self estimed of ancestry
    well_classified_father_when_father_self_estimed = confusion_matrix_father.iloc[0, 0]
    well_classified_father_when_father_self_estimed_prop = well_classified_father_when_father_self_estimed / total_self_father
    bad_classified_father_when_father_self_estimed = confusion_matrix_father.iloc[1, 0]
    bad_classified_father_when_father_self_estimed_prop = bad_classified_father_when_father_self_estimed / total_self_father

    # father when father not self estimed of ancestry
    well_classified_father_when_father_not_self_estimed = confusion_matrix_father.iloc[0, 1]
    well_classified_father_when_father_not_self_estimed_prop = well_classified_father_when_father_not_self_estimed / total_no_self_father
    bad_classified_father_when_father_not_self_estimed = confusion_matrix_father.iloc[1, 1]
    bad_classified_father_when_father_not_self_estimed_prop = bad_classified_father_when_father_not_self_estimed / total_no_self_father

    # for mother
    confusion_matrix_mother= confusion_matrix.loc[:, (f"yes_mother_self_estimated", f"no_mother_self_estimated")]

    total_genet_mother = confusion_matrix_mother.iloc[0, 0] + confusion_matrix_mother.iloc[0, 1] # out of total number of child genet of ancestry for all fathers that answered for ancestry (unknown is an answer)
    total_no_genet_mother = confusion_matrix_mother.iloc[1, 0] + confusion_matrix_mother.iloc[1, 1]
    total_self_mother = confusion_matrix_mother.iloc[0, 0] + confusion_matrix_mother.iloc[1, 0]
    total_no_self_mother = confusion_matrix_mother.iloc[0, 1] + confusion_matrix_mother.iloc[1, 1]

    # mother among child of ancestry
    well_classified_mother_when_child_of_ancestry = confusion_matrix_mother.iloc[0, 0]
    well_classified_mother_when_child_of_ancestry_prop = well_classified_mother_when_child_of_ancestry / total_genet_mother
    bad_classified_mother_when_child_of_ancestry = confusion_matrix_mother.iloc[0, 1]
    bad_classified_mother_when_child_of_ancestry_prop = bad_classified_mother_when_child_of_ancestry / total_genet_mother

    # mother among child not of ancestry
    well_classified_mother_when_child_not_of_ancestry = confusion_matrix_mother.iloc[1, 1]
    well_classified_mother_when_child_not_of_ancestry_prop = well_classified_mother_when_child_not_of_ancestry / total_no_genet_mother
    bad_classified_mother_when_child_not_of_ancestry = confusion_matrix_mother.iloc[1, 0]
    bad_classified_mother_when_child_not_of_ancestry_prop = bad_classified_mother_when_child_not_of_ancestry / total_no_genet_mother

    # mother when father self estimed of ancestry = out of the total of mother assessment as self estimed, how many child were indeed of ancestry
    well_classified_mother_when_mother_self_estimed = confusion_matrix_mother.iloc[0, 0]
    well_classified_mother_when_mother_self_estimed_prop = well_classified_mother_when_mother_self_estimed / total_self_mother
    bad_classified_mother_when_mother_self_estimed = confusion_matrix_mother.iloc[1, 0]
    bad_classified_mother_when_mother_self_estimed_prop = bad_classified_mother_when_mother_self_estimed / total_self_mother

    # mother when father not self estimed of ancestry =  out of the total of mother assessment as not self estimed, how many child were indeed not of ancestry
    well_classified_mother_when_mother_not_self_estimed = confusion_matrix_mother.iloc[0, 1]
    well_classified_mother_when_mother_not_self_estimed_prop = well_classified_mother_when_mother_not_self_estimed / total_no_self_mother
    bad_classified_mother_when_mother_not_self_estimed = confusion_matrix_mother.iloc[1, 1]
    bad_classified_mother_when_mother_not_self_estimed_prop = bad_classified_mother_when_mother_not_self_estimed / total_no_self_mother

    among_child_of_ancestry = [(well_classified_father_when_child_of_ancestry, well_classified_father_when_child_of_ancestry_prop), (well_classified_mother_when_child_of_ancestry, well_classified_mother_when_child_of_ancestry_prop), (bad_classified_father_when_child_of_ancestry, bad_classified_father_when_child_of_ancestry_prop), (bad_classified_mother_when_child_of_ancestry, bad_classified_mother_when_child_of_ancestry_prop)]
    among_child_not_of_ancestry = [(well_classified_father_when_child_not_of_ancestry, well_classified_father_when_child_not_of_ancestry_prop), (well_classified_mother_when_child_not_of_ancestry, well_classified_mother_when_child_not_of_ancestry_prop), (bad_classified_father_when_child_not_of_ancestry, bad_classified_father_when_child_not_of_ancestry_prop), (bad_classified_mother_when_child_not_of_ancestry, bad_classified_mother_when_child_not_of_ancestry_prop)]
    among_self_estimed = [(well_classified_father_when_father_self_estimed, well_classified_father_when_father_self_estimed_prop), (well_classified_mother_when_mother_self_estimed, well_classified_mother_when_mother_self_estimed_prop), (bad_classified_father_when_father_self_estimed, bad_classified_father_when_father_self_estimed_prop), (bad_classified_mother_when_mother_self_estimed, bad_classified_mother_when_mother_self_estimed_prop)]
    among_not_self_estimed = [(well_classified_father_when_father_not_self_estimed, well_classified_father_when_father_not_self_estimed_prop), (well_classified_mother_when_mother_not_self_estimed, well_classified_mother_when_mother_not_self_estimed_prop), (bad_classified_father_when_father_not_self_estimed, bad_classified_father_when_father_not_self_estimed_prop), (bad_classified_mother_when_mother_not_self_estimed, bad_classified_mother_when_mother_not_self_estimed_prop)]
    
    # barplots
    for i, l in enumerate([among_child_of_ancestry, among_child_not_of_ancestry, among_self_estimed, among_not_self_estimed]):
        ns = [elm[0] for elm in l]
        props = [elm[1] for elm in l]
        categories = ["Agreement father", "Agreement mother", "Disagreement father", "Disagreement mother"]

        # compute confidence interval for each category
        cis = [confidence_interval(props[i], ns[i], confidence=0.95) for i in range(0, len(categories))]

        # Compute error values as the difference between the proportion and the upper/lower CI bound
        errors = [ (props[i] - cis[i][0], cis[i][1] - props[i]) for i in range(0, len(categories)) ]

        # Convert tuple errors to upper and lower bounds
        lower_error = [e[0] for e in errors]
        upper_error = [e[1] for e in errors]
        
        # create bar plots
        fig, ax = plt.subplots(figsize=(14, 10))
        bar_plot = sns.barplot(x=categories, y=props, ax=ax, palette="muted")
         # Retrieve bar positions
        bar_positions = [bar.get_x() + bar.get_width() / 2.0 for bar in bar_plot.patches]

        # Add error bars to the plot using bar positions
        ax.errorbar(x=bar_positions, y=props, yerr=[lower_error, upper_error], fmt='none', capsize=5, color='black')

        for j, (n, prop) in enumerate(zip(ns, props)):
            plt.text(j, prop+upper_error[j]+0.025, f"{n} ({prop*100:.2f}%)", ha="center", va="bottom")

        if i == 0:
            suffix = "_on_total_child_of_ancestry_ab" # ab for a+b in the contingency table, same principle for below
        elif i == 1:
            suffix = "_on_total_child_not_of_ancestry_cd"
        elif i == 2:
            suffix = "_on_total_parent_self_estimed_ac"
        elif i == 3:
            suffix = "_on_total_parent_not_self_estimed_bd"

        plt.ylabel("Proportion (%)")
        plt.tight_layout()
        plt.savefig(output_filename+suffix+".pdf", format="pdf", dpi=300)
        plt.clf()
        plt.close()

def get_admixture_proportion_distributions_for_contingency_tables(declared_origins, estimated_origins, output_dir):
    """Faire tableau contingence avec à l'intérieur de chaque case la distribution (dict en fait pas table) des admixture proportions."""

    results = {}
    groups = estimated_origins["cluster"].unique() # cluster numbers to consider (no 3 and 5 because use of 8 which is Asian clusters merged (East + South))
    groups.sort()
    for group in groups:
        for admixed_ref_pop in ["Europe", "East_Asia", "Africa"]:
            # count ancestry
            declared_arr = declared_origins[["code_suj_crb", "zone_origine", admixed_ref_pop]].copy()
            declared_arr.loc[declared_arr["zone_origine"] != group, "zone_origine"] = -2
            declared_arr.loc[declared_arr["zone_origine"] == group, "zone_origine"] = -1

            estimated_arr = estimated_origins[["code_suj_crb", "cluster"]].copy()
            estimated_arr.loc[estimated_arr["cluster"] != group, "cluster"] = -2
            estimated_arr.loc[estimated_arr["cluster"] == group, "cluster"] = -1

            merged_arr = pd.merge(declared_arr, estimated_arr, on="code_suj_crb", how="inner")

            # create contingency tables with admixtures proportions as mean (+ std and margin of error computed as 1.96 * std / sqrt(total_n))
            up_left_distrib = merged_arr.loc[(merged_arr["zone_origine"] == -1) & (merged_arr["cluster"] == -1), admixed_ref_pop].to_numpy()
            up_right_distrib = merged_arr.loc[(merged_arr["zone_origine"] == -2) & (merged_arr["cluster"] == -1), admixed_ref_pop].to_numpy()
            low_left_distrib = merged_arr.loc[(merged_arr["zone_origine"] == -1) & (merged_arr["cluster"] == -2), admixed_ref_pop].to_numpy()
            low_right_distrib = merged_arr.loc[(merged_arr["zone_origine"] == -2) & (merged_arr["cluster"] == -2), admixed_ref_pop].to_numpy()

            cross_table = pd.DataFrame(
                np.array([
                    [up_left_distrib.mean(), up_right_distrib.mean()],
                    [low_left_distrib.mean(), low_right_distrib.mean()]
                ]), 
                columns = [f"yes_self_report_{estimated_clusters_mapping[group]}", f"no_self_report_{estimated_clusters_mapping[group]}"],
                index = [f"yes_genetic_ancestry_{estimated_clusters_mapping[group]}", f"no_genetic_ancestry_{estimated_clusters_mapping[group]}"]
            ) # col: yes / no (self-report) - row: yes / no (cluster)

            cross_table_std = pd.DataFrame(
                np.array([
                    [np.std(up_left_distrib), np.std(up_right_distrib)],
                    [np.std(low_left_distrib), np.std(low_right_distrib)]
                ]), 
                columns = [f"yes_self_report_{estimated_clusters_mapping[group]}", f"no_self_report_{estimated_clusters_mapping[group]}"],
                index = [f"yes_genetic_ancestry_{estimated_clusters_mapping[group]}", f"no_genetic_ancestry_{estimated_clusters_mapping[group]}"]
            )

            cross_table_se_normal = pd.DataFrame( # computed as 1.96 * std / sqrt(n_total = 2542)
                np.array([
                    [1.96 * np.std(up_left_distrib) / np.sqrt(len(up_left_distrib)), 1.96 * np.std(up_right_distrib) / np.sqrt(len(up_right_distrib))],
                    [1.96 * np.std(low_left_distrib) / np.sqrt(len(low_left_distrib)), 1.96 * np.std(low_right_distrib) / np.sqrt(len(low_right_distrib))]
                ]), 
                columns = [f"yes_self_report_{estimated_clusters_mapping[group]}", f"no_self_report_{estimated_clusters_mapping[group]}"],
                index = [f"yes_genetic_ancestry_{estimated_clusters_mapping[group]}", f"no_genetic_ancestry_{estimated_clusters_mapping[group]}"]
            )
            cross_table.to_csv(os.path.join(output_dir, f"admixture_contingency_{estimated_clusters_mapping[group]}_{admixed_ref_pop}_proportion_mean.csv"), sep=",")
            cross_table_std.to_csv(os.path.join(output_dir, f"admixture_contingency_{estimated_clusters_mapping[group]}_{admixed_ref_pop}_proportion_std.csv"), sep=",")
            cross_table_se_normal.to_csv(os.path.join(output_dir, f"admixture_contingency_{estimated_clusters_mapping[group]}_{admixed_ref_pop}_proportion_margin_error.csv"), sep=",")

            distrib_self_yes_genet_yes = merged_arr.loc[( (merged_arr["zone_origine"] == -1) & (merged_arr["cluster"] == -1)), ("code_suj_crb", admixed_ref_pop)]
            distrib_self_yes_genet_no = merged_arr.loc[( (merged_arr["zone_origine"] == -1) & (merged_arr["cluster"] == -2) ), ("code_suj_crb", admixed_ref_pop)]
            distrib_self_no_genet_yes = merged_arr.loc[( (merged_arr["zone_origine"] == -2) & (merged_arr["cluster"] == -1) ), ("code_suj_crb", admixed_ref_pop)]
            distrib_self_no_genet_no = merged_arr.loc[( (merged_arr["zone_origine"] == -2) & (merged_arr["cluster"] == -2) ), ("code_suj_crb", admixed_ref_pop)]

            table = pd.DataFrame(
                data = {
                    f"yes_self_estimated_&_yes_genetically_inferred_prop_{admixed_ref_pop}": distrib_self_yes_genet_yes[admixed_ref_pop],
                    f"yes_self_estimated_&_yes_genetically_inferred_id_{admixed_ref_pop}": distrib_self_yes_genet_yes["code_suj_crb"],

                    f"no_self_estimated_&_yes_genetically_inferred_prop_{admixed_ref_pop}": distrib_self_no_genet_yes[admixed_ref_pop],
                    f"no_self_estimated_&_yes_genetically_inferred_id_{admixed_ref_pop}": distrib_self_no_genet_yes["code_suj_crb"],

                    f"yes_self_estimated_&_no_genetically_inferred_prop_{admixed_ref_pop}": distrib_self_yes_genet_no[admixed_ref_pop],
                    f"yes_self_estimated_&_no_genetically_inferred_id_{admixed_ref_pop}": distrib_self_yes_genet_no["code_suj_crb"],

                    f"no_self_estimated_&_no_genetically_inferred_prop_{admixed_ref_pop}": distrib_self_no_genet_no[admixed_ref_pop],
                    f"no_self_estimated_&_no_genetically_inferred_id_{admixed_ref_pop}": distrib_self_no_genet_no["code_suj_crb"],
                }
            )

            results[f"{estimated_clusters_mapping[group]}_{admixed_ref_pop}"] = table
            table.to_csv(os.path.join(output_dir, f"admixture_distribution_{estimated_clusters_mapping[group]}_{admixed_ref_pop}_proportion.csv"), sep=",")
    
    return results

def admixture_prop_violin_plot(distrib_1: list | np.ndarray | pd.Series, distrib_2: list | np.ndarray | pd.Series, distrib_1_name: str, distrib_2_name: str, output_file: str):
    """"""
    
    # create violin plots
    fig, ax = plt.subplots(figsize=(14, 10))
     # Plot the violin
    sns.violinplot(
        data=[distrib_1, distrib_2],
        palette="pastel",
        scale='count',
        # inner="point",
        ax=ax,
        zorder=1
    )

    # Set x-axis labels
    ax.set_xticks([0, 1])  # Set the positions of the ticks
    ax.set_xticklabels([distrib_1_name, distrib_2_name])  # Set the labels for the ticks

    # Create a gap above y=1 by adding a blank space in the plot area
    plt.ylim(0, 1.05)  # Set y-limits to create a gap above y=1

    # Customize the ticks for y-axis
    plt.yticks(np.arange(0, 1.2, 0.2))  # Y ticks from 0 to 1.2

    def cut_violins_above_1(ax):
        for i, collection in enumerate(ax.collections):
            # Check if this collection is a part of a violin (edges typically have larger counts)
            if len(collection.get_paths()) > 0:
                for path in collection.get_paths():
                    # Get the vertices of the path
                    vertices = path.vertices
                    # Set y-values above 1 to 1
                    vertices[vertices[:, 1] > 1, 1] = 1
        return ax

    ax = cut_violins_above_1(ax)

    plt.ylabel("Proportion (%)")
    plt.tight_layout()
    plt.savefig(output_file, format="pdf", dpi=300)
    plt.clf()
    plt.close()


def student_test_compare_2_distributions(distrib_1: list | np.ndarray | pd.Series, distrib_2: list | np.ndarray | pd.Series):
    """
    T test independent
    """
     # Ensure inputs are numpy arrays
    distrib_1 = np.asarray(distrib_1)
    distrib_2 = np.asarray(distrib_2)
    
    distrib_1 = distrib_1[~np.isnan(distrib_1)]  # Remove NaN values from distrib_1
    distrib_2 = distrib_2[~np.isnan(distrib_2)]  # Remove NaN values from distrib_2

    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(distrib_1, distrib_2, equal_var=False)  # Use equal_var=True if variances are assumed equal

    mean1 = distrib_1.mean()
    mean2 = distrib_2.mean()
    std1 = distrib_1.std()
    std2 = distrib_2.std()

    return t_stat, p_value, mean1, mean2, std1, std2


def mannwhitney_test_compare_2_distributions(distrib_1: list | np.ndarray | pd.Series, distrib_2: list | np.ndarray | pd.Series):
    """
    mannwhitney - non parametric equivalent to t test
    """
    
    # Ensure inputs are numpy arrays
    distrib_1 = np.asarray(distrib_1)
    distrib_2 = np.asarray(distrib_2)
    
    distrib_1 = distrib_1[~np.isnan(distrib_1)]  # Remove NaN values from distrib_1
    distrib_2 = distrib_2[~np.isnan(distrib_2)]  # Remove NaN values from distrib_2

    # Perform the Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(distrib_1, distrib_2, alternative='two-sided')

    mean1 = distrib_1.mean()
    mean2 = distrib_2.mean()
    std1 = distrib_1.std()
    std2 = distrib_2.std()

    return u_stat, p_value, mean1, mean2, std1, std2

def normality_test(distrib_list: list, distrib_names: list):
    """
    distrib_list is a list of distributions: list[list | np.ndarray | pd.Series]

    Test for each distribution if it follows a normal distribution law.
    Return a dictionary with results for each tests depending on sample size.
    """

    results = {}
    for distrib, name in zip(distrib_list, distrib_names):

        distrib = np.asarray(distrib)
        distrib = distrib[~np.isnan(distrib)]  # Remove NaN values from distrib_1

        if len(distrib) >= 50: # if large sample size
            # perform Anderson-Darling test , Kolmogorov-Smirnov test, D'Agostino's K-squared test
            stat_ks, p_ks = kstest(distrib, 'norm', args=(np.mean(distrib), np.std(distrib)))
            result_anderson = anderson(distrib, dist='norm')
            stat_anderson, critical_values = result_anderson.statistic, result_anderson.critical_values
            stat_normaltest, p_normaltest = normaltest(distrib)

            results[name] = {"stat_ks": stat_ks, "p_ks": p_ks, "stat_anderson": stat_anderson, "critical_values_anderson": critical_values, "stat_normaltest": stat_normaltest, "p_normaltest": p_normaltest, "stat_shapiro": None, "p_shapiro": None}
        else: # if low sample size perform shapiro test
            try:
                assert(len(distrib) >= 3)
            except AssertionError:
                sys.stderr.write(f"[NotEnoughDataError] Not enough data for shapiro testing. Ignoring column '{name}'")
                results[name] = {"stat_ks": None, "p_ks": None, "stat_anderson": None, "critical_values_anderson": None, "stat_normaltest": None, "p_normaltest": None, "stat_shapiro": None, "p_shapiro": None}
                continue
            stat_shapiro, p_shapiro = shapiro(distrib)
            results[name] = {"stat_ks": None, "p_ks": None, "stat_anderson": None, "critical_values_anderson": None, "stat_normaltest": None, "p_normaltest": None, "stat_shapiro": stat_shapiro, "p_shapiro": p_shapiro}

    return results        

def qq_plot_against_normal(distrib: list | np.ndarray | pd.Series, output_file: str):
    """
    Test whether a distribution follows a normal distribution law graphically using a Q-Q plot.
    
    Parameters:
    distrib: list, np.ndarray, or pd.Series
        The distribution to compare against a normal distribution.
    
    distribution_name: str
        The name of the distribution being tested (used for labeling).
    
    output_file: str
        The file path to save the plot as a PDF.
    
    Returns:
    None: Saves a Q-Q plot to the specified file.
    """

    # Convert to numpy array if not already
    distrib = np.asarray(distrib)
    distrib = distrib[~np.isnan(distrib)]
    
    # Create a ProbPlot object and generate Q-Q plot
    pp = sm.ProbPlot(distrib, dist=norm, fit=True)
    plt.figure(figsize=(8, 8))
    pp.qqplot(line='45')
    
    # Customize plot
    plt.title(f'Q-Q Plot of the Sample Distribution Against Normal Distribution')
    plt.grid(True)

    # Save the plot as a PDF
    plt.savefig(output_file, dpi=300, format="pdf")
    plt.clf()
    plt.close()

def fisher_exact_test_mxn_tables(contingency_table: np.ndarray | pd.DataFrame):
    """
    Perform fisher test on 2x2 contingency table or mxn. Use R matrix with Monte Carlo simulation for > 2x2 contingency tables.
    Use classical fisher test from statsmodel for 2x2 contingency tables.
    """
    p_value_fisher = None # init

    if isinstance(contingency_table, pd.DataFrame):
        contingency_table = contingency_table.to_numpy()

    row_sums = contingency_table.sum(axis=1)
    col_sums = contingency_table.sum(axis=0)

    if any([v == 0 for v in contingency_table.flatten()]) or any(row_sums == 0) or any(col_sums == 0):
        print("[StatsError] Some values or marginal row / col totals in contingency_table are 0. Can't perform Fisher's exact test. Ignoring ...")
        return p_value_fisher
    
    # perform only if > 2x2 contingency table, otherwise use classic fisher test
    if (contingency_table.shape[0] >= 2 and contingency_table.shape[1] > 2) or (contingency_table.shape[1] >= 2 and contingency_table.shape[0] > 2):
        # convert np.array into r matrix
        r_matrix = ro.r.matrix(IntVector(contingency_table.flatten()), nrow=contingency_table.shape[0])

        try:
            result = stats_r.fisher_test(r_matrix, simulate_p_value=True, B=10000)
            p_value_fisher = result.rx2('p.value')[0]
        except Exception as e:
            sys.stderr.write(f"[StatsError] Fisher's test (mXn) failed in R: {e}\n")
            return None
        # Perform Fisher's Exact Test with Monte Carlo simulations for larger tables
        # simulate_p_value=True to perform Monte Carlo, B=10000 simulations

    # classical fisher test on 2x2 contingency table
    elif contingency_table.shape[0] == 2 and contingency_table.shape[1] == 2:
        results = fisher_exact(contingency_table)
        p_value_fisher = results.pvalue
    
    return p_value_fisher

def create_contingency_table_figure(df: pd.DataFrame, output_dir: str, cluster: int):
    """
    Divide contingency table in 2 tables showing marginal proportions
    """
    # Contingency table data
    data = df.to_numpy()

    # Labels for the cells()
    # Do not reverse row_labels here, we will use a different order later
    col_labels = df.index.to_list()
    row_labels = df.columns.to_list()
    col_labels.reverse()

    # Total sums for row, col, and overall
    row_totals = data.sum(axis=1)
    col_totals = data.sum(axis=0)

    # # Calculate percentages - need to transpose for create_fig() function
    row_perc = (data.T / row_totals).T  # Row-wise percentage
    col_perc = (data / col_totals).T  # Column-wise percentage

    def create_fig(data: np.ndarray, perc: np.ndarray, totals: np.ndarray, prefix: str, col_labels: list[str], row_labels: list[str], reverse_labels_cols_rows=False, self_report=True):
        """
        perc: transposed percentages (row or columns)
        """

        if reverse_labels_cols_rows:
            col_labels2 = col_labels
            col_labels = row_labels
            row_labels = col_labels2
            col_labels.reverse()
            row_labels.reverse()

        # Plotting
        fig, ax = plt.subplots(figsize=(20, 10))  # Larger figure size

        main_rectangles_to_plot = {
            "0": None, # 0,0 = upper left cell of 2x2 contingency table
            "1": None,
        }

        current_y = 0  # current y according to first element in draw_order
        current_x = 0
        for j in range(0, 2):
            
            width = 0.5
            height = 1 
            
            main_rectangles_to_plot[f"{j}"] = {
                "x": current_x,
                "y": current_y,
                "width": width,
                "height": height,
            }          

            # Update current_x to the right (move horizontally for next cell)
            current_x += width

        min_size = 0.035 # minimum size

        # # Create and draw inner subcells in main cells before drawing main cells
        for main_rectangle_k, main_rectangle_v in main_rectangles_to_plot.items(): # start for rrectangle 0,0 upper left one

            # min_size = min_size_absolute * main_rectangle_v.get("height")
            i = int(main_rectangle_k)
            
            subcell_up = {
                "x": main_rectangle_v.get("x"),
                "y": main_rectangle_v.get("y") + main_rectangle_v.get("height") - perc[i, 0],
                "height": perc[i, 0],
                "width": main_rectangle_v.get("width"),
                "color": "cornflowerblue"
            }

            subcell_down = {
                "x": main_rectangle_v.get("x"),
                "y": main_rectangle_v.get("y"),
                "height": perc[i, 1],
                "width": main_rectangle_v.get("width"),
                "color": "lightcoral"
            }

            # resize if too small
            if subcell_up.get("height") < min_size:
                subcell_up["y"] = 1 - min_size
                subcell_down["height"] = subcell_up.get("y")
                subcell_up["height"] = min_size

            if subcell_down.get("height") < min_size:
                old_subcell_up_height = subcell_up.get("height")
                subcell_down["height"] = min_size
                subcell_up["y"] = min_size
                subcell_up["height"] = subcell_up.get("height") - abs(min_size - old_subcell_up_height)
                
            subcell_up_rect = plt.Rectangle((subcell_up.get('x'), subcell_up.get('y')), subcell_up.get('width'), subcell_up.get('height'), 
                            facecolor=subcell_up.get('color'), edgecolor='black', linewidth=1)
            ax.add_patch(subcell_up_rect)

            subcell_down_rect = plt.Rectangle((subcell_down.get('x'), subcell_down.get('y')), subcell_down.get('width'), subcell_down.get('height'), 
                            facecolor=subcell_down.get('color'), edgecolor='black', linewidth=1)
            ax.add_patch(subcell_down_rect)

            shadow_shift = 0.003
            ax.text(subcell_down.get('x') + subcell_down.get('width') / 2 + shadow_shift, # shadow percentage value in white
                    subcell_down.get('y') + subcell_down.get('height') / 2, 
                    f'{perc[i, 1] * 100:.2f}%', 
                    ha='center', va='center', fontsize=16, color='white', fontweight="bold")
            
            ax.text(subcell_down.get('x') + subcell_down.get('width') / 2, #  percentage value in black
                    subcell_down.get('y') + subcell_down.get('height') / 2, 
                    f'{perc[i, 1] * 100:.2f}%', 
                    ha='center', va='center', fontsize=16, color='black', fontweight="bold")

            # Add text for percentages in the middle of subcell_up and subcell_down
            ax.text(subcell_up.get('x') + subcell_up.get('width') / 2 + shadow_shift, # shadow in white
                    subcell_up.get('y') + subcell_up.get('height') / 2, 
                    f'{perc[i, 0] * 100:.2f}%', 
                    ha='center', va='center', fontsize=16, color='white', fontweight="bold")
            
            ax.text(subcell_up.get('x') + subcell_up.get('width') / 2, # text in black
                    subcell_up.get('y') + subcell_up.get('height') / 2, 
                    f'{perc[i, 0] * 100:.2f}%', 
                    ha='center', va='center', fontsize=16, color='black', fontweight="bold")
            
            # bboxes
            bbox_margin = 0.15
            ax.text(subcell_up.get('x') + subcell_up.get('width') / 2 + bbox_margin, 
                subcell_up.get('y') + subcell_up.get('height') / 2,  # Position slightly below percentage
                f'N = {int(data[i, 0])}', 
                ha='center', va='center', fontsize=16, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

            ax.text(subcell_down.get('x') + subcell_down.get('width') / 2 + bbox_margin, 
                subcell_down.get('y') + subcell_down.get('height') / 2,  # Position slightly below percentage
                f'N = {int(data[i, 1])}', 
                ha='center', va='center', fontsize=16, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                 
        # plot main rectangles
        for main_rectangle_k, main_rectangle_v in main_rectangles_to_plot.items():
            i = int(main_rectangle_k)
            main_rect = plt.Rectangle((main_rectangles_to_plot.get(f'{i}').get('x'), main_rectangles_to_plot.get(f'{i}').get('y')), main_rectangles_to_plot.get(f'{i}').get('width'), main_rectangles_to_plot.get(f'{i}').get('height'), 
                        facecolor='none', edgecolor='black', linewidth=3)
            ax.add_patch(main_rect)
            ax.text(main_rectangles_to_plot[f'{i}'].get('x') + main_rectangles_to_plot[f'{i}'].get('width') / 2,
                -0.05,  # Position slightly below the plot
                f'N = {int(totals[i])}', ha='center', va='center', fontsize=16, color='black', fontweight='bold')

        # Set ticks for x and y axes, ensuring we match the number of labels and ticks
        ax.set_xticks([1-1/2*0.5, (1-1/2)*0.5])  # Place ticks in the center of each rectangle on x-axis
        ax.set_xticklabels(col_labels, fontsize=16, rotation=30)  # Rotate x-axis labels for clarity

        ax.set_yticks([1-1/2*0.5, (1-1/2)*0.5])  # Place ticks in the center of each rectangle on y-axis
        ax.set_yticklabels(row_labels, fontsize=16)  # Reverse row labels to match plot order

        # Set title and formatting
        plt.title(f"{prefix} {estimated_clusters_mapping[cluster]}", fontsize=16, pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add outer thick border around the entire plot
        plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, edgecolor='black', linewidth=4, facecolor='none'))

        # Add space between cells for clarity
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)

        if self_report:
            label_green = f"Self-reported {estimated_clusters_mapping[cluster]}"
            label_red = f"Not self-reported {estimated_clusters_mapping[cluster]}"
        else:
            label_green = f"{estimated_clusters_mapping[cluster]} genetic ancestry"
            label_red = f"Not {estimated_clusters_mapping[cluster]} genetic ancestry"

        legend_handles = [
            plt.Line2D([0], [0], color='cornflowerblue', lw=4, label=label_green),
            plt.Line2D([0], [0], color='lightcoral', lw=4, label=label_red)
        ]

        # Add the legend to the plot
        ax.legend(handles=legend_handles, loc='upper left', fontsize=10, 
          frameon=True, title="Legend", title_fontsize='12', borderpad=1,
          bbox_to_anchor=(1.05, 1))  # Adjust the x and y to position it

        # Add title and show plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"figure_contingency_{prefix}_{estimated_clusters_mapping[cluster]}.pdf"), dpi=300, format="pdf")
        plt.close()
        plt.clf()

    create_fig(data, row_perc, row_totals, "self_report_proportion_among_genetic_ancestry(total_rows)", col_labels, row_labels, reverse_labels_cols_rows=False, self_report=True)
    create_fig(data, col_perc, col_totals, "genetic_ancestry_proportions_among_self_report(total_cols)", col_labels, row_labels, reverse_labels_cols_rows=True, self_report=False)


def create_larger_contingency_table_figure(df: pd.DataFrame, output_dir: str, cluster: int):
    """
    Divide contingency table in 2 tables showing marginal proportions
    Function to be used for contingency tables with 2xm (m columns)
    
    """
    # Contingency table data

    data = df.to_numpy()

    row_labels = df.index.to_list()
    col_labels = df.columns.to_list()

     # change legends for better writing
    col_labels_splits = [label.split('_') for label in col_labels]

    for i in range(0, len(col_labels_splits)):
        lower_list = [elm.lower() for elm in col_labels_splits[i]] # list i with lower cases everywhere
        if "father" in lower_list:
            begin = "Father"
        elif "mother" in lower_list:
            begin = "Mother"
        elif "none" in lower_list or ("both" in lower_list and "no" in lower_list):
            begin = "None"
        elif "both" in lower_list:
            begin = "Both"
        else:
            begin = "UNKNOWN"

        col_labels[i] = f"{begin} self-reported {estimated_clusters_mapping[cluster]}"

    # Total sums for row, col, and overall
    row_totals = data.sum(axis=1)
    col_totals = data.sum(axis=0)

    # # Calculate percentages - need to transpose for create_fig() function
    row_perc = (data.T / row_totals)  # Row-wise percentage
    col_perc = (data / col_totals) # Column-wise percentage

    def create_fig(data: np.ndarray, perc: np.ndarray, totals: np.ndarray, prefix: str, col_labels: list[str], row_labels: list[str], reverse_labels_cols_rows=False):
        """
        perc: transposed percentages (row or columns)
        """
        
        # reverse labels for plotting
        if reverse_labels_cols_rows:
            tmp = row_labels
            row_labels = col_labels
            col_labels = tmp

        # Plotting
        fig, ax = plt.subplots(figsize=(20, 10))  # Larger figure size

        main_rectangles_to_plot = {
            str(i): None for i in range(0, data.shape[1])
        }

        # 1 - main rectangles creation
        current_y = 0  # current y according to first element in draw_order
        current_x = 0
        for j in range(0, data.shape[1]):
            
            width = 1/data.shape[1] # 1/nb_cols = number of columns
            height = 1 
            
            main_rectangles_to_plot[f"{j}"] = {
                "x": current_x,
                "y": current_y,
                "width": width,
                "height": height,
            }          

            # Update current_x to the right (move horizontally for next cell)
            current_x += width

        # 2 - subcell rectangles creation
        min_size = 0.035 # minimum size

        colors = ["cornflowerblue", "lightcoral", "moccasin", "palevioletred", "blue", "darkorange", "yellow", "tan"]

        # # # Create and draw inner subcells in main cells before drawing main cells
        for main_rectangle_k, main_rectangle_v in main_rectangles_to_plot.items():  # Start for rectangle 0,0 upper left one
            
            # Initiate first subcell as top subcell
            subcells = {
                0: {
                    "x": main_rectangle_v.get("x"),
                    "y": main_rectangle_v.get("height") - perc[0, int(main_rectangle_k)],  # Use perc for height
                    "height": perc[0, int(main_rectangle_k)],  # Use perc for height
                    "width": main_rectangle_v.get("width"),
                    "color": colors[0]
                }
            }

            # Create other subcells, from top to bottom
            for i in range(1, data.shape[0]):
                subcells[i] = {
                    "x": subcells.get(i - 1).get("x"),
                    "y": subcells.get(i - 1).get("y") - perc[i, int(main_rectangle_k)],
                    "height": perc[i, int(main_rectangle_k)],
                    "width": main_rectangle_v.get("width"),
                    "color": colors[i]
                }

                # Resizing logic: apply min_size if necessary
                if subcells.get(i - 1).get("height") < min_size:
                    # Handle the subcell just above the current one
                    subcells[i - 1]["height"] = min_size
                    subcells[i - 1]["y"] = subcells[i].get("y") + subcells[i].get("height")

                if subcells.get(i).get("height") < min_size:
                    # Ensure current subcell height is at least min_size
                    subcells[i]["height"] = min_size
                    # Adjust y-position of current subcell
                    subcells[i]["y"] = subcells[i - 1].get("y") - subcells[i].get("height")

            # draw subcells and add text               
            for subcell_k, subcell_v in subcells.items():
                subcell_rect = plt.Rectangle((subcell_v.get('x'), subcell_v.get('y')), subcell_v.get('width'), subcell_v.get('height'), 
                            facecolor=subcell_v.get('color'), edgecolor='black', linewidth=1)
                ax.add_patch(subcell_rect)

                if subcell_v.get("height") <= min_size:
                    lower_value = -0.009
                else:
                    lower_value = 0

                bbox_margin = 0.09

                # Add shadow text (white) for percentages in the middle of subcell_up and subcell_down
                ax.text(subcell_v.get('x') + subcell_v.get('width') / 2 + 0.003, 
                        subcell_v.get('y') + subcell_v.get('height') / 2 + lower_value, 
                        f'{perc[int(subcell_k), int(main_rectangle_k)] * 100:.2f}%', 
                        ha='center', va='center', fontsize=16, color='white', fontweight="bold")

                # Add text for percentages in the middle of subcell_up and subcell_down
                ax.text(subcell_v.get('x') + subcell_v.get('width') / 2, 
                        subcell_v.get('y') + subcell_v.get('height') / 2 + lower_value, 
                        f'{perc[int(subcell_k), int(main_rectangle_k)] * 100:.2f}%', 
                        ha='center', va='center', fontsize=16, color='black', fontweight="bold")
                

                
                # Add text for number of individuals in each subcell - white bbox
                ax.text(subcell_v.get('x') + subcell_v.get('width') / 2 + bbox_margin, 
                    subcell_v.get('y') + subcell_v.get('height') / 2,  # Position slightly below percentage
                    f'N = {int(data[int(subcell_k), int(main_rectangle_k)])}', 
                    ha='center', va='center', fontsize=16, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        # plot main rectangles
        for main_rectangle_k, main_rectangle_v in main_rectangles_to_plot.items():
            i = int(main_rectangle_k)
            main_rect = plt.Rectangle((main_rectangles_to_plot.get(f'{i}').get('x'), main_rectangles_to_plot.get(f'{i}').get('y')), main_rectangles_to_plot.get(f'{i}').get('width'), main_rectangles_to_plot.get(f'{i}').get('height'), 
                        facecolor='none', edgecolor='black', linewidth=3)
            ax.add_patch(main_rect)

            ax.text(main_rectangles_to_plot[f'{i}'].get('x') + main_rectangles_to_plot[f'{i}'].get('width') / 2,
                -0.025,  # Position slightly below the plot
                f'N = {int(totals[i])}', ha='center', va='center', fontsize=16, color='black', fontweight='bold')

        # Set ticks for x and y axes, ensuring we match the number of labels and ticks
        ax.set_xticks( [1/data.shape[1] * 0.5 + a*(1/data.shape[1]) for a in range(0, data.shape[1])] )  # Place ticks in the center of each rectangle on x-axis
        ax.set_xticklabels(col_labels, fontsize=16, rotation=30)  # Rotate x-axis labels for clarity

        ax.set_yticks( [1/data.shape[0] * 0.5 + a*(1/data.shape[0]) for a in range(0, data.shape[0])] )  # Place ticks in the center of each rectangle on y-axis
        ax.set_yticklabels(row_labels, fontsize=16)  # Reverse row labels to match plot order

        # Set title and formatting
        plt.title(f"{prefix} {estimated_clusters_mapping[cluster]}", fontsize=16, pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add outer thick border around the entire plot
        plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, edgecolor='black', linewidth=4, facecolor='none'))

        # Add space between cells for clarity
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)

        legend_handles = [ plt.Line2D([0], [0], color=colors[i], lw=4, label=row_labels[i]) for i in range(0, len(row_labels)) ]

        # Add the legend to the plot
        ax.legend(handles=legend_handles, loc='upper left', fontsize=10, 
          frameon=True, title="Legend", title_fontsize='12', borderpad=1,
          bbox_to_anchor=(1.05, 1))  # Adjust the x and y to position it

        # Add title and show plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"figure_contingency_{prefix}_parents_{estimated_clusters_mapping[cluster]}.pdf"), dpi=300, format="pdf")
        plt.close()
        plt.clf()

    create_fig(data, col_perc, col_totals, "genetic_ancestry_proportions_among_self_report(total_cols)", col_labels, row_labels, reverse_labels_cols_rows=False)
    create_fig(data.T, row_perc, row_totals, "self_report_proportion_among_genetic_ancestry(total_rows)", col_labels, row_labels, reverse_labels_cols_rows=True)


def create_admixture_table_figure(cross_distrib_afr: np.ndarray, cross_distrib_eur: np.ndarray, cross_distrib_asia: np.ndarray, cluster: int, output_dir: str):
    """
    Takes as input 2 cross tables (with distributions inside each cell): 1 for African genome proportion and one for European genome proportion and also add asian proportion.
    Split in 4 equal cells, and divide each cell depending on admixture proportions.
    
    distrib 1 = African genome proportions
    distrib 2 = European genome proportions

    Also compute mean observed +/- 1.96*std/sqrt(n)
    """

    # remove any nan values
    cross_distrib_afr = [cross_distrib_afr[i, j, ~np.isnan(cross_distrib_afr[i, j, :])] for i in range(cross_distrib_afr.shape[0]) for j in range(cross_distrib_afr.shape[1])]
    cross_distrib_eur = [cross_distrib_eur[i, j, ~np.isnan(cross_distrib_eur[i, j, :])] for i in range(cross_distrib_eur.shape[0]) for j in range(cross_distrib_eur.shape[1])]
    cross_distrib_asia = [cross_distrib_asia[i, j, ~np.isnan(cross_distrib_asia[i, j, :])] for i in range(cross_distrib_asia.shape[0]) for j in range(cross_distrib_asia.shape[1])]

    z = 1.96

    # compute mean and IQR and std for AFR and EUR genome proportions
    cross_distrib_mean_afr = np.array([
        [np.mean(cross_distrib_afr[0]), np.mean(cross_distrib_afr[1])],
        [np.mean(cross_distrib_afr[2]), np.mean(cross_distrib_afr[3])]
    ])

    cross_distrib_ic_afr = np.array([
        [z * ( np.std(cross_distrib_afr[0])/np.sqrt(len(cross_distrib_afr[0])) ), z * ( np.std(cross_distrib_afr[1])/np.sqrt(len(cross_distrib_afr[1])) )],
        [z * ( np.std(cross_distrib_afr[2])/np.sqrt(len(cross_distrib_afr[2])) ), z * ( np.std(cross_distrib_afr[3])/np.sqrt(len(cross_distrib_afr[3])) )],
    ])

    cross_distrib_mean_eur = np.array([
        [np.mean(cross_distrib_eur[0]), np.mean(cross_distrib_eur[1])],
        [np.mean(cross_distrib_eur[2]), np.mean(cross_distrib_eur[3])]
    ])

    cross_distrib_ic_eur = np.array([
        [z * (np.std(cross_distrib_eur[0])/np.sqrt(len(cross_distrib_eur[0])) ), z * ( np.std(cross_distrib_eur[1])/np.sqrt(len(cross_distrib_eur[1])) )],
        [z * ( np.std(cross_distrib_eur[2])/np.sqrt(len(cross_distrib_eur[2])) ), z * ( np.std(cross_distrib_eur[3])/np.sqrt(len(cross_distrib_eur[3])) )],
    ])

    cross_distrib_mean_asia = np.array([
        [np.mean(cross_distrib_asia[0]), np.mean(cross_distrib_asia[1])],
        [np.mean(cross_distrib_asia[2]), np.mean(cross_distrib_asia[3])]
    ])

    cross_distrib_ic_asia = np.array([
        [z * (np.std(cross_distrib_asia[0])/np.sqrt(len(cross_distrib_asia[0])) ), z * ( np.std(cross_distrib_asia[1])/np.sqrt(len(cross_distrib_asia[1])) )],
        [z * ( np.std(cross_distrib_asia[2])/np.sqrt(len(cross_distrib_asia[2])) ), z * ( np.std(cross_distrib_asia[3])/np.sqrt(len(cross_distrib_asia[3])) )],
    ])

    col_labels = [f"Self_reported_{estimated_clusters_mapping[cluster]}", f"Not_self_reported_{estimated_clusters_mapping[cluster]}"]
    col_labels.reverse()
    row_labels = [f"{estimated_clusters_mapping[cluster]}_genetic_ancestry", f"Not_{estimated_clusters_mapping[cluster]}_genetic_ancestry"]

    def create_fig(cross_table_mean_1: np.ndarray, cross_table_ic_1: np.ndarray, cross_table_mean_2: np.ndarray, cross_table_ic_2: np.ndarray, cross_table_mean_3: np.ndarray, cross_table_ic_3: np.ndarray, col_labels: list, row_labels: list, output_dir: str):
        """"""
        # Plotting
        fig, ax = plt.subplots(figsize=(20, 10))  # Larger figure size

        main_rectangles_to_plot = {
            "0,0": None, # 0,0 = upper left cell of 2x2 contingency table
            "0,1": None,
            "1,0": None,
            "1,1": None
        }

        current_y = 0.5  # current y according to first element in draw_order
        current_x = 0

        for i in range(0, 2):
            for j in range(0, 2):
                
                width = 0.5
                height = 0.5
                
                main_rectangles_to_plot[f"{i},{j}"] = {
                    "x": current_x,
                    "y": current_y,
                    "width": width,
                    "height": height,
                }          

                # Update current_x to the right (move horizontally for next cell)
                current_x += width

            current_x = 0
            current_y = 0

        min_size = 0.035 # minimum size

          # # Create and draw inner subcells in main cells before drawing main cells
        for main_rectangle_k, main_rectangle_v in main_rectangles_to_plot.items(): # start for rrectangle 0,0 upper left one

            # min_size = min_size_absolute * main_rectangle_v.get("height")
            i, j = main_rectangle_k.split(',')
            i, j = int(i), int(j)

            subcell_up = { # Afr proportion (1)
                "x": main_rectangle_v.get("x"),
                "y": main_rectangle_v.get("y") + main_rectangle_v.get("height") - (cross_table_mean_1[i, j] * main_rectangle_v.get("height")),
                "height": cross_table_mean_1[i, j] * main_rectangle_v.get("height"),
                "width": main_rectangle_v.get("width"),
                "color": "lightcoral"
            }

            subcell_bottom = {
                "x": main_rectangle_v.get("x"),
                "y": main_rectangle_v.get("y"),
                "height": cross_table_mean_3[i, j] * main_rectangle_v.get("height"),
                "width": main_rectangle_v.get("width"),
                "color": "lightyellow"
            }

            subcell_down = { # Eur proportion (2)
                "x": main_rectangle_v.get("x"),
                "y": main_rectangle_v.get("y") + subcell_bottom.get("height") ,
                "height": cross_table_mean_2[i, j] * main_rectangle_v.get("height"),
                "width": main_rectangle_v.get("width"),
                "color": "cornflowerblue"
            }

            # Apply minimum size adjustments, ensuring other subcells update accordingly
            if subcell_bottom.get("height") < min_size:
                # Adjust East Asian subcell height
                old_height_bottom = subcell_bottom["height"]
                subcell_bottom["height"] = min_size
                subcell_down["y"] = subcell_bottom["y"] + subcell_bottom["height"]  # Adjust Euro subcell y-position
                subcell_down["height"] -= min_size - old_height_bottom  # Adjust Euro subcell height
                subcell_up["y"] = subcell_down["y"] + subcell_down["height"]  # Adjust Afr subcell y-position

            if subcell_up.get("height") < min_size:
                # Adjust African subcell height
                old_height_up = subcell_up["height"]
                subcell_up["height"] = min_size
                subcell_up["y"] = main_rectangle_v.get("y") + main_rectangle_v.get("height") - min_size  # Adjust position
                subcell_down["height"] -= min_size - old_height_up  # Shrink European subcell to compensate

            if subcell_down.get("height") < min_size:
                # Adjust European subcell height
                old_height_down = subcell_down["height"]
                subcell_down["height"] = min_size
                subcell_down["y"] = subcell_up["y"] - min_size  # Adjust position
                subcell_up["height"] -= min_size - old_height_down  # Shrink African subcell to compensate

            subcell_up_rect = plt.Rectangle((subcell_up.get('x'), subcell_up.get('y')), subcell_up.get('width'), subcell_up.get('height'), 
                            facecolor=subcell_up.get('color'), edgecolor='black', linewidth=1)
            ax.add_patch(subcell_up_rect)

            subcell_down_rect = plt.Rectangle((subcell_down.get('x'), subcell_down.get('y')), subcell_down.get('width'), subcell_down.get('height'), 
                            facecolor=subcell_down.get('color'), edgecolor='black', linewidth=1)
            ax.add_patch(subcell_down_rect)

            subcell_bottom_rect = plt.Rectangle((subcell_bottom.get('x'), subcell_bottom.get('y')), subcell_bottom.get('width'), subcell_bottom.get('height'), 
                            facecolor=subcell_bottom.get('color'), edgecolor='black', linewidth=1)
            ax.add_patch(subcell_bottom_rect)

            # Add text shadow for percentages in the middle of subcell_up
            shadow_shift = 0.003
            ax.text(subcell_up.get('x') + subcell_up.get('width') / 2 + shadow_shift, 
                    subcell_up.get('y') + subcell_up.get('height') / 2, 
                    f'{cross_table_mean_1[i, j] * 100:.2f}% ± {cross_table_ic_1[i, j] * 100:.3f}%', 
                    ha='center', va='center', fontsize=12, color='white', fontweight="bold")

            # Add text for percentages in the middle of subcell_up
            ax.text(subcell_up.get('x') + subcell_up.get('width') / 2, 
                    subcell_up.get('y') + subcell_up.get('height') / 2, 
                    f'{cross_table_mean_1[i, j] * 100:.2f}% ± {cross_table_ic_1[i, j] * 100:.3f}%', 
                    ha='center', va='center', fontsize=12, color='black', fontweight="bold")
            
            # shadow down
            ax.text(subcell_down.get('x') + subcell_down.get('width') / 2 + shadow_shift, 
                    subcell_down.get('y') + subcell_down.get('height') / 2, 
                    f'{cross_table_mean_2[i, j] * 100:.2f}% ± {cross_table_ic_2[i, j] * 100:.3f}%', 
                    ha='center', va='center', fontsize=12, color='white', fontweight="bold")
            
            # text down
            ax.text(subcell_down.get('x') + subcell_down.get('width') / 2, 
                    subcell_down.get('y') + subcell_down.get('height') / 2, 
                    f'{cross_table_mean_2[i, j] * 100:.2f}% ± {cross_table_ic_2[i, j] * 100:.3f}%', 
                    ha='center', va='center', fontsize=12, color='black', fontweight="bold")

            # shadow bottom
            ax.text(subcell_bottom.get('x') + subcell_bottom.get('width') / 2 + shadow_shift, 
                    subcell_bottom.get('y') + subcell_bottom.get('height') / 2, 
                    f'{cross_table_mean_3[i, j] * 100:.2f}% ± {cross_table_ic_3[i, j] * 100:.3f}%', 
                    ha='center', va='center', fontsize=12, color='white', fontweight="bold")
            # text bottom     
            ax.text(subcell_bottom.get('x') + subcell_bottom.get('width') / 2, 
                    subcell_bottom.get('y') + subcell_bottom.get('height') / 2, 
                    f'{cross_table_mean_3[i, j] * 100:.2f}% ± {cross_table_ic_3[i, j] * 100:.3f}%', 
                    ha='center', va='center', fontsize=12, color='black', fontweight="bold")
                 
        # plot main rectangles
        for main_rectangle_k, main_rectangle_v in main_rectangles_to_plot.items():
            i, j = main_rectangle_k.split(',')
            i, j = int(i), int(j)

            main_rect = plt.Rectangle((main_rectangles_to_plot.get(f'{i},{j}').get('x'), main_rectangles_to_plot.get(f'{i},{j}').get('y')), main_rectangles_to_plot.get(f'{i},{j}').get('width'), main_rectangles_to_plot.get(f'{i},{j}').get('height'), 
                        facecolor='none', edgecolor='black', linewidth=3)
            ax.add_patch(main_rect)
    
        # Set ticks for x and y axes, ensuring we match the number of labels and ticks
        ax.set_xticks([1-1/2*0.5, (1-1/2)*0.5])  # Place ticks in the center of each rectangle on x-axis
        ax.set_xticklabels(col_labels, fontsize=12, rotation=30)  # Rotate x-axis labels for clarity

        ax.set_yticks([1-1/2*0.5, (1-1/2)*0.5])  # Place ticks in the center of each rectangle on y-axis
        ax.set_yticklabels(row_labels, fontsize=12)  # Reverse row labels to match plot order

        # Set title and formatting
        plt.title(f"Admixture proportion (AFR/EUR/ASIA) {estimated_clusters_mapping[cluster]}", fontsize=16, pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add outer thick border around the entire plot
        plt.gca().add_patch(plt.Rectangle((0, 0), 1, 1, edgecolor='black', linewidth=4, facecolor='none'))

        # Add space between cells for clarity
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)

        label_blue = "European genome proportion"
        label_red = "African genome proportion"
        label_yellow = "East-Asian genome proportion"

        legend_handles = [
            plt.Line2D([0], [0], color='lightcoral', lw=4, label=label_red),
            plt.Line2D([0], [0], color='cornflowerblue', lw=4, label=label_blue),
            plt.Line2D([0], [0], color="lightyellow", lw=4, label=label_yellow)
        ]

        # Add the legend to the plot
        ax.legend(handles=legend_handles, loc='upper left', fontsize=10, 
          frameon=True, title="Legend", title_fontsize='12', borderpad=1,
          bbox_to_anchor=(1.05, 1))  # Adjust the x and y to position it

        # Add title and show plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"figure_contingency_admixture_prop_{estimated_clusters_mapping[cluster]}.pdf"), dpi=300, format="pdf")
        plt.close()
        plt.clf()
    
    create_fig(cross_distrib_mean_afr, cross_distrib_ic_afr, cross_distrib_mean_eur, cross_distrib_ic_eur, cross_distrib_mean_asia, cross_distrib_ic_asia, col_labels, row_labels, output_dir)