#!/usr/bin/env python3

####################################
### ADMIXED ANCESTRY MODULE FILE ###
####################################

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import seaborn as sns
from shlex import split as ssplit
import os
import re
import numpy as np
from config_main3_admixed_ancestry_analysis import *
matplotlib.use('agg')

def generate_list_indiv_to_keep(plink_prefix: str, output_dir: str, superpopulations_to_keep: list, subpopulations_to_keep: list, kg_list_file: str):
    """
    Generate a text file containing the FID (=SUPERPOPULATION) and the IID (=SAMPLE_ID) of individuals to keep based on their respective superpopulation and subpopulation code.
    This file is used by --keep option of plink to keep only selected individuals.

    Parameters:
    plink_prefix: str - basename of plink file
    output_dir: str - output directory
    superpopulations_to_keep: list - list of superpopulation names to keep
    subpopulations_to_keep: list - list of subpopulation codes to keep

    Return: str - file path to list of individuals to keep
    """

    # read fam file and create df, keep only superpop and sample id columns
    fam_df = pd.read_table(plink_prefix+".fam", sep=' ', header=None).iloc[:, 0:2]
    fam_df.columns = ["SUPERPOP", "SAMPLE_ID"]
    
    # read 1kg list file and create df by keeping sample id and subpopulation code columuns only
    kg_df = pd.read_table(kg_list_file)[["Sample name", "Population code"]]
    kg_df.columns = ["SAMPLE_ID", "SUBPOP"]
    
    # merge df on sample id by keeping all individuals from fam_df
    merged_df = pd.merge(fam_df, kg_df, how="left", on="SAMPLE_ID")
    
    # free space
    del fam_df
    del kg_df

    # select only individuals to keep
    merged_df = merged_df[merged_df["SUPERPOP"].isin(superpopulations_to_keep) & (merged_df["SUBPOP"].isin(subpopulations_to_keep) | merged_df["SUBPOP"].isna())]
    merged_df.drop(["SUBPOP"], axis=1, inplace=True)

    # write to output file to be used by --keep option of plink
    output_file = os.path.join(output_dir, "indiv_to_keep.txt")
    merged_df.to_csv(output_file, sep = ' ', header=False, index=False) # remove header and index (number of lines)

    print(f"[NewDataset] Plink files of individuals to keep created at {output_file}.")

    return output_file

def change_ctrl_case_name(plink_prefix: str):
    """
    Replace Case and Control by CASE and CTRL
    """

    output_file_prefix = plink_prefix+"_tmp"
    with open(output_file_prefix+".fam", 'w') as f:
        awk_c = ["gawk", '{sub("Case", "CASE", $1); sub("Control", "CTRL", $1); print $0}', plink_prefix+".fam"]
        awk_p = Popen(awk_c, stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate()
    
    mv_c = ["mv", output_file_prefix+".fam", plink_prefix+".fam"]
    mv_p = Popen(mv_c, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    mv_p.communicate()

    return plink_prefix

def generate_plink_dataset_with_indiv_to_keep(plink_prefix: str, output_dir: str, indiv_to_keep_file: str):
    """
    Generate new plink dataset with only selected individuals and sort the dataset.

    Parameters:
    plink_prefix: str - basename of plink file
    output_dir: str - output directory
    indiv_to_keep_file: str - input file for plink --keep option

    Return: str - New plink prefix filepath
    """

    # create new plink prefix name
    new_plink_prefix = os.path.join(output_dir, os.path.basename(plink_prefix) + "_only_indiv_to_keep")

    # generate new filtered dataset in output directory
    plink_c = ["plink", "--bfile", plink_prefix, "--keep", indiv_to_keep_file, "--indiv-sort", "0", "--make-bed", "--out", new_plink_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p.communicate()

    print(f"[Keep][Sort] New plink dataset created and filtered to keep only selected individuals. Plink dataset sorted.")

    return new_plink_prefix

def generate_pop_file(plink_prefix: str):
    """"""
    pop_df = pd.read_table(plink_prefix+".fam", sep=' ', header=None).iloc[:, 0] # keep only first column with superpop
    pop_df.replace(["CASE", "CTRL"], '-', inplace=True)
    pop_file = plink_prefix+".pop"
    pop_df.to_csv(pop_file, sep=' ', header=False, index=False)

    return pop_file


def admixture(plink_prefix: str, k: int, output_dir: str):
    """
    Run admixture software (use admixture 32 bits version).

    Parameters:
    plink_prefix: str - basename of plink files
    k: int - admixture K parameter (number of categories / population used for admixed ancestry)
    output_dir: str - output directory

    Return: str - admixture output files paths
    """

    output_files_basename = os.path.splitext(os.path.basename(plink_prefix+".bed"))[0]
    Q_file = output_files_basename+f".{str(k)}.Q"
    P_file = output_files_basename+f".{str(k)}.P"
    log_file = output_files_basename+f"_log{k}.out"
    
    # run admixture and write output in log file
    with open(log_file, 'w') as log_f:
        admixture_c = ["admixture32", "--cv", "--supervised", plink_prefix+".bed", str(k)]
        admixture_p = Popen(admixture_c, stdout=log_f, stdin=PIPE, stderr=PIPE, text=True)
        admixture_p.communicate()

    # move created files (.P, .Q, log.out) to output_dir
    bash_c1 = f"mv {Q_file} {output_dir}"
    bash_p1 = Popen(ssplit(bash_c1), stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    bash_p1.communicate()

    bash_c2 = f"mv {P_file} {output_dir}"
    bash_p2 = Popen(ssplit(bash_c2), stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
    bash_p2.communicate()

    bash_c3 = f"mv {log_file} {output_dir}"
    bash_p3 = Popen(ssplit(bash_c3), stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
    bash_p3.communicate()

    print(f"[Admixture] Admixed ancestry files {Q_file} and {P_file} created in {output_dir}.")

    return os.path.join(output_dir, Q_file), os.path.join(output_dir, P_file), os.path.join(output_dir, log_file)

def merge_Q_and_fam_file(plink_prefix: str, Q_file: str, kg_list_file: str, output_dir:str):
    """
    Merge 2 first columns of fam file (SAMPLE_ID and FID) with all columns from Q file and also add the subpopulations associated to each individuals.

    Parameters:
    plink_prefix: str - basename of plink files
    Q_file: str - filepath .Q output file from admixture
    kg_list_file: str - 1000 genomes file list of individuals with subpopulations
    output_dir: str - output directory

    Return: str - merged Q file path + pd.DataFrame - merged Q and fam file
    """

    splitted = Q_file.split('.') # splitted Q file name on '.'
    new_Q_file = "".join(splitted[0]+"_fam"+'.'+splitted[1]+'.'+splitted[2]) # create new merged Q file name (include fam in the name)

    # merge fam file and Q file (must be (and should be) in the same order)
    fam_df = pd.read_table(plink_prefix+".fam", sep=' ', header=None).iloc[:, 0:2]
    fam_df.columns = ["SUPERPOP", "SAMPLE_ID"]
    
    Q_df = pd.read_table(Q_file, sep=' ', header=None)
    Q_df.columns = [str(i) for i in range(1, len(Q_df.columns)+1)]

    Q_df["SAMPLE_ID"] = fam_df["SAMPLE_ID"]

    #+["SAMPLE_ID"]
    merged_Q_df = pd.merge(fam_df, Q_df, how="inner", on="SAMPLE_ID")
    
    # free space
    del Q_df
    del fam_df

    # add subpopulations associated to each individual
    kg_df = pd.read_table(kg_list_file)[["Sample name", "Population code"]]
    kg_df.columns = ["SAMPLE_ID", "SUBPOP"]
    
    merged_Q_df = pd.merge(merged_Q_df, kg_df, "left", on="SAMPLE_ID") # left works, inner doesn't because we only keep common sample_id
    del kg_df # free space
    merged_Q_df.set_index("SAMPLE_ID", inplace=True)

    # save merged_Q_df as new_Q_file
    merged_Q_df.to_csv(new_Q_file, sep=' ') 
    
    return merged_Q_df, new_Q_file
    
def check_concordance_between_pop_and_subpop(Q_df: pd.DataFrame, Q_file: str, matching_superpop_subpop: dict):
    """
    For troubleshooting. Check if subpop code is in agreement with assigned superpopulation. Update superpopulation according to subcode
    
    Parameters:
    matching_superpop_subpop: dict - matching dictionary between superpopulations (keys) and subpopulations (list of values)
    """

    for index, row in Q_df.iterrows():
        if not row["SUBPOP"] in matching_superpop_subpop[row["SUPERPOP"]]: # wrong superpopulation
            print("[MISMATCH]")
            print("Subpop: ", row["SUBPOP"], "\nSuperpop: ", row["SUPERPOP"], "\nAvailable subpop code:", matching_superpop_subpop[row["SUPERPOP"]])
            for superpop, subpops in matching_superpop_subpop.items():
                if row["SUBPOP"] in (subpops):
                    Q_df["SUPERPOP"].iloc[index] = superpop 
    
    Q_df.set_index("SAMPLE_ID", inplace=True)
    # update Q file
    Q_df.to_csv(Q_file, sep=' ')

    return Q_df, Q_file

def label_columns(Q_df: pd.DataFrame, Q_file: str, superpopulations: list):
    """
    Find the superpopulation associated to each column in Q file (merged Q-fam file).\n
    
    Parameters:
    Q_df: pd.dataFrame - Q dataFrame
    Q_file: str - Q file path
    superpopulations: list of superpopulations names to consider (reference superpopulations)

    Return: Labelled Q - fam file path and dataframe
    """

    # dictionary of correspondance between 
    corres_dict = {}

    # label columns by retrieving the corresponding population to each column
    for pop in superpopulations: # for each reference superpopulation
        pop_df = Q_df[Q_df["SUPERPOP"]==pop] # select only one superpopulation
        sum_ancestry_columns = [] # to store the score associated to each column
        for i in range(1, len(Q_df.columns)-1): # -2 because of columns SUPERPOP and SUBPOP
            sum_ancestry_columns.append(pop_df[str(i)].sum())
        
        # find column number name associated to the current pop
        print(pop, "- sum ancestry proportions:", sum_ancestry_columns)

        associated_col_number = sum_ancestry_columns.index(max(sum_ancestry_columns))+1 # associated column number

        corres_dict[str(associated_col_number)] = pop # create an entry in the correspondance dict
    
    # label columns
    print("correspondance dictionary:", corres_dict)
    
    Q_df.columns = ["SUPERPOP"]+[corres_dict[str(j)] for j in range(1, len(Q_df.columns)-1)]+["SUBPOP"]

    output_Qfile_path = Q_file.split('.')[0]+"_labelled"+"."+Q_file.split('.')[1]+"."+Q_file.split('.')[2] # get output Qfile path
    Q_df.to_csv(output_Qfile_path, sep=' ')

    return Q_df, output_Qfile_path

def plot_admixed_ancestry(Q_df: pd.DataFrame, labelled_clusters: pd.DataFrame, output_dir: str, superpopulations: list, custom_colors: dict, cluster_colors: dict, sort_by="Africa"):
    """
    Create admixed ancestry plot and save the plot as pdf with PCA clusters displayed.
    
    Parameters:
    Q_df: pd.DataFrame - Q-fam labelled dataframe including PCA_CLUSTER
    labelled_clusters: pd.DataFrame - DataFrame containing SAMPLE_ID and associated cluster information
    output_dir: str - output directory
    superpopulations: list - list of superpopulation (reference superpopulations)
    custom_colors: dict - dictionary with keys = superpopulations and values = colors
    cluster_colors: dict - dictionary with keys = PCA clusters and values = colors
    
    Return: None
    """
    fontsize=45

    # replace FID "CASE" by "OFSEP_HD"
    Q_df.replace("CASE", "OFSEP_HD", inplace=True)

    # create custom order to sort dataframe
    if len(superpopulations) == 3:
        custom_order = superpopulations[0:2] + ["OFSEP_HD"] + [superpopulations[2]] + ["CTRL"]  # to get correct order
    else:
        print("[NeedToModify] Length of superpopulations list used as reference population isn't equal to 3.")
        exit(8)

    # sort dataframe by superpopulation according to custom order and by "sort_by" column in descending order
    Q_df_sorted = pd.DataFrame()  # initialize empty dataframe
    for superpop in custom_order:
        tmp_df = Q_df[Q_df["SUPERPOP"] == superpop]  # get current superpopulation
        tmp_df.sort_values(by=sort_by, ascending=False, inplace=True)  # sort values among superpopulation
        Q_df_sorted = pd.concat([Q_df_sorted, tmp_df], axis=0)  # update dataframe
    del Q_df  # free space

    # Create a subset of Q_df_sorted for "OFSEP_HD" individuals only
    Q_df_sorted_case_only = Q_df_sorted[Q_df_sorted["SUPERPOP"] == "OFSEP_HD"]

    # Merge the clusters DataFrame (labelled_clusters) based on SAMPLE_ID into Q_df_sorted_case_only
    Q_df_sorted_case_only = pd.merge(Q_df_sorted_case_only, labelled_clusters[['SAMPLE_ID', 'cluster']], on="SAMPLE_ID", how="left")

    # plot admixed ancestry
    fig, (ax, ax2) = plt.subplots(2, 1, width_ratios=[0.5], gridspec_kw={'height_ratios': [5, 0.5]}, figsize=(100, 75))  # Two subplots, ax2 is smaller

    # plot admixed ancestry in the main plot (ax)
    Q_df_sorted[superpopulations].plot(kind="bar", stacked=True, color=[custom_colors.get(col, 'gray') for col in superpopulations], ax=ax)  # default color is grey

    #Calculate the positions for the vertical lines based on the "FID" column
    Q_df_sorted.reset_index(inplace=True)
    x_coords = []
    for i, superpop in enumerate(Q_df_sorted["SUPERPOP"]):
        if i == 0:
            x_coords.append(i)
        elif superpop != Q_df_sorted.loc[i-1, "SUPERPOP"]:
            x_coords.append(i - 0.5)

    # Add vertical lines to separate populations based on "FID"
    for i, xc in enumerate(x_coords):
            # if i > 0 and ( (Q_df_sorted["SUPERPOP"].iloc[int(xc + 0.5)] == "OFSEP_HD" and Q_df_sorted["SUPERPOP"].iloc[int(xc + 1.5)] == superpopulations[2]) or
            #                (Q_df_sorted["SUPERPOP"].iloc[int(xc + 0.5)] == "Africa" and Q_df_sorted["SUPERPOP"].iloc[int(xc + 1.5)] == "OFSEP_HD") or
            #                (Q_df_sorted["SUPERPOP"].iloc[int(xc + 0.5)] == "Europe" and Q_df_sorted["SUPERPOP"].iloc[int(xc + 1.5)] == "CTRL") ):
        ax.axvline(x=xc, color='black', linestyle='-', linewidth=1.5)  # Thicker line
            # else:
            #     ax.axvline(x=xc, color='black', linestyle='-', linewidth=2)  # Regular lines

    # Plot cluster information below each individual (only for "OFSEP_HD")
    ax2.set_xlim(ax.get_xlim())  # Ensure the new axis matches the original one
    ax2.set_xticks([])  # Remove tick labels
    ax2.set_yticks([])  # Remove y-ticks
    ax2.set_xlabel("Genetic ancestry clusters OFSEP-HD", fontsize=fontsize, fontweight = "bold")
    ax2.xaxis.set_label_coords(0.40, -0.1) 

    # Add colored vertical lines or bars for PCA clusters under "OFSEP_HD" individuals only
    cluster_handles = []  # To store handles for legend
    max_pos_case_in_all_indivs = int(Q_df_sorted[Q_df_sorted['SUPERPOP'] == 'OFSEP_HD'].index.max())
    for i in range(len(Q_df_sorted)):
        if i <= max_pos_case_in_all_indivs:
            ind = Q_df_sorted["SAMPLE_ID"].iloc[i]  # Get the index of the individual in Q_df_sorted
            
            if ind in Q_df_sorted_case_only["SAMPLE_ID"].values:
                cluster = Q_df_sorted_case_only.loc[Q_df_sorted_case_only["SAMPLE_ID"] == ind, "cluster"].values[0]  # Get the cluster
                cluster_color = cluster_colors.get(estimated_clusters_mapping_plt_names[cluster], "gray")  # Get color for the cluster (default to gray)
                ax2.axvline(x=i, color=cluster_color, linestyle='-', linewidth=1)  # Draw vertical lines representing PCA clusters
                # Add a handle for the legend

     # Prepare legend handles
    cluster_handles = [plt.Line2D([0], [0], color=color, lw=4, label=cluster) for cluster, color in case_colors_clusters_plt_names.items()]
    # Add the legend for the cluster colors
    ax2.legend(handles=cluster_handles, loc='lower left', fontsize=fontsize, bbox_to_anchor=(0, -1.65))

    x_coords.append(len(Q_df_sorted))
    mid_x_coords = [(x_coords[j]+x_coords[j+1])/2 for j in range(0, len(x_coords)) if j < len(x_coords)-1]  # compute new x ticks position to put label under each part in the middle

    ax.set_xlabel("Individuals", fontsize=fontsize, fontweight="bold")
    ax.set_ylabel("Ancestry proportion (%)", fontsize=fontsize, fontweight="bold")
    ax.set_yticklabels([0, 20, 40, 60, 80, 100], fontsize=fontsize)  # set y labels

    # Fix legend placement in the top right
    ax.legend(loc="upper right", fontsize=fontsize)

    # Add x legends     
    ax.set_xticks(mid_x_coords)  # add xticks only under dotted lines
    ax.set_xticklabels(custom_order, rotation=45, ha="right", fontsize=fontsize)  # set labels

    plt.subplots_adjust(hspace=0.10)  # Adjust space between the two subplots
    plt.savefig(os.path.join(output_dir, f"admixed_ancestry_k_{len(superpopulations)}_with_clusters_sortby{sort_by}.pdf"), format="pdf")
    plt.close()
    plt.clf()
    print("[Plot] Admixed ancestry plot with PCA clusters created")

def get_cross_validation_error(log_file: str):
    """Get cross-validation error from log file output of admixture software.
    
    Parameters:
    log_file: str - output admixture log file path

    Return: float - cross-validation score (if found in log_file) else return None
    """
    pattern = r'CV error \(K=(\d+)\): (\d+\.\d+)' # 2 groups to capture (between ())

    with open(log_file, 'r') as log_f:
        content = log_f.read()
    
    match = re.search(pattern, content)

    if match:
        return float(match.group(2)) # return text obtained with the first 
    
    print(f"[Cross-validation error] Cross-validation error was not found in {log_file}.")
    return None      

def plot_elbow_cross_validation_error(cv_errors: dict, output_dir: str, output_file: str):
    """
    Create plot of cross-validation errors for each k value (admixture K) and save the plot as pdf.
    
    Parameters:
    cv_errors: dict - dictionary with key = k and value = cv_error
    output_dir: str - output directory path
    output_file: str - output file path
    """

    plt.plot(cv_errors.keys(), cv_errors.values(), "gs-")
    plt.xlabel("Values of k")
    plt.ylabel("Cross-Validation Error")
    plt.title("Cross-Validation errors for different k values", fontsize=8)
    plt.savefig(os.path.join(output_dir, output_file), format="pdf")
    plt.close()
    plt.clf()

    print(f"[Plot] {os.path.join(output_dir, output_file)} plot created.")

def plot_admixture_proportions_by_pca_cluster(admixed_df: pd.DataFrame, labelled_cluster_file: str, output_dir: str):
    """
    For case only. Violin plots of distribution of each admixed ancestry proportion (Eur, Afr, Asia) among each cluster identified through PCA analysis during ancestry analysis.
    """

    def get_distribution_statistics_dict(admixed_prop: pd.DataFrame):
        """
        admixed_prop: admixture proportion for one PCA cluster
        """
        distributions_by_pop = {
            "European": None,
            "African": None,
            "East-Asian": None
        }

        for col in admixed_prop.columns:

            current_admixture = np.asarray(admixed_prop[col], dtype=np.float32)

            match col:
                case "Europe": col = "European"
                case "Africa": col = "African"
                case "East_Asia": col = "East-Asian"
                case _: continue

            distributions_by_pop[col] = {
                "mean": np.mean(current_admixture),
                "median": np.median(current_admixture),
                "min": np.min(current_admixture),
                "max": np.max(current_admixture),
                "25th_quantile": np.percentile(current_admixture, 25),
                "75th_quantile": np.percentile(current_admixture, 75),
                "std": np.std(current_admixture),
                "nb_more_or_equal_90%": len(current_admixture[current_admixture >= 0.9]),
                "nb_more_or_equal_80%": len(current_admixture[current_admixture >= 0.8]),
                "nb_more_or_equal_75%": len(current_admixture[current_admixture >= 0.75]),
                "nb_more_or_equal_50%": len(current_admixture[current_admixture >= 0.50]),
                "nb_more_or_equal_25%": len(current_admixture[current_admixture >= 0.25]),
                "nb_more_or_equal_10%": len(current_admixture[current_admixture >= 0.1]),
                "nb_more_or_equal_5%": len(current_admixture[current_admixture >= 0.05])
            }

        return distributions_by_pop
    
    labelled_clusters_df = pd.read_csv(labelled_cluster_file, sep=',', header="infer")
    labelled_clusters_df.columns = ["SAMPLE_ID", "PC1", "PC2", "FID", "cluster"]
    admixed_df = admixed_df[admixed_df["SUPERPOP"]=="CASE"]

    merged_df = pd.merge(admixed_df, labelled_clusters_df, on="SAMPLE_ID", how="inner")
    
    clusters = list(np.unique(merged_df["cluster"]))

    # for each cluster identified on PCA
    for cluster in clusters:
        admixed_prop = merged_df.loc[merged_df["cluster"]==cluster, ("Europe", "East_Asia", "Africa")]

        fig, ax = plt.subplots(figsize=(16, 10))
        # Plot the violin
        sns.violinplot(
            data=[np.asarray(admixed_prop["Europe"], dtype=np.float32), np.asarray(admixed_prop["Africa"], dtype=np.float32), np.asarray(admixed_prop["East_Asia"], dtype=np.float32)],
            palette="pastel",
            scale='count',
            ax=ax,
            zorder=1,
        )

        # Set x-axis labels
        ax.set_xticks([0, 1, 2])  # Set the positions of the ticks
        ax.set_xticklabels(["European", "African", "East-Asian"])  # Set the labels for the ticks
        # Create a gap above y=1 by adding a blank space in the plot area
        plt.ylim(0, 1.05)  # Set y-limits to create a gap above y=1

        # Customize the ticks for y-axis
        plt.yticks(np.arange(0, 1.2, 0.2))  # Y ticks from 0 to 1.2
        
        # Access the lines of the violin plot

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
        plt.savefig(os.path.join(output_dir, f"admixture_proportions_among_{estimated_clusters_mapping[cluster]}.pdf"), format="pdf", dpi=300)
        plt.clf()
        plt.close()

        # get csv of distributions
        distribution_stats = get_distribution_statistics_dict(admixed_prop)

        distribution_stats = pd.DataFrame(
            data = {
            col: [distribution_stats.get(ancestry).get(col) for ancestry in list(distribution_stats.keys())] for col in distribution_stats.get("European").keys()
            },
            index=list(distribution_stats.keys())
        )
        distribution_stats.to_csv(os.path.join(output_dir, f"admixture_proportions_among_{estimated_clusters_mapping[cluster]}.csv"), sep=';')
    
    return labelled_clusters_df

def plot_agg_ancestry_proportions(admixed_df: pd.DataFrame, output_dir: str, case_only=True):
    """
    Violin plot of ancestry proportions distribution for each considered superpopulation (here African, European, East Asian)
    
    Parameters:
    admixed_df: pd.DataFrame - Q_df with superpopulation labels in SUPERPOPULATION column names and SAMPLE_ID column;
    """

    # ncheck index and reset index to avoid having additional automatically created "index" column after resetting index
    admixed_df.reset_index(drop=False, inplace=True)
    if "index" in admixed_df.columns:
        admixed_df.drop("index", axis=1, inplace=True)
    admixed_df.set_index("SAMPLE_ID", inplace=True)

    admixed_df.rename({"East_Asia": "East Asia"}, axis=1, inplace=True)

    col_names = ["Europe", "East Asia", "Africa"]
    # color_mapping = {
    #     "Europe": "lightblue",
    #     "East Asia": "lightyellow",
    #     "Africa": "lightcoral",
    #     # "OFSEP-HD Europe": "blue",
    #     # "OFSEP-HD East_Asia": "yellow",
    #     # "OFSEP-HD Africa": "red",
    #     # "Control Europe": "lightblue",
    #     # "Control East Asia": "lightyellow",
    #     # "Control Africa": "lightcoral"
    # }

    # preparing data for ploting
    # if case_only:
    #     melted_df = admixed_df[admixed_df["SUPERPOP"] == "CASE"].melt(id_vars=["SUPERPOP"], value_vars=["Europe", "East Asia", "Africa"], var_name="Ancestry", value_name="Proportion")
    #     output_file = "case_only_violinplot_admixture.pdf"
    # else:
    #     melted_df = admixed_df[admixed_df["SUPERPOP"].isin(["CASE", "CTRL"])].melt(id_vars=["SUPERPOP"], value_vars=["Europe", "East Asia", "Africa"], var_name="Ancestry", value_name="Proportion")
    #     output_file = "case_ctrl_violinplot_admixture.pdf"

    # # ploting
    # sns.set_theme(style="white")
    # sns.set_style("whitegrid")
    # ax = sns.violinplot(x="SUPERPOP" if not case_only else "Ancestry", y="Proportion", data=melted_df, palette=color_mapping) NOT WORKING
    # ax.set_ylim((0,1))
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))

 
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # plt.legend(loc="upper right")
    # plt.ylabel("Ancestry proportion")
    # plt.savefig(os.path.join(output_dir, output_file), format="pdf")

    def get_violin_plot_data(labels, data):
            rows_list = []
            for label in labels:
                dict1 = {}
                dict1['label'] = label
                dict1['mean'] = np.mean(data[label])
                dict1['median'] = np.median(data[label])
                dict1['min'] = np.min(data[label])
                dict1['max'] = np.max(data[label])
                dict1['25th_quantile'] = np.percentile(data[label], 25)
                dict1['75th_quantile'] = np.percentile(data[label], 75)
                dict1[f"nb_more_or_equal_85%"] = len(data[data[label] >= 0.85])
                dict1[f"prop_more_or_equal_85%"] = len(data[data[label] >= 0.85]) / len(data)
                dict1[f"nb_more_or_equal_84.75%"] = len(data[data[label] >= 0.8475])
                dict1[f"prop_more_or_equal_84.75%"] = len(data[data[label] >= 0.8475]) / len(data)
                dict1[f"nb_more_or_equal_84.5%"] = len(data[data[label] >= 0.845])
                dict1[f"prop_more_or_equal_84.5%"] = len(data[data[label] >= 0.845]) / len(data)
                dict1[f"nb_more_or_equal_84.25%"] = len(data[data[label] >= 0.8425])
                dict1[f"prop_more_or_equal_84.25%"] = len(data[data[label] >= 0.8425]) / len(data)
                dict1[f"nb_more_or_equal_84%"] = len(data[data[label] >= 0.84])
                dict1[f"prop_more_or_equal_84%"] = len(data[data[label] >= 0.84]) / len(data)
                dict1[f"nb_more_or_equal_83.75%"] = len(data[data[label] >= 0.8375])
                dict1[f"prop_more_or_equal_83.75%"] = len(data[data[label] >= 0.8375]) / len(data)
                dict1[f"nb_more_or_equal_83.65%"] = len(data[data[label] >= 0.8365])
                dict1[f"prop_more_or_equal_83.65%"] = len(data[data[label] >= 0.8365]) / len(data)
                dict1[f"nb_more_or_equal_83.6%"] = len(data[data[label] >= 0.836])
                dict1[f"prop_more_or_equal_83.6%"] = len(data[data[label] >= 0.836]) / len(data)
                dict1[f"nb_more_or_equal_83.55%"] = len(data[data[label] >= 0.8355])
                dict1[f"prop_more_or_equal_83.55%"] = len(data[data[label] >= 0.8355]) / len(data)
                dict1[f"nb_more_or_equal_83.5%"] = len(data[data[label] >= 0.835])
                dict1[f"prop_more_or_equal_83.5%"] = len(data[data[label] >= 0.835]) / len(data)
                dict1[f"nb_more_or_equal_83.25%"] = len(data[data[label] >= 0.8325])
                dict1[f"prop_more_or_equal_83.25%"] = len(data[data[label] >= 0.8325]) / len(data)
                dict1[f"nb_more_or_equal_83%"] = len(data[data[label] >= 0.83])
                dict1[f"prop_more_or_equal_83%"] = len(data[data[label] >= 0.83]) / len(data)
                dict1[f"nb_more_or_equal_82%"] = len(data[data[label] >= 0.82])
                dict1[f"prop_more_or_equal_82%"] = len(data[data[label] >= 0.82]) / len(data)
                dict1[f"nb_more_or_equal_81%"] = len(data[data[label] >= 0.81])
                dict1[f"prop_more_or_equal_81%"] = len(data[data[label] >= 0.81]) / len(data)
                dict1[f"nb_more_or_equal_80%"] = len(data[data[label] >= 0.80])
                dict1[f"prop_more_or_equal_80%"] = len(data[data[label] >= 0.80]) / len(data)
                dict1[f"nb_more_or_equal_75%"] = len(data[data[label] >= 0.75])
                dict1[f"prop_more_or_equal_75%"] = len(data[data[label] >= 0.75]) / len(data)
                dict1[f"nb_more_or_equal_70%"] = len(data[data[label] >= 0.70])
                dict1[f"prop_more_or_equal_70%"] = len(data[data[label] >= 0.70]) / len(data)
                dict1[f"nb_more_or_equal_50%"] = len(data[data[label] >= 0.5])
                dict1[f"prop_more_or_equal_50%"] = len(data[data[label] >= 0.5]) / len(data)
                dict1[f"nb_more_or_equal_45%"] = len(data[data[label] >= 0.45])
                dict1[f"prop_more_or_equal_45%"] = len(data[data[label] >= 0.45]) / len(data)
                dict1[f"nb_more_or_equal_40%"] = len(data[data[label] >= 0.40])
                dict1[f"prop_more_or_equal_40%"] = len(data[data[label] >= 0.40]) / len(data)
                dict1[f"nb_more_or_equal_35%"] = len(data[data[label] >= 0.35])
                dict1[f"prop_more_or_equal_35%"] = len(data[data[label] >= 0.35]) / len(data)
                dict1[f"nb_more_or_equal_33%"] = len(data[data[label] >= 0.33])
                dict1[f"prop_more_or_equal_33%"] = len(data[data[label] >= 0.33]) / len(data)
                dict1[f"nb_more_or_equal_32%"] = len(data[data[label] >= 0.32])
                dict1[f"prop_more_or_equal_32%"] = len(data[data[label] >= 0.32]) / len(data)
                dict1[f"nb_more_or_equal_30%"] = len(data[data[label] >= 0.30])
                dict1[f"prop_more_or_equal_30%"] = len(data[data[label] >= 0.30]) / len(data)
                dict1[f"nb_more_or_equal_28%"] = len(data[data[label] >= 0.28])
                dict1[f"prop_more_or_equal_28%"] = len(data[data[label] >= 0.28]) / len(data)
                dict1[f"nb_more_or_equal_25%"] = len(data[data[label] >= 0.25])
                dict1[f"prop_more_or_equal_25%"] = len(data[data[label] >= 0.25]) / len(data)
                dict1[f"nb_more_or_equal_22%"] = len(data[data[label] >= 0.22])
                dict1[f"prop_more_or_equal_22%"] = len(data[data[label] >= 0.22]) / len(data)
                dict1[f"nb_more_or_equal_21.5%"] = len(data[data[label] >= 0.215])
                dict1[f"prop_more_or_equal_21.5%"] = len(data[data[label] >= 0.215]) / len(data)
                dict1[f"nb_more_or_equal_21%"] = len(data[data[label] >= 0.21])
                dict1[f"nb_more_or_equal_21%"] = len(data[data[label] >= 0.21])
                dict1[f"nb_more_or_equal_20.5%"] = len(data[data[label] >= 0.205])
                dict1[f"prop_more_or_equal_20.5%"] = len(data[data[label] >= 0.205]) / len(data)
                dict1[f"nb_more_or_equal_20%"] = len(data[data[label] >= 0.2])
                dict1[f"prop_more_or_equal_20%"] = len(data[data[label] >= 0.2]) / len(data)
                dict1[f"nb_more_or_equal_19.5%"] = len(data[data[label] >= 0.195])
                dict1[f"prop_more_or_equal_19.5%"] = len(data[data[label] >= 0.195]) / len(data)
                dict1[f"nb_more_or_equal_19.3%"] = len(data[data[label] >= 0.193])
                dict1[f"prop_more_or_equal_19.3%"] = len(data[data[label] >= 0.193]) / len(data)
                dict1[f"nb_more_or_equal_19.25%"] = len(data[data[label] >= 0.1925])
                dict1[f"prop_more_or_equal_19.25%"] = len(data[data[label] >= 0.1925]) / len(data)
                dict1[f"nb_more_or_equal_19.2%"] = len(data[data[label] >= 0.192])
                dict1[f"prop_more_or_equal_19.2%"] = len(data[data[label] >= 0.192]) / len(data)
                dict1[f"nb_more_or_equal_19%"] = len(data[data[label] >= 0.19])
                dict1[f"prop_more_or_equal_19%"] = len(data[data[label] >= 0.19]) / len(data)
                dict1[f"nb_more_or_equal_18%"] = len(data[data[label] >= 0.18])
                dict1[f"prop_more_or_equal_18%"] = len(data[data[label] >= 0.18]) / len(data)
                dict1[f"nb_more_or_equal_13%"] = len(data[data[label] >= 0.13])
                dict1[f"prop_more_or_equal_13%"] = len(data[data[label] >= 0.13]) / len(data)
                dict1[f"nb_more_or_equal_12%"] = len(data[data[label] >= 0.12])
                dict1[f"prop_more_or_equal_12%"] = len(data[data[label] >= 0.12]) / len(data)
                dict1[f"prop_more_or_equal_11.85%"] = len(data[data[label] >= 0.1185]) / len(data)
                dict1[f"nb_more_or_equal_11.85%"] = len(data[data[label] >= 0.1185])
                dict1[f"prop_more_or_equal_11.8%"] = len(data[data[label] >= 0.118]) / len(data)
                dict1[f"nb_more_or_equal_11.8%"] = len(data[data[label] >= 0.118])
                dict1[f"prop_more_or_equal_11.75%"] = len(data[data[label] >= 0.1175]) / len(data)
                dict1[f"nb_more_or_equal_11.75%"] = len(data[data[label] >= 0.1175])
                dict1[f"prop_more_or_equal_11.7%"] = len(data[data[label] >= 0.117]) / len(data)
                dict1[f"nb_more_or_equal_11.7%"] = len(data[data[label] >= 0.117])
                dict1[f"prop_more_or_equal_11.5%"] = len(data[data[label] >= 0.115]) / len(data)
                dict1[f"nb_more_or_equal_11.5%"] = len(data[data[label] >= 0.115])
                dict1[f"prop_more_or_equal_11.5%"] = len(data[data[label] >= 0.115]) / len(data)
                dict1[f"nb_more_or_equal_11.5%"] = len(data[data[label] >= 0.115])
                dict1[f"nb_more_or_equal_11.25%"] = len(data[data[label] >= 0.1125])
                dict1[f"prop_more_or_equal_11.25%"] = len(data[data[label] >= 0.1125]) / len(data)

                dict1[f"nb_more_or_equal_11%"] = len(data[data[label] >= 0.11])
                dict1[f"prop_more_or_equal_11%"] = len(data[data[label] >= 0.11]) / len(data)
                dict1[f"nb_more_or_equal_10%"] = len(data[data[label] >= 0.10])
                dict1[f"prop_more_or_equal_10%"] = len(data[data[label] >= 0.10]) / len(data)
                dict1[f"nb_more_or_equal_5%"] = len(data[data[label] >= 0.05])
                dict1[f"prop_more_or_equal_5%"] = len(data[data[label] >= 0.05]) / len(data)
      
                rows_list.append(dict1)
                
            return pd.DataFrame(rows_list)


    # Merge the KDE data from both CASE and CTRL if not case_only
    if not case_only:
        violin_data_case = get_violin_plot_data(col_names, admixed_df[admixed_df["SUPERPOP"]=="OFSEP_HD"])
        violin_data_ctrl = get_violin_plot_data(col_names, admixed_df[admixed_df["SUPERPOP"]=="CTRL"])
        violin_data = pd.concat([violin_data_case, violin_data_ctrl], ignore_index=True)
    else:
        violin_data = get_violin_plot_data(col_names, admixed_df[admixed_df["SUPERPOP"]=="OFSEP_HD"])

    # plt.close()
    # plt.clf()

    violin_data.to_csv(os.path.join(output_dir, "violinplot_values.csv"), sep=",", index=False)

    return violin_data
