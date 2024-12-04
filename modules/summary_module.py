import os
import pandas as pd
from shlex import split as ssplit
from subprocess import PIPE, Popen
from functools import reduce
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def read_parameters_file(parameters_file: str):
    """Read parameters file and return a dictionary with keys as parameter name and values as filepath or parameter value"""

    with open(parameters_file) as f:
        return {field.split(" ")[0]:field.split(" ")[1].rsplit('\n')[0] for field in f.readlines() if not field.startswith('#') and not field.startswith('\n') }

def get_all_individuals(output_dir: str, initial_prefix: str):
    """Get all individuals from original file"""
    output_file = os.path.join(output_dir, "initial_list_of_individuals.txt")
    with open(output_file, 'w') as f:   
        awk_c = ["awk", '{print $2}', initial_prefix]
        awk_p = Popen(awk_c, text=True, stdout=f, stdin=PIPE, stderr=PIPE)
        awk_p.communicate()
    
    all_indiv_df = pd.read_csv(output_file, header=None)
    all_indiv_df.columns = ["IID"]
    all_indiv_df["IID"] = all_indiv_df["IID"].str.split('.').str[0]
    
    all_indiv_df.set_index("IID", inplace=True)

    return output_file, all_indiv_df

def get_fstats(sexcheck_file: str):
    """"""

    fstat = pd.read_csv(sexcheck_file, delim_whitespace=True, header="infer")
    fstat = fstat.loc[:, ("IID", "F")]
    fstat.set_index("IID", inplace=True)

    return fstat

def get_het_rate(het_file: str):
    
    het = pd.read_csv(het_file,delim_whitespace=True, header="infer") 
    het['HET_RATE'] = (het["N(NM)"] - het["O(HOM)"])/het["N(NM)"]
    het = het.loc[:, ("IID", "HET_RATE")]
    het.set_index("IID", inplace=True)

    return het

def get_relatedness(relatedness_file: str, ind_to_keep_frq_analysis: str):
    """"""
    ind_to_keep = pd.read_csv(ind_to_keep_frq_analysis, header=None).to_numpy()
    relatedness = pd.read_csv(relatedness_file, delim_whitespace=True, header="infer")
    relatedness = relatedness.loc[:, ("IID1", "IID2", "PI_HAT", "RT")]

    relatedness["IID1"] = relatedness["IID1"].str.split('.').str[0]
    relatedness["IID2"] = relatedness["IID2"].str.split('.').str[0]
    #relatedness.set_index(["IID1", "IID2"], inplace=True)
    
    relatedness["selection_freq_allele"] = relatedness.apply(lambda x: "1" if x["IID1"] in ind_to_keep else ("2" if x["IID2"] in ind_to_keep else "AllRemoved"), axis=1)

    return relatedness 

def get_eigenvectors(smartpca_file: str):
    """Get eigenvectors for Case"""

    eigen = pd.read_csv(smartpca_file, delim_whitespace=True, header="infer")
    eigen.reset_index(inplace=True, drop=False)
    pcs = [f"PC{i}" for i in range(1, len(eigen.columns)-1)]

    eigen.columns = ["IID", *pcs, "FID"]
    eigen = eigen.loc[eigen["FID"] == "Case", :]
    eigen.drop("FID", axis=1, inplace=True)
    eigen.set_index("IID", inplace=True)

    return eigen

def get_admixture(admixture_file: str):
    """Get admixture for case"""

    admixture = pd.read_csv(admixture_file, header="infer", delim_whitespace=True)
    admixture = admixture.loc[admixture["SUPERPOP"]=="CASE"]
    admixture.drop("SUPERPOP", inplace=True, axis=1)
    admixture.drop("SUBPOP", inplace=True, axis=1)
    admixture.rename(columns={"SAMPLE_ID": "IID"}, inplace=True)
    admixture.set_index("IID", inplace=True)

    return admixture

def remove_dup(plink_prefix: str):
    """"""
    output_file_prefix = plink_prefix+".2"
    plink_c = ["plink2", "--bfile", plink_prefix, "--rm-dup", "force-first", "--make-bed", "--out", output_file_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    plink_p.communicate()

    mv_c = ["mv", output_file_prefix, plink_prefix]
    mv_p = Popen(mv_c, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    mv_p.communicate()

    return output_file_prefix

def get_snp_matrix(imputed_dataset_prefix: str, variants_allele_or_file: str, output_dir: str):
    """
    Encode genotypes as number of occurence of minor allele variant within OFSEP-HD dataset. (allele A1)
    """

    # create file with only 1 columns listing the >200 MS snps in HG38 code
    snp_list_file = os.path.join(output_dir, "233_MSsnps_listHG38_snp.txt")
    with open(snp_list_file, 'w') as f:
        awk_c = ["awk" , "-F", ";", "{if(NR>1){print $1}}", variants_allele_or_file]
        awk_p = Popen(awk_c, stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate()

    output_file = os.path.join(output_dir, "233_MSsnps")

    # Extract SNPs
    plink_c1 = ["plink", "--bfile", imputed_dataset_prefix, "--extract", snp_list_file, "--allow-no-sex", "--make-bed", "--out", output_file]
    plink_p1 = Popen(plink_c1, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    plink_p1.communicate()

    # Recode to tab-delimited format
    recoded_file = os.path.join(output_dir, "233_MSsnps_recoded")
    plink_c2 = ["plink", "--bfile", output_file, "--recode", "--out", recoded_file]
    plink_p2 = Popen(plink_c2, stdout=PIPE, stderr=PIPE, stdin=PIPE)
    plink_p2.communicate()

    snp_matrix_file = os.path.join(output_dir, "snp_matrix.csv")

    # A1 minor allele frequency
    # A2 major allele frequency
    variants_allele_or_df = pd.read_csv(variants_allele_or_file, header="infer", sep=';')
    matching_allele = {
        "A":"T",
        "T":"A",
        "G":"C",
        "C":"G"
    }

    with open(recoded_file+".ped", 'r') as recodedin, open(output_file+".bim", 'r') as bimin, open(snp_matrix_file, 'w') as fout:
        
        variant_names = [line_variant.split("\t")[1] for line_variant in bimin.readlines()] 

        for line in recodedin:
            individual_id = line.strip().split(" ")[1]
            fields = line.strip().split()[6:]  # Extract genotype columns

            genotype = [ list([fields[j], fields[j+1]]) for j in range(0, len(fields)-1, 2)]
            
            recoded_genotype = []
            for variant_name, geno in zip(variant_names, genotype):

                # define which allele is the risk allele, the OR is calculated on A1
                if float(variants_allele_or_df.loc[variants_allele_or_df["HG38_variant_id"]==variant_name, "OR"].iloc[0]) > 1:
                    risk_allele = "A1"
                else:
                    risk_allele = "A2"
                # count the number of risk allele in the genotype for current locus
                code = int(sum(allele == variants_allele_or_df[variants_allele_or_df["HG38_variant_id"]==variant_name][risk_allele].iloc[0] for allele in geno))
                if code == 0:
                    code = int(sum(allele == matching_allele[variants_allele_or_df[variants_allele_or_df["HG38_variant_id"]==variant_name][risk_allele].iloc[0]] for allele in geno))  
                recoded_genotype.append(code)
            
            recoded_genotype = ';'.join([str(int(elm)) for elm in recoded_genotype])

            fout.write(f"{str(individual_id)};{recoded_genotype}\n")

    snp_matrix = pd.read_csv(snp_matrix_file, header=None, sep=';')
    snp_matrix.columns = ["IID", *variant_names]

    snp_matrix.loc[:, variant_names] = snp_matrix.loc[:, variant_names].astype(int)
    snp_matrix.to_csv(snp_matrix_file, header=True, index=False, sep=";") # use ; to not interefer with alleles of chr_posref,alt notation to be splitted.
    snp_matrix.set_index("IID", inplace=True)

    return snp_matrix_file, snp_matrix, variant_names

def merge_data(df_to_merge: list):
    """
    Merge data on index.
    """
    new_df_to_merge = []
    for df in df_to_merge:
        df.reset_index(inplace=True, drop=False)
        df["IID"] = df["IID"].str.split('.').str[0]
        df.set_index("IID", inplace=True)
        new_df_to_merge.append(df)

    return reduce(lambda left, right: pd.merge(left, right, right_index=True, left_index=True, how="outer") if right is not None else left, new_df_to_merge)

def get_list_indiv_kept(file: str):
    """Return the list of individuals kept after QC as a python list based on file.
    File 2 columns: FID IID"""

    with open(file, 'r') as f:
        return [line.split(" ")[1].rstrip('\n').split('.')[0] for line in f if line.split(" ")[0].rstrip('\n').lower() == "case"]


def collide_rows(data: pd.DataFrame):
    """
    Collide data from rows associated to the same patient.
    """
    data["short_id"] = data["IID"].apply(lambda x: x.split('_')[0])

    data.reset_index(inplace=True)
    groupby_data = data.groupby(by="short_id")
    data.set_index("IID", inplace=True)

    for _, group in groupby_data:
        if len(group) > 1 and (group["excludedQC"] == 0).any():
            correct_row = pd.Series(group[group["excludedQC"]==0].iloc[0])
            other_rows = group[group["excludedQC"]==1]
            for i in range(0, len(other_rows)):
                correct_row = correct_row.fillna(other_rows.iloc[i])

            sample_id = correct_row.name
            data.loc[sample_id, :] = correct_row

    return data   

def merge_ancestry_cluster_and_description_table(description_df: pd.DataFrame, ancestry_clusters_file: str):
    """
    Merge all descriptions with ancestry cluster determined in main2_ancestry_analysis
    """
    ancestry_clusters_df = pd.read_csv(ancestry_clusters_file, header="infer", sep=",")
    ancestry_clusters_df.columns = ["IID", "PC1", "PC2", "FID", "cluster"]
    ancestry_clusters_df = ancestry_clusters_df.loc[:, ["IID", "cluster"]]
    merged_df = description_df.merge(ancestry_clusters_df, how="outer", on="IID")
    merged_df.set_index("IID", inplace=True)

    return merged_df

def get_hla_variant_list(hla_variants_file: str):
    """"""
    df = pd.read_csv(hla_variants_file, header=None)
    return df[0].tolist()

def msgb(x, ms_snps_df: pd.DataFrame, snps_not_in_df: list[str], use_freq_as_dose_when_absent=True):
    """
    x: pd.Series (one row of df)
    """
    msgb = 0
    for snp, n_risk_alleles in x.items():
        o_r = ms_snps_df.loc[ms_snps_df["variantID"]==snp, "OR"].astype(float).squeeze()
        
        if o_r < 1:
            o_r = 1/o_r # compute risk allele OR
        msgb += ( n_risk_alleles * np.log(o_r) )
    
    # if dose imputation of absent snp by frequency in global population
    if use_freq_as_dose_when_absent:
        for snp in snps_not_in_df:
            o_r = ms_snps_df.loc[ms_snps_df["variantID"]==snp, "OR"].astype(float).squeeze()

            if o_r < 1:
                o_r = 1/o_r # compute risk allele OR
            msgb += ms_snps_df.loc[ms_snps_df["variantID"]==snp, "Freq"].astype(float).squeeze() * np.log(o_r)
    
    return msgb

def risk_alleles_count(x, ms_snps_df: pd.DataFrame, snps_not_in_df: list[str], use_freq_as_dose_when_absent=True):
    """
    x: pd.Series (one row of df)
    """

    count = x.sum()

    if use_freq_as_dose_when_absent:
        count_suppl = ms_snps_df.loc[ms_snps_df["variantID"].isin(snps_not_in_df), "Freq"].astype(float).sum()
        count += count_suppl
    
    return count

def compute_prs_by_individual(variant_or_freq_file: str, merged_table: pd.DataFrame, indivs_to_keep_for_frq_analysis: list[str], hla_variants_to_remove: list[str], sep=";", use_freq_as_dose_when_absent=True):
    """
    Compute PRS only for non-MHC variants.
    merged_table should have IID (patient IDs) as index.
    
    """
    ms_snps_df = pd.read_csv(variant_or_freq_file, sep=sep, header="infer")
    merged_table.reset_index(inplace=True, drop=False)

    # Compute MSGB and risk allele count
    snps_in_df = merged_table.columns[merged_table.columns.isin(ms_snps_df["variantID"])]
    snps_in_df_to_use = snps_in_df[~snps_in_df.isin(hla_variants_to_remove)].to_list()


    snps_not_in_df = ms_snps_df.loc[~ms_snps_df["variantID"].isin(snps_in_df_to_use), "variantID"]
    snps_not_in_df_to_use = snps_not_in_df[~snps_not_in_df.isin(hla_variants_to_remove)].to_list()
    
    # MSGB
    msgb_col = "MSGB"
    if use_freq_as_dose_when_absent:
        msgb_col = "MSGB_freq_as_dose_when_absent"
    merged_table.loc[merged_table["IID"].isin(indivs_to_keep_for_frq_analysis), msgb_col] = merged_table.loc[merged_table["IID"].isin(indivs_to_keep_for_frq_analysis), snps_in_df_to_use].apply(msgb, ms_snps_df=ms_snps_df, snps_not_in_df=snps_not_in_df_to_use, use_freq_as_dose_when_absent=use_freq_as_dose_when_absent, axis=1) # apply to each row

    # Risk allele count
    risk_all_count_col = "risk_alleles_count"
    if use_freq_as_dose_when_absent:
        risk_all_count_col = "risk_alleles_count_freq_when_absent"
    merged_table.loc[merged_table["IID"].isin(indivs_to_keep_for_frq_analysis), risk_all_count_col] = merged_table.loc[merged_table["IID"].isin(indivs_to_keep_for_frq_analysis), snps_in_df_to_use].apply(risk_alleles_count, ms_snps_df=ms_snps_df, snps_not_in_df=snps_not_in_df_to_use, use_freq_as_dose_when_absent=use_freq_as_dose_when_absent, axis=1) # apply to each row
    
    merged_table.set_index("IID", inplace=True)

    return merged_table, msgb_col, risk_all_count_col

def correlation_graph(melted_df: pd.DataFrame, prs_cols_to_compare: list[str] | tuple[str], output_dir: str):
    """
    Plot correlation graph  etween 2 PRS scores by fitting a linear regression line on the plot. Save plot in output_dir.
    prs_cols_to_compare: list of 2 prs computation to compare
    """
    df_to_plot = melted_df[ (melted_df["PRS"]==prs_cols_to_compare[0]) | (melted_df["PRS"]==prs_cols_to_compare[1]) ]
    
    # Extract the data points for the two PRS scores
    x_data = df_to_plot[df_to_plot["PRS"] == prs_cols_to_compare[0]]["Score"]
    y_data = df_to_plot[df_to_plot["PRS"] == prs_cols_to_compare[1]]["Score"]

    # Combine x_data and y_data into a new DataFrame for plotting
    df_to_plot = pd.DataFrame({prs_cols_to_compare[0]: x_data.values, prs_cols_to_compare[1]: y_data.values})

    r, _ = stats.pearsonr(df_to_plot[prs_cols_to_compare[0]], df_to_plot[prs_cols_to_compare[1]])
    r2 = r**2
    
    joint_plot = sns.jointplot(x=prs_cols_to_compare[0], y=prs_cols_to_compare[1], data=df_to_plot, kind="scatter", palette="Set2")
    # Plot regression line
    sns.regplot(x=prs_cols_to_compare[0], y=prs_cols_to_compare[1], data=df_to_plot, ax=joint_plot.ax_joint, scatter=False, color="red")

    # Add R² annotation
    joint_plot.ax_joint.annotate(f'R² = {r2:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='center', fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
    
    # rename axes removing '_' and adding maj to each word of axis name
    joint_plot.set_axis_labels(f"{' '.join([part[0].upper()+part[1:] if i==0 else part for i, part in enumerate(prs_cols_to_compare[0].split('_'))])}-like", f"Sum of {' '.join([part[0].upper()+part[1:] if i==0 else part for i, part in enumerate(prs_cols_to_compare[1].split('_'))])}")
    joint_plot.ax_joint.grid(axis='y')
    joint_plot.ax_joint.set_axisbelow(True)
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(output_dir, f"correlation_plot_between_{prs_cols_to_compare[0]}_and_{prs_cols_to_compare[1]}.pdf"), format="pdf", dpi=300)
    plt.clf()
    plt.close()

def plot_prs(merged_table: pd.DataFrame, prs_cols_to_plot: list[str], indivs_to_keep_for_freq_analysis: list[str], output_dir: str, scaling=1):
    """
    multiply by scaler [0:1] prs_col_to_plot[1] and prs_col_to_plot[3]
    """

    # reset index to create melted df
    merged_table.reset_index(inplace=True, drop=False)
    merged_table_to_plt = merged_table.loc[merged_table["IID"].isin(indivs_to_keep_for_freq_analysis), prs_cols_to_plot]
    
    # divide by 10 risk allele count to have same  scale on plots
    merged_table_to_plt[["Risk Alleles Count", "Risk Alleles Count Frequency as dose when missing"]] = merged_table_to_plt.loc[:,("Risk Alleles Count","Risk Alleles Count Frequency as dose when missing")].apply(lambda x: x / 10)

    merged_table_to_plt[prs_cols_to_plot[1]] = merged_table_to_plt[prs_cols_to_plot[1]]*scaling
    merged_table_to_plt[prs_cols_to_plot[3]] = merged_table_to_plt[prs_cols_to_plot[3]]*scaling
    melted_df = merged_table_to_plt.melt(var_name="PRS", value_name="Score")
    merged_table.set_index("IID", inplace=True) # reset back index on original df

    fig, ax = plt.subplots(figsize=(12,8))

    # Plot the violin
    sns.violinplot(
        y="Score", 
        x="PRS", 
        data=melted_df,
        palette="pastel",
        scale='count',
        # inner="point",
        ax=ax,
        zorder=1
    )
    
    # Overlay the boxplot
    sns.boxplot(y="Score", x="PRS", data=melted_df,width=0.3, color="white", ax=ax, showfliers=False, showcaps=True, boxprops={'zorder': 3}, whiskerprops={'zorder': 3}, medianprops={'zorder': 4, "color": "purple"}, meanprops={"zorder": 4, "color": "red"}, zorder=2, showmeans=True, showbox=True, meanline=True) # Outliers should be above boxplot

    # Access the matplotlib objects to change inner points colors
    #sns.swarmplot(x="PRS", y="Score", data=melted_df, ax=ax, color='black', alpha=0.5, marker="o", size=1.4, zorder=3)

    # Change axis labels, ticks and title
    ax.set_xticks(np.arange(0, len(prs_cols_to_plot), 1), prs_cols_to_plot)
    ax.set_xticklabels([" ".join([part[0].upper()+part[1:] if i==0 else part for i, part in enumerate(prs_col.split("_"))]) for prs_col in prs_cols_to_plot], rotation=45, ha="right") # rename x axis names removing '_' and adding maj to first letter
    ax.set_ylabel("Score")

    # Add horizontal grid
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "PRS_scores.pdf"), format="pdf", dpi=300)
    plt.clf()
    plt.close()

    # keep distribution statistics and write to a file
    statistics = melted_df.groupby(['PRS']).agg(
        mean=('Score', 'mean'),
        std=('Score', 'std'),
        median=('Score', 'median'),
        min=('Score', 'min'),
        max=('Score', 'max'),
        q25=('Score', lambda x: x.quantile(0.25)),
        q75=('Score', lambda x: x.quantile(0.75))
    ).reset_index()

    statistics.to_csv(os.path.join(output_dir, "statistics_prs_score_global_population.csv"), sep=";")

    correlation_graph(melted_df, ["MSGB", "Risk Alleles Count"], output_dir)
    correlation_graph(melted_df, ["MSGB Frequency as dose when missing", "Risk Alleles Count Frequency as dose when missing"], output_dir)

    return statistics

def plot_prs_by_ancestry(merged_table: pd.DataFrame, prs_cols_to_plot: list[str], indivs_to_keep_for_freq_analysis: list[str], estimated_cluster_mapping: dict, colors_mapping: dict, output_dir: str, scaling=1, merge_asian=True):
    """
    Plot PRS scores previously computed for each ancestral background
    Scale prs_col_to_plot 1 and 3 (alleles count)
    """

    merged_table.reset_index(inplace=True, drop=False)
    estimated_cluster_mapping = estimated_cluster_mapping.copy()

    if merge_asian:
        merged_table.loc[merged_table["cluster"]==3, "cluster"] = 5
        estimated_cluster_mapping[5] = "CloserToAsianReferenceClusters"

    melted_dict = {cluster:None for cluster in merged_table.loc[merged_table["IID"].isin(indivs_to_keep_for_freq_analysis), "cluster"].unique()}

    for cluster in melted_dict.keys():
        melted_dict[cluster] = merged_table.loc[ ( merged_table["IID"].isin(indivs_to_keep_for_freq_analysis)) & (merged_table["cluster"]==cluster), prs_cols_to_plot]
        melted_dict[cluster].loc[:, ("Risk Alleles Count","Risk Alleles Count Frequency as dose when missing")] = melted_dict[cluster].loc[:,("Risk Alleles Count","Risk Alleles Count Frequency as dose when missing")].apply(lambda x: x / 10) # scale SRAC method
        melted_dict[cluster] = melted_dict[cluster].melt(var_name="PRS", value_name="Score")
    merged_table.set_index("IID", inplace=True)

    for cluster in melted_dict.keys():
        fig, ax = plt.subplots(figsize=(12,8))
        melted_dict[cluster].loc[(melted_dict[cluster]["PRS"] == prs_cols_to_plot[1]) | (melted_dict[cluster]["PRS"] == prs_cols_to_plot[3]), "Score"] = melted_dict[cluster].loc[(melted_dict[cluster]["PRS"] == prs_cols_to_plot[1]) | (melted_dict[cluster]["PRS"] == prs_cols_to_plot[3]), "Score"]*scaling
        # Plot the violin
        sns.violinplot(
            y="Score", 
            x="PRS", 
            data=melted_dict[cluster],
            palette=colors_mapping[estimated_cluster_mapping[cluster]],
            scale='count',
            # inner="point",
            zorder=1,
            ax=ax
        )

        sns.boxplot(y="Score", x="PRS", data=melted_dict[cluster], width=0.3, color="white", ax=ax, showfliers=False, showcaps=True, boxprops={'zorder': 3}, whiskerprops={'zorder': 3}, medianprops={'zorder': 4, "color": "purple"}, meanprops={"zorder": 4, "color": "red"}, zorder=2, showmeans=True, showbox=True, meanline=True)  

        #sns.swarmplot(x="PRS", y="Score", data=melted_dict[cluster], ax=ax, color='black', alpha=0.6, marker="o", edgecolor='black', size=1.4, zorder=3)

        # Change axis labels, ticks and title
        ax.set_xticks(np.arange(0, len(prs_cols_to_plot), 1), [" ".join([part[0].upper()+part[1:] if i==0 else part for i, part in enumerate(prs_col.split("_"))]) for prs_col in prs_cols_to_plot])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel("Score")

        # Add horizontal grid
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        plt.title(f"PRS scores for {estimated_cluster_mapping[cluster]}")
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"PRS_scores_for_{estimated_cluster_mapping[cluster]}.pdf"), format="pdf", dpi=300)
        plt.clf()
        plt.close()

        # keep distribution statistics and write to a file
        statistics = melted_dict[cluster].groupby(['PRS']).agg(
            mean=('Score', 'mean'),
            std=('Score', 'std'),
            median=('Score', 'median'),
            min=('Score', 'min'),
            max=('Score', 'max'),
            q25=('Score', lambda x: x.quantile(0.25)),
            q75=('Score', lambda x: x.quantile(0.75))
        ).reset_index()

        statistics.to_csv(os.path.join(output_dir, f"statistics_prs_score_{estimated_cluster_mapping[cluster]}.csv"), sep=";")

    return statistics