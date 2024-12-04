#!/usr/bin/env python3

import sys
import os
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    """"""

    try:
        ms_hla_file = sys.argv[1]
        haplo_file = sys.argv[2]
        n_most_frequent = int(sys.argv[3])
        output_dir = sys.argv[4]
    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: python3 haplotypes_description.py <ms_hla_list_file> whaplotypes_imputed_file> <output_dir>\n")
        exit(1)

    # 1 - get list of ms HLA alleles
    with open(ms_hla_file, 'r') as f:
        ms_hla_list = [line.rstrip('\n').split('-')[1]for line in f]
    
    # 2 - get df of imputed haplotypes
    haplo_df = pd.read_csv(haplo_file, sep=',', header="infer")
    total_indiv_with_haplo = len(haplo_df)

    # 3 - get list of haplotypes with at least 1 ms hla allele
    filtered_haplotypes = haplo_df[
        haplo_df.apply(
            lambda row: any(allele in [a.rstrip('g') for a in row['Allele'].split("~")] for allele in ms_hla_list), # remove g for each allele because Haplomat format adds g at the end of each allele
            axis=1)
    ] 
    
    # 4 - get most frequent haplotypes (n most frequent) and count their occurences
    most_frequent_haplo = filtered_haplotypes.loc[:, ("Allele", "frequency")].sort_values(by="frequency", ascending=False, axis=0).iloc[0:n_most_frequent, :]
    counts_most_frq_haplo = {k:0 for k in most_frequent_haplo["Allele"]}
    for value in haplo_df["Allele"]:
        if value in most_frequent_haplo["Allele"].values:
            counts_most_frq_haplo[value] = counts_most_frq_haplo.get(value, 0) + 1

    # 5 - plot most frequent haplotypes on barplots
    categories = list(most_frequent_haplo["Allele"].to_numpy())
        
    # create bar plots
    fig, ax = plt.subplots(figsize=(18, 14))
    bar_plot = sns.barplot(x=categories, y=most_frequent_haplo["frequency"].to_numpy()*100, ax=ax, palette="muted")
  
    sorted_count = [counts_most_frq_haplo[haplo] for haplo in most_frequent_haplo["Allele"].values]
    for j, (n, prop) in enumerate(zip(sorted_count, most_frequent_haplo["frequency"].to_numpy())):
        # plt.text(j, prop+upper_error[j]+0.025, f"{n} ({prop*100:.2f}%)", ha="center", va="bottom")
        plt.text(j, prop*100+0.003, f"{prop*100:.2f}%", ha="center", va="bottom")

    plt.text(0.95, 0.95, f'n={total_indiv_with_haplo}', 
         horizontalalignment='right', verticalalignment='top', fontsize=16,
         transform=plt.gca().transAxes)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.ylabel("Proportion (%)")
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"most_frequent_haplotypes_ofsep_hd_barplots.pdf"), format="pdf", dpi=300)
    plt.clf()
    plt.close()

if __name__ == "__main__":
    main()