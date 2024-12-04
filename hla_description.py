#!/usr/bin/env python3

# Script to get the frequencies for each imputed hla allele
# plot the most represented HLA allele for each HLA gene available in hla_input_dir

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    """"""

    try:
        hla_input_dir = sys.argv[1] # directory with one file for each hla gene imputation
        output_dir = sys.argv[2]
    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: python3 hla_description.py <hla_input_dir> <hla_output_dir>\n")
        exit(1)

    os.makedirs(output_dir, exist_ok=True)
    hla_files = [ os.path.join(hla_input_dir, file) for file in os.listdir(hla_input_dir) if file.endswith(".csv") ]    
    
    hla = []
    for file in hla_files:
        # Read the HLA imputation file into a DataFrame
        df = pd.read_csv(file, sep=';')
        n_individuals = len(df)

        columns = [col for col in df.columns]
        # Split the HLA_A_1 and HLA_A_2 columns to extract individual alleles
        allele_chr1 = df[columns[1]]
        allele_chr2 = df[columns[2]]

        # Concatenate the alleles from both columns
        alleles = pd.concat([allele_chr1, allele_chr2])

        # Count the frequency of each allele
        allele_counts = alleles.value_counts()
        allele_counts = allele_counts.to_frame()
        allele_counts.reset_index(inplace=True)
        allele_counts.columns = ["HLA_allele", "Count"]
        hla_gene = "_".join(columns[1].split('_')[0:2])
        allele_counts["HLA_allele"] = allele_counts["HLA_allele"].apply(lambda x: f"{hla_gene}*{x}")
        allele_counts["Frequency(%)"] = allele_counts["Count"].apply(lambda x: x/n_individuals*100) # compute frequency
        allele_counts.sort_values(by="HLA_allele", inplace=True)
        allele_counts.set_index("HLA_allele", inplace=True)
        allele_counts.to_csv(os.path.join(output_dir, f"{hla_gene}_frequencies.csv"), sep=',')
        hla.append(allele_counts)

    # plotting most represented HLA alleles
    most_represented_hla = pd.DataFrame()
    for df in hla:
        most_represented = df.loc[df["Frequency(%)"].idxmax(), :] 
        most_represented = most_represented.to_frame().T
        most_represented = most_represented.loc[:, "Frequency(%)"]
        most_represented_hla = pd.concat([most_represented_hla, most_represented], ignore_index=False)

    most_represented_hla.columns = ["Frequency(%)"]
    most_represented_hla.reset_index(inplace=True)
    most_represented_hla.columns = ["HLA_allele", "Frequency(%)"]
    most_represented_hla.sort_values(by="HLA_allele", inplace=True)
    most_represented_hla.set_index("HLA_allele", inplace=True)
    
    # Plotting barplots
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    fig, ax = plt.subplots()
    bars = ax.bar(most_represented_hla.index, most_represented_hla['Frequency(%)'], color=colors)

    # Adding labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 0.1), 
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adding labels under each bar
    plt.xlabel('HLA Allele')

    # Adding title and ylabel
    plt.title('Frequency of HLA Alleles')
    plt.ylabel('Frequency (%)')

    # Show plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"HLA_histogram_most_represented_alleles.pdf"), format="pdf")



if __name__ == "__main__":
    main()