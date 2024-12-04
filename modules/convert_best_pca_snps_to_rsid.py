#!/usr/bin/env python3

import sys
import pandas as pd
import requests

## TOOL SCRIPT to convert chr_pos,alt,ref to RSIDs

# Function to fetch RSID from Ensembl for a given chromosome and position
def fetch_rsid_from_ensembl(chr, pos):
    url = f"https://rest.ensembl.org/overlap/region/human/{chr}:{pos}-{pos}?feature=variation"
    headers = {"Content-Type": "application/json"}
    
    response = requests.get(url, headers=headers)
    
    print("Response status: ", response.status_code)
    
    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            for variant in data:
                if 'id' in variant:
                    return variant['id']  # Return RSID (if available)
        else:
            print(f"No RSID found for {chr}:{pos}")
            return None
    else:
        print(f"Error fetching RSID for {chr}:{pos}: {response.status_code} - {response.text}")
        return None

def main():
    """"""
    try:
        hg38_list_file = sys.argv[1] # should contain 6 columns delimited by ; (pc;snp;chr;pos;eigenval;explained_variance)
    except IndexError:
        sys.stderr.write("[SyntaxError] Usage: python3 convert_best_pca_snps_to_rsid.py <hg38_snp_list - separated by ;>\n")
        exit(1)

    hg38_df = pd.read_csv(hg38_list_file, sep=";")
    hg38_snp_list = hg38_df.loc[:, ("snp","chr", "pos")]
    hg38_snp_list["alt"] = hg38_snp_list["snp"].apply(lambda x: x.split(',')[1].strip()) # strip to remove added spaces
    hg38_snp_list["ref"] = hg38_snp_list["snp"].apply(lambda x: x.split(',')[2].strip()) # strip to remove added spaces

    for index, row in hg38_snp_list.iterrows():
        print(row)
        chr, pos, alt, ref = row["chr"], row["pos"], row["alt"], row["ref"]
        rsid = fetch_rsid_from_ensembl(chr, pos)

        # if rsid is None: # revert alt ref alleles if didn't work
        #     rsid = fetch_rsid_from_ensembl(chr, pos)

        if rsid is None:
            continue

        print("rsid: ", rsid)

        # update output dataframe
        hg38_df.loc[hg38_df["snp"]==row["snp"], "rsid"] = rsid

    hg38_df.to_csv("explained_variance_by_snp_10best_snp_by_pc_rsids.csv", sep=";", index=False)

if __name__ == "__main__":
    main()