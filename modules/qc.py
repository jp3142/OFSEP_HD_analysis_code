#!/usr/bin/env python3

import os
import sys
from subprocess import Popen, PIPE
from shlex import split as ssplit
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg') # to use matplotlib in wsl (backend version: no show option possible, only savefig)
import matplotlib.pyplot as plt
from typing import Optional

def remove_cel(output_dir: str, file: str):
    """
    Remove ".CEL" from fam file

    file: fam file as input
    """
    out_file = os.path.join(output_dir, "tmp_fam.fam")
    with open(file, "r") as fin:
        with open(out_file,"w") as fout:
            awk_c = "sed 's/.CEL//g'"
            awk_p = Popen(ssplit(awk_c), text=True, stdin=fin, stdout=fout, stderr=PIPE)
            awk_p.communicate()

    mv_c = f"mv {out_file} {file}"
    mv_p = Popen(ssplit(mv_c), text=True, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    mv_p.communicate()

def get_bed_bim_fam_files(data_dir: str):
    """
    Automatically find bed, bim and fam  file in 'data_dir'
    and return bed, bim and fam relative paths.
    Arguments:
    data: directory containing 1 bed, 1 bim and 1 fam files
    """

    bed_file = str()
    bim_file = str()
    fam_file = str()

    for f in os.listdir(data_dir):
        file, ext = os.path.splitext(f)

        if ext == ".bed":
            bed_file = os.path.join(data_dir, file+ext)
        elif ext == ".bim":
            bim_file = os.path.join(data_dir, file+ext)
        elif ext == ".fam":
            fam_file = os.path.join(data_dir, file+ext)

    return bed_file, bim_file, fam_file


def get_sample_techno_QC_file(techno_QC_dir: str):
    """Get the 'sample_QC.txt' file in technology QC directory
    Arguments:
    techno_QC_dir: relative path to techno QC directory"""

    for f in os.listdir(techno_QC_dir):
        if f == "sample_QC.txt":
            return os.path.join(techno_QC_dir, f)


def get_bed_bim_fam_prefix(bed_file: str, bim_file: str, fam_file: str):
    """
    Check if a common basename exist between bed, bim and fam files relative path and eventually return it.
    Arguments:
    bed_file: bed file relative path
    bim_file: bim file relative path
    fam_file: fam file relative path
    """

    # check if prefix in map and ped files are identical
    try:
        assert(os.path.splitext(bed_file)[0] == os.path.splitext(bim_file)[0] == os.path.splitext(fam_file)[0])
    except AssertionError:
        sys.stderr.write("[AssertionError] .bed, .bim and .fam file names aren't identical. Please use the same name for all files.\n")
        exit(2)

    return os.path.splitext(bed_file)[0] # to keep only filename without relative path


def update_pheno_fam_file(fam_file: str, pheno_val: int):
    """Set all individuals phenotype variable value of fam_file to pheno_val
    fam_file: fam file relative filepath
    pheno_val: 1 for unaffected, 2 for affected"""

    try:
        assert(pheno_val in [1, 2])
    except AssertionError:
        sys.stderr.write(f"[AssertionError] Function: update_pheno_fam. The value of argument `pheno_val` must be equal to 1 or 2. Got pheno_val={pheno_val}\n")
        exit(3)

    # create and execute commands (in place modification of file with awk)
    with open(fam_file+"_tmp", 'w') as f:
        awk_c = ["awk", "{$6="+str(pheno_val)+"; print $0}", fam_file]
        awk_p = Popen(awk_c, stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate() # wait for process to end before executing the rest of the script

    mv_c = f"mv {fam_file}_tmp {fam_file}"
    mv_p = Popen(ssplit(mv_c), stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    mv_p.communicate()

def update_sex_from_previous_dataset(previous_fam_file: str, current_fam_file: str):
    """"""
    columns = ["FID", "IID", "PID", "MID", "Sex", "Phenotype"]
    old_fam = pd.read_csv(previous_fam_file, header=None, delim_whitespace=True)
    old_fam.columns = columns
    old_fam.sort_values(by="IID", inplace=True)

    current_fam = pd.read_csv(current_fam_file, header=None, delim_whitespace=True)
    current_fam.columns = columns
    current_fam.sort_values(by="IID", inplace=True)

    current_fam.set_index(["FID", "IID"], inplace=True)
    old_fam.set_index(["FID", "IID"], inplace=True) 
    current_fam.loc[:, "Sex"] = old_fam.loc[current_fam.index, "Sex"]

    current_fam.reset_index(drop=False, inplace=True)

    current_fam.to_csv(current_fam_file, sep=" ", index=False, header=False)

    return current_fam_file

def update_sex_fam_file(fam_file: str, sample_techno_QC_file: str, mapping_sex={"unknown": 0, "male": 1, "female": 2}):
    """Update fam file sex column by fetching sex annoted in sample_techno_QC_file
    Arguments:
    fam_file: fam file relative filepath
    sample_techno_QC_file: sample QC relative file path from AxiomAnalysisSuite technology
    mapping_sexe: dict mapping litteral sex value with corresponding integer value
    """

    # keep only columns 1 2 and 8 (sample_id, QC fail/pass, sex) of sample_techno_QC_file
    awk_c = " ".join(["awk", "'{print $1,$2,$8}'", sample_techno_QC_file])
    awk_p = Popen(ssplit(awk_c), stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)  # get 2 columns with sample id, QC(pass/fail), sex

    # remove ADN Control lines
    sed_c = "sed '/ADN\s/d'"
    sed_p = Popen(ssplit(sed_c), stdout=PIPE, stdin=awk_p.stdout, stderr=PIPE, text=True) # remove "ADN\sfloat_value" lines (corresponding to ADN control lines)

    # remove Failed QC samples from techno file (qc samples not BestRecommended)
    sed_c2 = "sed '/Fail/d'"
    sed_p2 = Popen(ssplit(sed_c2), stdout=PIPE, stdin=sed_p.stdout, stderr=PIPE, text=True)

    # get result of piped commands
    out_techno_QC, _ = sed_p2.communicate()

    # Transform out_techno_QC to pd.dataFrame
    out_techno_QC = np.array(out_techno_QC.split('\n')[1:-1], dtype=object) # remove header line and transform output of awk | sed into list
    out_techno_QC = pd.DataFrame(np.array([elm.split(' ') for elm in out_techno_QC], dtype=object), columns=["sample_id", "QC", "SEX"]) # sample_QC columns sample_id + sex as dataframe
    out_techno_QC.set_index('sample_id', inplace=True) # set index to sample_id column

    print("Sample_QC\n", out_techno_QC)

    # Load fam file as pd.Dataframe
    fam = pd.read_csv(fam_file, sep=' ', header=None) # header = None to avoid considering first sample as header
    fam.columns = ["FID", "IID", "PID", "MID", "SEX", "PHENO"] # get fam_file as dataframe and edit header
    #fam = pd.read_table(fam_file) pour remplacer les 2 lignes au dessus ???

    fam.set_index('IID', inplace=True) # set index to sample_id column

    ### Syntax: df.loc[df['column_name'] == "some_value"]

    # Get sex in sample_QC and modify sex in fam dataframe
    for sample_id in fam.index: #
        index_techno = out_techno_QC.index.get_loc(sample_id)
        index_fam = fam.index.get_loc(sample_id)
        fam['SEX'].iloc[index_fam] = mapping_sex[out_techno_QC['SEX'].iloc[index_techno]]

    # update fam file with updated sex (fam dataframe with updated sex)
    fam.reset_index(inplace=True)
    fam.set_index('FID', inplace=True)

    fam.to_csv(fam_file, sep=' ', header=False, mode='w') # overwrite original fam file


def keep_individuals_based_on_phenotype(file_prefix: str, fam_file: str, pheno_val_to_keep: int):
    """Keep individuals with given phenotype value (pheno_val_to_keep) in fam file
    Arguments:
    fam_file: relative path to fam file
    pheno_val_to_keep: 1 to keep controls, 2 to keep cases
    """
    # create a list of individuals to keep (with phenotype value = pheno_val_to_keep)
    output_dir = "/".join(file_prefix.split('/')[0:-1])
    ind_to_keep = os.path.join(output_dir, "ind_to_keep.txt")
    with open(ind_to_keep, 'w') as f:
        awk_c = "".join(["awk '{if($6==", "{}) print $1, $2".format(pheno_val_to_keep), "}' ", fam_file])
        awk_p = Popen(ssplit(awk_c), stdin=PIPE, stdout=f, stderr=PIPE, text=True)
        awk_p.communicate()

    # make a new dataset with only individuals to keep
    output_file_prefix = file_prefix+"_only_ctrl"
    plink_c = ["plink", "--bfile", file_prefix, "--keep", ind_to_keep, "--make-bed", "--out", output_file_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p.communicate()

    return output_file_prefix


def write_snp_list(output_dir: str, file_prefix: str, n_step: int, output_file="snp_list"):
    """
    STEP 1:
    plink --write-snplist function. Create files only if output file from write-snplist plink function already exists.
    Arguments:
    output_dir: output directory relative path
    files_prefix: relative path prefix of plink files
    """

    print(f"\n# {n_step} - Writing SNPs list ...")

    output_file = os.path.join(output_dir, f"{n_step}_"+output_file)

    if not os.path.exists(output_file + ".snplist"):
        plink_c = ["plink", "--bfile", file_prefix, "--write-snplist", "--out", output_file]
        plink_p = Popen(plink_c, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
        plink_p.communicate()
        print("[WriteSNPlist] Plink files have been created.")
    else:
        print("[WriteSNPlist] Plink write-snplist files already exists. No new file have been created.")


def get_SNP_freq(output_dir: str, file_prefix: str, n_step: int, output_file="freq"):
    """STEP 2 get frq file of variant frequencies"""
    print(f"\n# {n_step} - Getting SNPs frequencies ...")

    output_file = os.path.join(output_dir, f"{n_step}_"+output_file)
    frq_file = output_file + ".frq"

    if not os.path.exists(frq_file):
        plink_c = ["plink", "--bfile", file_prefix, "--freq", "--out", output_file]
        plink_p = Popen(plink_c, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
        plink_p.communicate()
        print("[Freq] Plink frq files have been created.")
    else:
        print("[Freq] Plink frq files already exists. No new file have been created.")

    return frq_file


def plot_maf_distribution(frq_file: str, n_bins=50):
    """
    Display minor allele frequency distribution
    """
    # delim_whitespace because the file isn't tab delimited but delimiter with several whitespaces
    frq_data = pd.read_table(frq_file, delim_whitespace=True)
    plt.hist(frq_data["MAF"], bins=n_bins, color='red')
    plt.title("MAF distribution of autosomal chromosome SNPs")
    plt.xlabel("MAF")
    plt.ylabel("count")
    plt.text(plt.xlim()[1]-(plt.xlim()[1]/6), plt.ylim()[1]-(plt.ylim()[1]/10), f"n={len(frq_data)}") # left, right = plt.xlim() / bottom, top = plt.ylim()
    plt.savefig(frq_file.split('.')[0] + "_hist", dpi=300, format='pdf')
    plt.close()
    print("[Figure] MAF distribution histogram created.")


def keep_only_needed_individuals_and_SNPs(output_dir: str, file_prefix: str, mind: float, geno: float, n_step: int, prefix=""):
    """
    STEP 3 (et refait après CheckSex): individual and snp missingness
    Keep only needed individuals and snps. Keep individuals based on missingness at individual level
    and keep SNPs based on missingness at SNP level.
    SNP filtering before individual filtering
    To use a prefix before the name to the output_file given by the function. Useful when another step calls this function, to keep the entire name of the step and understand that the files created by this function come after."""

    print(f"\n# {n_step} - Individual and SNP levels missingness ...")
    if prefix != "":
        output_file_prefix = os.path.join(output_dir, f"{n_step}_{prefix}_mind{str(mind)}_geno{str(geno)}")
    else:
        output_file_prefix = os.path.join(output_dir, f"{n_step}_mind{str(mind)}_geno{str(geno)}")

    new_bed = output_file_prefix + ".bed"
    new_bim = output_file_prefix + ".bim"
    new_fam = output_file_prefix + ".fam"

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

    file_prefix = remove_AT_GC_snps(file_prefix)

    if not (os.path.exists(new_bed) and os.path.exists(new_bim) and os.path.exists(new_fam)):
        plink_c = ["plink", "--bfile", file_prefix, "--geno", str(geno), "--mind", str(mind), "--make-bed", "--out", output_file_prefix]
        plink_p = Popen(plink_c, stderr=PIPE, stdin=PIPE, stdout=PIPE, text=True)
        plink_p.communicate()
        print("[Missingness] Individual level and SNP level missingness filter performed. New files have been created.")
    else:
        print("[Missingness] Individual level and SNP level missingness was already performed. No new files have been created.")

    return output_file_prefix


def plot_F_statistic(checksex_file: str, n_bins=10):
    """"""
    #checksex_df = pd.read_csv(checksex_file, delim_whitespace=True, header=0)
    checksex_df = pd.read_table(checksex_file, delim_whitespace=True)
    checksex_df = checksex_df.sort_values(by=['F'])
    plt.hist(checksex_df['F'], bins=n_bins)
    plt.title("F statistic (inbreeding coefficient) distribution")
    plt.xlabel("F")
    plt.ylabel("count")
    plt.text(plt.xlim()[1]-(plt.xlim()[1]/5), plt.ylim()[1]-(plt.ylim()[1]/10), f"n={len(checksex_df)}")
    plt.savefig(checksex_file.split('.')[0] + "_hist", dpi=300, format='pdf')
    plt.close()
    print("[Figure] F statistic distribution histogram created.")

def check_sex_impute_sex(output_dir: str, file_prefix: str, n_step: int, mind: float, geno: float, fstat_female_threshold: Optional[float] = 0.5, fstat_male_threshold: Optional[float] = 0.7):
    """
    STEP 4: check sex and remove individuals with sex discrepancies (PROBLEMS individuals). Update sex in dataset when F value allows to determine sex.
    Do not exclude individuals that can be updated from f stat value from dataset.
    Check sex, plot f stats, remove individuals and recheck for mind and geno to see if any new individuals need to be removed

    Parameters:
    fstat_male_threshold: f stat threshold above which we consider an individual as a male (usually 0.8)
    fstat_female_threshold: f stat threshold under which we consider an individual as female (usually 0.2)
    """
    print(f"\n# {n_step} - Checking sex discrepancies ...")

    check_sex_output_file_prefix = os.path.join(output_dir, f"{n_step}_checksex")
    
    # get sexcheck file
    checksex_file = check_sex_output_file_prefix+".sexcheck"

    if not os.path.exists(checksex_file):
        plink_c1 = ["plink", "--bfile", file_prefix, "--check-sex", f"{str(fstat_female_threshold)}", f"{str(fstat_male_threshold)}", "--out", check_sex_output_file_prefix]
        plink_p1 = Popen(plink_c1, stderr=PIPE, stdin=PIPE, stdout=PIPE, text=True)
        plink_p1.communicate()
        print("[CheckSex] check-sex files created.")

    else:
        print("[CheckSex] check-sex files already exists. No new files have been created")

    # plot F statistic (Normally F < 0.2 = female, F > 0.8 = Male)
    plot_F_statistic(checksex_file, n_bins=10)

    # exclude individuals with sex discrepancies + re run QC mind + geno

    # # get individuals with status = PROBLEM (sex discrepancies)
    ind_PROBLEM = check_sex_output_file_prefix + "_ind_PROBLEM_before_impute.txt"
    with open(ind_PROBLEM, 'w') as f1:
        awk_c1 = "".join(["awk -v OFS=' ' '{if($5==\"PROBLEM\") print $1,$2,$3,$4,$5,$6", "}' ", checksex_file])

        awk_p1 = Popen(ssplit(awk_c1), stderr=PIPE, stdin=PIPE, stdout=f1, text=True)
        awk_p1.communicate()

    print(f"[CheckSex] {ind_PROBLEM} file created.")

    imputed_sex_output_file_prefix = os.path.join(output_dir, f"{n_step}_imputed_sex")

    plink_c2 = ["plink", "--bfile", file_prefix, "--impute-sex", f"{str(fstat_female_threshold)}", f"{str(fstat_male_threshold)}", "--make-bed", "--out", imputed_sex_output_file_prefix]
    plink_p2 = Popen(plink_c2, stderr=PIPE, stdin=PIPE, stdout=PIPE, text=True)
    plink_p2.communicate()

    print("[ImputeSex] imputed-sex files created")

    # create list of individuals to remove that were not imputed
    ind_to_exclude = os.path.join(output_dir, f"{n_step}_ind_to_exclude_after_impute.txt")
    with open(ind_to_exclude, 'w') as f2:
        awk_c2 = "awk '{if($5==0){print $1,$2}}' " + imputed_sex_output_file_prefix + ".fam"
        awk_p2 = Popen(ssplit(awk_c2), stdout=f2, stderr=PIPE, stdin=PIPE, text=True)
        awk_p2.communicate()

    # remove individuals that weren't imputed
    removed_sex_output_file_prefix = imputed_sex_output_file_prefix + "_removedIndNotImputed"
    plink_c3 = ["plink", "--bfile", imputed_sex_output_file_prefix, "--remove", ind_to_exclude, "--make-bed", "--out", removed_sex_output_file_prefix]
    plink_p3 = Popen(plink_c3, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p3.communicate()
    
    # re run keep_only_needed_individuals_and_snp because snp and individuals missingness changed 
    output_file_prefix = keep_only_needed_individuals_and_SNPs(output_dir, removed_sex_output_file_prefix, mind=mind, geno=geno, n_step=n_step, prefix="imputed_sex")

    return output_file_prefix

def keep_only_autosomal_chr(output_dir: str, file_prefix: str, n_step: int):
    """
    AJOUTER PLUS TARD CONDITION CREATION FICHIER
    """
    print(f"\n# {n_step} - Keeping only autosomal chr ...")

    snp1_22_file = os.path.join(output_dir, f"{n_step}_snp1-22.txt")

    # get on,ly autosomal snps
    bim_file = file_prefix + ".bim"
    awk_c = "".join(["awk '", "{if ($1 >= 1 && $1 <= 22) print $2}' ", bim_file])
    with open(snp1_22_file, 'w') as f:
        awk_p = Popen(ssplit(awk_c), stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate()

    # create bfile with only autosomal snps
    output_file_prefix = os.path.join(output_dir, f"{n_step}_chr1-22only")
    plink_c = ["plink", "--bfile", file_prefix, "--extract", snp1_22_file, "--make-bed", "--out", output_file_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p.communicate()

    print(f"[AutosomalChr] File {n_step}_chr1-22only created.")

    return output_file_prefix


def filter_maf(output_dir: str, file_prefix: str, maf: float, n_step: int):
    """
    Filter minor allele frequencies
    """
    print(f"\n# {n_step} - Filtering MAF ...")
    output_file_prefix = os.path.join(output_dir, f"{n_step}_maf{maf}")

    if not (os.path.exists(output_file_prefix+'.bed') and os.path.exists(output_file_prefix+'.bim') and os.path.exists(output_file_prefix+'.fam')):
        plink_c = ["plink", "--bfile", file_prefix, "--maf", str(maf), "--make-bed", "--out", output_file_prefix]

        plink_p = Popen(plink_c, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
        plink_p.communicate()
        print(f"[MAF Filters] {output_file_prefix} files created.")
    else:
        print(f"[MAF Filters] {output_file_prefix} files already exist. No new files have been created. No --maf and --hwe re performed.")

    return output_file_prefix


def detect_strongly_deviating_snps_and_plot_hwe_distribution(output_dir: str, file_prefix: str, hwe: float, n_step: int, n_bins=50, dataset_type=0):
    """
    Generate files (with --hardy) to then plot the distribution of HWE p-values of all SNPs and snps with hwe < hwe threshold."""
    # --hardy writes a list of genotype counts and Hardy-Weinberg equilibrium exact test statistics to plink.hwe. Necessary for plotting
    # distribution of hwe pvalues of all snps

    hwe_test_cat = "UNAFF" # lines marked by this in the hwe file
    if dataset_type in [1, 2]:
        hwe_test_cat = "AFF"
    elif dataset_type == 0:
        hwe_test_cat = "ALL"
    else:
        sys.stderr.write(f"[ParameterError] 'dataset_type' set to '{dataset_type}'. Value should be in [0, 1, 2].\n")
        exit(7)

    print(f"\n# {n_step} - Detecting strongly hwe deviating variants ...")
  
    # generate file of hwe for all snps in order to then plot distribution of all snps
    hardy_prefix = os.path.join(output_dir, f"{n_step}_hardy")
    plink_c = ["plink", "--bfile", file_prefix, "--hardy", "--out", hardy_prefix]
    # plink --hwe only filters for controls
    # plink --hardy fonction filter for also cases
    plink_p = Popen(plink_c, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p.communicate()
    hardy_file = f"{hardy_prefix}.hwe"
    print("[Hardy] HWE exact test on all snps done. New files have been created.")

    # Selecting SNPs with HWE p-value below 0.00001, required for one of the two plot generated by the next function (plotting), allows to zoom in on
    # strongly deviating SNPs.
    # zooming to get distribution of hwe of snp with hwe < threshold
    # no header on this file
    hardy_zoom_file = f"{hardy_prefix}_zoom.hwe"
    with open(hardy_zoom_file, 'w') as f:
        awk_c = "".join(["awk '{ if ($9 < ", str(hwe),") print $0 }' ", hardy_file])
        awk_p = Popen(ssplit(awk_c), stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        awk_p.communicate()
        print(f"[Hardy] Filtering SNP with p < {hwe}")

    # plot all snp hwe distribution
    all_hwe = pd.read_table(hardy_file, delim_whitespace=True)
    all_hwe = all_hwe[all_hwe["TEST"] == hwe_test_cat] # keep only test=aff because we don't want to use the 5 missing sex sample (5 ambiguous)
    # -> difference between all (with 5 missing sex samples) and aff (ignoring the 5 missing sex samples)
    plt.hist(all_hwe['P'], bins=n_bins)
    plt.title("Distribution of HWE exact test p-values for all SNPs")
    plt.xlabel("p")
    plt.ylabel("count")
    plt.text(plt.xlim()[1]-(plt.xlim()[1]/2), plt.ylim()[1]-(plt.ylim()[1]/10), f"n={len(all_hwe)}")
    hist_file = f"{n_step}_distribution_pval_hwe_all_snp.pdf"
    plt.savefig(os.path.join(output_dir, hist_file), format="pdf", dpi=300)
    plt.close()
    print(f"[Figure] {hist_file} plot created.")

    # plot zoom hwe snp distribution

    if os.stat(hardy_zoom_file).st_size > 0:
        zoom_hwe = pd.read_table(hardy_zoom_file, delim_whitespace=True, header=None)
        zoom_hwe.columns = all_hwe.columns # set column names to all_hwe columns (no header on zoom hwe file)
        zoom_hwe = zoom_hwe[zoom_hwe["TEST"] == hwe_test_cat] # keep only results for affected
        zoom_hwe.columns = all_hwe.columns
        plt.hist(zoom_hwe['P'], bins=n_bins)
        plt.title(f"Distribution of HWE exact test p-values of SNPs with p < {hwe}.")
        plt.xlabel("p")
        plt.ylabel("count")
        plt.text(plt.xlim()[1]-(plt.xlim()[1]/7), plt.ylim()[1]-(plt.ylim()[1]/10), f"n={len(zoom_hwe)}")
        hist_file = f"{n_step}_distribution_pval_hwe_zoom{hwe}_snp.pdf"
        plt.savefig(os.path.join(output_dir, hist_file), format="pdf", dpi=300)
        plt.close()
        print(f"[Figure] {hist_file} plot created.")
    else:
        print(f"[ZoomHWE] No variants with HWE < {hwe}. No zoomed hwe file created.")


def hwe_filtering(output_dir: str, file_prefix: str, hwe: float, n_step: int, dataset_type: bool):
    """
    dataset_type: 0: controls, 1: cases, 2: reference
    """
    # set control=False for hwe filtering on case and ctrl at the same time to include in the statistic test ctrl, case and missing phenotype individuals
    # By default the --hwe option in plink only filters for controls.
    # Therefore, we use two steps, first we use a stringent HWE threshold for controls, followed by a less stringent threshold for the case data.
    print(f"\n# {n_step} - HWE filtering ...")

    if dataset_type == 0: # --hwe on SNP only on ctrl (ignore cases)
        output_file_prefix = os.path.join(output_dir, f"{n_step}_hwe{hwe}_only_ctrl")
        plink_c = ["plink", "--bfile", file_prefix, "--hwe", str(hwe), "--make-bed", "--out", output_file_prefix]
        plink_p = Popen(plink_c, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
        plink_p.communicate()
        print("[HWE Filtering] Filtering done for controls. New files have been created.")

    elif dataset_type in [1, 2]: # hwe on all SNPS from cases and control and ref(if case and control and ref separate dataset -> perform hwe only on cases or ref)
        output_file_prefix = os.path.join(output_dir, f"{n_step}_hwe-{hwe}_include-nonctrl")
        plink_c = ["plink", "--bfile", file_prefix, "--hwe", str(hwe), "include-nonctrl", "--make-bed", "--out", output_file_prefix]
        plink_p = Popen(plink_c, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
        plink_p.communicate()
        print("[HWE Filtering] Filtering done for cases. New files have been created.")
    else:
        sys.stderr.write(f"[ParameterError] 'dataset_type' set to '{dataset_type}'. Value should be in [0, 1, 2].\n")
        exit(7)
        

    return output_file_prefix

    # The HWE threshold for the cases filters out only SNPs which deviate extremely from HWE.
    # This second HWE step only focusses on cases because in the controls all SNPs with a HWE p-value < hwe 1e-6 were already removed

def heterozygosity_rate_filtering(output_dir: str, file_prefix: str, high_ld_regions_file: str, n_step: int, n_bins=10):
    """
    # Generate a plot of the distribution of the heterozygosity rate of your subjects.
    # And remove individls with a heterozygosity rate deviating more than 3 sd from the mean.

    Creates a pruned dataset for het rate filtering but output the non pruned dataset as output_file_prefix

    """

    # Checks for heterozygosity are performed on a set of SNPs which are not highly correlated. (But the "output_file_prefix" returned corresponds to the whole dataset filtered for het rate (it doesn't only contains pruned snp, we just check het on prenued dataset) )
    # Therefore, to generate a list of non-(highly)correlated SNPs (independant snps), we exclude high inversion regions (inversion.txt [High LD regions]) and prune the SNPs using the command --indep-pairwise�.
    # The parameters 50 5 0.2 stand
    #  respectively for: the window size, the number of SNPs to shift the window at each step, and the multiple correlation coefficient for a SNP being regressed on all other SNPs simultaneously.
    print(f"\n# {n_step} - High and low heterogozygosity rate filtering ...")

    #plink --bfile HapMap_3_r3_9 --exclude inversion.txt --range --indep-pairwise 50 5 0.2 --out indepSNP
    # Note, don't delete the file indepSNP.prune.in, we will use this file in later steps of the tutorial.
    # generate list of independant SNPs
    # --range flag is deprecated use --extract range <filename> ???
    indepSNP_prefix = os.path.join(output_dir, f"{n_step}_indepSNP")
    plink_c1 = ["plink", "--bfile", file_prefix, "--exclude", high_ld_regions_file, "--range", "--indep-pairwise", "50", "5", "0.2", "--out", indepSNP_prefix]
    plink_p1 = Popen(plink_c1, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p1.communicate()
    print("[Pruning] Independant SNPs list created.")

    # plink --bfile HapMap_3_r3_9 --extract indepSNP.prune.in --het --out R_check
    # This file contains your pruned data set.
    pruned_data_prefix = os.path.join(output_dir, f"{n_step}_pruned_dataset")
    plink_c2 = ["plink", "--bfile", file_prefix, "--extract", f"{indepSNP_prefix}.prune.in", "--het", "--out", pruned_data_prefix]
    plink_p2 = Popen(plink_c2, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p2.communicate()
    print("[Pruning] Independant SNPs dataset created.")

    # plotting heterozygosity rate distribution before filtering
    het = pd.read_table(f"{pruned_data_prefix}.het", delim_whitespace=True)
    het['HET_RATE'] = (het["N(NM)"] - het["O(HOM)"])/het["N(NM)"] # NM: nb non missing autosomal genotype obs, O(HOM): observed number of homozygote
    plt.hist(het["HET_RATE"], bins=n_bins)
    plt.title("Distribution of heterozygosity rate")
    plt.xlabel("Heterozygosity rate")
    plt.ylabel("count")
    plt.text(plt.xlim()[1]-plt.xlim()[1]/7, plt.ylim()[1]-(plt.ylim()[1]/10), f"n={len(het)}")
    plt.savefig(os.path.join(output_dir, f"{n_step}_distribution_het.pdf"), format="pdf", dpi=300)
    plt.close()
    print("[Figure] Distribution of heterozygosity rate histogram created.")

    # The following code generates a list of individuals who deviate more than 3 standard deviations from the heterozygosity rate mean.
    # generate list of heterozygosity outliers (who deviate more than 3SD from het_rate mean of the pop)
    het_fail = het.loc[ ( het["HET_RATE"] < (np.mean(het["HET_RATE"])-3*np.std(het["HET_RATE"])) ) | ( het["HET_RATE"] > (np.mean(het["HET_RATE"])+3*np.std(het["HET_RATE"])) ) ]
    het_fail_qc_file = os.path.join(output_dir, f"{n_step}_fail_het_qc.txt")
    het_fail.to_csv(het_fail_qc_file, sep="\t", index=False) # save all het_fail_qc
    print(f"[HET Filtering] File HET FAIL QC '{het_fail_qc_file}' created.")

    het_fail_ind_file = os.path.join(output_dir, f"{n_step}_fail_het_ind.txt")
    het_fail.iloc[:, 0:2].to_csv(het_fail_ind_file, sep="\t", index=False) # save only 2 first column of het_fail_qc necessary to make it compatible with plink format
    print(f"[HET Filtering] File HET FAIL IND '{het_fail_ind_file}' created.")

    # Output of the command above: fail-het-qc.txt .
    # When using our example data/the HapMap data this list contains 2 individuals (i.e., two individuals have a heterozygosity rate deviating more than 3 SD's from the mean).
    # Adapt this file to make it compatible for PLINK, by removing all quotation marks from the file and selecting only the first two columns.
    #sed 's/"// g' fail-het-qc.txt | awk '{print$1, $2}'> het_fail_ind.txt
    # remove heterozygosity rate outliers
    
    # /!\ NO REMOVING OF INDIVIDUALS WITH HIGH OR LOW HET RATE !!! Change the number of individuals depending on the maf since het not ocmputed on same snps
    # output_file_prefix = os.path.join(output_dir, f"{n_step}_het_removeInd")
    # plink_c3 = ["plink", "--bfile", file_prefix, "--remove", het_fail_ind_file, "--make-bed", "--out", output_file_prefix]
    # plink_p3 = Popen(plink_c3, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    # plink_p3.communicate()
    # print("[HET Filtering] Removed het outliers individuals")

    return pruned_data_prefix, indepSNP_prefix, file_prefix # file prefix is the input plink dataset, it doesn't only contains pruned snp. We just removed het rate deviating individuals and the list of these individuals was created using pruned snp dataset.

def relatedness_filtering(output_dir: str, file_prefix: str, indepSNP_prefix: str, min: float, n_step: int, n_bins: int):
    """"""

    # It is essential to check datasets you analyse for cryptic relatedness.
    # Assuming a random population sample we are going to exclude all individuals above the pihat threshold of 0.2
    # Check for relationships between individuals with a pihat > 0.2.
    # plink --bfile HapMap_3_r3_10 --extract indepSNP.prune.in --genome --min 0.2 --out pihat_min0.2

    print(f"\n# {n_step} - Checking for relatedness between every pair of individuals ...")

    prune_in_file = indepSNP_prefix + ".prune.in" # contient les snps pruned = independants (prun.out contient les snp non prune = non independant)
    relatedness_prefix = os.path.join(output_dir, f"{n_step}_pihat_min{min}")

    plink_c1 = ["plink", "--bfile", file_prefix, "--extract", prune_in_file, "--genome", "--min", str(min), "--out", relatedness_prefix]
    plink_p1 = Popen(plink_c1, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p1.communicate()
    print("[Relatedness] Relatedness files created.")


    # ( NOT DONE FOR MY DATASET ): The HapMap dataset is known to contain parent-offspring relations.
    # The following commands will visualize specifically these parent-offspring relations, using the z values.
    #awk '{ if ($8 >0.9) print $0 }' pihat_min0.2.genome>zoom_pihat.genome

    relatedness_file = relatedness_prefix + ".genome"
    relatedness = pd.read_table(relatedness_file, delim_whitespace=True)

    # plot pi-hat distribution
    plt.hist(relatedness['PI_HAT'], bins=n_bins, color="red")
    plt.title("pi_hat (IBD) distribution")
    plt.text(plt.xlim()[1]-plt.xlim()[1]/6, plt.ylim()[1]-(plt.ylim()[1]/10), f"n={len(relatedness)}", fontsize=12) # coordinates correspond to axes graduation of the plot
    plt.savefig(os.path.join(output_dir, f"{n_step}_pi-hat_distribution.pdf"), format="pdf")
    plt.close()

    # update relationship column (RT (5th column)) of genome file and set if 1st, 2nd, 3rd degree relationship or duplicate/twin
    for i, row in relatedness.iterrows():

        if row["PI_HAT"] >= 0.9:
            relatedness.loc[i, "RT"] = "DUP" # duplicates, identical twins 1

        elif row["PI_HAT"] >= 0.44 and row["PI_HAT"] < 0.9:
            relatedness.loc[i, "RT"] = "FDR" # first degree relatives 0.5 parents, siblings, children

        elif row["PI_HAT"] >= 0.2 and row["PI_HAT"] < 0.44:
            relatedness.loc[i, "RT"] = "SDR" # second degree relatives 0.25 grandparent, grandchildren, uncles, aunts, nephews, nieces, half siblings

        elif row["PI_HAT"] >= 0.10 and row["PI_HAT"] < 0.2:
            relatedness.loc[i, "RT"] = "TDR" # third degree relatives 0.125 first cousin, great grand parents, great aunt, great uncle, great niece, great nephew, great grandchild, half aunt, half uncle

    relatedness.to_csv(relatedness_file, sep="\t") # overwrite relatedness_file with updated RT column

    # Generate a plot to assess the type of relationship.
    # fig params
    plt.rcParams["figure.figsize"] = [7.50, 7.50]
    plt.rcParams["figure.autolayout"] = True

    # subsets to plot
    dup = relatedness.loc[relatedness["RT"] == "DUP"] # parent offspring (relation parent - progéniture (enfants))
    fdr = relatedness.loc[relatedness["RT"] == "FDR"] # full sibblings (frère et soeurs)
    sdr = relatedness.loc[relatedness["RT"] == "SDR"] # half sibblings (demi_frère et demi-soeur)
    tdr = relatedness.loc[relatedness["RT"] == "TDR"] # other relatedness

    # plotting (only Z0 et Z1 car avec 2 des coefficients k on peut déterminer le 3ème donc info redondante. Et avec seulement
    # 2 coefficients k on peut déterminer quel type de relationship il y a.)
    plt.scatter(dup["Z0"], dup["Z1"], c="red", label="duplicates/twins") # scatter plot permet de changer individuellement les couleurs des points pour chaque sous groupe
    plt.scatter(fdr["Z0"], fdr["Z1"], c="yellow", label="first degree relatives")
    plt.scatter(sdr["Z0"], sdr["Z1"], c="orange", label="second degree relatives")
    plt.scatter(tdr["Z0"], tdr["Z1"], c="blue", label="third degree relatives")
    plt.title("Relatedness between individuals")
    plt.xlabel("Z0 P(K=0)")
    plt.ylabel("Z1 P(K=1)")
    plt.legend(loc="upper right")
    plt.text(plt.xlim()[1]-(plt.xlim()[1]/7), plt.ylim()[1]-(plt.ylim()[1]/7), f"n={len(relatedness)}", fontsize=12)
    plt.savefig(os.path.join(output_dir, f"{n_step}_relatedness_plot.pdf"), format="pdf")
    plt.close()
    print("[Figure] Relatedness plot created.")

    # # (NOT DONE) Normally, family based data should be analyzed using specific family based methods. In this tutorial, for demonstrative purposes, we treat the relatedness as cryptic relatedness in a random population sample.
    # # In this tutorial, we aim to remove all 'relatedness' from our dataset.
    # # To demonstrate that the majority of the relatedness was due to parent-offspring we only include founders (individuals without parents in the dataset).
    # plink --bfile HapMap_3_r3_10 --filter-founders --make-bed --out HapMap_3_r3_11

    # #(NOT DONE) Now we will look again for individuals with a pihat >0.2.
    # plink --bfile HapMap_3_r3_11 --extract indepSNP.prune.in --genome --min 0.2 --out pihat_min0.2_in_founders
    # # The file 'pihat_min0.2_in_founders.genome' shows that, after exclusion of all non-founders, only 1 individual pair with a pihat greater than 0.2 remains in the HapMap data.
    # # This is likely to be a full sib or DZ twin pair based on the Z values. Noteworthy, they were not given the same family identity (FID) in the HapMap data.

    # # For each pair of 'related' individuals with a pihat > 0.2, we recommend to remove the individual with the lowest call rate.
    # plink --bfile HapMap_3_r3_11 --missing
    # # Use an UNIX text editor (e.g., vi(m) ) to check which individual has the highest call rate in the 'related pair'.

    # create missing call rates file
    missing_prefix = os.path.join(output_dir, f"{n_step}_missing")
    plink_c2 = ["plink", "--bfile", file_prefix, "--missing", "--out", missing_prefix]
    plink_p2 = Popen(plink_c2, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    plink_p2.communicate()
    print("[CallRates] Missing call rates file created.")

    # make a list of individuals to remove (with pihat > 0.2 and with the lowest missing call rate). For each line in relatedness file, remove from the dataset the individual with lowest call rate
    missing_file = missing_prefix + ".imiss"
    missing_df = pd.read_table(missing_file, delim_whitespace=True)
    missing_df["IID"] = missing_df["IID"].astype(object) # in case iid is an integer (can't merge df with col int and col object)

    relatedness_1 = relatedness[["FID1","IID1"]]
    relatedness_1.columns = ["FID", "IID"]
    relatedness_1["IID"] = relatedness_1["IID"].astype(object) # in case iid is an integer (can't merge df with col int and col object)

    relatedness_2 = relatedness[["FID2","IID2"]]
    relatedness_2.columns = ["FID", "IID"]
    relatedness_2["IID"] = relatedness_2["IID"].astype(object) # in case iid is an integer (can't merge df with col int and col object)

    call_rate_1 = pd.merge(relatedness_1, missing_df[["IID", "F_MISS"]], on="IID", how="left")
    call_rate_2 = pd.merge(relatedness_2, missing_df[["IID", "F_MISS"]], on="IID", how="left")

    individual_to_remove = []
    for i in range(0, len(call_rate_1)):
        line1 = call_rate_1.iloc[i, :]
        line2 = call_rate_2.iloc[i, :]

        if line1.loc["F_MISS"] <= line2.loc["F_MISS"]: # add to list of ind to remove the ind with smallest call rate
            individual_to_remove.append((line1.loc["FID"], line1.loc["IID"]))
        else:
            individual_to_remove.append((line2.loc["FID"], line2.loc["IID"]))

    individual_to_remove = set(individual_to_remove) # to erase any doubled

    individual_to_remove = pd.DataFrame(individual_to_remove, columns=["FID", "IID"])
    ind_to_remove_file = os.path.join(output_dir, f"{n_step}_low_call_rate_pihat{min}.txt")
    individual_to_remove.to_csv(ind_to_remove_file, sep='\t', header=False, index=False)


    # # Delete the individuals with the lowest call rate in 'related' pairs with a pihat > 0.2
    # plink --bfile HapMap_3_r3_11 --remove 0.2_low_call_rate_pihat.txt --make-bed --out HapMap_3_r3_12
    output_file_prefix = os.path.join(output_dir, f"{n_step}_low_call_rate_pihat{min}_ind_removed")
    plink_c3 = ["plink", "--bfile", file_prefix, "--remove", ind_to_remove_file, "--make-bed", "--out", output_file_prefix]
    plink_p3 = Popen(plink_c3, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
    plink_p3.communicate()

    print(f"[LowCallRate] Low call rate individuals with pihat > {min} removed. New dataset created.")

    return output_file_prefix

def split_pseudo_auto_region_x_chr(output_dir: str, file_prefix: str, n_step: int):
    """
    Use split x before check-sex to  split the x chromosome autosomal region and encode it as chr 25.
    Changes F statistics computation with --check-sex
    """

    print(f"\n# {n_step} - Splitting X autosomal region of X chromosome ...")
    output_file_prefix = os.path.join(output_dir, f"{n_step}_split_x")
    plink_c = ["plink", "--bfile", file_prefix, "--split-x", "b37", "--make-bed", "--out", output_file_prefix] # b37'/'hg19 build version of genome dataset
    plink_p = Popen(plink_c, stderr=PIPE, stdout=PIPE, stdin=PIPE, text=True)
    plink_p.communicate()
    print(f"[Split-X] Autosomal region of X chromosome split done. Set to chromosome code 25.")

    return output_file_prefix

def QC(output_dir: str, file_prefix: str, high_ld_regions_file: str,  mind: float, geno: float, maf: float, hwe: float, min: float, hwe_zoom: float, dataset_type: int, presence_of_non_auto_chr: bool, fstat_female_threshold: float, fstat_male_threshold: float):
    """FOLLOW ORDER OF STEPS IN BLACK NOTEBOOK
    file_prefix = prefix loaded with --bfile
    hwe_zoom: ~ 100 à 1000 x bigger than hwe threshold
    presence_of_sex_chr: wether chr 23 24 and 25 are present or not in the bim file (if non autosomal chromosomes are present)
    dataset_type: 0: controls, 1: cases, 2: ref
    min: pi_hat
    QC for GWAS aren't the same than QC for Population stratification.
    function for QC before imputation
    """

    if dataset_type == 0:
        dataset_type_name = "CONTROLS"
        output_dir = os.path.join(output_dir, "data_ctrl")
    elif dataset_type == 1:
        dataset_type_name = "CASES"
        output_dir = os.path.join(output_dir, "data_case")
    elif dataset_type == 2:
        dataset_type_name = "REFERENCE"
        output_dir = os.path.join(output_dir, "data_ref")
    else:
        sys.stderr.write(f"[ParameterError] 'dataset_type' set to '{dataset_type}'. Value should be in [0, 1, 2].\n")
        exit(7)

    print(f"\n#### Quality Check (QC) FOR {dataset_type_name} ####")
    
    os.makedirs(output_dir, exist_ok=True)

    # 1 -SNP LIST
    write_snp_list(output_dir, file_prefix, n_step=1)

    # 2 - INDIVIDUAL + SNP MISSINGNESS (no individuals remove with mind geno < 0.2 so directly set thresholds as 0.02)
    file_prefix = keep_only_needed_individuals_and_SNPs(output_dir, file_prefix, mind=mind, geno=geno, n_step=2) # no filtering with 0.2 value (0 indiv (mind) filtered and 0 variants (geno) filtered)

    # 3 - SEX DISCREPANCY
    if presence_of_non_auto_chr: # if sexual chr in dataset
        file_prefix = split_pseudo_auto_region_x_chr(output_dir, file_prefix, n_step=3)
        file_prefix = check_sex_impute_sex(output_dir, file_prefix, n_step=4, mind=mind, geno=geno, fstat_female_threshold=fstat_female_threshold, fstat_male_threshold=fstat_male_threshold)

        # 4 - keep only autosomal chr
        file_prefix = keep_only_autosomal_chr(output_dir, file_prefix, n_step=5)

    # 5 - GET FREQ AND PLOT MAF DISTRIBUTION
    frq_file = get_SNP_freq(output_dir, file_prefix, n_step=6, output_file="maf_all_SNP")
    plot_maf_distribution(frq_file, n_bins=50)

    # 6 - Filter MAF
    file_prefix = filter_maf(output_dir, file_prefix, maf=maf, n_step=7)

    # 7 - Delete SNPs which are not in Hardy-Weinberg equilibrium (HWE). Check the distribution of HWE p-values of all SNPs.
    detect_strongly_deviating_snps_and_plot_hwe_distribution(output_dir, file_prefix, hwe=hwe_zoom, n_step=8, n_bins=50, dataset_type=dataset_type)

    # 8 - HWE pvalue filtering
    file_prefix = hwe_filtering(output_dir, file_prefix, hwe=hwe, n_step=9, dataset_type=dataset_type)

    # 9 - Heterogosity rate filtering
    _, indepSNP_prefix, file_prefix = heterozygosity_rate_filtering(output_dir, file_prefix, high_ld_regions_file, n_step=10, n_bins=50) # file prefix is the non pruned dataset

    # 10 - relatedness checking
    file_prefix = relatedness_filtering(output_dir, file_prefix, indepSNP_prefix, min=min, n_step=11, n_bins=50)

    return file_prefix, indepSNP_prefix # file_prefix doesn't only contains independent snps !!! Return indepSNP prefix but useless because we will re create a new one after imputation


def recode_to_vcf(file_prefix: str):
    """
    Recode bed bim fam files to a single vcf file.
    Return newly created vcf file path.
    # recode last dataset (marked 10 from cases) to vcf before imputation (and lift over cases to HG38)
    #plink --bfile [filename prefix] --recode vcf --out [VCF prefix]
    #The reverse VCF -> .bed/.bim/.fam conversion is also supported:
    #plink --vcf [VCF filename] --out [.bed/.bim/.fam prefix]
    """
    plink_c = ["plink", "--bfile", file_prefix, "--recode", "vcf", "--out", file_prefix]
    plink_p = Popen(plink_c, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    plink_p.communicate()
    print(f"\n[Recode] Recoded {file_prefix} bed, bim and fam files to vcf done.")

    return file_prefix+".vcf" # return vcf file

def split_vcf_by_chr(vcf_file: str, n_chr=22):
    """
    Split vcf file by chromosome and compress each file with bgzip (one vcf file by chrom). Needed prior to SNP imputation using TopMed
    n_chr = number of chromosomes
    TopMed takes bgzipped files .vcf.gz
    """

    # # Pas besoin de recréer index car topMed ne prend pas index en compte
    file_prefix = ".".join(vcf_file.split('.')[0:2])

    # create .vcf.gz file containing all chromosomes
    vcf_gz_file = vcf_file + ".gz"
    with open(vcf_gz_file, 'w') as f:
        # bgzip -> bgzip -c 10_low_call_rate_pihat0.1_ind_removed.vcf > 10_low_call_rate_pihat0.1_ind_removed.vcf.gz
        bgzip_c1 = ["bgzip", "-c", vcf_file]
        bgzip_p1 = Popen(bgzip_c1, stdout=f, stdin=PIPE, stderr=PIPE, text=True)
        bgzip_p1.communicate()

    # create tabix index file (.tbi) for .vcf.gz file
    # tabix -> tabix -p vcf 10_low_call_rate_pihat0.1_ind_removed.vcf.gz (OU bcftools index 10_low_call_rate_pihat0.1_ind_removed.vcf.gz ) mais doc topmed recommende utiliser bgzip
    tabix_c = ["tabix", "-p", "vcf", vcf_gz_file]
    tabix_p = Popen(tabix_c, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    tabix_p.communicate()

    # get compressed vcf file for each chromosome
    vcf_gz_files_by_chr = []
    for chr in range(1, n_chr+1):
        # get only current chromosome vcf file from vcf.gz file
        vcf_file_chr = file_prefix+f"_chr{chr}.vcf"
        with open(vcf_file_chr, 'w') as f1:
            # bcftools -> bcftools view -r 1 10_low_call_rate_pihat0.1_ind_removed.vcf.gz > 10_low_call_rate_pihat0.1_ind_removed_chr1.vcf
            bcftools_c = ["bcftools", "view", "-r", str(chr), vcf_gz_file]
            bcftools_p = Popen(bcftools_c, stdout=f1, stdin=PIPE, stderr=PIPE, text=True)
            bcftools_p.communicate()

        # compress the vcf_file_chr of the current chr for TopMed
        vcf_gz_file_chr = vcf_file_chr + ".gz"
        with open(vcf_gz_file_chr, "w") as f2:
            # bgzip -> bgzip -c 10_low_call_rate_pihat0.1_ind_removed_chr1.vcf > 10_low_call_rate_pihat0.1_ind_removed_chr1.vcf.gz
            bgzip_c2 = ["bgzip", "-c", vcf_file_chr]
            bgzip_p2 = Popen(bgzip_c2, stdout=f2, stdin=PIPE, stderr=PIPE, text=True)
            bgzip_p2.communicate()

        vcf_gz_files_by_chr.append(vcf_gz_file_chr)

    print(f"\n[Recode] Split {vcf_file} by chromosome file done.")

    return vcf_gz_files_by_chr # return list of vcf.gz files