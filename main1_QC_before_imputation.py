#!/usr/bin/env python3

from modules.qc import *
from config_main1_QC_before_imputation import *

def main():
    """"""

    try:
        data_case_dir = str(sys.argv[1])
        data_ctrl_dir = str(sys.argv[2])
        data_naf_dir = str(sys.argv[3])
        techno_QC_dir = str(sys.argv[4])
        high_ld_regions_file = str(sys.argv[5])
        output_dir = str(sys.argv[6]) # directory in which to store all outputs of the script (intermediaite datasets and others)
    except IndexError:
        sys.stderr.write("[SyntaxError] python3 main1_qc_before_imputation.py <data_case_directory> <data_ctrl_directory> <data_naf_dir> <sampleQC_axiomAnlaysis_directory> <high_ld_regions_file> <output_directory>\n")
        # Exemple: python3 GWAS_OFSEP_HD.py data/data_case/ data/data_ctrl/ data/techno_QC_Axiom/ data/data_reference test/
        exit(1)

    os.makedirs(output_dir, exist_ok=True) # create output directory

    # get cases and ctrl bed bim fam files relative paths
    case_bed_file, case_bim_file, case_fam_file = get_bed_bim_fam_files(data_case_dir)
    remove_cel(output_dir, case_fam_file) # remove .CEL from case fam file
    ctrl_bed_file, ctrl_bim_file, ctrl_fam_file = get_bed_bim_fam_files(data_ctrl_dir)
    naf_bed_file, naf_bim_file, naf_fam_file = get_bed_bim_fam_files(data_naf_dir)
    sample_techno_QC_file = get_sample_techno_QC_file(techno_QC_dir)

    # get cases and ctrl bed bim fam common prefix filename
    case_prefix = get_bed_bim_fam_prefix(case_bed_file, case_bim_file, case_fam_file)
    ctrl_prefix = get_bed_bim_fam_prefix(ctrl_bed_file, ctrl_bim_file, ctrl_fam_file)
    naf_prefix = get_bed_bim_fam_prefix(naf_bed_file, naf_bim_file, naf_fam_file)

    # PRE TREATMENT #

    # set cases phenotype to 2 (=affected)
    if update_pheno_case_fam: # DONE
        update_pheno_fam_file(case_fam_file, pheno_val=case_pheno_value)

    # update fam file with sex value from sample_techno_QC_file
    if update_sex_case_fam: # DONE
        update_sex_fam_file(case_fam_file, sample_techno_QC_file)

    # keep only ctrl individuals in ctrl dataset
    if keep_only_ctrl_individuals_in_ctrl_data:  # DONE
        _ = keep_individuals_based_on_phenotype(ctrl_prefix, ctrl_fam_file, pheno_val_to_keep=ctrl_pheno_value)

    # QC for population stratification (enlever ld 50 0.5 2, cmh, data)
    case_prefix, _ = QC(output_dir, case_prefix, high_ld_regions_file, mind=QC_case_params["mind"], geno=QC_case_params["geno"], maf=QC_case_params["maf"], hwe=QC_case_params["hwe"], min=QC_case_params["min"], hwe_zoom=QC_case_params["hwe_zoom"], dataset_type=QC_case_params["dataset_type"], presence_of_non_auto_chr=QC_case_params["presence_of_non_auto_chr"], fstat_female_threshold=QC_case_params["fstat_female_threshold"], fstat_male_threshold=QC_case_params["fstat_male_threshold"]) # QC for case data
    ctrl_prefix, _ = QC(output_dir, ctrl_prefix, high_ld_regions_file, mind=QC_ctrl_params["mind"], geno=QC_ctrl_params["geno"], maf=QC_ctrl_params["maf"], hwe=QC_ctrl_params["hwe"], min=QC_ctrl_params["min"], hwe_zoom=QC_ctrl_params["hwe_zoom"], dataset_type=QC_ctrl_params["dataset_type"], presence_of_non_auto_chr=QC_ctrl_params["presence_of_non_auto_chr"], fstat_female_threshold=QC_ctrl_params["fstat_female_threshold"], fstat_male_threshold=QC_ctrl_params["fstat_male_threshold"]) # QC for control data
    naf_prefix, _ = QC(output_dir, naf_prefix, high_ld_regions_file, mind=QC_naf_params["mind"], geno=QC_naf_params["geno"], maf=QC_naf_params["maf"], hwe=QC_naf_params["hwe"], min=QC_naf_params["min"], hwe_zoom=QC_naf_params["hwe_zoom"], dataset_type=QC_naf_params["dataset_type"], presence_of_non_auto_chr=QC_naf_params["presence_of_non_auto_chr"], fstat_female_threshold=QC_naf_params["fstat_female_threshold"], fstat_male_threshold=QC_naf_params["fstat_male_threshold"])
    
    # prepare case data for imputation using TopMed
    vcf_file_case = recode_to_vcf(case_prefix)
    vcf_file_naf = recode_to_vcf(naf_prefix)
    vcf_file_ctrl = recode_to_vcf(ctrl_prefix)

    # split vcf into one compressed vcf file per chr
    _ = split_vcf_by_chr(vcf_file_case, n_chr=22)
    _ = split_vcf_by_chr(vcf_file_naf, n_chr=22)
    _ = split_vcf_by_chr(vcf_file_ctrl, n_chr=22)

    ######## MANUAL STEP #######
    # After main1_QC --> perform imputation using TopMed
    # Imputation using TopMed
    # Configuration:
    # array Build: GRCh37/hg19 car data case en 37
    # rsq filter: 0.3 To minimize the file size, Michigan Imputation Server includes a r2 filter option, excluding all imputed SNPs with a r2-value (= imputation quality) smaller then the specified value.
    # Phasing Eagle v2.4 (phased output) Phasing involves separating maternally and paternally inherited copies of each chromosome into haplotypes to get a complete picture of genetic variation. Au lieu d'avoir les variations de la mère + du père sur une même séquence, on sépare en 2 séquence
    # population: vs TopMed Panel
    # Mode: Quality control & Imputation

    # Post imputation:
    # Annotation with rsid of cases data

    ###################################################################
    
    # QC for GWAS - to execute after imputation using Michigan Imputation 
    # TO UNCOMMENT AFTER SNPS IMPUTATION

    # case_prefix, indSNP_case_prefix = QC(output_dir, case_prefix, high_ld_regions_file, mind=0.02, geno=0.02, maf=0.01, hwe=1e-10, min=0.1, hwe_zoom=1e-7, control=False)
    # ctrl_prefix, indSNP_ctrl_prefix = QC(output_dir, ctrl_prefix, high_ld_regions_file, mind=0.02, geno=0.02, maf=0.01, hwe=1e-6, min=0.1, hwe_zoom=1e-3, control=True, presence_of_non_auto_chr=False)
    print("\n")
    
if __name__ == "__main__":
    main()