# HOW TO USE THE ANCESTRY ANALYSIS PIPELINE ?

**Author's note:**

This ancestry analysis pipeline was designed by Julien Paris, phD student in Nantes Université, CR2TI, France, under the responsability of Nicolas Vince, researcher at CR2TI, and Pierre-Antoine Gourraud, PU-PH at Nantes University, CR2TI.

**Requirements:**

- `plink 1.9`
- `plink 2`
- `bcftools 1.13`
- `smartpca.perl` script from eigensoft https://github.com/chrchang/eigensoft/blob/master/bin/smartpca.perl. Don't forget to add the custom path of this script to your `.bashrc`.
- `convertf` binary executable https://github.com/argriffing/eigensoft/tree/master/CONVERTF.
- `python 3.10` or higher.
- `conda environment` with the required packages and their dependencies:
  - pandas 2.1.1
  - numpy 1.24.3
  - matplotlib 3.8.0
  - matplotlib-base 3.8.0
  - scikit-learn 1.3.0
  
## A. Setting working directory

**NOTE:** This pipeline assumes that the quality control for the technology chip / array used were already performed. (For example by using ThermoFisher Axiom Analysis Tool). Make sure you know which is the genome build of each of your datasets.

1. **Create a directory for initial data**

   Within this directory, you need to create several subdirectories:
   - Case directory containing case data before any qc and imputation. It should contain at least the plink dataset constituted of .bed .fam and .bim files, and the 'BestRecommended.txt' file produced by `step0_pretreatment.sh` containing the SNP to keep in the final dataset (the best quality snps). The sex will be lost during pretreatments steps. You can use the sample_QC.txt from technology chip/array quality control steps to reassign samples sex. You can store this file with sex for each samples in a subdirectory of the case directory. The sex will be automatically reassigned during main1_QC_before_imputation.py.
   - Control directory containing control data (plink files) before any qc and imputation. It should contain at least the plink dataset constituted of .bed .fam and .bim files and an additional subdirectory "for_update_sex_fam" containing the .fam file with assigned sex for each individual. Indeed the control fam file doesn't contain the sex of the individuals after the pre-treatment steps and you should re assign it manually.
   - Reference data directory containing a reference dataset, for example from 1000Genomes project. It should contain .bed .fam and .bim files.
   - An additional reference data directory for a reference dataset (plink files) on a population not well represented in reference dataset. For example, a North-African reference dataset. It should contain .bed, .bim and .fam files.
   - A regions to exclude file like `inversion.txt` in the initial data directory to exclude high inversion regions and hla regions. This will be used to created a pruned dataset for ancestry analysis. Example of such a file:
    "6 25500000 33500000 8 HLA
    8 8135000 12000000 Inversion8"

2. **Pre-treatment steps for case data before `main1_QC_before_imputation.py`**
   - Execute `step0_pretreatment.sh` to remove any ADN control, create BestRecommended.txt file. Usages: `./step0_pretreatment.sh <input_vcf_file> <probeQC.txt>`. Both files are from technology quality controls steps.
   - Execute `step1_check_ref.sh` to fix the reference alleles. It uses a fasta file containing a reference genome (.fa) and its index file (.fai). Make sure to take the most similar reference genome (hg37 or hg38 depending on the genome build of your files) to the one used for imputation (TopMed Michigan Imputation Server). Usage: `./step1_check_ref.sh <vcf/bcf file to fixref> <reference.fa> <convert_to_plink_mode_1|0> <set_chr_pos_alt_ref_id_mode_0|1>`. This script can also be used to convert vcf.gz to plink dataset and convert snp ids to chr_pos,alt,ref format*.
   - Replace your plink data files in the initial data case subdirectory by the newly created files after this step.

3. **Pre-treatment steps for control data before `main1_QC_before_imputation.py`**
   - Execute `step1_check_ref.sh` to fix the reference alleles. It uses a fasta file containing a reference genome (.fa) and its index file (.fai). Make sure to take the most similar reference genome (hg37 or hg38 depending on the genome build of your files) to the one used for imputation (TopMed Michigan Imputation Server). Usage: `./step1_check_ref.sh <vcf/bcf file to fixref> <reference.fa> <convert_to_plink_mode_1|0> <set_chr_pos_alt_ref_id_mode_0|1>`. This script can also be used to convert vcf.gz to plink dataset and convert snp ids to chr_pos,alt,ref format*.
   - Replace your plink data files in the initial data case subdirectory by the newly created files after this step.

4. **Pre-treatment for 1Kg reference data**
   - Execute `step1_check_ref.sh` to fix the reference alleles. It uses a fasta file containing a reference genome (.fa) and its index file (.fai). Make sure to take the most similar reference genome (hg37 or hg38 depending on the genome build of your files) to the one used for imputation (TopMed Michigan Imputation Server). Usage: `./step1_check_ref.sh <vcf/bcf file to fixref> <reference.fa> <convert_to_plink_mode_1|0> <set_chr_pos_alt_ref_id_mode_0|1>`. This script can also be used to convert vcf.gz to plink dataset and convert snp ids to chr_pos,alt,ref format*.
   - Replace your plink data files in the initial data case subdirectory by the newly created files after this step.

5. **Pre-treatment for North-African reference data**
   - Execute `step1_check_ref.sh` to fix the reference alleles. It uses a fasta file containing a reference genome (.fa) and its index file (.fai). Make sure to take the most similar reference genome (hg37 or hg38 depending on the genome build of your files) to the one used for imputation (TopMed Michigan Imputation Server). Usage: `./step1_check_ref.sh <vcf/bcf file to fixref> <reference.fa> <convert_to_plink_mode_1|0> <set_chr_pos_alt_ref_id_mode_0|1>`. This script can also be used to convert vcf.gz to plink dataset and convert snp ids to chr_pos,alt,ref format*.
   - Replace your plink data files in the initial data case subdirectory by the newly created files after this step.

***NOTE**: when using `step1_check_ref.sh`, you can add parameters <convert_to_plink_mode_1|0>=1 <set_chr_pos_alt_ref_id_mode_0|1>=1 at the end of the parameters list to convert to plink format and also convert snp ids to chr_pos,alt,ref format.

## B. Quality controls and preprocessing

1. **main1_QC_before_imputation.py**
   - Configure `config_main1_QC_before_imputation.py`
   - Execute the first script using this syntax: `python3 main1_QC_before_imputation.py <data_case_directory> <data_ctrl_directory> <data_North_African_directory> <techno_QC_directory> <high_LD_regions_file> <output_directory>`
   - This script performs quality controls of the control and case dataset.
   - The `<output_directory>` will contain all quality control and processing results.

2. **Imputation**
   - Impute using TopMed Michigan Imputation Server: "https://imputation.biodatacatalyst.nhlbi.nih.gov/#!"
   - Use parameters:
      - Reference panel: Topmedr3
      - rsq filter = 0.3
   - Once the imputation is done use a download script using wget commands provided by TopMed. Download directly on a cluster, since files could be large. The download may take a while. You can find an example of a download script in the root directory of the pipeline.

3. **Post imputation processing**  
A few steps are needed after imputation process (for any imputed dataset):
   - Execute the `main2_preprocessing.sh` script from the root of `main2_preprocessing_all_steps` directory using this syntax `./main2_preprocessing.sh <input_dir_containing_zip_data_by_chr -p <zip_topmed_password> -maf <value> -r2 <value> -ctrl <mode_0|1> -o <output_prefix>`. This script will post process data after topmed imputation, filter for Minor Allele Frequency (MAF), R2 (imputation quality) and HWE (1e-6 for ctrl data (if -ctrl 1), 1e-10 for case data (if -ctrl 0)). This script calls another script called `update_geno_remove_dup_snps_part2_main2_preprocessing.sh` which will remode duplicated variant IDs and update missing genotypes. Indeed, topmed may induce duplicated ids with swapped alleles and thus may impute one of the variants and keep the typed variants with missing values. The aim here is to update the missing and non-imputed genotypes of the typed variant with the imputed genotypes () of the corresponding swapped imputed variant. To update the missing genotypes in the typed variants, the genotypes of the imputed variants are swapped.
   - We recommend using a less stringent maf threshold (for example 0.01) at first and then filter again using an adequate maf value depending on your analysis. You can use the following plink command to do so `plink --bfile <input_prefix> --maf <value> --make-bed --out <output_prefix>`

## C. Ancestry analysis

1. **Prepare data for main2_ancestry_post_imputation_analysis.py**

- For ancestry analysis, you should filter again your dataset with a maf value of 0.1 if you used a lower maf threshold during post imputation processing. You can use this command: `plink --bfile <input_prefix> --maf 0.1 --make-bed --out <output_prefix>`.

- Prepare the `data_for_main2` directory. This directory should contain several subdirectories and files:
  - `data_case_post_imputed_QC` subdirectory containing the plink fileset (.bim, .fam and .bed) of the preprocessed and imputed case dataset.
  - `data_ctrl_imputed_post_QC` subdirectory containing the plink fileset (.bim, .fam and .bed) of the preprocessed and imputed control dataset.
  - `data_ref` subdirectory containing the plink fileset (.bim, .fam and .bed) of the preprocessed non-imputed 1000Genomes reference dataset.
  - `data_NAF` subdirectory containing the plink fileset (.bim, .fam and .bed) of the preprocessed non-imputed North-African reference dataset.
  - `for_update_sex` subdirectory containing the original plink .fam files for 1000Genomes, North-African, control and case datasets. Those .fam should contain the information about the sex of each individual (a value != -9 indicating missing sex).
  - `geographic_origins.csv` file containing 2 coma delimited columns: Population, n. Population refers to self-reported ancestry and n refers to the number of individuals associated to each of the self-reported origins.
  - `inversion.txt` file containing high inversion and hla regions to exclude. This will be used to created a pruned dataset for ancestry analysis. Example of such a file:
  
  6 25500000 33500000 8 HLA

  8 8135000 12000000 Inversion8

2. **main2_ancestry_post_imputation_analysis.py**

- Configure `config_main2_ancestry.py`.
- Execute `main2_ancestry_post_imputation_analysis.py` using this syntax `python3 main2_ancestry_post_imputation_analysis.py <data_case_post_imputed_QC_dirpath> <data_ctrl_imputed_post_QC_dirpath> <data_ref_dirpath> <data_NAF_dirpath> <original_case_fam_file_with_sex_path> <original_ctrl_fam_file_with_sex_path> <original_1kg_reference_fam_file_with_sex_path> <original_NAF_reference_fam_file_with_sex_path> <geographic_origins_csv_path> <inversion_txt_path> <output_directory>`.
- This script performs ancestry analysis using Principal Component Analysis (PCA) and Kmeans clustering.
- The `output_directory` will contain all ancestry analysis results.

## D. Admixture analysis

1. **Prepare data for main3_admixed_ancestry_analysis.py**

- Prepare the `data_for_main3` directory. This directory should contains several files:
  
  - Plink fileset (.bim, .fam and .bed) from the output of `main2_ancestry_post_imputation_analysis.py`. If you used outlier removals (set in the `config_main2_ancestry.py`), use the merged dataset from the step before the PCA so every individuals are used to determine admixed ancestry.
  - The `igsr_samples.tsv file` from the 1000Genomes project. This file contains the population and superpopulation ancestry for each of the 1000Genomes individuals. Download it from https://www.internationalgenome.org/data/.

2. **main3_admixed_ancestry_analysis.py**

- Configure `config_main3_admixed_ancestry_analysis.py`.
- Execute `main3_admixed_ancestry_analysis.py` using this syntax: `python3 main3_admixed_ancestry_analysis.py <input_plink_prefix> <igsr_samples_tsv_file> <output_directory>`.
- This script performs admixture analysis using predefined reference populations. We recommend using only the least admixed reference population like East_Asian, European and African reference ancestry. Indeed, South_Asian, North-African and even American reference populations are admixed by design and thus are not suitable to be used as reference population for admixture analysis.
- The `output_directory` will contain all admixture analysis results.

## E. Analysis summary

- Configure the `parameters_file_main4.txt` and add the correct paths to each of the parameters.
- Execute `main4_summary_table.py` using this syntax: `python3 main4_summary_table.py <parameters_file> <output_directory>`.
- The resulting `output_directory` will contain all summary tables describing the analysis.
