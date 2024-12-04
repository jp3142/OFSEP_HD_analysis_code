#!/usr/bin/env python3

import sys
from modules.qc import *
from modules.ancestry2 import *
from config_main2_ancestry import *

# script ancestry analysis - post imputation analysis

##############################################################################
# MANUAL STEP TO DO BEFORE HAND - main2_preprocessing.sh (automated)
    # check which is allele ref and allele alt (plink: A1=alt=minor, A2=ref=major, VCF c'est l'inverse par défaut)
    # gunzip vcf_file and grep alt or ref and see if it's alt/ref or ref/alt (PLINK = ALT (minor) REF (major), VCF = REF ALT)

    #  keep probability of each allele (dosage) and rmdup
    #example: plink2 --vcf ../DATA_IMP-GENONLY_QC_rsID/CKiD_138kids_Hg38_Imputed-Genonly_MAF03_rsID_noAT-noGC_nodupl_sort.VERSION10102022.vcf.gz 'dosage=HDS' --rm-dup 'force-first' --export vcf 'vcf-dosage=HDS' --out CKiD_138kids_Hg38_Imputed-Genonly_MAF03_rsID_noAT-noGC_nodupl_sort_uniq.VERSION10102022
    
    # force allele 2 as ref and transform file to .gen (.gen contains dosage) mais (quand on fait cela sur plink l'allele 2 est le ref et l'allele 1 et le ALT donc il faut force l'allele 2 comme ref
    # example: plink2 --vcf CKiD_138kids_Hg38_Imputed-Genonly_MAF03_25072023.vcf.gz 'dosage=DS' --ref-allele 'force' CKiD_138kids_Hg38_Imputed-Genonly_MAF03_25072023.vcf 5 3 '#' --export oxford --out CKiD_138kids_Hg38_Imputed-Genonly_MAF03_25072023

    # convert vcf.gz to plink format (imputed dataset = case dataset)
    # bcftools concat -f merge.list.txt -Oz -o -> merge.list.txt = list of chr to merge into one dataset
    # plink --vcf file.vcf.gz (--const-fid) --make-bed --out output_file_prefix
    
    # replace 2 first characters 0_ of column 2 -> awk -i inplace '{sub("0_", " ", $2); print $0}' file_case_imputed_annotated.fam 
    # (pour NAF faire en sorte d'avoir 6 colonne avec FID North Africa)
    # Remplacer rsid par id de type chr_position sur jeux de ref, cas et ctrl, et naf

    # remove duplicated variants from dataset -> plink2 !
    #  plink2 --bfile data_case_allchr_concat_removed_21SUJ110291 --rm-dup force-first --make-bed --out data_case_allchr_concat_removed_21SUJ110291_rmdup
##############################################################################

def main():
    # """EXAMPLE exec: python3 main2_ancestry_post_imputation_analysis.py data_for_main2/data_case_imputed_HG38/ data_for_main2/data_ctrl_already_imputed/ data_for_main2/data_ref_1000_genome_HG38_already_imputed/ data_for_main2/for_UPDATE_SEX_CASE_FAM/sample_QC.txt data_for_main2/data_ref_1000_genome_HG38_already_imputed/1kg_list.txt data_for_main2/inversion.txt main2_all_chr_output_2D_3D_plt_kdefull_alphagood_ctrlabove/"""
    
    ###### PARAMETERS ######
    try:
        # all data are imputed data
        data_case_dir_path = sys.argv[1] 
        data_ctrl_dir_path = sys.argv[2]
        data_ref_dir_path = sys.argv[3]
        data_naf_dir_path = sys.argv[4]
        previous_topmed_case_fam_file = sys.argv[5]
        previous_ctrl_fam_file = sys.argv[6]
        previous_ref_fam_file = sys.argv[7]
        previous_naf_fam_file = sys.argv[8]
        kg_list_subpop = sys.argv[9]
        high_ld_regions_file = sys.argv[10]
        output_dir = sys.argv[11]
    except IndexError:
        sys.stderr.write("[ArgumentError] Usage: main_2_ancestry_post_imputation_analysis.py <data_case_vcf_gz_path> <data_ctrl_dir_path> <data_ref_dir_path> <sample_techno_QC_file> <reference_pop_list_with_superpop> <output_dir>")
        exit(0)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # get bed, bim, fam file path for each dataset
    case_bed_file, case_bim_file, case_fam_file = get_bed_bim_fam_files(data_case_dir_path)
    ctrl_bed_file, ctrl_bim_file, ctrl_fam_file = get_bed_bim_fam_files(data_ctrl_dir_path)
    ref_bed_file, ref_bim_file, ref_fam_file = get_bed_bim_fam_files(data_ref_dir_path)
    naf_bed_file, naf_bim_file, naf_fam_file = get_bed_bim_fam_files(data_naf_dir_path)

    # update sex case_fam_file
    if UPDATE_SEX_CASE_FAM:
        update_sex_from_previous_dataset(previous_topmed_case_fam_file, case_fam_file)
    
    if UPDATE_SEX_CTRL_FAM:
        update_sex_from_previous_dataset(previous_ctrl_fam_file, ctrl_fam_file)

    if UPDATE_SEX_REF_FAM:
        update_sex_from_previous_dataset(previous_ref_fam_file, ref_fam_file)
    
    if UPDATE_SEX_NAF_FAM:
        update_sex_from_previous_dataset(previous_naf_fam_file, naf_fam_file)

    if UPDATE_PHENO_CASE_FAM:
        update_pheno_fam_file(case_fam_file, 2)

    if UPDATE_PHENO_CTRL_FAM:
        update_pheno_fam_file(ctrl_fam_file, 1)

    # get case, ctrl and ref file prefix
    case_prefix = get_bed_bim_fam_prefix(case_bed_file, case_bim_file, case_fam_file)
    ctrl_prefix = get_bed_bim_fam_prefix(ctrl_bed_file, ctrl_bim_file, ctrl_fam_file)
    ref_prefix = get_bed_bim_fam_prefix(ref_bed_file, ref_bim_file, ref_fam_file)
    naf_prefix = get_bed_bim_fam_prefix(naf_bed_file, naf_bim_file, naf_fam_file)

    # replace specific origins of naf individuals by North Africa
    if REPLACE_FID_FAM_NAF:
        naf_prefix = replace_fid_fam_by(naf_prefix, "North_Africa")

    # merge case control and 1000 genomes
    merged_prefix = merge_case_ctrl_and_ref_datasets(case_prefix, ctrl_prefix, ref_prefix, naf_prefix, output_dir) # normalement que pruned snp ici car déjà pruned dans référence dataset

    # update ids 
    update_ids_txt = create_update_ids_list(output_dir, merged_prefix, kg_list_subpop)
    merged_updated_ids_prefix = update_ids(merged_prefix, update_ids_txt)

    # regénérer le dataset pruned snp (au cas où certains snp en LD restants mais normalement non)
    merged_pruned_prefix = keep_indep_snps(output_dir, merged_updated_ids_prefix, high_ld_regions_file)
    
    # create list of finnish individuals in order to exclude them
    indiv_to_exclude_file = create_list_fin_indiv(kg_list_subpop, output_dir)

    # remove finnish individuals
    merged_pruned_prefix_no_fin = remove_individuals(indiv_to_exclude_file, merged_pruned_prefix)

    # remove AT GC snps for ancestry analysis
    merged_pruned_prefix = remove_AT_GC_snps(merged_pruned_prefix_no_fin)
    
    # # PCA for population stratification    
    #merged_pruned_prefix_no_fin=os.path.join(output_dir, "merged_pruned_dataset_removedInd_removedFreqDiffSnps0.1_noATGC")
    pca_output_prefix, map_file = eigensoft_smart_pca(output_dir, merged_pruned_prefix, n_pc=N_PC, n_pc_outliers=N_PC_OUTLIERS, outlier_mode=OUTLIER_MODE, n_outliers_iter=N_OUTLIERS_ITER)

    # # ##### FOR TESTING ###
    # it is possible that smartpca.map file doesn't exist if eigensoft didn't generate it. In this case you should use .map from previous step. 
    # The script renames eigen_file_prefix.map to "smartpca.map".
    # In this case uncomment the following block of code below
    #############
    # pca_output_prefix = os.path.join(output_dir, "merged_pruned_dataset_removedInd_removedFreqDiffSnps0.1_noATGC") 
    # get number of snps
    # if pca_output_prefix.split('/')[-1]+".map" in os.listdir(output_dir):
    # nb_snps = get_number_of_snps(pca_output_prefix+".map")
    # else:
    #     print(f"[DevError] Variable 'pca_output_prefix'.map = '{pca_output_prefix}.map' doesn't exist. Use previous step map file. Error due to the use of eigensoft - behavior not always predictable.")
    #     exit(6)
    #############
    
    # get nb snps and compute explained variance for each PC
    nb_snps = get_number_of_snps(map_file)
    explained_var = compute_explained_variance(pca_output_prefix, output_dir, output_file_name="pca_explained_variance.csv")
    
    # get eigenvectors
    eigenvec_df = get_eigenvec(pca_output_prefix)

    ####### PLOTING #######

    # for PC1 and PC2

    # plot ref without naf (set xlim and ylim )
    xlim, ylim = plot_pca_2d(output_dir, 
                             eigenvec_df, 
                             explained_var, 
                             pca_plot_name="ref_only_no_naf_PCA_PC1_PC2_ancestry.pdf", 
                             plot_title="1000 genomes reference population",
                             number_of_snps=nb_snps, 
                             ref_colors_dict=REF_COLS_DICT, 
                             pc_list=[1, 2], 
                             alpha=ALPHA_REF, 
                             ctrl_case_only=False, 
                             ref_only=True, 
                             plot_naf=False, 
                             loc_legend="upper right")
    
    # plot ref with naf
    _, _ = plot_pca_2d(output_dir, 
                             eigenvec_df, 
                             explained_var, 
                             pca_plot_name="ref_only_with_naf_PCA_PC1_PC2_ancestry.pdf", 
                             plot_title="1000 genomes and North-african reference populations",
                             number_of_snps=nb_snps, 
                             ref_colors_dict=REF_COLS_DICT, pc_list=[1, 2], 
                             alpha=ALPHA_REF, 
                             ctrl_case_only=False, 
                             ref_only=True, 
                             plot_naf=True, 
                             loc_legend="upper right")

    # plot case - ctrl non clustered without ref (case above)
    _, _ = plot_pca_2d(output_dir, 
                       eigenvec_df, 
                       explained_var, 
                       pca_plot_name="ctrl_case_only_PCA_PC1_PC2_ancestry.pdf", 
                       plot_title="Ctrl and OFSEP-HD populations",
                       number_of_snps=nb_snps, 
                       case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS,
                       pc_list=[1, 2], 
                       xlim=xlim, 
                       ylim=ylim, 
                       ctrl_case_only=True, 
                       plot_ctrl_above_case=False, 
                       loc_legend="upper left")
    
    # plot case - ctrl non clustered without ref (ctrl above)
    _, _ = plot_pca_2d(output_dir, 
                       eigenvec_df, 
                       explained_var, 
                       pca_plot_name="ctrl_case_only_PCA_PC1_PC2_ancestry_ctrl_above.pdf", 
                       plot_title="Ctrl and OFSEP-HD populations",
                       number_of_snps=nb_snps, 
                       case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS,
                       pc_list=[1, 2], 
                       xlim=xlim, 
                       ylim=ylim, 
                       ctrl_case_only=True, 
                       plot_ctrl_above_case=True, 
                       loc_legend="upper left")    
                                            
    # define number of clusters in case_ctrl dataset
    find_nb_clusters_elbow_method(eigenvec_df, output_dir, output_file="ctrl_case_define_number_of_clusters_elbow_method.pdf", pc_list=[1, 2], max_nb_clusters=10)
    scores = find_nb_clusters_silhouette_score(eigenvec_df, pc_list=[1, 2], max_nb_clusters=10)
    _ = write_silhouette_scores(scores, output_dir)

    # clustering configuration/correction and ploting
    labelled_clusters_case_ctrl, cluster_labels, cluster_centers = clustering_case_ctrl(eigenvec_df, n_clusters=N_CLUSTERS, pc_list=[1, 2], homemade_cluster_barycenters=HOMEMADE_CLUSTER_CENTERS, cluster_to_modify=CLUSTER_TO_MODIFY)
    n_clusters = len(cluster_centers)
    labelled_clusters_case_ctrl.to_csv(os.path.join(output_dir,"labelled_clusters_case_ctrl.csv"), sep=',') # keep assigned cluster for each individual
    
    # print cluster labels number and associated colors to know which population is which cluster
    print("Summary clustering:")
    print("cluster labels", cluster_labels)
    print("case colors", CASE_COLORS_CLUSTERS)
    print("ctrl colors", CTRL_COLORS_CLUSTERS)
    print("n clusters", n_clusters)
    # write clustering summary to file
    with open(os.path.join(output_dir, "clustering_info.txt"), "w") as f:
        f.write("Cluster barycenters:\n")
        for i, elm in enumerate(cluster_centers):
            f.write(str(i)+" "+str(elm[0])+" "+str(elm[1])+"\n") 

    # plot case ctrl clustered only
    plot_case_ctrl_ref_clustered_data(labelled_clusters_case_ctrl, 
                                      cluster_labels, 
                                      cluster_centers, 
                                      output_dir, 
                                      "ctrl_case_KMeans_clustering.pdf", 
                                      nb_snps, 
                                      explained_var, 
                                      ctrl_colors=CTRL_COLORS_CLUSTERS, 
                                      case_colors=CASE_COLORS_CLUSTERS, 
                                      pc_list=[1, 2], 
                                      xlim=xlim, 
                                      ylim=ylim, 
                                      plot_title="KMeans clustering of OFSEP-HD and Ctrl populations", 
                                      loc_legend="upper right",
                                      plot_case=True,
                                      plot_ctrl=True,
                                      plot_ref=False,
                                      plot_naf=False)

    # plot case clustered only
    plot_case_ctrl_ref_clustered_data(labelled_clusters_case_ctrl, 
                                      cluster_labels, 
                                      cluster_centers, 
                                      output_dir, 
                                      "case_only_KMeans_clustering.pdf",
                                      nb_snps, 
                                      explained_var, 
                                      case_colors=CASE_COLORS_CLUSTERS, 
                                      pc_list=[1, 2], 
                                      xlim=xlim, 
                                      ylim=ylim, 
                                      plot_title="KMeans clustering of OFSEP-HD population", 
                                      loc_legend="upper right", 
                                      plot_case=True,
                                      plot_ctrl=False,
                                      plot_naf=False,
                                      plot_ref=False)
    
    # plot ctrl clustered only
    plot_case_ctrl_ref_clustered_data(labelled_clusters_case_ctrl, 
                                      cluster_labels, 
                                      cluster_centers, 
                                      output_dir, 
                                      "ctrl_only_KMeans_clustering.pdf",
                                      nb_snps, 
                                      explained_var, 
                                      ctrl_colors=CTRL_COLORS_CLUSTERS, 
                                      pc_list=[1, 2], 
                                      xlim=xlim, 
                                      ylim=ylim, 
                                      plot_title="KMeans clustering of Ctrl population", 
                                      loc_legend="upper right", 
                                      plot_ctrl=True,
                                      plot_case=False,
                                      plot_naf=False,
                                      plot_ref=False)

     
    # plot ref without naf + cas clustered
    plot_case_ctrl_ref_clustered_data(labelled_clusters_case_ctrl,
                                      cluster_labels,
                                      cluster_centers,
                                      output_dir,
                                      "case_ref_KMeans_clustering_noNAF.pdf",
                                      nb_snps,
                                      explained_var,
                                      alpha=ALPHA_REF,
                                      eigenvec=eigenvec_df,
                                      case_colors=CASE_COLORS_CLUSTERS,
                                      ref_colors_dict=REF_COLS_DICT,
                                      pc_list=[1, 2],
                                      xlim=xlim,
                                      ylim=ylim,
                                      plot_title="Clustered OFSEP-HD population and 1000 genomes reference population",
                                      loc_legend="upper right",
                                      plot_case=True,
                                      plot_ctrl=False,
                                      plot_naf=False,
                                      plot_ref=True)

    # plot ref without naf + ctrl clustered
    plot_case_ctrl_ref_clustered_data(labelled_clusters_case_ctrl,
                                      cluster_labels,
                                      cluster_centers,
                                      output_dir,
                                      "ctrl_ref_KMeans_clustering_noNAF.pdf",
                                      nb_snps,
                                      explained_var,
                                      alpha=ALPHA_REF,
                                      eigenvec=eigenvec_df,
                                      ctrl_colors=CTRL_COLORS_CLUSTERS,
                                      ref_colors_dict=REF_COLS_DICT,
                                      pc_list=[1, 2],
                                      xlim=xlim,
                                      ylim=ylim,
                                      plot_title="Clustered Ctrl population and 1000 genomes reference population",
                                      plot_case=False,
                                      plot_ctrl=True,
                                      plot_naf=False,
                                      plot_ref=True,
                                      loc_legend="upper right")
    
    # plot ref with naf + cas clustered
    plot_case_ctrl_ref_clustered_data(labelled_clusters_case_ctrl,
                                      cluster_labels,
                                      cluster_centers,
                                      output_dir,
                                      "case_ref_KMeans_clustering_NAF.pdf",
                                      nb_snps,
                                      explained_var,
                                      alpha=ALPHA_REF,
                                      eigenvec=eigenvec_df,
                                      case_colors=CASE_COLORS_CLUSTERS,
                                      ref_colors_dict=REF_COLS_DICT,
                                      pc_list=[1, 2],
                                      xlim=xlim,
                                      ylim=ylim,
                                      plot_title="Clustered OFSEP-HD population with 1000 genomes and North-african reference populations",
                                      plot_case=True,
                                      plot_ctrl=False,
                                      plot_naf=True,
                                      plot_ref=True,
                                      loc_legend="upper right")

    # plot ref avec naf + ctrl clustered
    plot_case_ctrl_ref_clustered_data(labelled_clusters_case_ctrl,
                                      cluster_labels,
                                      cluster_centers,
                                      output_dir,
                                      "ctrl_ref_KMeans_clustering_NAF.pdf",
                                      nb_snps,
                                      explained_var,
                                      alpha=ALPHA_REF,
                                      eigenvec=eigenvec_df,
                                      ctrl_colors=CTRL_COLORS_CLUSTERS,
                                      ref_colors_dict=REF_COLS_DICT,
                                      pc_list=[1, 2],
                                      xlim=xlim,
                                      ylim=ylim,
                                      plot_title="Clustered Ctrl population with 1000 genomes and North-african reference populations",
                                      plot_case=False,
                                      plot_ctrl=True,
                                      plot_naf=True,
                                      plot_ref=True,
                                      loc_legend="upper right")
    
    # plot ref naf ctrl case clustered
    plot_case_ctrl_ref_clustered_data(labelled_clusters_case_ctrl,
                                      cluster_labels,
                                      cluster_centers,
                                      output_dir,
                                      "ctrl_case_ref_KMeans_clustering_NAF.pdf",
                                      nb_snps,
                                      explained_var,
                                      alpha=ALPHA_REF,
                                      eigenvec=eigenvec_df,
                                      ctrl_colors=CTRL_COLORS_CLUSTERS,
                                      case_colors=CASE_COLORS_CLUSTERS, 
                                      ref_colors_dict=REF_COLS_DICT,
                                      pc_list=[1, 2],
                                      xlim=xlim,
                                      ylim=ylim,
                                      plot_title="Clustered Ctrl and OFSEP-HD populations with 1000 genomes and North-african reference populations",
                                      plot_case=True,
                                      plot_ctrl=True,
                                      plot_naf=True,
                                      plot_ref=True,
                                      loc_legend="upper right")
    
    # PC1 - PC3
    # plot ctrl case + ref avec naf
    xlim, ylim = plot_pca_2d(output_dir, 
                       eigenvec_df, 
                       explained_var, 
                       pca_plot_name="ctrl_case_ref_PCA_PC1_PC3_ancestry.pdf", 
                       plot_title="Ctrl and OFSEP-HD populations with 1000 genome and North-African reference populations",
                       number_of_snps=nb_snps, 
                       case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS,
                       ref_colors_dict=REF_COLS_DICT,
                       pc_list=[1, 3], 
                       alpha=ALPHA_REF,
                       ctrl_case_only=False, 
                       plot_ctrl_above_case=False, 
                       plot_naf=True,
                       loc_legend="upper right")

    # plot ctrl case only
    _, _ = plot_pca_2d(output_dir, 
                       eigenvec_df, 
                       explained_var, 
                       pca_plot_name="ctrl_case_only_PCA_PC1_PC3_ancestry.pdf", 
                       plot_title="Ctrl and OFSEP-HD populations",
                       number_of_snps=nb_snps, 
                       case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS,
                       pc_list=[1, 3], 
                       xlim=xlim, 
                       ylim=ylim, 
                       ctrl_case_only=True, 
                       plot_ctrl_above_case=False, 
                       loc_legend="lower left",
                       plot_naf=False)


    # PC2 PC3
    # plot ctrl case + ref with naf
    xlim, ylim = plot_pca_2d(output_dir, 
                       eigenvec_df, 
                       explained_var, 
                       pca_plot_name="ctrl_case_ref_PCA_PC2_PC3_ancestry.pdf", 
                       plot_title="Ctrl and OFSEP-HD populations with 1000 genome and North-African reference populations",
                       number_of_snps=nb_snps, 
                       case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS,
                       ref_colors_dict=REF_COLS_DICT,
                       alpha=ALPHA_REF,
                       pc_list=[2, 3], 
                       ctrl_case_only=False, 
                       plot_ctrl_above_case=False,
                       plot_naf=True,
                       loc_legend="upper right")
    
    # plot ctrl case + ref with naf
    _, _ = plot_pca_2d(output_dir, 
                       eigenvec_df, 
                       explained_var, 
                       pca_plot_name="ctrl_case_only_PCA_PC2_PC3_ancestry.pdf", 
                       plot_title="Ctrl and OFSEP-HD populations",
                       number_of_snps=nb_snps, 
                       case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS,
                       pc_list=[2, 3], 
                       xlim=xlim, 
                       ylim=ylim, 
                       ctrl_case_only=True, 
                       plot_ctrl_above_case=False, 
                       loc_legend="lower left",
                       plot_naf=False)

    # 3D

    # plot ctrl case + ref avec naf
    xlim, ylim, zlim = plot_pca_3d(output_dir, 
                eigenvec_df, 
                explained_var, 
                "ctrl_case_ref_3D_PCA_NAF.pdf", 
                nb_snps, 
                plot_title="Ctrl and OFSEP-HD populations with 1000 genomes and North-african reference populations",
                ref_colors_dict=REF_COLS_DICT,
                case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS, 
                alpha=ALPHA_REF,
                ctrl_case_only=False,
                ref_only=False,
                plot_ctrl_above_case=False,
                plot_naf=True,
                loc_legend="upper left")
    
    _, _, _ = plot_pca_3d(output_dir, 
                eigenvec_df, 
                explained_var, 
                "ctrl_case_only_3D_PCA.pdf", 
                nb_snps, 
                plot_title="Ctrl and OFSEP-HD populations",
                ref_colors_dict=REF_COLS_DICT,
                case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS, 
                ctrl_case_only=True,
                ref_only=False,
                plot_ctrl_above_case=False,
                plot_naf=False,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                loc_legend="upper left")
    
    _, _, _ = plot_pca_3d(output_dir, 
                eigenvec_df, 
                explained_var, 
                "ctrl_case_ref_3D_PCA_noNAF.pdf", 
                nb_snps, 
                plot_title="Ctrl and OFSEP-HD populations with 1000 genomes reference populations",
                ref_colors_dict=REF_COLS_DICT,
                case_ctrl_colors=NON_CLUSTERED_CASE_CTRL_COLORS, 
                ctrl_case_only=False,
                ref_only=False,
                alpha=ALPHA_REF,
                plot_ctrl_above_case=False,
                plot_naf=False,
                xlim=xlim,
                ylim=ylim,
                zlim=zlim,
                loc_legend="upper left")
    
    print("[END] Step2 ancestry analysis. DONE.")

if __name__ == "__main__":
    main()
