#!/usr/bin/env python3

# admixed ancestry plot superpopulation colors matching
custom_colors = {
"Europe": "blue",
"America": "orange",
"Africa": "red",
"East_Asia": "yellow",
"South_Asia": "green",
"North_Africa": "purple",
}

case_colors_clusters_plt_names = {
    "European cluster": "slateblue",
    "Afro-Caribbean cluster": "mediumvioletred",
    "North-African cluster": "magenta",
    "East-Asian cluster": "#e8f332",
    "African cluster": "indianred",
    "South-Asian cluster": "lime",
    "Admixed European / North-African cluster": "rebeccapurple",
}

estimated_clusters_mapping_plt_names = { # the one to use for plotting
        0: "European cluster",
        1: "Afro-Caribbean cluster",
        2: "North-African cluster",
        3: "East-Asian cluster",
        4: "African cluster",
        5: "South-Asian cluster",
        6: "Admixed European / North-African cluster",
        7: "unknown",
}

estimated_clusters_mapping = {
        0: "CloserToEuropeanReferenceCluster",
        1: "InBetweenNorthAfricanAndAfricanAndAfrocaribbeanReferenceClusters",
        2: "CloserToNorthAfricanReferenceCluster",
        3: "CloserToEastAsianReferenceCluster",
        4: "CloserToAfricanAndAfrocaribbeanReferenceCluster",
        5: "CloserToSouthAsianReferenceCluster",
        6: "InBetweenNorthAfricanAndEuropeanReferenceClusters",
        7: "unknown",
} # used in statistical analysis also

# uncomment one line for fast exec (one iteration)
# matching between each k value and the superpopulations considered as reference
corres_k_superpop = {
    3: ["East_Asia", "Africa", "Europe"]
    #4: ["Africa", "East_Asia", "South_Asia", "Europe"],
    #5: ["America", "Africa", "East_Asia", "South_Asia", "Europe"],
    #6: ["America", "Africa", "East_Asia", "South_Asia", "Europe", "North_Africa"] 
} # dict: k:superpop_list

# matching between all subpop codes in 1kg project and superpopulations
matching_superpop_subpop = {
    "Africa": ["YRI", "LWK", "GWD", "MSL", "ESN", "ASW", "ACB"],
    "East_Asia": ["CHB", "JPT", "CHS", "CDX", "KHV", "CHD"],
    "Europe": ["CEU", "TSI", "GBR", "FIN", "IBS", "IBS,MSL"]
    #South_Asia: [],
    #America: [],
}

# subpopulations codes to remove (leading to noise in admixed ancestry analysis)
subpop_code_to_remove = ["ASW", "ACB", "FIN", "IBS,MSL"]

# how to sort admixed ancestry plot within each superpopulation
sort_by="Africa"
