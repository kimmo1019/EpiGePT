import numpy as np 
import sys, os 
import gzip
import pandas as pd



def merge_rseq_reps(cell_id, id2name, data_path="data/encode/raw_data"):
    """Merge gene expression across replicates give one cell type or tissue id.
    Args:
        cell_id: ID for a celltype or tissue. E.g., 1
        id2name: dic mapping gene id to gene name
        data_path: path to data
    Return:
        A dic map gene name to its TPM value.
    """
    num_reps = 0
    name2tpm = {}
    for exp in os.listdir('%s/%d/rseq' % (data_path, cell_id)):
        for rseq_file in os.listdir('%s/%d/rseq/%s' % (data_path, cell_id, exp)):
            if rseq_file[-3:] == "tsv":
                tsv_file = '%s/%d/rseq/%s/%s' % (data_path, cell_id, exp, rseq_file)
                for line in open(tsv_file).readlines()[1:]:
                    gene_id = line.split('\t')[0].split('.')[0]
                    if gene_id not in id2name.keys():
                        continue
                    tpm_value = float(line.split('\t')[5])
                    gene_name = id2name[gene_id]
                    if gene_name in name2tpm.keys():
                        name2tpm[gene_name].append(tpm_value)
                    else:
                        name2tpm[gene_name] = [tpm_value]
                num_reps += 1

    for item in name2tpm.items():
        if len(item[1]) != num_reps:
            pass
            #print(item[0],len(item[1]))

    average_tpm = {item[0]:np.mean(item[1]) for item in name2tpm.items()}
    return average_tpm

def get_cell_ids(num_celltypes=39):
    """get valid cell ids by removing cell types with missing data.
    Return:
        A cell id list.
    """
    missing_ids = [8,23,25,30,32,33,34,35,38,39,17]
    return [item for item in list(range(1,num_celltypes+1)) if item not in missing_ids]
    
def get_gene_annot(gene_annot_file='data/encode/gencode.v19.annotation.gtf.gz'):
    """ get gene annotation from a annotation file.
    Arg:
        cell_id: ID for a celltype or tissue. E.g., 1
        gene_annot_file: gene annotation file from ENCODE 
            URL: https://www.encodeproject.org/files/gencode.v19.annotation/@@download/gencode.v19.annotation.gtf.gz
            FORMAT: https://www.gencodegenes.org/pages/data_format.html
    Returns:
        Two dics
            First dic mapping gene id (ENSGXXXXXXXXXXX) to gene name;
            Second dic mapping gene name to gene location.
    """
    id2name = {}
    gene2loc = {}
    lines = gzip.open(gene_annot_file).readlines()
    for line in lines:
        if line.decode("utf-8").startswith('#'):
            continue
        trans_type = line.decode("utf-8").split('\t')[-1].strip().split('; ')[5].split(' ')[1].strip('"').split('.')[0]
        gene_type = line.decode("utf-8").split('\t')[-1].strip().split('; ')[2].split(' ')[1].strip('"').split('.')[0]
        if trans_type != 'protein_coding' or gene_type != 'protein_coding':
            continue
        if line.decode("utf-8").split('\t')[2] != 'gene':
            continue
        chrom = line.decode("utf-8").split('\t')[0]
        start = line.decode("utf-8").split('\t')[3]
        end = line.decode("utf-8").split('\t')[4]
        loc = chrom + ':' + start + '-' + end
        gene_id = line.decode("utf-8").split('\t')[-1].strip().split('; ')[0].split(' ')[1].strip('"').split('.')[0]
        gene_name = line.decode("utf-8").split('\t')[-1].strip().split('; ')[4].split(' ')[1].strip('"')
        id2name[gene_id] = gene_name
        gene2loc[gene_name] = loc
    #NR2E3 and ZSCAN26 not protein coding, but associated with motif
    #We manually add the two genes.
    id2name['ENSG00000031544'] = 'NR2E3'
    gene2loc['NR2E3'] = 'chr15:72084977-72110600'
    id2name['ENSG00000197062'] = 'ZSCAN26'
    gene2loc['ZSCAN26'] = 'chr6:28234788-28245974'
    return id2name, gene2loc

    
def get_tf():
    """need to be improved, generate a two-column file, motif-tf
    Return:
        A list of TF(gene) name (711 TFs).
    """
    from scipy.io import loadmat
    motif_info = loadmat('data/motif/MotifMatch_human_rmdup.mat') 
    motif_list = [item.split('\t')[1] for item in open('data/motif/all_motif_rmdup.motif').readlines() \
        if item.startswith('>') ]
    motif2tf = {}
    for each in motif_info['Match2']:
        motif = each[0][0]
        tf = each[1][0]
        if motif not in motif2tf.keys():
            motif2tf[motif] = [tf]
        else:
            motif2tf[motif]+=[tf]
    motif_list_filtered = [item for item in motif2tf.keys() if item in motif_list]
    tf_list = []
    for each in motif_list_filtered:
        tf_list += motif2tf[each]
    tf_list = list(set(tf_list))
    tf_list.sort()
    return tf_list


def quantile_norm(df):
    """quantile normalization
    Arg:
        df: Pandas DataFrame with genes x cells
    Return:
        normalized Pandas DataFrame 
    """
    rank_mean = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
    return df.rank(method='min').stack().astype(int).map(rank_mean).unstack()

def get_sorted_gene(gene2loc):
    """sorted genes according to the genomic coordinate
    Return:
        A list of sorted genes.
    """
    chrom2index = {'chr%d' % item : item for item in range(1,23)}
    chrom2index['chrX'] = 23
    chrom2index['chrY'] = 24
    chrom2index['chrM'] = 25

    locs = [[item[0], chrom2index[item[1].split(':')[0]], int(item[1].split(':')[1].split('-')[0]), 
            int(item[1].split('-')[1])] for item in gene2loc.items()]
    locs.sort(key = lambda x : (x[1],x[2]))
    return [item[0] for item in locs]



def aggregate_rseq(cell_ids, id2name, gene2loc, tf_list, use_norm = True, save = True):
    """aggregate RNA-seq data across all filtered cell types/tissues.
    generate full with gene location, generate only tf, save two panda files.
    """
    sorted_genes = get_sorted_gene(gene2loc)
    sorted_locs = [gene2loc[item] for item in sorted_genes]
    aggregated_tf_expr = np.empty((len(tf_list),len(cell_ids)),dtype='float32')
    aggregated_full_expr = np.empty((len(sorted_genes),len(cell_ids)),dtype='float32')
    for cell_id in cell_ids:
        average_tpm = merge_rseq_reps(cell_id, id2name)
        tf_tpm = [average_tpm[item] for item in tf_list]
        full_tpm = [average_tpm[item] for item in sorted_genes]
        aggregated_tf_expr[:, cell_ids.index(cell_id)] = tf_tpm
        aggregated_full_expr[:, cell_ids.index(cell_id)] = full_tpm
    pd_aggregated_tf_expr = pd.DataFrame(data = aggregated_tf_expr,
                            index = tf_list,
                            columns = cell_ids)
    pd_aggregated_full_expr = pd.DataFrame(data = aggregated_full_expr,
                            index = [sorted_genes, sorted_locs],
                            columns = cell_ids)

    if use_norm:
        pd_aggregated_tf_expr = quantile_norm(pd_aggregated_tf_expr)
        pd_aggregated_full_expr = quantile_norm(pd_aggregated_full_expr)

    if save:
        pd_aggregated_tf_expr.to_csv('data/encode/aggregated_tf_expr.csv',sep = '\t')
        pd_aggregated_full_expr.to_csv('data/encode/aggregated_full_expr.csv',sep = '\t')

    


    

if __name__ == "__main__":
    cell_ids = get_cell_ids()
    tf_list = get_tf()
    id2name, gene2loc = get_gene_annot()
    aggregate_rseq(cell_ids, id2name ,gene2loc, tf_list)