import pickle
import numpy as np

GENESET_FILENAME = "realdata/c1.all.v5.0.entrez.gmt"

def read_geneset_file():
    """
    Read geneset file. Contains positional geneset C1 from MSigDB
    """
    geneset_dict = {}
    with open(GENESET_FILENAME) as geneset_file:
        lines = geneset_file.readlines()
        i = 0
        for l in lines:
            i += 1
            entries = l.split()
            geneset_name = entries[0]
            geneset = set(entries[2:])
            geneset_dict[geneset_name] = geneset

    return geneset_dict


def get_geneset_from_dict(geneset_dict, geneid):
    """
    Returns the geneset of the given Entrez gene id
    If not found, returns None
    """
    for k, v in geneset_dict.iteritems():
        if geneid in v:
            return k
    return None


def shuffle_and_split_data(X_genesets, y, train_size, validate_size):
    """
    shuffle and split data
    """
    permutation = np.random.permutation(y.size)
    y_permuted = y[permutation]
    X_genesets_permuted = [Xg[permutation, :] for Xg in X_genesets]
    X_groups_train = [Xg[0:train_size, :] for Xg in X_genesets_permuted]
    X_groups_validate = [Xg[train_size: validate_size + train_size, :] for Xg in X_genesets_permuted]
    X_groups_test = [Xg[validate_size + train_size:, :] for Xg in X_genesets_permuted]
    y_train = y_permuted[0:train_size]
    y_validate = y_permuted[train_size: validate_size + train_size]
    y_test = y_permuted[validate_size + train_size:]
    return X_groups_train, y_train, X_groups_validate, y_validate, X_groups_test, y_test


def shuffle_and_split_data_full_cv(X_genesets, y, train_validate_size):
    """
    shuffle and split data
    """
    permutation = np.random.permutation(y.size)
    y_permuted = y[permutation]
    X_genesets_permuted = [Xg[permutation, :] for Xg in X_genesets]
    X_groups_train_validate = [Xg[0:train_validate_size, :] for Xg in X_genesets_permuted]
    X_groups_test = [Xg[train_validate_size:, :] for Xg in X_genesets_permuted]
    y_train_validate = y_permuted[0:train_validate_size]
    y_test = y_permuted[train_validate_size:]
    return X_groups_train_validate, y_train_validate, X_groups_test, y_test


def get_num_nonzero_betas(betas, genesets, threshold=1e-6):
    """
    Just analyzes the results and returns the number of nonzero coefficients and
    the number of groups with nonzero coefficients
    """
    total_nonzeros = 0
    total_genesets = 0
    for idx, b in enumerate(betas):
        geneset_nonzeros = sum(np.greater(b, threshold))
        total_nonzeros += geneset_nonzeros
        if geneset_nonzeros > 0:
            total_genesets += 1
            print "geneset found", genesets[idx], "nonzeros", geneset_nonzeros, "total genes", b.size
    return total_nonzeros[0,0], total_genesets


def normalize_data(X_genesets):
    """
    Center X data and make variance 1
    """
    normalized_datasets = []
    for Xg in X_genesets:
        averages = np.average(Xg, axis=0)
        Xg_centered = Xg - np.repeat(averages, Xg.shape[0], axis=0)
        std_devs = np.sqrt(np.var(Xg_centered, axis=0))
        Xg_normalized = np.divide(Xg, np.repeat(std_devs, Xg.shape[0], axis=0))
        normalized_datasets.append(Xg_normalized)
    return normalized_datasets

def pickle_data(filename, X_groups_train, y_train, X_groups_validate, y_validate, X_groups_test, y_test, genesets):
    data_output = open(filename, 'wb')
    pickle.dump(X_groups_train, data_output, -1)
    pickle.dump(y_train, data_output, -1)
    pickle.dump(X_groups_validate, data_output, -1)
    pickle.dump(y_validate, data_output, -1)
    pickle.dump(X_groups_test, data_output, -1)
    pickle.dump(y_test, data_output, -1)
    pickle.dump(genesets, data_output, -1)
    data_output.close()


def load_pickled_data(filename):
    data_file = open(filename, 'r')
    X_groups_train = pickle.load(data_file)
    y_train = pickle.load(data_file)
    X_groups_validate = pickle.load(data_file)
    y_validate = pickle.load(data_file)
    X_groups_test = pickle.load(data_file)
    y_test = pickle.load(data_file)
    genesets = pickle.load(data_file)
    data_file.close()
    return X_groups_train, y_train, X_groups_validate, y_validate, X_groups_test, y_test, genesets


def pickle_betas(filename, hc_betas, gs_grouped_betas, gs_betas):
    betas_output = open(filename, 'wb')
    pickle.dump(hc_betas, betas_output, -1)
    pickle.dump(gs_grouped_betas, betas_output, -1)
    pickle.dump(gs_betas, betas_output, -1)
    betas_output.close()
