import getopt
import sys
import time
import traceback
from multiprocessing import Pool
import numpy as np
from Bio import Geo

import hillclimb_realdata_grouped_lasso_fullcv as hc
import gridsearch_grouped_lasso as gs_grouped
from common import testerror_logistic_grouped
from realdata_common import *
from iteration_models import Simulation_Settings, Iteration_Data
from method_results import MethodResults
from method_results import MethodResult

CONTROL_LABEL = 0
DISEASE_LABEL = 1
TRAIN_SIZE = 40
VALIDATE_SIZE = 10
KFOLDS = 5
# Download dataset GDS1615 from the Gene Expression Omnibus (GEO) repository
GENE_EXPR_FILENAME = "realdata/GDS1615_full.soft"

class Shuffled_Gene_Data:
    def __init__(self, X_genesets, y, genesets, train_size=TRAIN_SIZE, validate_size=VALIDATE_SIZE):
        X_groups_train_validate, y_train_validate, X_groups_test, y_test = shuffle_and_split_data_full_cv(
            X_genesets, y, train_size + validate_size)
        feature_group_sizes = [Xg.shape[1] for Xg in X_groups_train_validate]
        X_test = np.hstack(X_groups_test)
        self.X_groups_train_validate = X_groups_train_validate
        self.y_train_validate = y_train_validate
        self.X_groups_test = X_groups_test
        self.y_test = y_test
        self.X_test = X_test
        self.genesets = genesets
        self.feature_group_sizes = feature_group_sizes

class RealDataSettings(Simulation_Settings):
    results_folder = "results/realdata"
    method_result_keys = [
        "log_likelihood",
        "nonzeros_genes",
        "nonzero_genesets",
        "validation_err",
        "test_rate",
        "false_positive_rate",
        "false_negative_rate",
        "runtime",
    ]

def main(argv):
    num_threads = 1
    num_runs = 1

    try:
        opts, args = getopt.getopt(argv,"m:t:r:")
    except getopt.GetoptError:
        print "Bad argument given to realdata_eval.py"
        sys.exit(2)

    settings = RealDataSettings()
    for opt, arg in opts:
        if opt == "-m":
            assert(arg in ["HC", "GS"])
            settings.method = arg
        elif opt == "-t":
            num_threads = int(arg)
        elif opt == "-r":
            num_runs = int(arg)

    print "TOTAL NUM RUNS %d" % num_runs
    sys.stdout.flush()

    geneset_dict = read_geneset_file()
    X_genesets, y, genesets = read_gene_expr_data(geneset_dict)
    print "num features", sum([X_genesets[i].shape[1] for i in range(0, len(X_genesets))])
    print "total genesets ever", len(X_genesets)
    X_genesets = normalize_data(X_genesets)

    run_data = []
    for i in range(num_runs):
        data = Shuffled_Gene_Data(X_genesets, y, genesets)
        run_data.append(Iteration_Data(i, data, settings))

    if num_threads > 1:
        print "Do multiprocessing"
        pool = Pool(num_threads)
        results = pool.map(fit_data_for_iter_safe, run_data)
    else:
        print "Avoiding multiprocessing"
        results = map(fit_data_for_iter_safe, run_data)

    method_results = MethodResults(settings.method, settings.method_result_keys)
    num_crashes = 0
    for r in results:
        if r is not None:
            method_results.append(r)
        else:
            num_crashes += 1
    print "==========TOTAL RUNS %d============" % method_results.get_num_runs()
    method_results.print_results()
    print "num crashes %d" % num_crashes

#########
# FUNCTIONS FOR CHILD THREADS
#########
def fit_data_for_iter_safe(iter_data):
    result = None
    try:
        result = fit_data_for_iter(iter_data)
    except Exception as e:
        print "Exception caught in iter %d: %s" % (iter_data.i, e)
        traceback.print_exc()
    return result

def fit_data_for_iter(iter_data):
    settings = iter_data.settings
    initial_lambdas = [0.5, 0.2]
    method = iter_data.settings.method

    str_identifer = "%s_%d" % (method, iter_data.i)
    log_file_name = "%s/tmp/log_%s.txt" % (settings.results_folder, str_identifer)
    print "log_file_name", log_file_name
    # set file buffer to zero so we can see progress
    with open(log_file_name, "w", buffering=0) as f:
        start_time = time.time()
        if method == "GS":
            complete_beta, validate_cost = gs_grouped.run_classify_fullcv(
                iter_data.data.X_groups_train_validate,
                iter_data.data.y_train_validate,
                iter_data.data.feature_group_sizes,
                KFOLDS
            )
        elif method == "HC":
            complete_beta, validate_cost, _ = hc.run_for_lambdas(
                iter_data.data.X_groups_train_validate,
                iter_data.data.y_train_validate,
                iter_data.data.feature_group_sizes,
                KFOLDS,
                init_lambdas=initial_lambdas
            )
        grouped_betas = get_grouped_betas(complete_beta, iter_data.data.feature_group_sizes)
        method_res = create_method_result(
            iter_data.data,
            grouped_betas,
            validate_cost,
            time.time() - start_time
        )

        f.write("SUMMARY\n%s" % method_res)
    return method_res

def get_grouped_betas(beta, feature_group_sizes):
    final_betas =[]
    start_feature_idx = 0
    for feature_group_size in feature_group_sizes:
        end_feature_idx = start_feature_idx + feature_group_size
        final_betas.append(
            beta[start_feature_idx:end_feature_idx]
        )
        start_feature_idx = end_feature_idx
    return final_betas

def create_method_result(data, grouped_betas, validate_cost, runtime, threshold=1e-6):
    log_likelihood, test_rate, false_positive_rate, false_negative_rate = testerror_logistic_grouped(
        data.X_test,
        data.y_test,
        grouped_betas
    )
    nonzeros_genes, nonzero_genesets = get_num_nonzero_betas(grouped_betas, data.genesets, threshold=threshold)

    return MethodResult({
        "log_likelihood":log_likelihood,
        "nonzeros_genes":nonzeros_genes,
        "nonzero_genesets":nonzero_genesets,
        "validation_err":validate_cost,
        "test_rate":test_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "runtime":runtime
    })

def read_gene_expr_data(geneset_dict):
    """
    Read gene expression data from file. Returns
    X - array of unnormalized gene expression data, grouped by genesets
    y - control and disease labels
    geneset - filtered genesets that match the given geneset dictionary, same order as returned X array
    """
    handle = open(GENE_EXPR_FILENAME)
    records = Geo.parse(handle)

    # gsm ids of the normal subjects
    normal_subjects = []

    # geneset row ids
    X_groups = {}
    for k in geneset_dict.keys():
        X_groups[k] = set()

    X = []
    y = []

    i = 0
    for record in records:
        i += 1
        if i == 3:
            # Read patient labels so we can make the y vector
            attr = record.entity_attributes
            assert(attr["subset_description"] == "normal")
            normal_subjects = attr["subset_sample_id"].split(",")

        if i == 7:
            # Read actual gene expression data
            col_names = record.table_rows[0]
            gsm_idxs = []
            for idx, col_name in enumerate(col_names):
                if "GSM" == col_name[0:3]:
                    gsm_idxs.append(idx)

                    # populate the y matrix
                    # 1 means diseased. 0 means control.
                    y.append(CONTROL_LABEL if col_name in normal_subjects else DISEASE_LABEL)

            geneid_idx = col_names.index("Gene ID")

            feature_idx = 0
            for row in record.table_rows[1:]:
                geneid = row[geneid_idx]
                geneset = get_geneset_from_dict(geneset_dict, geneid)
                if geneset is not None:
                    # add feature idx to correct geneset
                    X_groups[geneset].add(feature_idx)

                    # append the gene expression data
                    X.append([float(row[i]) for i in gsm_idxs])

                    feature_idx += 1

    # Make feature groups
    X = np.matrix(X).T
    X_genesets = []
    genesets_included = []
    for geneset_key, geneset_col_idxs in X_groups.iteritems():
        if len(geneset_col_idxs) == 0:
            continue
        X_genesets.append(X[:, list(geneset_col_idxs)])
        genesets_included.append(geneset_key)

    y = np.matrix(y).T
    return X_genesets, y, genesets_included

if __name__ == "__main__":
    main(sys.argv[1:])
