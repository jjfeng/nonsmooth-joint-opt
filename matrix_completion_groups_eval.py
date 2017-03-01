import getopt
import time
import sys
import traceback
import numpy as np
from multiprocessing import Pool

from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult
from iteration_models import Simulation_Settings, Iteration_Data

from matrix_completion_groups_hillclimb import Matrix_Completion_Groups_Hillclimb, Matrix_Completion_Groups_Hillclimb_Simple
from matrix_completion_groups_neldermead import Matrix_Completion_Groups_Nelder_Mead, Matrix_Completion_Groups_Nelder_Mead_Simple
from matrix_completion_groups_grid_search import Matrix_Completion_Groups_Grid_Search
from matrix_completion_groups_spearmint import Matrix_Completion_Groups_Spearmint, Matrix_Completion_Groups_Spearmint_Simple

from common import *

class Matrix_Completion_Group_Settings(Simulation_Settings):
    results_folder = "results/matrix_completion_groups"
    num_nonzero_s = 1
    num_rows = 4
    num_cols = 4
    num_row_groups = 1
    num_col_groups = 1
    num_row_features = 6 # num features per group
    num_col_features = 6 # num features per group
    num_nonzero_row_groups = 1
    num_nonzero_col_groups = 1
    train_perc = 0.3
    validate_perc = 0.15
    spearmint_numruns = 100
    snr = 2
    gamma_to_row_col_m = 0.5
    feat_factor = 1.
    gs_lambdas1 = np.power(10, np.arange(0, -2, -2.0/10))
    gs_lambdas2 = gs_lambdas1
    assert(gs_lambdas1.size == 10)
    method_result_keys = [
        "test_err",
        "validation_err",
        "alpha_err",
        "beta_err",
        "gamma_err",
        "gamma_num_s",
        "runtime",
        "num_solves",
    ]

    def print_settings(self):
        obj_str = "SETTINGS\n method %s\n" % self.method
        obj_str += "interaction matrix dim %d x %d (nonzero: %d)\n" % (self.num_rows, self.num_cols, self.num_nonzero_s)
        obj_str += "num_features %d x %d\n" % (self.num_row_features, self.num_col_features)
        obj_str += "num_nonzero_groups %d x %d\n" % (self.num_nonzero_row_groups, self.num_nonzero_col_groups)
        obj_str += "num_groups %d x %d\n" % (self.num_row_groups, self.num_col_groups)
        obj_str += "t/v/t size %f/%f\n" % (self.train_perc, self.validate_perc)
        obj_str += "snr %f\n" % self.snr
        obj_str += "sp runs %d\n" % self.spearmint_numruns
        obj_str += "nm_iters %d\n" % self.nm_iters
        obj_str += "gamma_to_row_col_m %f\n" % self.gamma_to_row_col_m
        obj_str += "feat_factor %f\n" % self.feat_factor
        print obj_str

#########
# MAIN FUNCTION
#########
def main(argv):
    num_threads = 1
    num_runs = 1

    try:
        opts, args = getopt.getopt(argv,"d:z:f:g:a:v:s:m:t:r:i:")
    except getopt.GetoptError:
        print "Bad argument given"
        sys.exit(2)

    settings = Matrix_Completion_Group_Settings()
    for opt, arg in opts:
        if opt == '-d':
            arg_split = arg.split(",")
            settings.num_rows = int(arg_split[0])
            settings.num_cols = int(arg_split[1])
        elif opt == '-z':
            arg_split = arg.split(",")
            settings.num_nonzero_row_groups = int(arg_split[0])
            settings.num_nonzero_col_groups = int(arg_split[1])
        elif opt == '-f':
            arg_split = arg.split(",")
            settings.num_row_features = int(arg_split[0])
            settings.num_col_features = int(arg_split[1])
        elif opt == '-g':
            arg_split = arg.split(",")
            settings.num_row_groups = int(arg_split[0])
            settings.num_col_groups = int(arg_split[1])
        elif opt == '-a':
            arg_split = arg.split(",")
            settings.train_perc = float(arg_split[0])
            settings.validate_perc = float(arg_split[1])
            assert(settings.train_perc + settings.validate_perc <= 1.0)
        elif opt == "-v":
            settings.num_nonzero_s = int(arg)
        elif opt == "-s":
            settings.snr = float(arg)
        elif opt == "-m":
            assert(arg in METHODS)
            settings.method = arg
        elif opt == "-t":
            num_threads = int(arg)
        elif opt == "-r":
            num_runs = int(arg)
        elif opt == "-i":
            settings.gamma_to_row_col_m = float(arg)

    assert(settings.num_nonzero_s <= settings.num_rows and settings.num_nonzero_s <= settings.num_cols)

    settings.matrix_size = settings.num_rows * settings.num_cols

    print "TOTAL NUM RUNS %d" % num_runs
    settings.print_settings()
    sys.stdout.flush()

    data_gen = DataGenerator(settings)

    run_data = []
    for i in range(num_runs):
        observed_data = data_gen.matrix_completion_groups(gamma_to_row_col_m=settings.gamma_to_row_col_m, feat_factor=settings.feat_factor)
        run_data.append(Iteration_Data(i, observed_data, settings))

    if settings.method != "SP" and num_threads > 1:
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

    one_vec_all = np.ones(1 + settings.num_row_groups + settings.num_col_groups)
    initial_lambdas_set = [one_vec_all * 0.005, one_vec_all * 0.003]

    one_vec2_all = np.ones(2)
    simple_initial_lambdas_set = [one_vec2_all * 0.005, one_vec2_all * 0.003]

    method = iter_data.settings.method

    str_identifer = "%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%s_%d" % (
        settings.num_rows,
        settings.num_cols,
        settings.num_nonzero_s,
        settings.num_row_features,
        settings.num_col_features,
        settings.num_row_groups,
        settings.num_col_groups,
        settings.num_nonzero_row_groups,
        settings.num_nonzero_col_groups,
        int(settings.train_perc * 100),
        int(settings.validate_perc * 100),
        settings.snr,
        int(settings.gamma_to_row_col_m * 100),
        settings.feat_factor,
        method,
        iter_data.i,
    )
    log_file_name = "%s/tmp/log_%s.txt" % (settings.results_folder, str_identifer)
    print "log_file_name", log_file_name

    # set file buffer to zero so we can see progress
    with open(log_file_name, "w", buffering=0) as f:
        if method == "NM":
            algo = Matrix_Completion_Groups_Nelder_Mead(iter_data.data, settings)
            algo.run(initial_lambdas_set, num_iters=settings.nm_iters, log_file=f)
        elif method == "NM0":
            algo = Matrix_Completion_Groups_Nelder_Mead_Simple(iter_data.data, settings)
            algo.run(simple_initial_lambdas_set, num_iters=settings.nm_iters, log_file=f)
        elif method == "GS":
            algo = Matrix_Completion_Groups_Grid_Search(iter_data.data, settings)
            algo.run(lambdas1=settings.gs_lambdas1, lambdas2=settings.gs_lambdas2, log_file=f)
        elif method == "HC":
            algo = Matrix_Completion_Groups_Hillclimb(iter_data.data, settings)
            algo.run(initial_lambdas_set, debug=False, log_file=f)
        elif method == "HC0":
            algo = Matrix_Completion_Groups_Hillclimb_Simple(iter_data.data, settings)
            algo.run(simple_initial_lambdas_set, debug=False, log_file=f)
        elif method == "SP":
            algo = Matrix_Completion_Groups_Spearmint(iter_data.data, str_identifer, settings)
            algo.run(settings.spearmint_numruns, log_file=f)
        elif method == "SP0":
            algo = Matrix_Completion_Groups_Spearmint_Simple(iter_data.data, str_identifer, settings)
            algo.run(settings.spearmint_numruns, log_file=f)
        else:
            raise ValueError("Method not implemented yet: %s" % method)
        sys.stdout.flush()
        method_res = create_method_result(iter_data.data, algo.fmodel)

        f.write("SUMMARY\n%s" % method_res)
    return method_res

def create_method_result(data, algo, zero_threshold=1e-6):
    test_err = testerror_matrix_completion_groups(
        data,
        data.test_idx,
        algo.best_model_params
    )
    alphas_guess = algo.best_model_params["alphas"]
    betas_guess = algo.best_model_params["betas"]
    gamma_guess = algo.best_model_params["gamma"]
    u, s, v = np.linalg.svd(gamma_guess)

    return MethodResult({
            "test_err": test_err,
            "validation_err": algo.best_cost,
            "alpha_err": betaerror(np.vstack(data.real_betas), np.vstack(alphas_guess)),
            "beta_err": betaerror(np.vstack(data.real_betas), np.vstack(betas_guess)),
            "gamma_err": betaerror(data.real_gamma, gamma_guess),
            "gamma_num_s": np.sum(s > zero_threshold),
            "runtime": algo.runtime,
            "num_solves": algo.num_solves,
        },
        lambdas=algo.best_lambdas
    )

if __name__ == "__main__":
    main(sys.argv[1:])
