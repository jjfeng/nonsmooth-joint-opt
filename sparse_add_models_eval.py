import getopt
import time
import sys
import traceback
import pickle
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

from sparse_add_models_hillclimb import Sparse_Add_Model_Hillclimb, Sparse_Add_Model_Hillclimb_Simple
from sparse_add_models_neldermead import Sparse_Add_Model_Nelder_Mead, Sparse_Add_Model_Nelder_Mead_Simple
from sparse_add_models_grid_search import Sparse_Add_Model_Grid_Search
from sparse_add_models_spearmint import Sparse_Add_Model_Spearmint, Sparse_Add_Model_Spearmint_Simple
from data_generator import DataGenerator
from method_results import MethodResults
from method_results import MethodResult
from iteration_models import Simulation_Settings, Iteration_Data

from common import *

########
# FUNCTIONS FOR ADDITIVE MODEL
########
def identity_fcn(x):
    return x.reshape(x.size, 1)

def big_sin(x):
    return identity_fcn(9 * np.sin(x*3))

def big_cos_sin(x):
    return identity_fcn(6 * (np.cos(x * 1.25) + np.sin(x/2 + 0.5)))

def crazy_down_sin(x):
    return identity_fcn(x * np.sin(x) - x)

def pwr_small(x):
    return identity_fcn(np.power(x/2,2) - 10)

def const_zero(x):
    return np.zeros(x.shape)

########
# Model settings
########
class Sparse_Add_Models_Settings(Simulation_Settings):
    results_folder = "results/sparse_add_models"
    num_funcs = 3
    num_zero_funcs = 1 #20
    gs_lambdas1 = np.power(10, np.arange(-4, 2, 6.0/10))
    assert(gs_lambdas1.size == 10)
    gs_lambdas2 = gs_lambdas1
    smooth_fcns = [big_sin, identity_fcn, big_cos_sin, crazy_down_sin, pwr_small]
    plot = False
    method = "HC"
    method_result_keys = [
        "test_err",
        "validation_err",
        "runtime",
        "num_solves",
    ]

    def print_settings(self):
        print "SETTINGS"
        obj_str = "method %s\n" % self.method
        obj_str += "num_funcs %d\n" % self.num_funcs
        obj_str += "num_zero_funcs %d\n" % self.num_zero_funcs
        obj_str += "t/v/t size %d/%d/%d\n" % (self.train_size, self.validate_size, self.test_size)
        obj_str += "snr %f\n" % self.snr
        obj_str += "sp runs %d\n" % self.spearmint_numruns
        obj_str += "nm_iters %d\n" % self.nm_iters
        print obj_str

#########
# MAIN FUNCTION
#########
def main(argv):
    num_threads = 1
    num_runs = 1

    try:
        opts, args = getopt.getopt(argv,"f:z:a:b:c:s:m:t:r:")
    except getopt.GetoptError:
        sys.exit(2)

    settings = Sparse_Add_Models_Settings()
    for opt, arg in opts:
        if opt == '-f':
            settings.num_funcs = int(arg)
        elif opt == '-z':
            settings.num_zero_funcs = int(arg)
        elif opt == '-a':
            settings.train_size = int(arg)
        elif opt == '-b':
            settings.validate_size = int(arg)
        elif opt == '-c':
            settings.test_size = int(arg)
        elif opt == "-s":
            settings.snr = float(arg)
        elif opt == "-m":
            assert(arg in METHODS)
            settings.method = arg
        elif opt == "-t":
            num_threads = int(arg)
        elif opt == "-r":
            num_runs = int(arg)

    print "TOTAL NUM RUNS %d" % num_runs
    settings.print_settings()
    sys.stdout.flush()

    assert(settings.num_funcs <= len(settings.smooth_fcns))
    smooth_fcn_list = settings.smooth_fcns[:settings.num_funcs] + [const_zero] * settings.num_zero_funcs
    data_gen = DataGenerator(settings)

    run_data = []
    for i in range(num_runs):
        observed_data = data_gen.make_additive_smooth_data(smooth_fcn_list)
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

    initial_lambdas = np.ones(1 + settings.num_funcs + settings.num_zero_funcs)
    initial_lambdas[0] = 10
    initial_lambdas_set = [initial_lambdas * 0.01, initial_lambdas]

    init_lambda_simple = np.ones(2)
    init_lambda_simple[0] = 10
    initial_lambdas_set_simple = [init_lambda_simple * 0.01, init_lambda_simple]

    method = iter_data.settings.method

    str_identifer = "%d_%d_%d_%d_%d_%d_%s_%d" % (
        settings.num_funcs,
        settings.num_zero_funcs,
        settings.train_size,
        settings.validate_size,
        settings.test_size,
        settings.snr,
        method,
        iter_data.i,
    )
    log_file_name = "%s/tmp/log_%s.txt" % (settings.results_folder, str_identifer)
    print "log_file_name", log_file_name
    # set file buffer to zero so we can see progress
    with open(log_file_name, "w", buffering=0) as f:
        if method == "NM":
            algo = Sparse_Add_Model_Nelder_Mead(iter_data.data)
            algo.run(initial_lambdas_set, num_iters=settings.nm_iters, log_file=f)
        elif method == "NM0":
            algo = Sparse_Add_Model_Nelder_Mead_Simple(iter_data.data)
            algo.run(initial_lambdas_set_simple, num_iters=settings.nm_iters, log_file=f)
        elif method == "GS":
            algo = Sparse_Add_Model_Grid_Search(iter_data.data)
            algo.run(lambdas1=settings.gs_lambdas1, lambdas2=settings.gs_lambdas2, log_file=f)
        elif method == "HC":
            algo = Sparse_Add_Model_Hillclimb(iter_data.data)
            algo.run(initial_lambdas_set, debug=False, log_file=f)
        elif method == "HC0":
            algo = Sparse_Add_Model_Hillclimb_Simple(iter_data.data)
            algo.run(initial_lambdas_set_simple, debug=False, log_file=f)
        elif method == "SP":
            algo = Sparse_Add_Model_Spearmint(iter_data.data, str_identifer)
            algo.run(settings.spearmint_numruns, log_file=f)
        elif method == "SP0":
            algo = Sparse_Add_Model_Spearmint_Simple(iter_data.data, str_identifer)
            algo.run(settings.spearmint_numruns, log_file=f)
        sys.stdout.flush()
        method_res = create_method_result(iter_data.data, algo.fmodel, settings.num_funcs, settings.num_zero_funcs)
        if settings.plot:
            plot(iter_data.data, algo.fmodel, settings, label="Gradient Descent")

        f.write("SUMMARY\n%s" % method_res)
    return method_res

def create_method_result(data, algo, num_funcs, num_zero_funcs):
    test_err = testerror_sparse_add_smooth(
        data.y_test,
        data.test_idx,
        algo.best_model_params
    )

    return MethodResult({
            "test_err":test_err,
            "validation_err":algo.best_cost,
            "runtime":algo.runtime,
            "num_solves": algo.num_solves,
        },
        lambdas=algo.current_lambdas
    )

def plot(data, algo_model, settings, label=None, func_indices=range(6), ylim=[-15,15], xlim=[-5,5]):
    file_name = "figures/sparse_add_model_%d_%d_%d_%d_%d_%s" % (
        settings.num_funcs,
        settings.num_zero_funcs,
        data.num_train,
        data.num_validate,
        data.num_test,
        settings.method
    )

    # Pickle the plot data
    pickle_file = "%s.pkl" % file_name
    with open(pickle_file, "wb") as f:
        pickle.dump({
            "data": data,
            "algo_model": algo_model,
            "settings": settings
        }, f)

    for func_idx in func_indices:
        plt.clf()
        x_i = data.X_full[:,func_idx]
        print "x_i", x_i.shape
        order_indices = np.argsort(x_i, axis=0)
        sort_x_i = x_i[order_indices]
        if func_idx < settings.num_funcs:
            plt.plot(sort_x_i, settings.smooth_fcns[func_idx](sort_x_i), label="Real", color="green", linestyle="--")
        else:
            plt.plot(sort_x_i, const_zero(sort_x_i), label="Real", color="green", linestyle="--")
        plt.plot(sort_x_i, algo_model.best_model_params[order_indices, func_idx], label=label, color="brown")
        plt.ylim(ylim)
        plt.xlim(xlim)
        figname = "%s_func%d.png" % (file_name, func_idx)
        print "figname", figname
        plt.savefig(figname)

if __name__ == "__main__":
    main(sys.argv[1:])
