# Lasso simulation to show that the best lasso parameters to minimize the validation
# error are not at the knots along the lasso path.

import time
import sys
import pickle
import numpy as np
from multiprocessing import Pool

from iteration_models import Simulation_Settings, Iteration_Data
from method_results import MethodResult
from convexopt_solvers import LassoProblemWrapper

from data_generator import DataGenerator

from common import *
from sklearn import linear_model
import matplotlib.pyplot as plt

class Lasso_Settings(Simulation_Settings):
    results_folder = "results/lasso"
    num_features = 100
    num_nonzero_features = 3
    train_size = 40
    validate_size = 30
    test_size = 40

def plot_lasso_path(alphas, coefs):
    # alphas is the lasso param
    plt.figure()
    neg_log_alphas = -np.log10(alphas)
    for coef in coefs:
        plt.plot(neg_log_alphas, coef)
    plt.xlabel("new log lasso param")
    plt.ylabel("coef value")
    plt.show()

def get_dist_of_closest_lambda(lam, lambda_path):
    lambda_knot_dists = np.abs(lambda_path - lam)
    min_idx = np.argmin(lambda_knot_dists)
    return lambda_knot_dists[min_idx], min_idx

def do_lasso_simulation(data, NUM_LAMBDA_SPLITS=3000):
    # Make lasso path
    lasso_path, coefs, _ = linear_model.lasso_path(
        data.X_train,
        np.array(data.y_train.flatten().tolist()[0]), # reshape appropriately
        method='lasso'
    )
    prob = LassoProblemWrapper(
        data.X_train,
        data.y_train
    )

    val_errors = []
    for i, l in enumerate(lasso_path):
        beta = prob.solve(np.array([l]))
        val_error = testerror_lasso(data.X_validate, data.y_validate, beta)
        val_errors.append(val_error)
    sorted_idx = np.argsort(val_errors)

    max_lam = lasso_path[np.min(sorted_idx[:3])]
    min_lam = lasso_path[np.max(sorted_idx[:3])]

    # Search non-knot lambdas to see if there is a lambda with a lower validation loss
    finer_lam_range = []
    for i, l_idx in enumerate(range(np.min(sorted_idx[:3]) - 1, np.max(sorted_idx[:3]) + 1)):
        fudge = 0
        if i == 0:
            fudge = 1e-10
        l_min = lasso_path[l_idx + 1] if lasso_path.size - 1 >= l_idx + 1 else 0
        l_max = lasso_path[l_idx] if l_idx >= 0 else lasso_path[0] + 0.1

        add_l = np.arange(start=l_min, stop=l_max + fudge, step=(l_max - l_min)/NUM_LAMBDA_SPLITS)
        finer_lam_range.append(add_l)
    finer_lam_range = np.concatenate(finer_lam_range)

    fine_val_errors = []
    for i, l in enumerate(finer_lam_range):
        beta = prob.solve(np.array([l]))
        val_error = testerror_lasso(data.X_validate, data.y_validate, beta)
        fine_val_errors.append(val_error)

    fine_sorted_idx = np.argsort(fine_val_errors)
    best_lam = finer_lam_range[fine_sorted_idx[0]]
    print "best_lam", best_lam

    min_dist, idx = get_dist_of_closest_lambda(best_lam, lasso_path)
    print "min_dist", min_dist
    return min_dist

def plot_min_dists():
    figure_file_name = "results/lasso/lasso_knot_locations.png"
    with open("results/lasso/lasso_knot_locations.pkl", "r") as f:
        min_dists = pickle.load(f)
    plt.hist(min_dists, bins=np.logspace(np.log(np.min(min_dists)), np.log(np.max(min_dists)), 50))
    plt.gca().set_xscale("log")
    plt.xlim(1e-7, 1e-2)
    plt.xlabel("Distance Between $\hat{\lambda}$ and Closest Knot")
    plt.ylabel("Frequency")
    print "figure_file_name", figure_file_name
    plt.savefig(figure_file_name)

np.random.seed(10)
NUM_RUNS = 500
num_threads = 12

settings = Lasso_Settings()
data_gen = DataGenerator(settings)
initial_lambdas_set = [np.ones(1) * 0.1]

# Make data
datas = []
for i in range(NUM_RUNS):
    datas.append(
        data_gen.make_simple_linear(settings.num_features, settings.num_nonzero_features)
    )

pool = Pool(num_threads)
min_dists = pool.map(do_lasso_simulation, datas)

pickle_file_name = "%s/lasso_knot_locations.pkl" % settings.results_folder
print "pickle_file_name", pickle_file_name
with open(pickle_file_name, "wb") as f:
    pickle.dump(min_dists, f)

mean_best_l_dists = np.mean(min_dists)
print "Mean distance between the best lasso vs. lasso path knots", mean_best_l_dists
print "min_dists equal zero", np.sum(min_dists == 0)

plot_min_dists()
