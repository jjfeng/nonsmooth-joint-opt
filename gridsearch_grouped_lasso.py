import sys
import numpy as np

from common import *
from convexopt_solvers import GroupedLassoProblemWrapperSimple
from convexopt_solvers import GroupedLassoClassifyProblemWrapperSimple
from convexopt_solvers import GroupedLassoClassifyProblemWrapperSimpleFullCV

from realdata_colitis_models import AllKFoldsData

LAMBDA_MIN_FACTOR = 1e-5

def run(X_train, y_train, X_validate, y_validate, group_feature_sizes):
    infty_norm = 0
    start_feature = 0
    for group_feature_size in group_feature_sizes:
        infty_norm = max(infty_norm, np.linalg.norm(X_train[:, start_feature : start_feature + group_feature_size].T * y_train, 2))
        start_feature += group_feature_size

    num_lambdas = 10
    max_power = np.log(infty_norm)
    min_power = np.log(LAMBDA_MIN_FACTOR * max_power)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 0.01) / (num_lambdas - 1)))
    print "gridsearch: lambda_guesses", lambda_guesses

    problem_wrapper = GroupedLassoProblemWrapperSimple(X_train, y_train, group_feature_sizes)

    best_cost = 1e5
    best_betas = []
    best_regularization = [lambda_guesses[0], lambda_guesses[0]]

    for l1 in lambda_guesses:
        for l2 in lambda_guesses:
            betas = problem_wrapper.solve([l1, l2], high_accur=False)
            current_cost = testerror_grouped(X_validate, y_validate, betas)
            if best_cost > current_cost:
                best_cost = current_cost
                best_betas = betas
                best_regularization = [l1, l2]
                print "best_cost so far", best_cost, "best_regularization", best_regularization

    print "gridsearch: best_validation_error", best_cost
    print "gridsearch: best lambdas:", best_regularization

    return best_betas, best_cost

def run_classify(X_groups_train, y_train, X_groups_validate, y_validate):
    """
    Runs a logistic regression instead
    Uses 8 lambdas instead of 10!
    """
    num_lambdas = 8
    X_validate = np.hstack(X_groups_validate)

    max_power = np.log(50)
    min_power = np.log(1e-4)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 1e-5) / (num_lambdas - 1)))
    print "gridsearch: lambda_guesses", lambda_guesses

    problem_wrapper = GroupedLassoClassifyProblemWrapperSimple(X_groups_train, y_train)

    best_cost = 1e5
    best_betas = []
    best_regularization = [lambda_guesses[0], lambda_guesses[0]]
    for l1 in reversed(lambda_guesses):
        for l2 in reversed(lambda_guesses):
            betas = problem_wrapper.solve([l1, l2])
            current_cost, _, _, _ = testerror_logistic_grouped(X_validate, y_validate, betas)
            if best_cost > current_cost:
                best_cost = current_cost
                best_betas = betas
                best_regularization = [l1, l2]
                print "best_cost so far", best_cost, "best_regularization", best_regularization
                sys.stdout.flush()

    print "gridsearch: best_validation_error", best_cost
    print "gridsearch: best lambdas:", best_regularization

    return best_betas, best_cost

def run_classify_fullcv(X_groups_train_validate, y_train_validate, feature_group_sizes, kfolds):
    """
    Runs a logistic regression instead
    Uses 5 lambdas instead of 10!
    """
    num_lambdas = 5

    max_power = np.log(5)
    min_power = np.log(1e-4)
    lambda_guesses = np.power(np.e, np.arange(min_power, max_power, (max_power - min_power - 1e-5) / (num_lambdas - 1)))
    print "gridsearch: lambda_guesses", lambda_guesses

    X_train_validate = np.hstack(X_groups_train_validate)
    full_problem = GroupedLassoClassifyProblemWrapperSimpleFullCV(X_train_validate, y_train_validate, feature_group_sizes)
    all_kfolds_data = AllKFoldsData(X_train_validate, y_train_validate, feature_group_sizes, kfolds, GroupedLassoClassifyProblemWrapperSimpleFullCV)

    best_cost = 1e5
    best_regularization = [lambda_guesses[0], lambda_guesses[0]]
    for l1 in reversed(lambda_guesses):
        for l2 in reversed(lambda_guesses):
            regularization = [l1, l2]
            _, cost = all_kfolds_data.solve(regularization)
            if best_cost > cost:
                best_cost = cost
                best_regularization = regularization
                print "best_cost so far", best_cost, "best_regularization", best_regularization
                sys.stdout.flush()

    print "gridsearch: best_validation_error", best_cost
    print "gridsearch: best lambdas:", best_regularization

    betas = full_problem.solve(best_regularization)

    return betas, best_cost
