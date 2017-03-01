import time
from cvxpy import *
import cvxopt
from common import *
import scipy as sp
import cvxpy
from matrix_completion_groups_solver import MatrixCompletionGroupsProblem

SCS_MAX_ITERS = 10000
SCS_EPS = 1e-3 # default eps
SCS_HIGH_ACC_EPS = 1e-6
ECOS_TOL = 1e-12
REALDATA_MAX_ITERS = 4000

# Objective function: 0.5 * norm(y - Xb)^2 + lambda1 * lasso + 0.5 * lambda2 * ridge
class Lambda12ProblemWrapper:
    def __init__(self, X, y):
        n = X.shape[1]
        self.beta = Variable(n)
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")
        objective = Minimize(0.5 * sum_squares(y - X * self.beta)
            + self.lambda1 * norm(self.beta, 1)
            + 0.5 * self.lambda2 * sum_squares(self.beta))
        self.problem = Problem(objective, [])

    def solve(self, lambda1, lambda2, quick_run=None):
        self.lambda1.value = lambda1
        self.lambda2.value = lambda2
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        # print "self.problem.status", self.problem.status
        return self.beta.value

# Objective function: 0.5 * norm(y - Xb)^2 + lambda1 * lasso
class LassoProblemWrapper:
    def __init__(self, X, y):
        num_train = X.shape[0]
        self.beta = Variable(X.shape[1])
        self.lambda1 = Parameter(sign="positive")
        objective = Minimize(0.5/num_train * sum_squares(y - X * self.beta)
            + self.lambda1 * norm(self.beta, 1))
        self.problem = Problem(objective)

    def solve(self, lambdas, quick_run=None, warm_start=True):
        self.lambda1.value = lambdas[0]
        result = self.problem.solve(verbose=VERBOSE)
        return self.beta.value


# Objective function: 0.5 * norm(y - Xb)^2 + lambda1 * lasso + 0.5 * lambda2 * ridge
class ElasticNetProblemWrapper:
    def __init__(self, X, y):
        n = X.shape[1]
        self.beta = Variable(n)
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")
        objective = Minimize(0.5 * sum_squares(y - X * self.beta)
            + self.lambda1 * norm(self.beta, 1)
            + 0.5 * self.lambda2 * sum_squares(self.beta))
        self.problem = Problem(objective, [])

    def solve(self, lambdas, quick_run=None, warm_start=True):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]
        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value

class GroupedLassoProblemWrapper:
    def __init__(self, X, y, group_feature_sizes):
        self.group_range = range(0, len(group_feature_sizes))
        self.betas = [Variable(feature_size) for feature_size in group_feature_sizes]
        self.lambda1s = [Parameter(sign="positive") for i in self.group_range]
        self.lambda2 = Parameter(sign="positive")

        feature_start_idx = 0
        model_prediction = 0
        group_lasso_regularization = 0
        sparsity_regularization = 0
        for i in self.group_range:
            end_feature_idx = feature_start_idx + group_feature_sizes[i]
            model_prediction += X[:, feature_start_idx : end_feature_idx] * self.betas[i]
            feature_start_idx = end_feature_idx
            group_lasso_regularization += self.lambda1s[i] * norm(self.betas[i], 2)
            sparsity_regularization += norm(self.betas[i], 1)

        objective = Minimize(0.5 / y.size * sum_squares(y - model_prediction)
            + group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = Problem(objective, [])

    def solve(self, lambdas, quick_run=False):
        for idx in self.group_range:
            self.lambda1s[idx].value = lambdas[idx]

        self.lambda2.value = lambdas[-1]

        ecos_iters = 200
        try:
            self.problem.solve(solver=ECOS, verbose=VERBOSE, abstol=ECOS_TOL, reltol=ECOS_TOL, abstol_inacc=ECOS_TOL, reltol_inacc=ECOS_TOL, max_iters=ecos_iters)
        except SolverError:
            self.problem.solve(solver=SCS, verbose=VERBOSE, eps=SCS_HIGH_ACC_EPS/100, max_iters=SCS_MAX_ITERS * 4, use_indirect=False, normalize=False, warm_start=True)

        return [b.value for b in self.betas]

class GroupedLassoClassifyProblemWrapper:
    def __init__(self, X_groups, y):
        group_feature_sizes = [g.shape[1] for g in X_groups]
        self.group_range = range(0, len(group_feature_sizes))
        self.betas = [Variable(feature_size) for feature_size in group_feature_sizes]
        self.lambda1s = [Parameter(sign="positive") for i in self.group_range]
        self.lambda2 = Parameter(sign="positive")

        feature_start_idx = 0
        model_prediction = 0
        group_lasso_regularization = 0
        sparsity_regularization = 0
        for i in self.group_range:
            model_prediction += X_groups[i] * self.betas[i]
            group_lasso_regularization += self.lambda1s[i] * norm(self.betas[i], 2)
            sparsity_regularization += norm(self.betas[i], 1)

        log_sum = 0
        for i in range(0, X_groups[0].shape[0]):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - model_prediction.T * y
            + group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        for idx in self.group_range:
            self.lambda1s[idx].value = lambdas[idx]

        self.lambda2.value = lambdas[-1]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=REALDATA_MAX_ITERS, use_indirect=False, normalize=True)
        print "self.problem.status", self.problem.status
        return [b.value for b in self.betas]

class GroupedLassoClassifyProblemWrapperFullCV:
    def __init__(self, X, y, group_feature_sizes):
        self.group_range = range(0, len(group_feature_sizes))
        total_features = np.sum(group_feature_sizes)
        self.beta = Variable(total_features)
        self.lambda1s = [Parameter(sign="positive") for i in self.group_range]
        self.lambda2 = Parameter(sign="positive")

        start_feature_idx = 0
        group_lasso_regularization = 0
        for i, group_feature_size in enumerate(group_feature_sizes):
            end_feature_idx = start_feature_idx + group_feature_size
            group_lasso_regularization += self.lambda1s[i] * norm(self.beta[start_feature_idx:end_feature_idx], 2)
            start_feature_idx = end_feature_idx

        model_prediction = X * self.beta
        log_sum = 0
        num_samples = X.shape[0]
        for i in range(0, num_samples):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - (X * self.beta).T * y
            + group_lasso_regularization
            + self.lambda2 * norm(self.beta, 1))
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        for idx in self.group_range:
            self.lambda1s[idx].value = lambdas[idx]

        self.lambda2.value = lambdas[-1]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=REALDATA_MAX_ITERS, use_indirect=False, normalize=True)
        print "self.problem.status", self.problem.status
        return self.beta.value

class GroupedLassoProblemWrapperSimple:
    def __init__(self, X, y, group_feature_sizes):
        self.group_range = range(0, len(group_feature_sizes))
        self.betas = [Variable(feature_size) for feature_size in group_feature_sizes]
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")

        feature_start_idx = 0
        model_prediction = 0
        group_lasso_regularization = 0
        sparsity_regularization = 0
        for i in self.group_range:
            end_feature_idx = feature_start_idx + group_feature_sizes[i]
            model_prediction += X[:, feature_start_idx : end_feature_idx] * self.betas[i]
            feature_start_idx = end_feature_idx
            group_lasso_regularization += norm(self.betas[i], 2)
            sparsity_regularization += norm(self.betas[i], 1)

        objective = Minimize(0.5 / y.size * sum_squares(y - model_prediction)
            + self.lambda1 * group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = Problem(objective, [])

    def solve(self, lambdas, quick_run=False):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]

        if not quick_run:
            ecos_iters = 400
            tol=ECOS_TOL * 100
            try:
                self.problem.solve(solver=ECOS, verbose=VERBOSE, reltol=tol, abstol_inacc=tol, reltol_inacc=tol, max_iters=ecos_iters)
            except SolverError:
                self.problem.solve(solver=SCS, verbose=VERBOSE, eps=SCS_HIGH_ACC_EPS, max_iters=SCS_MAX_ITERS * 4, use_indirect=False, normalize=False, warm_start=True)
        else:
            try:
                self.problem.solve(solver=ECOS, verbose=VERBOSE)
            except SolverError:
                self.problem.solve(solver=SCS, verbose=VERBOSE, use_indirect=False, normalize=False, warm_start=True)
        return [b.value for b in self.betas]

class GroupedLassoClassifyProblemWrapperSimple:
    def __init__(self, X_groups, y):
        self.group_range = range(0, len(X_groups))
        self.betas = [Variable(Xg.shape[1]) for Xg in X_groups]
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")

        feature_start_idx = 0
        model_prediction = 0
        group_lasso_regularization = 0
        sparsity_regularization = 0
        for i, Xg in enumerate(X_groups):
            model_prediction += Xg * self.betas[i]
            group_lasso_regularization += norm(self.betas[i], 2)
            sparsity_regularization += norm(self.betas[i], 1)

        log_sum = 0
        for i in range(0, X_groups[0].shape[0]):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - model_prediction.T * y
            + self.lambda1 * group_lasso_regularization
            + self.lambda2 * sparsity_regularization)
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return [b.value for b in self.betas]

class GroupedLassoClassifyProblemWrapperSimpleFullCV:
    def __init__(self, X, y, feature_group_sizes):
        total_features = np.sum(feature_group_sizes)
        self.beta = Variable(total_features)
        self.lambda1 = Parameter(sign="positive")
        self.lambda2 = Parameter(sign="positive")

        start_feature_idx = 0
        group_lasso_regularization = 0
        for feature_group_size in feature_group_sizes:
            end_feature_idx = start_feature_idx + feature_group_size
            group_lasso_regularization += norm(self.beta[start_feature_idx:end_feature_idx], 2)
            start_feature_idx = end_feature_idx

        model_prediction = X * self.beta
        log_sum = 0
        num_samples = X.shape[0]
        for i in range(0, num_samples):
            one_plus_expyXb = vstack(0, model_prediction[i])
            log_sum += log_sum_exp(one_plus_expyXb)

        objective = Minimize(
            log_sum
            - model_prediction.T * y
            + self.lambda1 * group_lasso_regularization
            + self.lambda2 * norm(self.beta, 1))
        self.problem = Problem(objective, [])

    def solve(self, lambdas):
        self.lambda1.value = lambdas[0]
        self.lambda2.value = lambdas[1]

        result = self.problem.solve(solver=SCS, verbose=VERBOSE)
        return self.beta.value

class SparseAdditiveModelProblemWrapperSimple:
    # A two lambda version
    def __init__(self, X, train_indices, y, tiny_e=0):
        self.tiny_e = tiny_e
        self.y = y

        num_samples, num_features = X.shape
        self.num_samples = num_samples
        self.num_features = num_features

        # Create smooth penalty matrix for each feature
        self.diff_matrices = []
        for i in range(num_features):
            D = _make_discrete_diff_matrix_ord2(X[:,i])
            self.diff_matrices.append(D)

        self.train_indices = train_indices
        num_train = train_indices.size

        self.lambdas = [Parameter(sign="positive"), Parameter(sign="positive")]
        self.thetas = Variable(self.num_samples, self.num_features)
        objective = 0.5/num_train * sum_squares(self.y - sum_entries(self.thetas[self.train_indices,:], axis=1))
        # group-level sparsity penalty
        objective += sum([1.0/num_train * self.lambdas[0] * pnorm(self.thetas[:,i], 2) for i in range(self.num_features)])
        for i in range(len(self.diff_matrices)):
            # individual sparsity penalty - same lambda
            objective += 1.0/num_train * self.lambdas[1] * pnorm(self.diff_matrices[i] * self.thetas[:,i], 1)
        objective += 0.5 * self.tiny_e * sum_squares(self.thetas)
        self.problem = Problem(Minimize(objective))

    def solve(self, lambdas, warm_start=True, quick_run=False):
        for i,l in enumerate(lambdas):
            self.lambdas[i].value = lambdas[i]

        if not quick_run:
            eps = SCS_HIGH_ACC_EPS * 1e-3
            max_iters = SCS_MAX_ITERS * 10
        else:
            eps = SCS_EPS
            max_iters = SCS_MAX_ITERS

        if quick_run:
            self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=max_iters, eps=eps, warm_start=warm_start)
        else:
            self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=max_iters, use_indirect=False, eps=eps, normalize=False, warm_start=warm_start)

        return self.thetas.value

class SparseAdditiveModelProblemWrapper:
    def __init__(self, X, train_indices, y, tiny_e=0):
        self.tiny_e = tiny_e
        self.y = y

        num_samples, num_features = X.shape
        self.num_samples = num_samples
        self.num_features = num_features

        # Create smooth penalty matrix for each feature
        self.diff_matrices = []
        for i in range(num_features):
            D = _make_discrete_diff_matrix_ord2(X[:,i])
            self.diff_matrices.append(D)

        self.train_indices = train_indices
        self.lambdas = [Parameter(sign="positive")]
        for i in range(self.num_features):
            self.lambdas.append(Parameter(sign="positive"))

        self.thetas = Variable(self.num_samples, self.num_features)
        num_train = train_indices.size
        objective = 0.5/num_train * sum_squares(self.y - sum_entries(self.thetas[self.train_indices,:], axis=1))
        objective += sum([1.0/num_train * self.lambdas[0] * pnorm(self.thetas[:,i], 2) for i in range(self.num_features)])
        for i in range(len(self.diff_matrices)):
            objective += 1.0/num_train * self.lambdas[i + 1] * pnorm(self.diff_matrices[i] * self.thetas[:,i], 1)
        objective += 0.5/num_train * self.tiny_e * sum_squares(self.thetas)
        self.problem = Problem(Minimize(objective))

    def solve(self, lambdas, warm_start=True, quick_run=False):
        for i in range(lambdas.size):
            self.lambdas[i].value = lambdas[i]

        if not quick_run:
            eps = SCS_HIGH_ACC_EPS * 1e-3
            max_iters = SCS_MAX_ITERS * 10
        else:
            eps = SCS_EPS
            max_iters = SCS_MAX_ITERS

        if quick_run:
            self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=max_iters, eps=eps, warm_start=warm_start)
        else:
            self.problem.solve(solver=SCS, verbose=VERBOSE, max_iters=max_iters, use_indirect=False, eps=eps, normalize=False, warm_start=warm_start)

        if self.problem.value > 0 and self.problem.status in [OPTIMAL,  OPTIMAL_INACCURATE]:
            return self.thetas.value
        else:
            if self.problem.value < 0:
                print "Warning: Negative problem solution from cvxpy"
            return None

class MatrixCompletionGroupsProblemWrapperCustom:
    def __init__(self, data, tiny_e=0):
        assert(tiny_e == 0)
        self.problem = MatrixCompletionGroupsProblem(data)

    def solve(self, lambdas, warm_start=True, quick_run=False):
        # this always does warm starts
        start_time = time.time()
        self.problem.update(lambdas)
        if quick_run:
            tol = 1e-7
            max_iters = 20000
        else:
            tol = 1e-14
            max_iters = 80000

        alphas, betas, gamma = self.problem.solve(max_iters=max_iters, tol=tol)
        return {
            "alphas": alphas,
            "betas": betas,
            "gamma": gamma
        }

class MatrixCompletionGroupsProblemWrapperSimple:
    def __init__(self, data, tiny_e=0):
        assert(tiny_e == 0)
        self.problem = MatrixCompletionGroupsProblem(data)
        self.num_lambda1s = data.num_alphas + data.num_betas

    def solve(self, lambdas, warm_start=True, quick_run=False):
        # this always does warm starts
        start_time = time.time()
        exploded_lambdas = np.array([lambdas[0]] + [lambdas[1]] * self.num_lambda1s)
        self.problem.update(exploded_lambdas)
        if quick_run:
            tol = 1e-7
            max_iters = 20000
        else:
            tol = 1e-14
            max_iters = 80000

        alphas, betas, gamma = self.problem.solve(max_iters=max_iters, tol=tol)
        return {
            "alphas": alphas,
            "betas": betas,
            "gamma": gamma
        }

def _make_discrete_diff_matrix_ord2(x_features):
    num_samples = len(x_features)
    d1_matrix = np.matrix(np.zeros((num_samples, num_samples)))
    # 1st, figure out ordering of samples for the feature
    sample_ordering = np.argsort(x_features)
    ordered_x = x_features[sample_ordering]
    d1_matrix[range(num_samples - 1), sample_ordering[:-1]] = -1
    d1_matrix[range(num_samples - 1), sample_ordering[1:]] = 1
    inv_dists = 1.0 / (ordered_x[np.arange(1, num_samples)] - ordered_x[np.arange(num_samples - 1)])
    inv_dists = np.append(inv_dists, 0)

    # Check that the inverted distances are all greater than zero
    assert(np.min(inv_dists) >= 0)
    D = d1_matrix * np.matrix(np.diagflat(inv_dists)) * d1_matrix
    return D
