import sys
import time
import scipy as sp
from scipy.sparse.linalg import spsolve, lsqr
from fitted_model import Fitted_Model
import numpy as np
import cvxpy

from common import *
from convexopt_solvers import SparseAdditiveModelProblemWrapper, SparseAdditiveModelProblemWrapperSimple
from gradient_descent_algo import Gradient_Descent_Algo

class BetaForm:
    eps = 1e-8

    def __init__(self, idx, theta, diff_matrix, log_file):
        self.log_file = log_file
        self.log("create beta form")
        self.idx = idx
        self.theta = theta
        self.theta_norm = sp.linalg.norm(theta, ord=None)
        self.diff_matrix = diff_matrix

        # Find the null space for the subsetted diff matrix
        start = time.time()
        zero_theta_idx = self._get_zero_theta_indices(diff_matrix * theta)
        u, s, v = sp.linalg.svd(diff_matrix[zero_theta_idx,:])
        self.log("SVD done %f" % (time.time() - start))
        null_mask = np.ones(v.shape[1])
        null_mask[:s.size] = s <= self.eps
        null_space = sp.compress(null_mask, v, axis=0)
        null_matrix = np.matrix(sp.transpose(null_space))
        start = time.time()
        beta, istop, itn, normr, normar, norma, conda, normx = sp.sparse.linalg.lsmr(null_matrix, theta.A1, atol=self.eps, btol=self.eps)
        self.log("sp.sparse.linalg.lsmr done %f, istop %d, itn %d" % ((time.time() - start), istop, itn))
        self.beta = np.matrix(beta).T
        self.u = null_matrix

        # Check that we reformulated theta but it is still very close to the original theta
        if sp.linalg.norm(self.u * self.beta - self.theta, ord=2) > self.eps:
            self.log("Warning: Reformulation is off: diff %f" % sp.linalg.norm(self.u * self.beta - self.theta, ord=2))
        self.log("create beta form success")

    def log(self, log_str):
        if self.log_file is None:
            print log_str
        else:
            self.log_file.write("%s\n" % log_str)

    def __str__(self):
        return "beta %s, theta %s" % (self.beta, self.theta)

    @staticmethod
    def _get_zero_theta_indices(theta, threshold=1e-10):
        return np.less(np.abs(theta), threshold).A1

class Sparse_Add_Model_Hillclimb_Base(Gradient_Descent_Algo):
    def _create_descent_settings(self):
        self.num_iters = 15
        self.step_size_init = 1
        self.step_size_min = 1e-6
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-4 * 5
        self.use_boundary = True
        self.boundary_factor = 0.999999
        self.backtrack_alpha = 0.001

    def get_validate_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            model_params
        )

class Sparse_Add_Model_Hillclimb(Sparse_Add_Model_Hillclimb_Base):
    method_label = "Sparse_Add_Model_Hillclimb"

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6] * (self.data.num_features + 1)

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapper(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )
        self.train_I = np.matrix(np.eye(self.data.num_samples)[self.data.train_idx,:])

    def _get_lambda_derivatives(self):
        # First filter out the thetas that are completely zero
        nonzero_thetas_idx = self._get_nonzero_theta_vectors(self.fmodel.current_model_params)
        self.log("nonzero_thetas_idx %s" % nonzero_thetas_idx)
        # Now reformulate the remaining thetas using the differentiable space
        nonzeros_idx = np.where(nonzero_thetas_idx)[0]

        if nonzeros_idx.size == 0:
            return np.array([0] * self.fmodel.num_lambdas)

        beta_u_forms = map(
            lambda i: BetaForm(i, self.fmodel.current_model_params[:,i], self.problem_wrapper.diff_matrices[i], log_file=self.log_file),
            nonzeros_idx
        )
        sum_dtheta_dlambda = self._get_sum_dtheta_dlambda(beta_u_forms, nonzero_thetas_idx)
        fitted_y_validate = np.sum(self.fmodel.current_model_params[self.data.validate_idx, :], axis=1)
        dloss_dlambda = -1.0/self.data.y_validate.size * sum_dtheta_dlambda[self.data.validate_idx, :].T * (self.data.y_validate - fitted_y_validate)
        return dloss_dlambda.A1 # flatten the matrix

    def _get_sum_dtheta_dlambda(self, beta_u_forms, nonzero_thetas_idx):
        def create_b_diag_elem(i):
            u = beta_u_forms[i].u
            beta = beta_u_forms[i].beta
            theta_norm = beta_u_forms[i].theta_norm
            b_diag_elem = 1.0/theta_norm * (np.eye(beta.size) - beta * beta.T/(theta_norm**2))
            return b_diag_elem

        def make_rhs_col0(i):
            theta_norm = beta_u_forms[i].theta_norm
            beta = beta_u_forms[i].beta
            return beta/theta_norm

        def make_diag_rhs(i):
            u = beta_u_forms[i].u
            theta = beta_u_forms[i].theta
            diff_matrix = beta_u_forms[i].diff_matrix
            theta_norm = beta_u_forms[i].theta_norm
            # Zero out the entries that are essentially zero.
            # Otherwise np.sign will give non-zero values
            zeroed_diff_theta = self._zero_theta_indices(diff_matrix * theta)
            return u.T * diff_matrix.T * np.sign(zeroed_diff_theta)

        num_nonzero_thetas = len(beta_u_forms)

        # Create part of the Hessian matrix
        b_diag_elems = map(create_b_diag_elem, range(num_nonzero_thetas))
        b_diag = sp.linalg.block_diag(*b_diag_elems)

        # Create rhs elements
        rhs_col0 = map(make_rhs_col0, range(num_nonzero_thetas))
        rhs_col0 = np.vstack(rhs_col0)
        rhs_diag = map(make_diag_rhs, range(num_nonzero_thetas))
        rhs_diag = sp.linalg.block_diag(*rhs_diag)
        insert_idx = np.minimum(np.arange(self.data.num_features)[~nonzero_thetas_idx], rhs_diag.shape[1])
        # insert zero columns that corresponded to the zero thetas
        rhs_diag = np.insert(rhs_diag, insert_idx, np.zeros((rhs_diag.shape[0], 1)), axis=1)
        rhs_matrix = np.hstack((rhs_col0, rhs_diag))

        lambda0 = self.fmodel.current_lambdas[0]
        u_matrices = map(lambda i: beta_u_forms[i].u, range(num_nonzero_thetas))
        u_matrices = np.hstack(u_matrices)
        uu = u_matrices.T * self.train_I.T * self.train_I * u_matrices
        tiny_e_matrix = self.problem_wrapper.tiny_e * np.eye(uu.shape[0])
        hessian = uu + lambda0 * b_diag + tiny_e_matrix

        start = time.time()
        dbeta_dlambda = map(
            lambda j: np.matrix(sp.sparse.linalg.lsmr(hessian, -1 * rhs_matrix[:,j].A1)[0]).T,
            range(rhs_matrix.shape[1])
        )
        dbeta_dlambda = np.hstack(dbeta_dlambda)
        self.log("lsmr time %f" % (time.time() - start))
        sum_dtheta_dlambda = u_matrices * dbeta_dlambda
        return sum_dtheta_dlambda

    @staticmethod
    def _get_nonzero_theta_vectors(thetas, threshold=1e-8):
        nonzero_thetas_idx = map(lambda i: sp.linalg.norm(thetas[:,i], ord=2) > threshold, range(thetas.shape[1]))
        return np.array(nonzero_thetas_idx)

    @staticmethod
    def _zero_theta_indices(theta, threshold=1e-8):
        return np.multiply(theta, np.greater(np.abs(theta), threshold))

class Sparse_Add_Model_Hillclimb_Simple(Sparse_Add_Model_Hillclimb_Base):
    method_label = "Sparse_Add_Model_Hillclimb_Simple"

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6, 1e-6]

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapperSimple(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )
        self.train_I = np.matrix(np.eye(self.data.num_samples)[self.data.train_idx,:])

    def _get_lambda_derivatives(self):
        # First filter out the thetas that are completely zero
        nonzero_thetas_idx = self._get_nonzero_theta_vectors(self.fmodel.current_model_params)
        self.log("nonzero_thetas_idx %s" % nonzero_thetas_idx)
        # Now reformulate the remaining thetas using the differentiable space
        nonzeros_idx = np.where(nonzero_thetas_idx)[0]

        if nonzeros_idx.size == 0:
            return np.array([0] * self.fmodel.num_lambdas)

        beta_u_forms = map(
            lambda i: BetaForm(i, self.fmodel.current_model_params[:,i], self.problem_wrapper.diff_matrices[i], log_file=self.log_file),
            nonzeros_idx
        )
        sum_dtheta_dlambda = self._get_sum_dtheta_dlambda(beta_u_forms, nonzero_thetas_idx)
        fitted_y_validate = np.sum(self.fmodel.current_model_params[self.data.validate_idx, :], axis=1)
        dloss_dlambda = -1.0/self.data.y_validate.size * sum_dtheta_dlambda[self.data.validate_idx, :].T * (self.data.y_validate - fitted_y_validate)
        return dloss_dlambda.A1 # flatten the matrix

    def _get_sum_dtheta_dlambda(self, beta_u_forms, nonzero_thetas_idx):
        def create_b_diag_elem(i):
            u = beta_u_forms[i].u
            beta = beta_u_forms[i].beta
            theta_norm = beta_u_forms[i].theta_norm
            b_diag_elem = 1.0/theta_norm * (np.eye(beta.size) - beta * beta.T/(theta_norm**2))
            return b_diag_elem

        def make_rhs_col0(i):
            theta_norm = beta_u_forms[i].theta_norm
            beta = beta_u_forms[i].beta
            return beta/theta_norm

        def make_rhs_col1(i):
            u = beta_u_forms[i].u
            theta = beta_u_forms[i].theta
            diff_matrix = beta_u_forms[i].diff_matrix
            theta_norm = beta_u_forms[i].theta_norm
            # Zero out the entries that are essentially zero.
            # Otherwise np.sign will give non-zero values
            zeroed_diff_theta = self._zero_theta_indices(diff_matrix * theta)
            return u.T * diff_matrix.T * np.sign(zeroed_diff_theta)

        num_nonzero_thetas = len(beta_u_forms)

        # Create part of the Hessian matrix
        b_diag_elems = map(create_b_diag_elem, range(num_nonzero_thetas))
        b_diag = sp.linalg.block_diag(*b_diag_elems)

        # Create rhs elements
        rhs_col0 = map(make_rhs_col0, range(num_nonzero_thetas))
        rhs_col0 = np.vstack(rhs_col0)
        rhs_col1 = map(make_rhs_col1, range(num_nonzero_thetas))
        rhs_col1 = np.vstack(rhs_col1)
        rhs_matrix = np.hstack((rhs_col0, rhs_col1))

        lambda0 = self.fmodel.current_lambdas[0]
        u_matrices = map(lambda i: beta_u_forms[i].u, range(num_nonzero_thetas))
        u_matrices = np.hstack(u_matrices)
        uu = u_matrices.T * self.train_I.T * self.train_I * u_matrices
        tiny_e_matrix = self.problem_wrapper.tiny_e * np.eye(uu.shape[0])
        hessian = uu + lambda0 * b_diag + tiny_e_matrix

        start = time.time()
        dbeta_dlambda = map(
            lambda j: np.matrix(sp.sparse.linalg.lsmr(hessian, -1 * rhs_matrix[:,j].A1)[0]).T,
            range(rhs_matrix.shape[1])
        )
        dbeta_dlambda = np.hstack(dbeta_dlambda)
        self.log("lsmr time %f" % (time.time() - start))
        sum_dtheta_dlambda = u_matrices * dbeta_dlambda
        return sum_dtheta_dlambda

    @staticmethod
    def _get_nonzero_theta_vectors(thetas, threshold=1e-8):
        nonzero_thetas_idx = map(lambda i: sp.linalg.norm(thetas[:,i], ord=2) > threshold, range(thetas.shape[1]))
        return np.array(nonzero_thetas_idx)

    @staticmethod
    def _zero_theta_indices(theta, threshold=1e-8):
        return np.multiply(theta, np.greater(np.abs(theta), threshold))
