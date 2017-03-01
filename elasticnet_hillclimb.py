import numpy as np
import scipy as sp
from common import testerror_elastic_net
from convexopt_solvers import ElasticNetProblemWrapper
from gradient_descent_algo import Gradient_Descent_Algo

class Elastic_Net_Hillclimb(Gradient_Descent_Algo):
    method_label = "Elastic_Net_Hillclimb"

    def _create_descent_settings(self):
        self.num_iters = 30
        self.step_size_init = 1
        self.step_size_min = 1e-8
        self.shrink_factor = 0.1
        self.use_boundary = False
        self.boundary_factor = 0.7
        self.decr_enough_threshold = 1e-4 * 5
        self.backtrack_alpha = 0.001

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-5] * 2

    def _create_problem_wrapper(self):
        self.problem_wrapper = ElasticNetProblemWrapper(
            self.data.X_train,
            self.data.y_train
        )

    def get_validate_cost(self, model_params):
        return testerror_elastic_net(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )

    def _get_lambda_derivatives(self):
        betas = self.fmodel.current_model_params
        nonzero_indices = self.get_nonzero_indices(betas)

        # If everything is zero, gradient is zero
        if np.sum(nonzero_indices) == 0:
            return np.zeros((1,2))

        X_train_mini = self.data.X_train[:, nonzero_indices]
        X_validate_mini = self.data.X_validate[:, nonzero_indices]
        betas_mini = betas[nonzero_indices]

        eye_matrix = np.matrix(np.identity(betas_mini.size))
        # Note: on certain computers, it will be difficult to run X_train_mini.T * X_train_mini in parallel
        to_invert_matrix = X_train_mini.T * X_train_mini + self.fmodel.current_lambdas[1] * eye_matrix

        dbeta_dlambda1, istop, itn, normr, normar, norma, conda, normx = sp.sparse.linalg.lsmr(to_invert_matrix, -1 * np.sign(betas_mini).A1)
        dbeta_dlambda1 = np.matrix(dbeta_dlambda1).T
        dbeta_dlambda2, istop, itn, normr, normar, norma, conda, normx = sp.sparse.linalg.lsmr(to_invert_matrix, -1 * betas_mini.A1)
        dbeta_dlambda2 = np.matrix(dbeta_dlambda2).T

        err_vector = self.data.y_validate - X_validate_mini * betas_mini
        gradient_lambda1 = -1 * (X_validate_mini * dbeta_dlambda1).T * err_vector
        gradient_lambda2 = -1 * (X_validate_mini * dbeta_dlambda2).T * err_vector

        return np.array([gradient_lambda1[0,0], gradient_lambda2[0,0]])

    @staticmethod
    def get_nonzero_indices(some_vector, threshold=1e-4):
        return np.reshape(np.array(np.greater(np.abs(some_vector), threshold).T), (some_vector.size, ))
