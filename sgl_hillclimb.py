import numpy as np
import scipy as sp
from common import testerror_grouped, get_norm2
from gradient_descent_algo import Gradient_Descent_Algo
from convexopt_solvers import GroupedLassoProblemWrapper, GroupedLassoProblemWrapperSimple

class SGL_Hillclimb_Base(Gradient_Descent_Algo):
    def _create_descent_settings(self):
        self.num_iters = 20
        self.step_size_init = 1
        self.step_size_min = 1e-6
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-4 * 5
        self.use_boundary = False
        self.boundary_factor = 0.999999
        self.backtrack_alpha = 0.001

    def get_validate_cost(self, model_params):
        return testerror_grouped(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )

    def _get_lambda_derivatives(self):
        # restrict the derivative to the differentiable space
        beta_minis = []
        beta_nonzeros = []
        for beta in self.fmodel.current_model_params:
            beta_nonzero = self._get_nonzero_indices(beta)
            beta_nonzeros.append(beta_nonzero)
            beta_minis.append(beta[beta_nonzero])

        complete_beta_nonzero = np.concatenate(beta_nonzeros)
        X_train_mini = self.data.X_train[:, complete_beta_nonzero]
        X_validate_mini = self.data.X_validate[:, complete_beta_nonzero]

        if complete_beta_nonzero.size == 0:
            return np.zeros(self.fmodel.current_lambdas)

        return self._get_lambda_derivatives_mini(X_train_mini, X_validate_mini, beta_minis)

class SGL_Hillclimb(SGL_Hillclimb_Base):
    method_label = "SGL_Hillclimb"

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6] * (self.settings.expert_num_groups + 1)

    def _create_problem_wrapper(self):
        self.problem_wrapper = GroupedLassoProblemWrapper(
            self.data.X_train,
            self.data.y_train,
            self.settings.get_expert_group_sizes()
        )

    def _get_lambda_derivatives_mini(self, X_train_mini, X_validate_mini, beta_minis):
        def _get_block_diag_component(idx):
            beta = beta_minis[idx]
            if beta.size == 0:
                return np.matrix(np.zeros((0,0))).T

            betabeta = beta * beta.T
            block_diag_component = -1 * self.fmodel.current_lambdas[idx] / get_norm2(beta, power=3) * betabeta
            return block_diag_component

        def _get_diagmatrix_component(idx):
            beta = beta_minis[idx]
            if beta.size == 0:
                return np.matrix(np.zeros((0,0))).T
            return self.fmodel.current_lambdas[idx] / get_norm2(beta) * np.identity(beta.size)

        def _get_dbeta_dlambda1(beta, matrix_to_invert, num_features_before):
            if beta.size == 0:
                return np.zeros((matrix_to_invert.shape[0], 1))
            else:
                normed_beta = beta / get_norm2(beta)
                zero_normed_beta = np.concatenate([
                    np.matrix(np.zeros(num_features_before)).T,
                    normed_beta,
                    np.matrix(np.zeros(total_features - normed_beta.size - num_features_before)).T
                ])

                dbeta_dlambda1 = sp.sparse.linalg.lsmr(matrix_to_invert, -1 * zero_normed_beta.A1)[0]
                return np.matrix(dbeta_dlambda1).T

        total_features = X_train_mini.shape[1]
        complete_beta = np.concatenate(beta_minis)

        XX = X_train_mini.T * X_train_mini

        block_diag_components = [_get_block_diag_component(idx) for idx in range(0, self.settings.expert_num_groups)]
        diagonal_components = [_get_diagmatrix_component(idx) for idx in range(0, self.settings.expert_num_groups)]
        dgrouplasso_dlambda = sp.linalg.block_diag(*block_diag_components) + sp.linalg.block_diag(*diagonal_components)

        matrix_to_invert = 1.0 / self.data.num_train * XX + dgrouplasso_dlambda

        dbeta_dlambda1s = None
        num_features_before = 0
        for beta in beta_minis:
            dbeta_dlambda1 = _get_dbeta_dlambda1(beta, matrix_to_invert, num_features_before)
            num_features_before += beta.size

            if dbeta_dlambda1s is None:  # not initialized yet
                dbeta_dlambda1s = dbeta_dlambda1
            else:
                dbeta_dlambda1s = np.hstack([dbeta_dlambda1s, dbeta_dlambda1])

        dbeta_dlambda1s = np.matrix(dbeta_dlambda1s)
        dbeta_dlambda2 = np.matrix(sp.sparse.linalg.lsmr(matrix_to_invert, -1 * np.sign(complete_beta).A1)[0]).T

        err_vector = self.data.y_validate - X_validate_mini * complete_beta
        df_dlambda1s = -1.0 / self.data.num_validate * (X_validate_mini * dbeta_dlambda1s).T * err_vector
        df_dlambda1s = np.reshape(np.array(df_dlambda1s), df_dlambda1s.size)
        df_dlambda2 = -1.0 / self.data.num_validate * (X_validate_mini * dbeta_dlambda2).T * err_vector
        return np.concatenate((df_dlambda1s, [df_dlambda2[0,0]]))

    @staticmethod
    def _get_nonzero_indices(beta, threshold=1e-4):
        return np.reshape(np.array(np.greater(np.abs(beta), threshold).T), (beta.size, ))


class SGL_Hillclimb_Simple(SGL_Hillclimb_Base):
    method_label = "SGL_Hillclimb_Simple"

    def _create_lambda_configs(self):
        self.lambda_mins = [1e-6, 1e-6]

    def _create_problem_wrapper(self):
        self.problem_wrapper = GroupedLassoProblemWrapperSimple(
            self.data.X_train,
            self.data.y_train,
            self.settings.get_expert_group_sizes()
        )

    def _get_lambda_derivatives_mini(self, X_train_mini, X_validate_mini, beta_minis):
        def _get_block_diag_component(idx):
            beta = beta_minis[idx]
            if beta.size == 0:
                return np.matrix(np.zeros((0,0))).T

            betabeta = beta * beta.T
            block_diag_component = -1 * self.fmodel.current_lambdas[0] / get_norm2(beta, power=3) * betabeta
            return block_diag_component

        def _get_diagmatrix_component(idx):
            beta = beta_minis[idx]
            if beta.size == 0:
                return np.matrix(np.zeros((0,0))).T
            return self.fmodel.current_lambdas[0] / get_norm2(beta) * np.identity(beta.size)

        def _get_dbeta_dlambda1(beta_minis, matrix_to_invert):
            if np.concatenate(beta_minis).size == 0:
                return np.zeros((matrix_to_invert.shape[0], 1))
            else:
                normed_betas = [beta / get_norm2(beta) for beta in beta_minis]
                all_normed_betas = np.concatenate(normed_betas)
                dbeta_dlambda1 = sp.sparse.linalg.lsmr(matrix_to_invert, -1 * all_normed_betas.A1)
                return np.matrix(dbeta_dlambda1[0]).T

        total_features = X_train_mini.shape[1]
        complete_beta = np.concatenate(beta_minis)

        XX = X_train_mini.T * X_train_mini

        block_diag_components = [_get_block_diag_component(idx) for idx in range(0, self.settings.expert_num_groups)]
        diagonal_components = [_get_diagmatrix_component(idx) for idx in range(0, self.settings.expert_num_groups)]
        dgrouplasso_dlambda = sp.linalg.block_diag(*block_diag_components) + sp.linalg.block_diag(*diagonal_components)

        matrix_to_invert = 1.0 / self.data.num_train * XX + dgrouplasso_dlambda

        dbeta_dlambda1 = _get_dbeta_dlambda1(beta_minis, matrix_to_invert)
        dbeta_dlambda2 = np.matrix(sp.sparse.linalg.lsmr(matrix_to_invert, -1 * np.sign(complete_beta).A1)[0]).T

        err_vector = self.data.y_validate - X_validate_mini * complete_beta
        df_dlambda1 = -1.0 / self.data.num_validate * (X_validate_mini * dbeta_dlambda1).T * err_vector
        df_dlambda2 = -1.0 / self.data.num_validate * (X_validate_mini * dbeta_dlambda2).T * err_vector
        return np.concatenate(([df_dlambda1[0,0]], [df_dlambda2[0,0]]))

    @staticmethod
    def _get_nonzero_indices(beta, threshold=1e-4):
        return np.reshape(np.array(np.greater(np.abs(beta), threshold).T), (beta.size, ))
