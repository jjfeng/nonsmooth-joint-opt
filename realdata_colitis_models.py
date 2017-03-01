import numpy as np
import scipy as sp
from common import *

CLOSE_TO_ZERO_THRESHOLD = 1e-5

class KFoldData:
    def __init__(self, X_train, y_train, X_validate, y_validate, feature_group_sizes, problem_wrapper_klass):
        self.feature_group_sizes = feature_group_sizes
        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate
        self.problem_wrapper = problem_wrapper_klass(X_train, y_train, feature_group_sizes)
        self.beta = []

    def solve(self, lambda_vals):
        """
        Solves for the given lambda values BUT does not set the beta value.
        Returns both the optimal beta and the validation cost
        """
        beta = self.problem_wrapper.solve(lambda_vals)
        cost, _, _, _ = testerror_logistic_grouped(self.X_validate, self.y_validate, beta)
        return beta, cost

    def set_beta(self, beta):
        self.beta = beta

    def get_betas(self):
        """
        Returns the betas split into feature groups
        """
        betas = []
        start_feature_idx = 0
        for feature_group_size in self.feature_group_sizes:
            end_feature_idx = start_feature_idx + feature_group_size
            betas.append(
                self.beta[start_feature_idx:end_feature_idx]
            )
            start_feature_idx = end_feature_idx
        return betas

    def get_lambda_derivatives(self, lambda_vals):
        """
        Returns df_dlambdas
        Performs minification first
        """
        betas = self.get_betas()

        # first minify the data
        beta_minis = []
        beta_nonzeros = []
        for beta in betas:
            beta_nonzero = get_nonzero_indices(beta, threshold=CLOSE_TO_ZERO_THRESHOLD)
            beta_nonzeros.append(beta_nonzero)
            beta_minis.append(beta[beta_nonzero])

        complete_beta_nonzero = np.concatenate(beta_nonzeros)
        X_train_mini = self.X_train[:, complete_beta_nonzero]
        X_validate_mini = self.X_validate[:, complete_beta_nonzero]

        # If too small, just exist with zeros
        if X_train_mini.size == 0:
            return np.array([0] * len(lambda_vals))

        return self._get_lambda_derivatives_mini(X_train_mini, X_validate_mini, beta_minis, lambda_vals)

    def _get_lambda_derivatives_mini(self, X_train_mini, X_validate_mini, beta_minis, lambda_vals):
        """
        Accepts the minified version of X_train and X_validate and betas
        The only function that should be calling this is get_lambda_derivatives

        Returns df_dlambdas
        """
        lambda1s = lambda_vals[0:-1]

        def _get_block_diag_component(idx):
            beta = beta_minis[idx]
            if beta.size == 0:
                return np.matrix(np.zeros((0,0))).T

            repeat_hstacked_beta = np.tile(beta, (1, beta.size)).T
            block_diag_component = -1 * lambda1s[idx] / get_norm2(beta, power=3) * np.diagflat(beta) * repeat_hstacked_beta
            return block_diag_component

        def _get_diagmatrix_component(idx):
            beta = beta_minis[idx]
            if beta.size == 0:
                return np.matrix(np.zeros((0,0))).T
            return lambda1s[idx] / get_norm2(beta) * np.identity(beta.size)

        def _get_dbeta_dlambda1(beta, inverted_matrix, num_features_before):
            if beta.size == 0:
                return np.matrix(np.zeros((inverted_matrix.shape[0], 1)))
            else:
                normed_beta = beta / get_norm2(beta)
                zero_normed_beta = np.concatenate([
                    np.matrix(np.zeros(num_features_before)).T,
                    normed_beta,
                    np.matrix(np.zeros(total_features - normed_beta.size - num_features_before)).T
                ])

                dbeta_dlambda1 = -1 * inverted_matrix * zero_normed_beta
                return dbeta_dlambda1

        num_feature_groups = len(beta_minis)
        total_features = X_train_mini.shape[1]
        complete_beta = np.matrix(np.concatenate(beta_minis))

        exp_Xb = np.matrix(np.exp(X_train_mini * complete_beta))
        diag_expXb_components = np.diagflat(np.multiply(np.power(1 + exp_Xb, -2), exp_Xb))

        block_diag_components = [_get_block_diag_component(idx) for idx in range(0, num_feature_groups)]
        diagonal_components = [_get_diagmatrix_component(idx) for idx in range(0, num_feature_groups)]
        dgrouplasso_dlambda = sp.linalg.block_diag(*block_diag_components) + sp.linalg.block_diag(*diagonal_components)

        matrix_to_invert = X_train_mini.T * diag_expXb_components * X_train_mini + dgrouplasso_dlambda
        inverted_matrix = sp.linalg.pinvh(matrix_to_invert, rcond=1e-10)
        dbeta_dlambda1s = np.matrix(np.zeros((0,0))).T
        num_features_before = 0

        for beta in beta_minis:
            dbeta_dlambda1 = _get_dbeta_dlambda1(beta, inverted_matrix, num_features_before)
            num_features_before += beta.size

            if dbeta_dlambda1s.size == 0:  # not initialized yet
                dbeta_dlambda1s = dbeta_dlambda1
            else:
                dbeta_dlambda1s = np.hstack([dbeta_dlambda1s, dbeta_dlambda1])

        dbeta_dlambda2 = inverted_matrix * -1 * np.sign(complete_beta)

        expXvBeta = np.exp(X_validate_mini * complete_beta)
        dloss_dbeta = X_validate_mini.T * (-1 * self.y_validate + 1 - np.power(1 + expXvBeta, -1))
        df_dlambda1s = dloss_dbeta.T * dbeta_dlambda1s
        df_dlambda1s = np.reshape(np.array(df_dlambda1s), df_dlambda1s.size)
        df_dlambda2 = dloss_dbeta.T * dbeta_dlambda2
        return np.concatenate((df_dlambda1s, [df_dlambda2[0,0]]))



class AllKFoldsData:
    def __init__(self, X_train_validate, y_train_validate, feature_group_sizes, kfolds, problem_wrapper_klass):
        self.kfolds = kfolds
        self.feature_group_sizes = feature_group_sizes
        self.all_kfolds_data = []
        all_indices = np.arange(0, y_train_validate.size)
        for k in range(0, kfolds):
            validate_start = k * y_train_validate.size / kfolds
            validate_end = (k + 1) * y_train_validate.size / kfolds
            indices_wanted = np.concatenate((all_indices[0:validate_start], all_indices[validate_end:]))

            X_train_k = X_train_validate[indices_wanted, :]
            y_train_k = y_train_validate[indices_wanted]
            X_validate_k = X_train_validate[validate_start:validate_end, :]
            y_validate_k = y_train_validate[validate_start:validate_end]

            self.all_kfolds_data.append(KFoldData(
                X_train_k,
                y_train_k,
                X_validate_k,
                y_validate_k,
                feature_group_sizes,
                problem_wrapper_klass
            ))

    def solve(self, lambda_vals):
        """
        Solve for the given lambda values
        Returns the set of beta values for each of the k-folds and total validation cost across folds
        """
        total_cost = 0
        betas = []
        for kfold in self.all_kfolds_data:
            beta, cost = kfold.solve(lambda_vals)
            betas.append(beta)
            total_cost += cost
        return betas, total_cost

    def set_betas(self, betas):
        """
        Set the beta values for each of the folds to the given betas parameter
        """
        for i, kfold in enumerate(self.all_kfolds_data):
            kfold.set_beta(betas[i])

    def get_lambda_derivatives(self, lambda_vals):
        """
        Calculate the lambda derivative, aggregating across all k folds
        """
        derivs = 0
        for kfold in self.all_kfolds_data:
            derivs += kfold.get_lambda_derivatives(lambda_vals)
        return derivs
