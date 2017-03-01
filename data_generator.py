import numpy as np

from common import *

class ObservedData:
    # data structure for storing simple vector-valued observations
    def __init__(self, X_train, y_train, X_validate, y_validate, X_test, y_test):
        self.num_features = X_train.shape[1]

        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate
        self.X_test = X_test
        self.y_test = y_test

        self.num_train = y_train.size
        self.num_validate = y_validate.size
        self.num_test = y_test.size
        self.num_samples = self.num_train + self.num_validate + self.num_test

        self.X_full = np.vstack((X_train, X_validate, X_test))

        self.train_idx = np.arange(0, self.num_train)
        self.validate_idx = np.arange(self.num_train, self.num_train + self.num_validate)
        self.test_idx = np.arange(self.num_train + self.num_validate, self.num_train + self.num_validate + self.num_test)

class MatrixGroupsObservedData:
    # special data structure for storing matrix-value observations
    def __init__(self, row_features, col_features, train_idx, validate_idx, test_idx, observed_matrix, alphas, betas, gamma, real_matrix):
        self.num_rows = real_matrix.shape[0]
        self.num_cols = real_matrix.shape[1]

        self.num_alphas = len(alphas)
        self.num_betas = len(betas)

        self.row_features = row_features
        self.col_features = col_features
        self.train_idx = train_idx
        self.validate_idx = validate_idx
        self.test_idx = test_idx
        self.observed_matrix = observed_matrix

        self.real_matrix = real_matrix
        self.real_alphas = alphas
        self.real_betas = betas
        self.real_gamma = gamma

class DataGenerator:
    def __init__(self, settings):
        self.settings = settings
        self.train_size = settings.train_size
        self.validate_size = settings.validate_size
        self.test_size = settings.test_size
        self.total_samples = settings.train_size + settings.validate_size + settings.test_size
        self.snr = settings.snr
        self.feat_range = settings.feat_range

    def make_additive_smooth_data(self, smooth_fcn_list):
        self.num_features = len(smooth_fcn_list)
        all_Xs = map(lambda x: self._make_shuffled_uniform_X(), range(self.num_features))
        X_smooth = np.column_stack(all_Xs)

        y_smooth = 0
        for idx, fcn in enumerate(smooth_fcn_list):
            y_smooth += fcn(X_smooth[:, idx]).reshape(self.total_samples, 1)

        return self._make_data(y_smooth, X_smooth)

    def make_simple_linear(self, num_features, num_nonzero_features):
        self.num_features = num_features
        X = np.matrix(np.random.randn(self.total_samples, num_features))

        # beta real is a shuffled array of zeros and iid std normal values
        beta_real = np.matrix(
            np.concatenate((
                np.ones((num_nonzero_features, 1)),
                np.zeros((num_features - num_nonzero_features, 1))
            ))
        )

        true_y = X * beta_real
        data = self._make_data(true_y, X)
        data.beta_real = beta_real
        return data

    def make_correlated(self, num_features, num_nonzero_features):
        self.num_features = num_features
        # Multiplying by the cholesky decomposition of the covariance matrix should suffice: http://www.sitmo.com/article/generating-correlated-random-numbers/
        correlation_matrix = np.matrix([[np.power(0.5, abs(i - j)) for i in range(0, num_features)] for j in range(0, num_features)])
        X = np.matrix(np.random.randn(self.total_samples, num_features)) * np.matrix(np.linalg.cholesky(correlation_matrix)).T

        # beta real is a shuffled array of zeros and iid std normal values
        beta_real = np.matrix(
            np.concatenate((
                np.ones((num_nonzero_features, 1)),
                np.zeros((num_features - num_nonzero_features, 1))
            ))
        )
        np.random.shuffle(beta_real)

        true_y = X * beta_real
        data = self._make_data(true_y, X)
        data.beta_real = beta_real
        return data

    def sparse_groups(self, base_nonzero_coeff=[1, 2, 3, 4, 5]):
        group_feature_sizes = self.settings.get_true_group_sizes()
        nonzero_features = len(base_nonzero_coeff)

        X = np.matrix(np.random.randn(self.total_samples, np.sum(group_feature_sizes)))
        betas = [
            np.matrix(np.concatenate((base_nonzero_coeff, np.zeros(num_features - nonzero_features)))).T
            for num_features in group_feature_sizes
        ]
        beta = np.matrix(np.concatenate(betas))

        true_y = X * beta
        data = self._make_data(true_y, X)
        data.beta_real = beta
        return data

    def matrix_completion_groups(self, gamma_to_row_col_m, feat_factor):
        matrix_shape = (self.settings.num_rows, self.settings.num_cols)

        def _make_feature_vec(num_feat, num_nonzero_groups, num_total_groups, feat_factor):
            return (
                [(i + 1) * feat_factor * np.matrix(np.ones(num_feat)).T for i in range(num_nonzero_groups)]
                + [np.matrix(np.zeros(num_feat)).T] * (num_total_groups - num_nonzero_groups)
            )

        def _create_feature_matrix(num_samples, num_feat):
            return np.matrix(np.random.randn(num_samples, num_feat))

        alphas = _make_feature_vec(
            self.settings.num_row_features,
            self.settings.num_nonzero_row_groups,
            self.settings.num_row_groups,
            feat_factor=feat_factor
        )
        betas = _make_feature_vec(
            self.settings.num_col_features,
            self.settings.num_nonzero_col_groups,
            self.settings.num_col_groups,
            feat_factor=feat_factor
        )

        row_features = [
            _create_feature_matrix(self.settings.num_rows, self.settings.num_row_features)
            for i in range(self.settings.num_row_groups)
        ]
        col_features = [
            _create_feature_matrix(self.settings.num_cols, self.settings.num_col_features)
            for i in range(self.settings.num_col_groups)
        ]

        gamma = 0
        for i in range(self.settings.num_nonzero_s):
            u = np.random.randn(self.settings.num_rows)
            v = np.random.randn(self.settings.num_cols)
            gamma += np.matrix(u).T * np.matrix(v)

        only_row_col = get_matrix_completion_groups_fitted_values(
            row_features,
            col_features,
            alphas,
            betas,
            np.zeros(gamma.shape),
        )
        xz_feat_factor = 1.0/gamma_to_row_col_m * 1/np.linalg.norm(only_row_col, ord="fro") * np.linalg.norm(gamma, ord="fro")
        row_features = [xz_feat_factor * m for m in row_features]
        col_features = [xz_feat_factor * m for m in col_features]

        true_matrix = get_matrix_completion_groups_fitted_values(
            row_features,
            col_features,
            alphas,
            betas,
            gamma,
        )

        epsilon = np.random.randn(matrix_shape[0], matrix_shape[1])
        SNR_factor = self._make_snr_factor(np.linalg.norm(true_matrix, ord="fro"), np.linalg.norm(epsilon, ord="fro"))
        observed_matrix = true_matrix + 1.0 / SNR_factor * epsilon

        # sample out parts of each column and parts of each row
        train_indices = set()
        validate_indices = set()
        num_train_sample = max(int(self.settings.train_perc * matrix_shape[0]), 1)
        num_val_sample = max(int(self.settings.validate_perc * matrix_shape[0]), 1)
        # sample from each column
        for i in range(matrix_shape[1]):
            shuffled_idx = i * matrix_shape[1] + np.random.permutation(matrix_shape[0])
            train_indices.update(shuffled_idx[:num_train_sample])
            validate_indices.update(shuffled_idx[num_train_sample:num_train_sample + num_val_sample])

        # sample from each row
        for j in range(matrix_shape[0]):
            shuffled_idx = j + matrix_shape[1] * np.random.permutation(matrix_shape[0])
            train_indices.update(shuffled_idx[:num_train_sample])
            validate_indices.update(shuffled_idx[num_train_sample:num_train_sample + num_val_sample])
        validate_indices.difference_update(train_indices)
        test_indices = set(range(matrix_shape[0] * matrix_shape[1]))
        test_indices.difference_update(train_indices)
        test_indices.difference_update(validate_indices)

        train_indices = np.array(list(train_indices), dtype=int)
        validate_indices = np.array(list(validate_indices), dtype=int)
        test_indices = np.array(list(test_indices), dtype=int)

        return MatrixGroupsObservedData(
            row_features,
            col_features,
            train_indices,
            validate_indices,
            test_indices,
            observed_matrix,
            alphas,
            betas,
            gamma,
            true_matrix
        )

    def _make_data(self, true_y, observed_X):
        # Given the true y and corresponding observed X values, this will add noise so that the SNR is correct
        epsilon = np.matrix(np.random.randn(self.total_samples, 1))
        SNR_factor = self._make_snr_factor(np.linalg.norm(true_y), np.linalg.norm(epsilon))
        observed_y = true_y + 1.0 / SNR_factor * epsilon

        X_train, X_validate, X_test = self._split_data(observed_X)
        y_train, y_validate, y_test = self._split_y_vector(observed_y)

        return ObservedData(X_train, y_train, X_validate, y_validate, X_test, y_test)

    def _split_y_vector(self, y):
        return y[0:self.train_size], y[self.train_size:self.train_size + self.validate_size], y[self.train_size + self.validate_size:]

    def _split_data(self, X):
        return X[0:self.train_size, :], X[self.train_size:self.train_size + self.validate_size, :], X[self.train_size + self.validate_size:, :]

    def _make_shuffled_uniform_X(self, eps=0.0001):
        step_size = (self.feat_range[1] - self.feat_range[0] + eps)/self.total_samples
        # start the uniformly spaced X at a different start point, jitter by about 1/20 of the step size
        jitter = np.random.uniform(0, 1) * step_size/10
        equal_spaced_X = np.arange(self.feat_range[0] + jitter, self.feat_range[1] + jitter, step_size)
        np.random.shuffle(equal_spaced_X)
        return equal_spaced_X

    def _make_snr_factor(self, true_sig_norm, noise_norm):
        return self.snr/ true_sig_norm * noise_norm
