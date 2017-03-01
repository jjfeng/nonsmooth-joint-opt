import time
from cvxpy import *
import cvxopt
import numpy as np
import scipy as sp
from common import VERBOSE
from common import print_time, get_norm2
from common import testerror_matrix_completion_groups, get_matrix_completion_groups_fitted_values
from common import make_column_major_flat, make_column_major_reshape
from gradient_descent_algo import Gradient_Descent_Algo
from convexopt_solvers import MatrixCompletionGroupsProblemWrapperCustom, MatrixCompletionGroupsProblemWrapperSimple

class Lamdba_Deriv_Problem_Wrapper:
    # A problem wrapper for solving for implicit derivatives.
    # The system of linear equations are quite complicated.
    # We will use cvxpy to solve them.
    max_iters = 100000
    eps = 1e-6
    solver=LS # a new solver in CVXPY for linear constraints and quadratic objective
    acceptable_status = [OPTIMAL, OPTIMAL_INACCURATE]

    # @print_time
    def __init__(self, alphas, betas, u_hat, sigma_hat, v_hat):
        # you should send in minified versions of the SVD decomposition.
        # we are not interested in the eigenvectors for sigma = 0
        self.constraints_uu_vv = []
        self.dgamma_dlambda = 0
        self.obj = 0

        self.dSigma_dlambda = None
        if sigma_hat.size > 0:
            self.dU_dlambda = Variable(u_hat.shape[0], u_hat.shape[1])
            self.dV_dlambda = Variable(v_hat.shape[0], v_hat.shape[1])
            self.dSigma_dlambda = Variable(sigma_hat.shape[0], 1)
            # Constraint from definition of U^T U = I and same for V
            uu = u_hat.T * self.dU_dlambda + self.dU_dlambda.T * u_hat
            vv = self.dV_dlambda.T * v_hat + v_hat.T * self.dV_dlambda
            self.constraints_uu_vv = [
                uu == 0,
                vv == 0,
            ]
            self.dgamma_dlambda = (
                self.dU_dlambda * sigma_hat * v_hat.T
                + u_hat * diag(self.dSigma_dlambda) * v_hat.T
                + u_hat * sigma_hat * self.dV_dlambda.T
            )

            self.obj += sum_squares(uu) + sum_squares(vv)

        def _create_var_vec(vec):
            return Variable(vec.size, 1) if vec.size > 0 else None

        self.dalphas_dlambda = [_create_var_vec(a) for a in alphas]
        self.dbetas_dlambda = [_create_var_vec(b) for b in betas]

    @print_time
    # @param obj: backup is to minimize this objective function
    def solve(self, obj=0, big_thres=0.01):
        # hence we want some things that were originally in the constraints to be in the objective

        grad_problem = Problem(Minimize(self.obj + obj))
        grad_problem.solve(
            solver=self.solver,
            verbose=VERBOSE,
        )
        print "grad_problem.status", grad_problem.status, "value", grad_problem.value
        if grad_problem.value > big_thres:
            grad_problem.solve(
                solver=self.solver,
                verbose=VERBOSE,
            )
            print "grad_problem.status (do again)", grad_problem.status, "value", grad_problem.value

        def _extract_values(var_vec):
            return var_vec.value if var_vec is not None else 0

        return {
            "dalphas_dlambda": [_extract_values(a) for a in self.dalphas_dlambda],
            "dbetas_dlambda": [_extract_values(b) for b in self.dbetas_dlambda],
            "dgamma_dlambda": self.dgamma_dlambda.value if self.dSigma_dlambda is not None else 0,
            "dU_dlambda": self.dU_dlambda.value if self.dSigma_dlambda is not None else 0,
            "dV_dlambda": self.dV_dlambda.value if self.dSigma_dlambda is not None else 0,
            "dSigma_dlambda": self.dSigma_dlambda.value if self.dSigma_dlambda is not None else 0,
        }

class Matrix_Completion_Groups_Hillclimb_Base(Gradient_Descent_Algo):
    def _create_descent_settings(self):
        self.num_iters = 20
        self.step_size_init = 1
        self.step_size_min = 1e-6
        self.shrink_factor = 0.1
        self.decr_enough_threshold = 1e-4 * 5
        self.use_boundary = True
        self.boundary_factor = 0.8
        self.backtrack_alpha = 0.001

        self.zero_thres = 1e-6 # determining which values are zero

        assert(self.settings.num_rows == self.settings.num_cols)

        self.train_vec = self._get_vec_mask(self.data.train_idx)
        self.onesT_row = np.matrix(np.ones(self.settings.num_rows))
        self.onesT_col = np.matrix(np.ones(self.settings.num_cols))
        self.num_train = self.data.train_idx.size
        self.num_val = self.data.validate_idx.size

    def get_validate_cost(self, model_params):
        return testerror_matrix_completion_groups(
            self.data,
            self.data.validate_idx,
            model_params
        )

    def get_test_cost(self, model_params):
        return testerror_matrix_completion_groups(
            self.data,
            self.data.test_idx,
            model_params
        )

    def _print_model_details(self):
        # overriding the function in Gradient_Descent_Algo
        alphas = self.fmodel.current_model_params["alphas"]
        betas = self.fmodel.current_model_params["betas"]
        gamma = self.fmodel.current_model_params["gamma"]
        u, s, v = self._get_svd_mini(gamma)
        self.log("model_deet alphas %s" % alphas)
        self.log("model_deet betas %s" % betas)
        self.log("model_deet sigma %s" % np.diag(s))
        # self.log("model_deet gamma %s" % gamma)

        # check that the matrices are similar - sanity check
        self.log("data.real_matrix row 1 %s" % self.data.real_matrix[1,:])
        fitted_m = get_matrix_completion_groups_fitted_values(
            self.data.row_features,
            self.data.col_features,
            self.fmodel.current_model_params["alphas"],
            self.fmodel.current_model_params["betas"],
            self.fmodel.current_model_params["gamma"]
        )
        self.log("fitted_m row 1 %s" % fitted_m[1,:])

    def _get_lambda_derivatives(self):
        # override the function in Gradient_Descent_Algo
        # wrapper around calculating lambda derivatives
        alphas = self.fmodel.current_model_params["alphas"]
        betas = self.fmodel.current_model_params["betas"]
        gamma = self.fmodel.current_model_params["gamma"]

        u_hat, sigma_hat, v_hat = self._get_svd_mini(gamma)

        alphas, row_features, alpha_nonzero_idx = self._get_nonzero_mini(alphas, self.data.row_features)
        betas, col_features, beta_nonzero_idx = self._get_nonzero_mini(betas, self.data.col_features)

        lambda_idxs, lambdas = self._get_minified_lambdas(
            self.fmodel.current_lambdas,
            sigma_hat,
            alpha_nonzero_idx,
            beta_nonzero_idx,
        )

        self.log("alpha_nonzero_idx %s" % alpha_nonzero_idx)
        self.log("alphas %s" % alphas)
        self.log("beta_nonzero_idx %s" % beta_nonzero_idx)
        self.log("betas %s" % betas)
        self.log("sigma_hat %s" % np.diag(sigma_hat))

        imp_derivs = Lamdba_Deriv_Problem_Wrapper(
            alphas,
            betas,
            u_hat,
            sigma_hat,
            v_hat,
        )

        dval_dlambda = np.zeros(self.num_lambdas)
        for i, lambda_idx in enumerate(lambda_idxs):
            self.log("Solve idx %d, LAMBDA %d" % (i, lambda_idx))
            grad_dict_i = self._get_dmodel_dlambda(
                i,
                imp_derivs,
                alphas,
                betas,
                gamma,
                row_features,
                col_features,
                u_hat,
                sigma_hat,
                v_hat,
                lambdas,
            )
            # for k, v in grad_dict_i.iteritems():
            #     self.log("grad_dict %d: %s %s" % (i, k, v))
            dval_dlambda_i = self._get_val_gradient(
                grad_dict_i,
                alphas,
                betas,
                gamma,
                row_features,
                col_features
            )
            dval_dlambda[lambda_idx] = dval_dlambda_i
        return np.array(dval_dlambda).flatten()

    def _get_vec_mask(self, idx):
        # returns a diagonal matrix multiplier that masks the specified indices
        vec_mask = np.zeros(self.settings.num_rows * self.settings.num_cols)
        vec_mask[idx] = 1
        return np.matrix(vec_mask).T

    # @print_time
    def _get_svd(self, gamma):
        # zeros out the singular values if close to zero
        # also transpose v
        u, s, v = np.linalg.svd(gamma)
        u_hat = u
        sigma_hat = np.diag((np.abs(s) > self.zero_thres) * s)
        v_hat = v.T
        return u_hat, sigma_hat, v_hat

    # @print_time
    def _get_svd_mini(self, gamma):
        # similar to _get_svd, but also
        # drops the zero singular values and the corresponding u and v columns
        u, s, v = np.linalg.svd(gamma)
        u_hat = u
        nonzero_mask = (np.abs(s) > self.zero_thres)
        u_mini = u[:, nonzero_mask]
        v_mini = v[nonzero_mask, :].T
        sigma_mini = np.diag(s[nonzero_mask])
        return u_mini, sigma_mini, v_mini

    def _get_minified_lambdas(self, lambdas, sigma_hat, alpha_idxs, beta_idxs):
        lambda_array =[0] if sigma_hat.size > 0 else []

        if lambdas.size == 2:
            if len(alpha_idxs) + len(beta_idxs) > 0:
                lambda_idxs = lambda_array + [1]
            else:
                lambda_idxs = lambda_array
            return lambda_idxs, lambdas
        else:
            lambda_idxs = np.array(np.concatenate(
                (lambda_array, 1 + alpha_idxs, 1 + self.settings.num_row_groups + beta_idxs)
            ), dtype=int)
            print "lambda_idxs", lambda_idxs
            return lambda_idxs, lambdas[lambda_idxs]

    def _get_nonzero_mini(self, vecs, feats):
        # return a smaller version of alpha and beta with zero elements removed
        # also returns a smaller version of the feature vectors
        vec_minis = []
        feat_minis = []
        idxs = []
        for i, v_f_tuple in enumerate(zip(vecs, feats)):
            v, f = v_f_tuple
            # get the mini mask
            v_mask = np.where(np.abs(v) > self.zero_thres)[0]
            if v_mask.size > 0:
                idxs.append(i)
                vec_minis.append(np.reshape(v[v_mask], (v.size, 1)))
                feat_minis.append(f[:, v_mask])
        return vec_minis, feat_minis, np.array(idxs)

    def _get_val_gradient(self, grad_dict, alphas, betas, gamma, row_features, col_features):
        # get gradient of the validation loss wrt lambda given the gradient of the
        # model parameters wrt lambda
        model_grad = grad_dict["dgamma_dlambda"]
        for da_dlambda, row_f in zip(grad_dict["dalphas_dlambda"], row_features):
            model_grad += row_f * da_dlambda * self.onesT_row
        for db_dlambda, col_f in zip(grad_dict["dbetas_dlambda"], col_features):
            model_grad += (col_f * db_dlambda * self.onesT_col).T

        dval_dlambda = - 1.0/self.num_val * (make_column_major_flat(
                self.data.observed_matrix
                - get_matrix_completion_groups_fitted_values(
                    row_features,
                    col_features,
                    alphas,
                    betas,
                    gamma
                )
            )
        )[self.data.validate_idx].T * make_column_major_flat(model_grad)[self.data.validate_idx]
        return dval_dlambda

    def _create_sigma_mask(self, sigma_hat):
        # mask with a zero along diagonal where sigma_hat is zero.
        # everywhere else is a one
        sigma_mask = np.ones(sigma_hat.shape)
        for i in range(sigma_hat.shape[0]):
            if sigma_hat[i,i] == 0:
                sigma_mask[i,i] = 0
        sigma_mask = make_column_major_flat(sigma_mask)
        return np.diag(sigma_mask.flatten())

    # @print_time
    def _get_d_square_loss(self, alphas, betas, gamma, row_features, col_features):
        # get first derivative of the square loss wrt X = gamma + stuff
        d_square_loss = - 1.0/self.num_train * np.multiply(
            self.train_vec,
            make_column_major_flat(
                self.data.observed_matrix
                - get_matrix_completion_groups_fitted_values(
                    row_features,
                    col_features,
                    alphas,
                    betas,
                    gamma
                )
            )
        )
        return d_square_loss

    # @print_time
    def _get_dd_square_loss(self, imp_derivs, row_features, col_features):
        # get double derivative of the square loss wrt X = gamma + stuff
        # imp_derivs should be a Lamdba_Deriv_Problem_Wrapper instance
        dd_square_loss = imp_derivs.dgamma_dlambda
        if len(imp_derivs.dalphas_dlambda) > 0:
            dd_square_loss += np.hstack(row_features) * vstack(*imp_derivs.dalphas_dlambda) * self.onesT_row
        if len(imp_derivs.dbetas_dlambda) > 0:
            dd_square_loss += (np.hstack(col_features) * vstack(*imp_derivs.dbetas_dlambda) * self.onesT_col).T
        return 1.0/self.num_train * mul_elemwise(self.train_vec, vec(dd_square_loss))

    def _get_dd_square_loss_mini(self, imp_derivs, row_features, col_features):
        # get double derivative of the square loss wrt X = gamma + stuff
        # imp_derivs should be a Lamdba_Deriv_Problem_Wrapper instance
        dd_square_loss = imp_derivs.dgamma_dlambda
        if len(imp_derivs.dalphas_dlambda) > 0:
            dd_square_loss += np.hstack(row_features) * vstack(*imp_derivs.dalphas_dlambda) * self.onesT_row
        if len(imp_derivs.dbetas_dlambda) > 0:
            dd_square_loss += (np.hstack(col_features) * vstack(*imp_derivs.dbetas_dlambda) * self.onesT_col).T
        return 1.0/self.num_train * vec(dd_square_loss)[self.data.train_idx]

    def _double_check_derivative_indepth(self, lambda_idx, model1, model2, model0, eps):
        # if it is not differentiable at that point, grad not necessarily zero
        print "alpha1 - deriv", model1["alphas"]
        print "alpha2 - deriv", model2["alphas"]

        if len(model1["alphas"]) > 0:
            dalpha0_dlambda = (model1["alphas"][0] - model2["alphas"][0])/(eps * 2)
            print "dalpha0_dlambda", dalpha0_dlambda
        if len(model1["betas"]) > 0:
            dbeta0_dlambda = (model1["betas"][0] - model2["betas"][0])/(eps * 2)
            print "dbeta0_dlambda", dbeta0_dlambda


        gamma1 = model1["gamma"]
        u1, s1, v1 = self._get_svd_mini(gamma1)
        gamma2 = model2["gamma"]
        u2, s2, v2 = self._get_svd_mini(gamma2)
        dU_dlambda = (u1 - u2)/(eps * 2)
        dV_dlambda = (v1 - v2)/(eps*2)
        dSigma_dlambda = (s1 - s2)/(eps * 2)
        dgamma_dlambda = (gamma1 - gamma2)/(eps * 2)

        # print "dalpha_dlambda, %s" % (dalpha_dlambda)
        # print "dBeta_dlambda, %s" % (dbeta_dlambda)
        print "ds_dlambda, %s" % (dSigma_dlambda)
        print "dU_dlambda", dU_dlambda
        print "dV_dlambda", dV_dlambda
        # print "dgamma_dlambda, %s" % (dgamma_dlambda)

class Matrix_Completion_Groups_Hillclimb(Matrix_Completion_Groups_Hillclimb_Base):
    method_label = "Matrix_Completion_Groups_Hillclimb"

    def _create_lambda_configs(self):
        self.num_lambdas = 1 + self.settings.num_row_groups + self.settings.num_col_groups

        # have a bigger nuclear norm lambda parameter because otherwise calculating
        # the derivative is really slow
        self.lambda_mins = np.ones(self.num_lambdas) * 1e-6
        self.lambda_mins[0] = 5 * 1e-4

    def _create_problem_wrapper(self):
        # self.problem_wrapper = MatrixCompletionProblemWrapper(self.data)
        self.problem_wrapper = MatrixCompletionGroupsProblemWrapperCustom(self.data)

    def _get_dmodel_dlambda(
            self,
            lambda_idx,
            imp_derivs,
            alphas,
            betas,
            gamma,
            row_features,
            col_features,
            u_hat,
            sigma_hat,
            v_hat,
            lambdas,
        ):
        # this fcn accepts mini-fied model parameters - alpha, beta, and u/sigma/v
        # returns the gradient of the model parameters wrt lambda
        num_alphas = len(alphas)
        dd_square_loss_mini = self._get_dd_square_loss_mini(imp_derivs, row_features, col_features)
        sigma_mask = self._create_sigma_mask(sigma_hat)
        obj = 0
        lambda_offset = 1 if sigma_hat.size > 0 else 0

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt gamma
        if sigma_hat.size > 0:
            d_square_loss = self._get_d_square_loss(alphas, betas, gamma, row_features, col_features)
            d_square_loss_reshape = make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols))

            dd_square_loss = self._get_dd_square_loss(imp_derivs, row_features, col_features)
            dd_square_loss_reshape = reshape(
                dd_square_loss,
                self.data.num_rows,
                self.data.num_cols,
            )

            # left multiply U^T and implicit derivative
            dgamma_left_imp_deriv_dlambda = (
                imp_derivs.dU_dlambda.T * d_square_loss_reshape
                + u_hat.T * dd_square_loss_reshape
                + lambdas[0] * np.sign(sigma_hat) * imp_derivs.dV_dlambda.T
            )

            # right multiply V and implicit derivative
            dgamma_right_imp_deriv_dlambda = (
                d_square_loss_reshape * imp_derivs.dV_dlambda
                + dd_square_loss_reshape * v_hat
                + lambdas[0] * imp_derivs.dU_dlambda * np.sign(sigma_hat)
            )
            if lambda_idx == 0:
                dgamma_left_imp_deriv_dlambda += np.sign(sigma_hat) * v_hat.T
                dgamma_right_imp_deriv_dlambda += u_hat * np.sign(sigma_hat)

            obj += sum_squares(dgamma_left_imp_deriv_dlambda) + sum_squares(dgamma_right_imp_deriv_dlambda)

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt
        # alpha and beta, respectively

        for i, a_tuple in enumerate(zip(row_features, alphas, imp_derivs.dalphas_dlambda)):
            row_f, alpha, da_dlambda = a_tuple
            for j in range(alpha.size):
                dalpha_imp_deriv_dlambda = (
                    dd_square_loss_mini.T * vec(row_f[:, j] * self.onesT_row)[self.data.train_idx]
                    + lambdas[i + lambda_offset] * (
                        da_dlambda[j]/get_norm2(alpha, power=1)
                        - alpha[j]/get_norm2(alpha, power=3) * (alpha.T * da_dlambda)
                    )
                )
                if lambda_idx == i + lambda_offset:
                    dalpha_imp_deriv_dlambda += alpha[j]/get_norm2(alpha, power=1)
                obj += sum_squares(dalpha_imp_deriv_dlambda)

        for i, b_tuple in enumerate(zip(col_features, betas, imp_derivs.dbetas_dlambda)):
            col_f, beta, db_dlambda = b_tuple
            for j in range(beta.size):
                dbeta_imp_deriv_dlambda = (
                    dd_square_loss_mini.T * vec(
                        (col_f[:, j] * self.onesT_col).T
                    )[self.data.train_idx]
                    + lambdas[i + lambda_offset + num_alphas] * (
                        db_dlambda[j]/get_norm2(beta, power=1)
                        - beta[j]/get_norm2(beta, power=3) * (beta.T * db_dlambda)
                    )
                )
                if lambda_idx == i + lambda_offset + num_alphas:
                    dbeta_imp_deriv_dlambda += beta[j]/get_norm2(beta, power=1)
                obj += sum_squares(dbeta_imp_deriv_dlambda)

        return imp_derivs.solve(obj)

    def _check_optimality_conditions(self, model_params, lambdas, opt_thres=1e-2):
        # sanity check function to see that cvxpy is solving to a good enough accuracy
        # check that the gradient is close to zero
        # can use this to check that our implicit derivative assumptions hold
        # lambdas must be an exploded lambda matrix
        print "check_optimality_conditions!"

        alphas = model_params["alphas"]
        betas = model_params["betas"]
        gamma = model_params["gamma"]

        u_hat, sigma_hat, v_hat = self._get_svd_mini(gamma)
        a = self.data.observed_matrix - get_matrix_completion_groups_fitted_values(
            self.data.row_features,
            self.data.col_features,
            alphas,
            betas,
            gamma
        )

        d_square_loss = self._get_d_square_loss(
            alphas,
            betas,
            gamma,
            self.data.row_features,
            self.data.col_features,
        )

        left_grad_at_opt_gamma = (
            make_column_major_reshape(
                d_square_loss,
                (self.data.num_rows, self.data.num_cols)
            ) * v_hat
            + lambdas[0] * u_hat * np.sign(sigma_hat)
        )
        right_grad_at_opt_gamma = (
            u_hat.T * make_column_major_reshape(
                d_square_loss,
                (self.data.num_rows, self.data.num_cols)
            )
            + lambdas[0] * np.sign(sigma_hat) * v_hat.T
        )
        print "left grad_at_opt wrt gamma (should be zero)", get_norm2(left_grad_at_opt_gamma)
        print "right grad_at_opt wrt gamma (should be zero)", get_norm2(right_grad_at_opt_gamma)
        # assert(get_norm2(left_grad_at_opt_gamma) < opt_thres)
        # assert(get_norm2(right_grad_at_opt_gamma) < opt_thres)

        for i, a_f_tuple in enumerate(zip(alphas, self.data.row_features)):
            alpha, row_f = a_f_tuple
            if np.linalg.norm(alpha) > 1e-5:
                grad_at_opt_alpha = []
                for j in range(alpha.size):
                    grad_at_opt_alpha.append((
                        d_square_loss.T * make_column_major_flat(
                            row_f[:, j] * self.onesT_row
                        )
                        + lambdas[1 + i] * alpha[j]/np.linalg.norm(alpha, ord=None)
                    )[0,0])
                print "grad_at_opt wrt alpha (should be zero)", get_norm2(grad_at_opt_alpha)
                # assert(np.linalg.norm(grad_at_opt_alpha) < opt_thres)

        for i, b_f_tuple in enumerate(zip(betas, self.data.col_features)):
            beta, col_f = b_f_tuple
            if np.linalg.norm(beta) > 1e-5:
                grad_at_opt_beta = []
                for j in range(beta.size):
                    grad_at_opt_beta.append((
                        d_square_loss.T * make_column_major_flat(
                            (col_f[:, j] * self.onesT_col).T
                        )
                        + lambdas[1 + self.settings.num_row_groups + i] * beta[j]/np.linalg.norm(beta, ord=None)
                    )[0,0])
                print "grad_at_opt wrt beta (should be zero)", get_norm2(grad_at_opt_beta)
                # assert(get_norm2(grad_at_opt_beta) < opt_thres)

class Matrix_Completion_Groups_Hillclimb_Simple(Matrix_Completion_Groups_Hillclimb_Base):
    method_label = "Matrix_Completion_Groups_Hillclimb_Simple"

    def _create_lambda_configs(self):
        self.num_lambdas = 2

        # have a bigger nuclear norm lambda parameter because otherwise calculating
        # the derivative is really slow
        self.lambda_mins = [5 * 1e-4, 1e-6]

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionGroupsProblemWrapperSimple(self.data)

    def _get_dmodel_dlambda(
            self,
            lambda_idx,
            imp_derivs,
            alphas,
            betas,
            gamma,
            row_features,
            col_features,
            u_hat,
            sigma_hat,
            v_hat,
            lambdas,
        ):
        # this fcn accepts mini-fied model parameters - alpha, beta, and u/sigma/v
        # returns the gradient of the model parameters wrt lambda
        num_alphas = len(alphas)
        dd_square_loss_mini = self._get_dd_square_loss_mini(imp_derivs, row_features, col_features)
        sigma_mask = self._create_sigma_mask(sigma_hat)
        obj = 0
        lambda_offset = 1 if sigma_hat.size > 0 else 0

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt gamma
        if sigma_hat.size > 0:
            d_square_loss = self._get_d_square_loss(alphas, betas, gamma, row_features, col_features)
            d_square_loss_reshape = make_column_major_reshape(d_square_loss, (self.data.num_rows, self.data.num_cols))

            dd_square_loss = self._get_dd_square_loss(imp_derivs, row_features, col_features)
            dd_square_loss_reshape = reshape(
                dd_square_loss,
                self.data.num_rows,
                self.data.num_cols,
            )

            # left multiply U^T and implicit derivative
            dgamma_left_imp_deriv_dlambda = (
                imp_derivs.dU_dlambda.T * d_square_loss_reshape
                + u_hat.T * dd_square_loss_reshape
                + lambdas[0] * np.sign(sigma_hat) * imp_derivs.dV_dlambda.T
            )

            # right multiply V and implicit derivative
            dgamma_right_imp_deriv_dlambda = (
                d_square_loss_reshape * imp_derivs.dV_dlambda
                + dd_square_loss_reshape * v_hat
                + lambdas[0] * imp_derivs.dU_dlambda * np.sign(sigma_hat)
            )
            if lambda_idx == 0:
                dgamma_left_imp_deriv_dlambda += np.sign(sigma_hat) * v_hat.T
                dgamma_right_imp_deriv_dlambda += u_hat * np.sign(sigma_hat)

            obj += sum_squares(dgamma_left_imp_deriv_dlambda) + sum_squares(dgamma_right_imp_deriv_dlambda)

        # Constraint from implicit differentiation of the optimality conditions
        # that were defined by taking the gradient of the training objective wrt
        # alpha and beta, respectively

        for i, a_tuple in enumerate(zip(row_features, alphas, imp_derivs.dalphas_dlambda)):
            row_f, alpha, da_dlambda = a_tuple
            for j in range(alpha.size):
                dalpha_imp_deriv_dlambda = (
                    dd_square_loss_mini.T * vec(row_f[:, j] * self.onesT_row)[self.data.train_idx]
                    + lambdas[1] * (
                        da_dlambda[j]/get_norm2(alpha, power=1)
                        - alpha[j]/get_norm2(alpha, power=3) * (alpha.T * da_dlambda)
                    )
                )
                if lambda_idx == 1:
                    dalpha_imp_deriv_dlambda += alpha[j]/get_norm2(alpha, power=1)
                obj += sum_squares(dalpha_imp_deriv_dlambda)

        for i, b_tuple in enumerate(zip(col_features, betas, imp_derivs.dbetas_dlambda)):
            col_f, beta, db_dlambda = b_tuple
            for j in range(beta.size):
                dbeta_imp_deriv_dlambda = (
                    dd_square_loss_mini.T * vec(
                        (col_f[:, j] * self.onesT_col).T
                    )[self.data.train_idx]
                    + lambdas[1] * (
                        db_dlambda[j]/get_norm2(beta, power=1)
                        - beta[j]/get_norm2(beta, power=3) * (beta.T * db_dlambda)
                    )
                )
                if lambda_idx == 1:
                    dbeta_imp_deriv_dlambda += beta[j]/get_norm2(beta, power=1)
                obj += sum_squares(dbeta_imp_deriv_dlambda)

        return imp_derivs.solve(obj)
