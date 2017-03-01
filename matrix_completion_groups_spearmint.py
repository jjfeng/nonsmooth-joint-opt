from spearmint_algo import Spearmint_Algo
from common import testerror_matrix_completion_groups
from convexopt_solvers import MatrixCompletionGroupsProblemWrapperCustom
from convexopt_solvers import MatrixCompletionGroupsProblemWrapperSimple

class Matrix_Completion_Groups_Spearmint(Spearmint_Algo):
    method_label = "Matrix_Completion_Groups_Spearmint"
    result_folder_prefix = "spearmint_descent/matrix_completion_groups"

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionGroupsProblemWrapperCustom(
            self.data
        )
        self.num_lambdas = self.data.num_alphas + self.data.num_betas + 1

    def get_validation_cost(self, model_params):
        return testerror_matrix_completion_groups(
            self.data,
            self.data.validate_idx,
            model_params
        )

class Matrix_Completion_Groups_Spearmint_Simple(Spearmint_Algo):
    method_label = "Matrix_Completion_Groups_Spearmint_Simple"
    result_folder_prefix = "spearmint_descent/matrix_completion_groups"

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionGroupsProblemWrapperSimple(
            self.data
        )
        self.num_lambdas = 2

    def get_validation_cost(self, model_params):
        return testerror_matrix_completion_groups(
            self.data,
            self.data.validate_idx,
            model_params
        )
