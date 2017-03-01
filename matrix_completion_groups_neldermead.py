from common import testerror_matrix_completion_groups
from neldermead import Nelder_Mead_Algo
from convexopt_solvers import MatrixCompletionGroupsProblemWrapperCustom, MatrixCompletionGroupsProblemWrapperSimple

class Matrix_Completion_Groups_Nelder_Mead(Nelder_Mead_Algo):
    method_label = "Matrix_Completion_Groups_Nelder_Mead"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionGroupsProblemWrapperCustom(self.data)

    def get_validation_cost(self, lambdas):
        # if any are not positive, then just return max value
        for l in lambdas:
            if l <= 0:
                return self.MAX_COST

        model_params = self.problem_wrapper.solve(lambdas, quick_run=True)
        validation_cost = testerror_matrix_completion_groups(
            self.data,
            self.data.validate_idx,
            model_params
        )
        self.log("validation_cost %f, lam: %s" % (validation_cost, lambdas))
        return validation_cost

class Matrix_Completion_Groups_Nelder_Mead_Simple(Nelder_Mead_Algo):
    method_label = "Matrix_Completion_Groups_Nelder_Mead_Simple"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = MatrixCompletionGroupsProblemWrapperSimple(self.data)

    def get_validation_cost(self, lambdas):
        # if any are not positive, then just return max value
        for l in lambdas:
            if l <= 0:
                return self.MAX_COST

        model_params = self.problem_wrapper.solve(lambdas, quick_run=True)
        validation_cost = testerror_matrix_completion_groups(
            self.data,
            self.data.validate_idx,
            model_params
        )
        self.log("validation_cost %f, lam: %s" % (validation_cost, lambdas))
        return validation_cost
