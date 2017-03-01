from common import testerror_grouped
from neldermead import Nelder_Mead_Algo
from convexopt_solvers import GroupedLassoProblemWrapper, GroupedLassoProblemWrapperSimple

class SGL_Nelder_Mead(Nelder_Mead_Algo):
    method_label = "SGL_Nelder_Mead"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = GroupedLassoProblemWrapper(
            self.data.X_train,
            self.data.y_train,
            self.settings.get_expert_group_sizes()
        )

    def get_validation_cost(self, lambdas):
        # if any are not positive, then just return max value
        for l in lambdas:
            if l <= 0:
                return self.MAX_COST

        model_params = self.problem_wrapper.solve(lambdas, quick_run=True)
        validation_cost = testerror_grouped(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )
        self.log("validation_cost %f" % validation_cost)
        return validation_cost


class SGL_Nelder_Mead_Simple(Nelder_Mead_Algo):
    method_label = "SGL_Nelder_Mead_simple"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = GroupedLassoProblemWrapperSimple(
            self.data.X_train,
            self.data.y_train,
            self.settings.get_expert_group_sizes()
        )

    def get_validation_cost(self, lambdas):
        # if any are not positive, then just return max value
        for l in lambdas:
            if l <= 0:
                return self.MAX_COST

        model_params = self.problem_wrapper.solve(lambdas, quick_run=True)
        validation_cost = testerror_grouped(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )
        self.log("validation_cost %f" % validation_cost)
        return validation_cost
