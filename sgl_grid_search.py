from grid_search import Grid_Search
from common import testerror_grouped
from convexopt_solvers import GroupedLassoProblemWrapperSimple

class SGL_Grid_Search(Grid_Search):
    method_label = "SGL_Grid_Search"

    def _create_problem_wrapper(self):
        self.problem_wrapper = GroupedLassoProblemWrapperSimple(
            self.data.X_train,
            self.data.y_train,
            self.settings.get_expert_group_sizes()
        )

    def get_validation_cost(self, model_params):
        return testerror_grouped(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )
