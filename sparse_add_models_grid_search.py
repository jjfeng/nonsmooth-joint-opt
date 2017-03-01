from common import testerror_sparse_add_smooth
from grid_search import Grid_Search
from convexopt_solvers import SparseAdditiveModelProblemWrapperSimple

class Sparse_Add_Model_Grid_Search(Grid_Search):
    method_label = "Sparse_Add_Model_Grid_Search"

    def _create_problem_wrapper(self):
        self.problem_wrapper = SparseAdditiveModelProblemWrapperSimple(
            self.data.X_full,
            self.data.train_idx,
            self.data.y_train
        )

    def get_validation_cost(self, model_params):
        return testerror_sparse_add_smooth(
            self.data.y_validate,
            self.data.validate_idx,
            model_params
        )
