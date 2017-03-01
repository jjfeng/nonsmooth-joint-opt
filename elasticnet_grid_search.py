from common import testerror_elastic_net
from grid_search import Grid_Search
from convexopt_solvers import ElasticNetProblemWrapper

class Elastic_Net_Grid_Search(Grid_Search):
    method_label = "Elastic_Net_Grid_Search"

    def _create_problem_wrapper(self):
        self.problem_wrapper = ElasticNetProblemWrapper(
            self.data.X_train,
            self.data.y_train
        )

    def get_validation_cost(self, model_params):
        return testerror_elastic_net(
            self.data.X_validate,
            self.data.y_validate,
            model_params
        )
