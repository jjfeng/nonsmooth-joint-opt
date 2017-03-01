from spearmint_algo import Spearmint_Algo
from common import testerror_elastic_net
from convexopt_solvers import ElasticNetProblemWrapper

class Elastic_Net_Spearmint(Spearmint_Algo):
    method_label = "Elastic_Net_Spearmint"
    result_folder_prefix = "spearmint_descent/elastic_net"

    def _create_problem_wrapper(self):
        self.problem_wrapper = ElasticNetProblemWrapper(
            self.data.X_train,
            self.data.y_train
        )
        self.num_lambdas = 2

    def get_validation_cost(self, thetas):
        validation_cost = testerror_elastic_net(
            self.data.X_validate,
            self.data.y_validate,
            thetas
        )
        return validation_cost
