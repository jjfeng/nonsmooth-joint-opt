from common import testerror_elastic_net
from neldermead import Nelder_Mead_Algo
from convexopt_solvers import ElasticNetProblemWrapper

class Elastic_Net_Nelder_Mead(Nelder_Mead_Algo):
    method_label = "Elastic_Net_Nelder_Mead"
    MAX_COST = 100000

    def _create_problem_wrapper(self):
        self.problem_wrapper = ElasticNetProblemWrapper(
            self.data.X_train,
            self.data.y_train
        )

    def get_validation_cost(self, lambdas):
        # if any are not positive, then just return max value
        for l in lambdas:
            if l <= 0:
                return self.MAX_COST

        betas = self.problem_wrapper.solve(lambdas, quick_run=True)
        validation_cost = testerror_elastic_net(
            self.data.X_validate,
            self.data.y_validate,
            betas
        )
        self.log("validation_cost %f" % validation_cost)
        return validation_cost
