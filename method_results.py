import numpy as np

class MethodResults:
    def __init__(self, method_name, stats_keys):
        self.method_name = method_name
        self.stats_keys = stats_keys
        self.run_stats = {k: [] for k in stats_keys}
        self.lambda_sets = []

    def get_num_runs(self):
        return len(self.run_stats[self.stats_keys[0]])

    def print_results(self):
        num_runs = self.get_num_runs()
        def get_std_err(values):
            return np.sqrt(np.var(values)/num_runs)

        if num_runs > 0:
            print self.method_name, "Results: (mean, std dev)"
            for k in self.stats_keys:
                stats_for_key = self.run_stats[k]
                print "%s: %.5f (%.5f)" % (k, np.average(stats_for_key), get_std_err(stats_for_key))

            if len(self.lambda_sets):
                print "average lambdas: %s" % np.mean(np.vstack(self.lambda_sets), axis=0)

    def append(self, result):
        for stats_key in self.stats_keys:
            if stats_key not in result.stats or result.stats[stats_key] is None:
                raise Exception("Stats missing for %s" % stats_key)
            self.run_stats[stats_key].append(result.stats[stats_key])

        if result.lambdas is not None:
            self.lambda_sets.append(result.lambdas)

class MethodResult:
    def __init__(self, stats, lambdas=None):
        self.stats = stats
        self.lambdas = lambdas

    def __str__(self):
        return_str = ""
        for key in self.stats.keys():
            return_str += "%s: %f\n" % (key, self.stats[key])
        return_str += "lambdas: %s\n" % self.lambdas
        return return_str
