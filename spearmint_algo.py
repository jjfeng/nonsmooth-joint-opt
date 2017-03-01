import sys
import time
from argparse import Namespace

import numpy as np
import scipy as sp

from common import *
from fitted_model import Fitted_Model
sys.path.insert(0, 'spearmint-master/spearmint')
sys.path.insert(0, 'spearmint-master/spearmint-lite')
spearmint_lite = __import__('spearmint-lite')

class Spearmint_Algo:
    MAX_COST = 1e6

    def __init__(self, data, folder_suffix, settings=None):
        self.data = data
        self.settings = settings
        # Call problem wrapper first to determine number of lambdas
        self._create_problem_wrapper()
        self._check_make_configs(folder_suffix)
        self.result_file = "%s/results.dat" % self.result_folder

    def run(self, num_runs, log_file=None):
        self.log_file = log_file
        # Run spearmint to get next experiment parameters
        self.fmodel = Fitted_Model(self.num_lambdas)

        # Find new experiment
        best_cost = None
        runtime = 0
        for i in range(num_runs):
            start_time = time.time()
            self.log("%s: iter %d" % (self.method_label, i))
            self.run_spearmint_command(self.result_folder)
            self.log("Spearmint command completed")

            with open(self.result_file,'r') as resfile:
                newlines = []
                for line in resfile.readlines():
                    values = line.split()
                    if len(values) < 3:
                        continue
                    val = values.pop(0)
                    dur = values.pop(0)
                    lambdas = np.array([10**float(v) for v in values[:self.num_lambdas]])
                    if (val == 'P'):
                        # P means pending experiment to run
                        # Run experiment
                        self.log("lambdas %s" % lambdas)
                        model_params = self.problem_wrapper.solve(lambdas, quick_run=True)
                        if model_params is None:
                            current_cost = self.MAX_COST
                        else:
                            current_cost = self.get_validation_cost(model_params)

                        if best_cost is None or best_cost > current_cost:
                            best_cost = current_cost
                            self.fmodel.update(lambdas, model_params, current_cost)
                            self.log("fmodel: %s" % self.fmodel)

                        newlines.append(str(current_cost) + " 0 "
                                        + " ".join(values) + "\n")
                    else:
                        # Otherwise these are previous experiment results
                        newlines.append(line)

            runtime += time.time() - start_time
            # Don't record time spent on writing files?
            with open(self.result_file, 'w') as resfile:
                resfile.writelines(newlines)

            sys.stdout.flush()

        self.fmodel.set_runtime(runtime)
        self.fmodel.set_num_solves(num_runs)
        self.log("%s: runtime %s" % (self.method_label, runtime))
        self.log("fmodel: %s" % self.fmodel)
        self.log("fmodel best_model_params: %s" % self.fmodel.best_model_params)

        # VERY IMPORTANT to clean spearmint results
        self.run_spearmint_clean(self.result_folder)

    def log(self, log_str):
        if self.log_file is None:
            print log_str
        else:
            self.log_file.write("%s\n" % log_str)

    def _check_make_configs(self, folder_suffix):
        self.result_folder = "%s%s" % (self.result_folder_prefix, folder_suffix)
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        config_file_name = "%s/config.json" % self.result_folder
        if not os.path.exists(config_file_name):
            with open(config_file_name, 'w') as config_file:
                config_file.write(self._create_config_string(self.num_lambdas))

    @staticmethod
    @print_time
    def run_spearmint_command(experiment_folder, use_multiprocessing=True, gridsize=20000):
        ## Note: not all machines have the capability of running spearmint without crashing.
        ## Gosset servers will crash!
        multiprocessing_option = ""
        if not use_multiprocessing:
            multiprocessing_option = ",use_multiprocessing=0"
        options = Namespace(
            chooser_module="GPEIOptChooser",
            chooser_args="noiseless=1%s" % multiprocessing_option,
            grid_size=gridsize,
            config_file="config.json",
            results_file="results.dat",
            grid_seed=1,
            num_jobs=1,
            max_finished_jobs=1000
        )
        spearmint_lite.main_controller(options, [experiment_folder])

    @staticmethod
    def run_spearmint_clean(experiment_folder):
        cmd = "rm %s/chooser* %s/results.dat" % (experiment_folder, experiment_folder)
        os.system(cmd)

    @staticmethod
    def _create_config_string(num_lambdas):
        return '{"lambda": {"name": "lambda","type": "float","size":%d,"min":  -6,"max":  2}}' % num_lambdas
