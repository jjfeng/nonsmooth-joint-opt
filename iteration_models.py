class Simulation_Settings:
    train_size = 100
    validate_size = 50
    test_size = 50
    snr = 2
    gs_num_lambdas = 10
    spearmint_numruns = 100
    nm_iters = 50
    feat_range = [-5,5]
    method = "HC"
    plot = False
    method_result_keys = [
        "test_err",
        "validation_err",
        "test_rate",
        "runtime",
    ]

class Iteration_Data:
    def __init__(self, i, data, settings, init_lambdas=None):
        self.data = data
        self.settings = settings
        self.i = i
        self.init_lambdas = init_lambdas
