class NoPruning:
    sampling = False
    name = "NoPruning"

    def __init__(self, model):
        self.model = model

    def check_and_prune(self, iteration):
        return False
