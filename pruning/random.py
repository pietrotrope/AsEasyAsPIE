from pruning.components.scheduling import Iterative
from pruning.components.scoring import RandomStructured
from pruning.components.tuning import weightRelearning
from pruning.tools import retrieve_tuples


class RandomPruningStructured:
    sampling = False

    def __init__(self, model, frequency, percentage):
        self.name = "randoms"
        if frequency:
            if frequency[0] == 0:
                self.name += "_ini"
        self.frequency = frequency.copy()
        self.frequency = Iterative(frequency)
        self.scoring = RandomStructured(percentage, model)
        self.model = model

    def check_and_prune(self, iteration):
        if self.frequency.next_check == iteration:
            self.frequency.update_next()
            self.scoring.compute_new_masks()
            return True
        return False


class RandomPruningWRStructured:
    sampling = False

    def __init__(self, model, frequency, percentage, device="cuda:0"):
        self.name = "random-WRs"
        self.frequency = frequency.copy()
        self.frequency = Iterative(frequency)
        self.scoring = RandomStructured(percentage, model)
        self.model = model
        self.tuples = retrieve_tuples(model, True, True, True)
        self.wr = weightRelearning(self.tuples, device)

    def check_and_prune(self, iteration):
        if self.frequency.next_check == iteration:

            self.frequency.update_next()
            self.scoring.compute_new_masks()

            self.wr.reset_weights()
            return True
        return False
