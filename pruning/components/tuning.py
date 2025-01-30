import torch


class weightRelearning:
    def __init__(self, tuples, device="cuda:0"):
        self.weights = None
        self.tuples = tuples
        self.device = device

        self.set_weights()
        self.memorize_weights()

    def set_weights(self):
        self.weights = [None] * len(self.tuples)

    def reset_weights(self):
        with torch.no_grad():

            for i, (layer, name) in enumerate(self.tuples):
                tmp = torch.tensor(self.weights[i]).to(self.device)

                for n, p in layer.named_parameters():
                    if n == name or n == name+"_orig":
                        p.data = p.data - p.data + tmp
                        p.requires_grad_()

    def memorize_weights(self):
        with torch.no_grad():
            for i, (layer, name) in enumerate(self.tuples):
                with torch.no_grad():
                    n = name
                    self.weights[i] = layer.__getattr__(
                        n).cpu().detach().numpy()
