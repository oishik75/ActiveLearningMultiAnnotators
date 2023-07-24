import random

class InstanceSelector:
    def __init__(self, seed=0) -> None:
        random.seed(seed)

    def select_random_instances(self, indices):
        return random.choice(indices)