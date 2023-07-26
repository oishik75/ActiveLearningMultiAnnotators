import random
from scipy.stats import entropy
import numpy as np

class InstanceSelector:
    def __init__(self, strategy, seed=0) -> None:
        self.strategy = strategy
        random.seed(seed)

    def select_random_instances(self, indices):
        return random.choice(indices)
    
    def select_highest_entropy(self, x, classifier, indices):
        probabilities = classifier.get_probabilities(x)
        entropies = entropy(probabilities, axis=1)
        masked_entropies = np.zeros_like(entropies)
        masked_entropies[indices] = entropies[indices]
        selected_index = np.argmax(masked_entropies)
        if selected_index not in indices:
            print("Error!!! Something went wrong in entropy instance selection.")
            exit()
        
        return selected_index

    def select_instances(self, x, classifier, indices):
        if self.strategy == "random":
            return self.select_random_instances(indices)
        elif self.strategy == "entropy":
            return self.select_highest_entropy(x, classifier, indices)