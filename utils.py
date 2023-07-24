from collections import defaultdict, Counter
import numpy as np

def get_weighted_labels(labels, weights, mask=None):
    weighted_labels = []
    label_weights = []

    if mask is None: # If mask is None, make mask to be all 1s
        mask = np.ones_like(labels)
    # If mask not of type bool, convert to bool
    if  mask.dtype != bool:
        mask = mask.astype(bool)
    
    for i in range(len(labels)):
        label_weight = defaultdict(list)
        for idx, label in enumerate(labels[i]):
            if mask[i][idx]:
                label_weight[label].append(weights[i][idx])
        
        weighted_label = list(label_weight.keys())[0]
        for label in label_weight:
            label_weight[label] = np.mean(label_weight[label])
            if label_weight[label] > label_weight[weighted_label]:
                weighted_label = label

        label_weights.append(label_weight)
        weighted_labels.append(weighted_label)

    return weighted_labels, label_weights

def get_majority_labels(labels, mask=None):
    majority_labels = []
    
    if mask is None: # If mask is None, make mask to be all 1s
        mask = np.ones_like(labels)
    # If mask not of type bool, convert to bool
    if  mask.dtype != bool:
        mask = mask.astype(bool)
    
    for i in range(len(labels)):
        counts = Counter(labels[i][mask[i]])
        majority_label = max(counts, key=counts.get)
        majority_labels.append(majority_label)

    return majority_labels

def get_max_labels(labels, weights, mask=None):
    max_labels = []
    max_weights = []

    if mask is None: # If mask is None, make mask to be all 1s
        mask = np.ones_like(labels)
    # If mask not of type bool, convert to bool
    if  mask.dtype != bool:
        mask = mask.astype(bool)

    weights = weights * mask

    for i in range(len(labels)):
        max_idx = np.argmax(weights[i])
        max_labels.append(labels[i][max_idx])
        max_weights.append(weights[i][max_idx])

    return max_labels, max_weights