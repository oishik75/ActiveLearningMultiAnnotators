import argparse
import pandas as pd
import numpy as np
from itertools import compress
from sklearn.cluster import KMeans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ionosphere-simulated-x.csv")
    parser.add_argument("--dataset_path", default="data/")
    parser.add_argument("--save_name", default="data/new/ionosphere.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_annotators_random", type=int, default=3)
    parser.add_argument("--n_annotators_inst_dep", type=int, default=4)
    parser.add_argument("--n_annotators_class_dep", type=int, default=5)
    parser.add_argument("--annotator_random_probs", default="0.6, 0.2, 0.4")
    parser.add_argument("--annotator_inst_dep_probs", default="0.6, 0.7, 0.7, 0.85")
    parser.add_argument("--annotator_class_dep_probs", default="0.7, 0.85, 0.9, 0.9, 0.8")
    parser.add_argument("--annotator_class_dep_classes", default="0, 1, 1, 0, 1")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    data = pd.read_csv(args.dataset_path + args.dataset)
    
    columns = list(data.columns.values)
    feature_cols = list(compress(columns, [c.startswith('x_') for c in columns]))
    ground_truth_labels = data["y"].to_numpy()
    n_classes = len(np.unique(ground_truth_labels))

    annotator_random_probs = list(map(float, args.annotator_random_probs.split(",")))
    annotator_inst_dep_probs = list(map(float, args.annotator_inst_dep_probs.split(",")))
    annotator_class_dep_probs = list(map(float, args.annotator_class_dep_probs.split(",")))

    annotator_class_dep_classes = list(map(float, args.annotator_class_dep_classes.split(",")))

    # Generate annotations for random annotators
    random_annotators = []
    for i in range(args.n_annotators_random):
        annotations = [label if np.random.rand() < annotator_random_probs[i] else np.random.choice(list(set(range(n_classes)) - set([label]))) for label in ground_truth_labels]
        random_annotators.append(annotations)

    # Generate annotations for instance dependent annotators
    n_clusters = args.n_annotators_inst_dep // 2
    model = KMeans(n_clusters=n_clusters)
    cluster_labels = model.fit_predict(data[feature_cols])
    annotator_inst_dep_clusters = [i%n_clusters for i in range(args.n_annotators_inst_dep)]

    inst_dep_annotators = []
    for i in range(args.n_annotators_inst_dep):
        annotations = []
        for idx, label in enumerate(ground_truth_labels):
            if cluster_labels[idx] == annotator_inst_dep_clusters[i]: # instance belongs to the cluster specialised by annotator
                annotations.append(label if np.random.rand() < annotator_inst_dep_probs[i] else np.random.choice(list(set(range(n_classes)) - set([label]))))
            else:
                annotations.append(np.random.choice(list(range(n_classes))))
        inst_dep_annotators.append(annotations)

    # Generate annotations for class dependent annotators
    class_dep_annotators = []
    for i in range(args.n_annotators_class_dep):
        annotations = []
        for label in ground_truth_labels:
            if label == annotator_class_dep_classes[i]: # instance belongs to the class specialised by annotator
                annotations.append(label if np.random.rand() < annotator_class_dep_probs[i] else np.random.choice(list(set(range(n_classes)) - set([label]))))
            else:
                annotations.append(np.random.choice(list(range(n_classes))))
        class_dep_annotators.append(annotations)


    annotated_data = data[feature_cols + ["y"]].copy()
    annotator_idx = 0
    # Random Annotators
    for annotations in random_annotators:
        annotated_data[f"y_{annotator_idx}"] = annotations
        annotator_idx += 1
    # Instance Dependent Annotators
    for annotations in inst_dep_annotators:
        annotated_data[f"y_{annotator_idx}"] = annotations
        annotator_idx += 1
    # Class Dependent Annotators
    for annotations in class_dep_annotators:
        annotated_data[f"y_{annotator_idx}"] = annotations
        annotator_idx += 1

    annotated_data.to_csv(args.save_name, index=False)


    



if __name__ == "__main__":
    main()