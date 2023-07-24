import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from annotator.annotator_selector import AnnotatorSelector

data = pd.read_csv("data/medical.csv")

n_annotators = 6
n_features = 62

features = data[["x_" + str(i) for i in range(n_features)]].to_numpy()

annotator_labels = data[["y_" + str(i) for i in range(n_annotators)]].to_numpy()
ground_truth_label = data["y"].to_numpy()

annotator_ground_truth_weights = []
for i in range(len(ground_truth_label)):
    annotator_ground_truth_weights.append((annotator_labels[i] == ground_truth_label[i]).astype("int"))

annotator_ground_truth_weights = np.array(annotator_ground_truth_weights)

train_x, test_x, train_annotator_labels, test_annotator_labels, train_annotator_weights, test_annotator_weights, train_y, test_y = train_test_split(features, annotator_labels, annotator_ground_truth_weights, ground_truth_label, test_size=0.3, random_state=0)

device = "cuda"

annotator_selector = AnnotatorSelector(n_features=n_features, n_annotators=n_annotators, device=device, log_dir="logs/run_30", report_to_tensorboard=True)

annotator_selector.train(args={}, train_x=train_x, train_annotator_labels=train_annotator_labels, train_y=train_y,
                         eval_x=test_x, eval_annotator_labels=test_annotator_labels, eval_y=test_y) # train_annotator_weights=train_annotator_weights, eval_annotator_weights=test_annotator_weights, 