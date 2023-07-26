from itertools import compress
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch

from annotator.annotator_selector import AnnotatorSelector
from instance_selector import InstanceSelector
from classifier.classifier import ClassifierModel
from utils import get_weighted_labels, get_max_labels

class MultiAnnotatorActiveLearner:
    def __init__(self, data, args):

        self.args = args
        self.create_data_splits(data)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.annotator_selector = AnnotatorSelector(n_features=self.n_features, n_annotators=self.n_annotators, device=self.device, log_dir=args.log_dir + args.exp_name, report_to_tensorboard=True)  
        self.instance_selector = InstanceSelector(self.args.instance_strategy, self.args.seed)
        self.classifier = ClassifierModel(self.args.classifier_name, self.n_features, self.n_classes, device=self.device)
    
    def create_data_splits(self, data):
        columns = list(data.columns.values)
        feature_cols = list(compress(columns, [c.startswith('x_') for c in columns]))
        annotator_cols = list(compress(columns, [c.startswith('y_') for c in columns]))
        
        features = data[feature_cols].to_numpy()
        annotator_labels = data[annotator_cols].to_numpy()
        ground_truth_label = data["y"].to_numpy()

        train_x, test_x, train_annotator_labels, test_annotator_labels, train_y, test_y = train_test_split(features, annotator_labels, ground_truth_label, test_size=self.args.test_size, random_state=self.args.seed)

        boot_x, active_x, boot_annotator_labels, active_annotator_labels, boot_y, active_y = train_test_split(train_x, train_annotator_labels, train_y, test_size=(1-self.args.boot_size), random_state=self.args.seed)

        self.boot_x = boot_x
        self.boot_annotator_labels = boot_annotator_labels
        self.boot_y = boot_y

        # self.boot_x = train_x
        # self.boot_annotator_labels = train_annotator_labels
        # self.boot_y = train_y

        self.active_x = active_x
        self.active_annotator_labels = active_annotator_labels
        self.active_y = active_y

        self.test_x = test_x
        self.test_annotator_labels = test_annotator_labels
        self.test_y = test_y

        self.n_features = self.boot_x.shape[1]
        self.n_annotators = self.boot_annotator_labels.shape[1]
        self.n_classes = len(np.unique(ground_truth_label))

        print(f"Boot Instances: {self.boot_x.shape[0]} \t Active Instances: {active_x.shape[0]} \t Test Instances: {self.test_x.shape[0]}")
        print(f"n_features: {self.boot_x.shape[1]}")
        print(f"n_annotators: {self.boot_annotator_labels.shape[1]}")

    def boot_phase(self):
        training_args = {}
        training_args["lr"] = self.args.boot_lr
        training_args["n_epochs"] = self.args.boot_n_epochs
        training_args["log_epochs"] = self.args.boot_log_epochs
        training_args["labeling_type"] = self.args.labeling_type
        # Train annotator model on boot data with all 5 annotators
        self.annotator_selector.train(args=training_args, train_x=self.boot_x, train_annotator_labels=self.boot_annotator_labels, train_y=self.boot_y,
                                      eval_x=self.test_x, eval_annotator_labels=self.test_annotator_labels, eval_y=self.test_y)
        

    def active_learning_phase(self, report_to_tensorboard=True, verbose=True):
        budget = self.args.budget
        active_instances = [i for i in range(len(self.active_x))] # Indices of the current active learning instances that could be queries. If all annotators of an instance is queried it is moved out of active_instances.
        queried_instances = [] # Indices of instances that have been queried atleast once.
        annotator_training_instances = [] # Indicess of instances that will be used for training the annotators.
        annotator_mask = np.array([[0 for _ in range(self.n_annotators)] for _ in range(len(self.active_x))]) # Tracks which annotators have been queried for each instance

        if report_to_tensorboard:
            writer = SummaryWriter(log_dir=self.args.log_dir + self.args.exp_name)

        # Annotator training arguments
        training_args = {}
        training_args["lr"] = self.args.active_lr
        training_args["n_epochs"] = self.args.active_n_epochs
        training_args["log_epochs"] = self.args.active_log_epochs
        training_args["labeling_type"] = self.args.labeling_type
        # Begin active learning cycle
        for b in tqdm(range(budget)):
            # Select instance from active list
            instance_idx = self.instance_selector.select_instances(self.active_x, self.classifier, active_instances)
            if instance_idx not in queried_instances:
                queried_instances.append(instance_idx)

            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
            if verbose:
                print(f"Instance selected in cycle {b+1}: {instance_idx}")
                print(f"Annotators already queried: {annotator_mask[instance_idx]}")

            if report_to_tensorboard:
                writer.add_text("Instance Selected", f"{instance_idx}", b)
                writer.add_text("Annotators Queried", f"{annotator_mask[instance_idx]}", b)
                writer.flush()
            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
            
            # Select Annotator for the given instance
            annotator_idx = self.annotator_selector.get_annotator(self.active_x[instance_idx], annotator_mask=annotator_mask[instance_idx])
            
            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
            if verbose:
                print(f"Annotator selected: {annotator_idx}")
            if report_to_tensorboard:
                writer.add_text("Annotator Selected", f"{annotator_idx}", b)
                writer.flush()
            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
            
            annotator_mask[instance_idx][annotator_idx] = 1
            # Remove instance from active learning list if all annotators are queried.
            if annotator_mask[instance_idx].sum() == self.n_annotators:
                if verbose:
                    print(f"Removing {instance_idx} from active list since all annotators are queried.")
                active_instances.remove(instance_idx)
            # If instance is queried for the first time, there is no change in annotator training data, so do not train the model.
            train_annotator_model = True
            if annotator_mask[instance_idx].sum() == 1:
                train_annotator_model = False

            # If 2 or more annotators are queried for an instace, we use it for training the annotator model.
            # So whenever annotator is queried 2 times, add it to the annotator training list.
            if annotator_mask[instance_idx].sum() == 2:
                annotator_training_instances.append(instance_idx)

            # Train annotator
            if train_annotator_model:
                annotator_loss, annotator_accuracy, annotator_f1 = self.annotator_selector.train(
                    args = training_args, 
                    train_x = np.concatenate((self.active_x[annotator_training_instances], self.boot_x)), 
                    train_annotator_labels = np.concatenate((self.active_annotator_labels[annotator_training_instances], self.boot_annotator_labels)), 
                    train_y = np.concatenate((self.active_y[annotator_training_instances], self.boot_y)),
                    train_annotator_mask = np.concatenate((annotator_mask[annotator_training_instances], np.ones_like(self.boot_annotator_labels))),
                    train_phase = "active_learning",
                    plot_to_tensorboard = False
                )

                ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
                if verbose:
                    print(f"Annotator Training Loss: {annotator_loss} \t Annotator Training Accuracy: {annotator_accuracy} \t Annotator Training F1: {annotator_f1}")
                if report_to_tensorboard:
                    writer.add_scalar("ActiveLearning/Annotator/loss", annotator_loss, b)
                    writer.add_scalar("ActiveLearning/Annotator/accuracy", annotator_accuracy, b)
                    writer.add_scalar("ActiveLearning/Annotator/f1", annotator_f1, b)
                    writer.flush()
                ### --------------------------------------------------- Logging steps --------------------------------------------------- ###

            # Train classifier
            classifier_training_x = np.concatenate((self.active_x[queried_instances], self.boot_x))
            classifier_training_annotator_model_weights = self.annotator_selector.get_annotator_model_weights(classifier_training_x)
            classifier_training_annotator_mask = np.concatenate((annotator_mask[queried_instances], np.ones_like(self.boot_annotator_labels)))
            classifier_training_annotator_labels = np.concatenate((self.active_annotator_labels[queried_instances], self.boot_annotator_labels))
            # Calculate label for training the classification_model.
            if self.args.labeling_type == "weighted":
                classifier_training_y, _ = get_weighted_labels(classifier_training_annotator_labels, classifier_training_annotator_model_weights, classifier_training_annotator_mask)
            elif self.args.labeling_type == "max":
                classifier_training_y, _ = get_max_labels(classifier_training_annotator_labels, classifier_training_annotator_model_weights, classifier_training_annotator_mask)
            classifier_true_y = np.concatenate((self.active_y[queried_instances], self.boot_y))

            self.classifier.train(classifier_training_x, classifier_training_y)
            classifier_accuracy, classifier_f1 = self.classifier.eval(self.test_x, self.test_y)

            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
            if verbose:
                print(f"Label accuracy score: {accuracy_score(classifier_true_y, classifier_training_y)} \t Label f1 score: {f1_score(classifier_true_y, classifier_training_y)}")
                print(f"Classifier accuracy score: {classifier_accuracy} \t Classifier F1 score: {classifier_f1}")
            if report_to_tensorboard:
                writer.add_scalar("ActiveLearning/Label/accuracy", accuracy_score(classifier_true_y, classifier_training_y), b)
                writer.add_scalar("ActiveLearning/Label/f1", f1_score(classifier_true_y, classifier_training_y), b)
                writer.add_scalar("ActiveLearning/Classifier/accuracy", classifier_accuracy, b)
                writer.add_scalar("ActiveLearning/Classifier/f1", classifier_f1, b)
                writer.flush()
            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###

        if report_to_tensorboard:
            writer.close()    

        print(f"Classifier accuracy score after Active Learning: {classifier_accuracy}")
        print(f"Classifier F1 score after Active Learning: {classifier_f1}")

        # If labels were 100% accurate
        classifier_training_x = np.concatenate((self.active_x[queried_instances], self.boot_x))
        classifier_training_y = np.concatenate((self.active_y[queried_instances], self.boot_y))
        self.classifier.train(classifier_training_x, classifier_training_y)
        classifier_accuracy, classifier_f1 = self.classifier.eval(self.test_x, self.test_y)
        print(f"Classifier accuracy score after Active Learning with oracle annotator: {classifier_accuracy}")
        print(f"Classifier F1 score after Active Learning with oracle annotator: {classifier_f1}")



    def fully_supervised(self):
        train_x = np.concatenate((self.active_x, self.boot_x))
        train_y = np.concatenate((self.active_y, self.boot_y))

        self.classifier.train(train_x, train_y)
        accuracy, f1 = self.classifier.eval(self.test_x, self.test_y)

        print(f"Fully Supervised Classifier Accuracy: {accuracy}")
        print(f"Fully Supervised Classifier F1: {f1}")