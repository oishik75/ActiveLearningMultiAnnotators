from itertools import compress
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch

from annotator.annotator_selector import AnnotatorSelector
from instance_selector import InstanceSelector
from knowledgebase import Knowledgebase
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
        if self.args.use_knowledgebase:
            self.knowledgebase = Knowledgebase(self.n_annotators, self.args.similarity_threshold)
        else:
            self.knowledgebase = None
    
    
    def create_boot_split_clustering(self, train_x, train_annotator_labels, train_y):
        boot_size = int(len(train_x) * self.args.boot_size)

        # Cluster Size will be boot size / n where n is the no of samples we want to choose per cluster.
        n_clusters = boot_size // self.args.samples_per_cluster
        model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        cluster_labels = model.fit_predict(train_x)
        clusters = {i: [] for i in range(n_clusters)}
        for idx in range(len(cluster_labels)):
            clusters[cluster_labels[idx]].append(idx)

        boot_indices = []
        active_indices = []
        
        # Select mediod
        for cluster in clusters:
            X = train_x[clusters[cluster]]
            pairwise_distance = pairwise_distances(X=X)
            pairwise_distances_sum = pairwise_distance.sum(axis=1)
            sorted_indices = np.argsort(pairwise_distances_sum)
            if len(sorted_indices) <= self.args.samples_per_cluster:
                boot_indices += clusters[cluster]
            else:
                boot_indices += list(np.array(clusters[cluster])[sorted_indices[:self.args.samples_per_cluster]])
                active_indices += list(np.array(clusters[cluster])[sorted_indices[self.args.samples_per_cluster:]])
        
        boot_x = train_x[boot_indices]
        active_x = train_x[active_indices]
        boot_annotator_labels = train_annotator_labels[boot_indices]
        active_annotator_labels = train_annotator_labels[active_indices]
        boot_y = train_y[boot_indices]
        active_y = train_y[active_indices]
        
        return boot_x, active_x, boot_annotator_labels, active_annotator_labels, boot_y, active_y
    
    def create_data_splits(self, data):
        columns = list(data.columns.values)
        feature_cols = list(compress(columns, [c.startswith('x_') for c in columns]))
        annotator_cols = list(compress(columns, [c.startswith('y_') for c in columns]))
        
        features = data[feature_cols].to_numpy()
        annotator_labels = data[annotator_cols].to_numpy()
        ground_truth_label = data["y"].to_numpy()

        train_x, test_x, train_annotator_labels, test_annotator_labels, train_y, test_y = train_test_split(features, annotator_labels, ground_truth_label, test_size=self.args.test_size, random_state=self.args.seed)

        if self.args.boot_strategy == "random":
            boot_x, active_x, boot_annotator_labels, active_annotator_labels, boot_y, active_y = train_test_split(train_x, train_annotator_labels, train_y, test_size=(1-self.args.boot_size), stratify=train_y, random_state=self.args.seed)
        elif self.args.boot_strategy == "cluster":
            boot_x, active_x, boot_annotator_labels, active_annotator_labels, boot_y, active_y = self.create_boot_split_clustering(train_x, train_annotator_labels, train_y)

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
        print(f"n_classes: {self.n_classes}")

        if self.args.budget < 1:
            self.args.budget = int(len(train_x) * self.n_annotators * self.args.budget - len(boot_x) * self.n_annotators)
            print("Budget is:", self.args.budget )

    def boot_phase(self):
        training_args = {}
        training_args["lr"] = self.args.boot_lr
        training_args["batch_size"] = self.args.boot_batch_size
        training_args["n_epochs"] = self.args.boot_n_epochs
        training_args["log_epochs"] = self.args.boot_log_epochs
        training_args["labeling_type"] = self.args.labeling_type
        # Train annotator model on boot data with all 5 annotators
        self.annotator_selector.train(args=training_args, train_x=self.boot_x, train_annotator_labels=self.boot_annotator_labels, train_y=self.boot_y,
                                      eval_x=self.test_x, eval_annotator_labels=self.test_annotator_labels, eval_y=self.test_y)
        
        # Train classifier model on boot data and annotation from annotators
        classifier_training_annotator_model_weights = self.annotator_selector.get_annotator_model_weights(self.boot_x)
        classifier_training_annotator_mask = np.ones_like(self.boot_annotator_labels)

        if self.args.labeling_type == "weighted":
            classifier_training_y, _ = get_weighted_labels(self.boot_annotator_labels, classifier_training_annotator_model_weights, classifier_training_annotator_mask)
        elif self.args.labeling_type == "max":
            classifier_training_y, _ = get_max_labels(self.boot_annotator_labels, classifier_training_annotator_model_weights, classifier_training_annotator_mask)

        self.classifier.train(self.boot_x, classifier_training_y)
        classifier_accuracy, classifier_f1 = self.classifier.eval(self.test_x, self.test_y)
        print(f"Boot Phase Classifier accuracy score: {classifier_accuracy} \t Classifier F1 score: {classifier_f1}")

        if self.args.use_knowledgebase:
            self.create_knowledgebase()

    def create_knowledgebase(self):
        for idx, instance in enumerate(self.boot_x):
            annotator, weight = self.annotator_selector.get_annotator(instance, np.zeros(self.n_annotators)) # Passing zero as mask since get_annotator only looks at annotators with mask=0
            # if weight > self.args.weight_threshold:
            self.knowledgebase.add_instance_to_knowledgebase(instance, self.boot_annotator_labels[idx][annotator], self.boot_y[idx], annotator)

        self.knowledgebase.print_knowledgebase_info()

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
        training_args["batch_size"] = self.args.active_batch_size
        training_args["n_epochs"] = self.args.active_n_epochs
        training_args["log_epochs"] = self.args.active_log_epochs
        training_args["labeling_type"] = self.args.labeling_type

        active_learning_cycle = -1
        b = 0
        pbar = tqdm(total = budget)
        n_kb_labels = 0
        n_kb_labels_correct = 0

        # Begin active learning cycle
        while b < budget:
            active_learning_cycle += 1
        # for b in tqdm(range(budget)):
            # Select instance from active list
            instance_idx = self.instance_selector.select_instances(self.active_x, self.classifier, active_instances)
            if instance_idx not in queried_instances:
                queried_instances.append(instance_idx)

            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
            if verbose:
                print(f"Instance selected in cycle {active_learning_cycle + 1}: {instance_idx}")
                print(f"Annotators already queried: {annotator_mask[instance_idx]}")

            if report_to_tensorboard:
                writer.add_text("Instance Selected", f"{instance_idx}", active_learning_cycle)
                writer.add_text("Annotators Queried", f"{annotator_mask[instance_idx]}", active_learning_cycle)
                writer.flush()
            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
            
            # Select Annotator for the given instance
            annotator_idx, annotator_weight = self.annotator_selector.get_annotator(self.active_x[instance_idx], annotator_mask=annotator_mask[instance_idx])
            label_obtained_from_kb = False
            # If use_knowledgebase, check if label can be obtained from knowledgebase
            if self.args.use_knowledgebase:
                kb_output = self.knowledgebase.get_label(self.active_x[instance_idx])
                if kb_output is not None:
                    label, kb_annotator_idx = kb_output
                    if kb_annotator_idx == annotator_idx: # If annotator model agree with knowledgebase expert then use label from annotator model instead of querying.
                        self.active_annotator_labels[instance_idx, kb_annotator_idx] = label
                        label_obtained_from_kb = True
                        n_kb_labels += 1
                        n_kb_labels_correct += int(label==self.active_y[instance_idx])
            
            if self.args.use_knowledgebase and (not label_obtained_from_kb) and annotator_weight > self.args.weight_threshold:
                # If instance already present in KB then add instance to new annotator if weight of new annotator is greater tham curremt KB annotator 
                kb_annotator_idx =  self.knowledgebase.check_instance_in_knowledgebase(self.active_x[instance_idx])
                if kb_annotator_idx is not None:
                    weights = self.annotator_selector.get_annotator_model_weights(self.active_x[instance_idx])
                    if weights[kb_annotator_idx] < weights[annotator_idx]:
                        self.knowledgebase.add_instance_to_knowledgebase(self.active_x[instance_idx], self.active_annotator_labels[instance_idx][annotator_idx], self.active_y[instance_idx], annotator_idx)
                else:
                    self.knowledgebase.add_instance_to_knowledgebase(self.active_x[instance_idx], self.active_annotator_labels[instance_idx][annotator_idx], self.active_y[instance_idx], annotator_idx)
                
            
            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###
            if verbose:
                print(f"Annotator selected: {annotator_idx}")
            if report_to_tensorboard:
                writer.add_text("Annotator Selected", f"{annotator_idx}", active_learning_cycle)
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

            # If 3 or more annotators are queried for an instace, we use it for training the annotator model.
            # So whenever annotator is queried 3 times, add it to the annotator training list.
            if annotator_mask[instance_idx].sum() == 3:
                annotator_training_instances.append(instance_idx)

            # Train annotator
            if train_annotator_model and not label_obtained_from_kb and (active_learning_cycle+1)%self.args.train_after_every==0:
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
                    writer.add_scalar("ActiveLearning/Annotator/loss", annotator_loss, active_learning_cycle)
                    writer.add_scalar("ActiveLearning/Annotator/accuracy", annotator_accuracy, active_learning_cycle)
                    writer.add_scalar("ActiveLearning/Annotator/f1", annotator_f1, active_learning_cycle)
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
                print(f"Label accuracy score: {accuracy_score(classifier_true_y, classifier_training_y)} \t Label f1 score: {f1_score(classifier_true_y, classifier_training_y, average='macro')}")
                print(f"Classifier accuracy score: {classifier_accuracy} \t Classifier F1 score: {classifier_f1}")
            if report_to_tensorboard:
                writer.add_scalar("ActiveLearning/Label/accuracy", accuracy_score(classifier_true_y, classifier_training_y), active_learning_cycle)
                writer.add_scalar("ActiveLearning/Label/f1", f1_score(classifier_true_y, classifier_training_y, average='macro'), active_learning_cycle)
                writer.add_scalar("ActiveLearning/Classifier/accuracy", classifier_accuracy, active_learning_cycle)
                writer.add_scalar("ActiveLearning/Classifier/f1", classifier_f1, active_learning_cycle)
                if n_kb_labels > 0:
                    writer.add_scalar("ActiveLearning/KB/label_accuracy", n_kb_labels_correct/n_kb_labels, active_learning_cycle)
                else:
                    writer.add_scalar("ActiveLearning/KB/label_accuracy", 0, active_learning_cycle)
                writer.flush()
            ### --------------------------------------------------- Logging steps --------------------------------------------------- ###

            if not label_obtained_from_kb:
                pbar.update(1)
                b += 1

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

        self.knowledgebase.print_knowledgebase_info()


    def fully_supervised(self):
        train_x = np.concatenate((self.active_x, self.boot_x))
        train_y = np.concatenate((self.active_y, self.boot_y))

        self.classifier.train(train_x, train_y)
        accuracy, f1 = self.classifier.eval(self.test_x, self.test_y)

        print(f"Fully Supervised Classifier Accuracy: {accuracy}")
        print(f"Fully Supervised Classifier F1: {f1}")