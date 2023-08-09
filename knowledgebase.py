import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score

class AnnotatorKnowledgebase:
    def __init__(self) -> None:
        self.instances = []
        self.labels = []
        self.true_labels = []
        self.similar_instances = {}

    def add_instance(self, instance, label, true_label):
        self.instances.append(instance)
        self.labels.append(label)
        self.true_labels.append(true_label) # Used to calculate accuracy of knowledgebase. Not used for getting labels
        self.similar_instances[len(self.instances)-1] = []

    def get_max_similarity(self, instance):
        if len(self.instances) == 0:
            return -1, np.array([0])
        similarities = cosine_similarity(self.instances, [instance])
        max_similarity_idx = np.argmax(similarities)
        return max_similarity_idx, similarities[max_similarity_idx]
    
    def get_label(self, instance, idx=None, add_instance_to_similar_instances=True):
        if idx == None:
            idx, _ = self.get_max_similarity(instance)

        label = self.labels[idx]
        if add_instance_to_similar_instances:
            self.similar_instances[idx].append(instance)

        return label
    
    def get_accuracy(self):
        if len(self.labels) == 0:
            return 0
        return accuracy_score(self.true_labels, self.labels)
    
    def get_f1(self):
        if len(self.labels) == 0:
            return 0
        return f1_score(self.true_labels, self.labels, average='macro')



class Knowledgebase:
    def __init__(self, n_annotators, similarity_threshold=0.95) -> None:
        self.annotator_knowledgebases = [AnnotatorKnowledgebase() for _ in range(n_annotators)]
        self.similarity_threshold = similarity_threshold

    def add_instance_to_knowledgebase(self, instance, label, true_label, annotator):
        self.annotator_knowledgebases[annotator].add_instance(instance, label, true_label)

    def get_label(self, instance):
        similarities = [] 
        similarity_idxs = []

        for annotator_kb in self.annotator_knowledgebases:
            sim_idx, similarity  = annotator_kb.get_max_similarity(instance)
            similarities.append(similarity)
            similarity_idxs.append(sim_idx)

        annotator_idx = np.argmax(similarities)
        # print(similarities[annotator_idx])
        # print(self.similarity_threshold)
        # print(similarities[annotator_idx] < self.similarity_threshold)
        # input()
        if similarities[annotator_idx] < self.similarity_threshold: # If max similarity is less than threshold then no annotator has expertise
            return None
        
        label = self.annotator_knowledgebases[annotator_idx].get_label(instance, similarity_idxs[annotator_idx])

        return label, annotator_idx
    
    def print_knowledgebase_info(self):
        knowledgebase_lengths = {i: len(self.annotator_knowledgebases[i].instances) for i in range(len(self.annotator_knowledgebases))}
        print("Knowledgebase Lengths: ", knowledgebase_lengths)
        knowledgebase_accuracy = {i: self.annotator_knowledgebases[i].get_accuracy() for i in range(len(self.annotator_knowledgebases))}
        print("Knowledgebase Accuracies: ", knowledgebase_accuracy)
        knowledgebase_f1 = {i: self.annotator_knowledgebases[i].get_f1() for i in range(len(self.annotator_knowledgebases))}
        print("Knowledgebase F1 scores: ", knowledgebase_f1)


