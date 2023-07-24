from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from classifier.models import NeuralNetClassifier

class ClassifierModel:
    def __init__(self, classifier_name, n_features, n_classes, device="cpu"):
        self.classifier_name = classifier_name
        self.n_features = n_features
        self.n_classes = n_classes
        self.device = device
        if self.classifier_name == "logistic_regression":
            self.model = LogisticRegression(max_iter=8000)
        elif self.classifier_name == "random_forest":
            self.model = RandomForestClassifier(random_state=0)
        elif self.classifier_name == "svm":
            self.model = SVC()
        elif self.classifier_name == "xgboost":
            self.model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
        elif self.classifier_name == "neural_net":
            self.model = NeuralNetClassifier(n_features=self.n_features, n_classes=self.n_classes)
            self.model.to(self.device)
    
    def train(self, x, y):
        if self.classifier_name in ["logistic_regression", "random_forest", "svm", "xgboost"]:
            self.model.fit(x, y)
        elif self.classifier_name == "neural_net":
            self.train_neural_net_model(x, y)

    def eval(self, x, y):
        if self.classifier_name in ["logistic_regression", "random_forest", "svm", "xgboost"]:
            y_pred = self.model.predict(x)
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)

        if self.classifier_name == "neural_net":
            x = torch.tensor(x).to(self.device).float()
            with torch.no_grad():
                out = self.model(x).cpu().numpy()
            y_pred = np.argmax(out, axis=1)
            print(y_pred)
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            
        return accuracy, f1
    
    def train_neural_net_model(self, x, y):
        self.model = NeuralNetClassifier(n_features=self.n_features, n_classes=self.n_classes)
        self.model.to(self.device)

        optimizer = Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.tensor(x).float(), torch.tensor(y).long())
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        losses = []
        for epoch in range(100):
            batch_loss = []
            for batch in dataloader:
                optimizer.zero_grad()

                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                out = self.model(x)
                loss = criterion(out, y)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            losses.append(sum(batch_loss) / len(batch_loss))

            # print(f"Epoch: {epoch} \t Loss: {losses[-1]}")

        
