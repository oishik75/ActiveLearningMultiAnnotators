import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from scipy.optimize import linprog

from annotator.models import AnnotatorModel
from utils import get_weighted_labels, get_majority_labels, get_max_labels

class AnnotatorSelector:
    def __init__(self, n_features, n_annotators, device, log_dir="logs", report_to_tensorboard=False):
        self.n_features = n_features
        self.n_annotators = n_annotators
        self.device = device

        self.model = self.initialize_model()

        if report_to_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def initialize_model(self):
        torch.manual_seed(0)
        return AnnotatorModel(n_features=self.n_features, n_annotators=self.n_annotators).to(self.device)
    

    def write_to_tensorboard(self, epoch, data, type="train"):
        for key in data:
            self.writer.add_scalar(f"annotator/{key}/{type}", data[key], epoch)

    # Annotator weights where majority agreeing annotators have weight 1 and rest 0
    def get_annotator_majority_weights(self, annotator_labels, annotator_mask):
        annotator_weights = []
        majority_labels = get_majority_labels(annotator_labels, annotator_mask)
        for i in range(len(annotator_labels)):
            annotator_weights.append((annotator_labels[i] == majority_labels[i]).astype("int"))
            # print(annotator_labels[i])
            # print(annotator_mask[i])
            # print(annotator_weights[-1])
            # input()
        return np.array(annotator_weights)
    
    # Create annotator weight using agreement-disagreement between annotator labels using linear programming
    def get_annotator_lp_weights(self, annotator_labels, annotator_mask):
        annotator_weights = []
        n_annotators = len(annotator_labels[0])
        for i in range(len(annotator_labels)):
            if annotator_mask[i].sum() == 1: # If only a single annotator, its weight will be 1 and rest 0.
                annotator_weights.append(annotator_mask[i].copy())
                continue
            A_eq = np.zeros((n_annotators, n_annotators))
            b_eq = np.zeros(n_annotators)
            for j in range(n_annotators):
                if annotator_mask[i][j] == 0: # Annotator is not queried. So not part of equation
                    continue
                for k in range(n_annotators):
                    if j == k:
                        A_eq[j, k] = annotator_mask[i].sum() - 1
                    elif annotator_mask[i][k] == 0:
                        A_eq[j, k] = 0
                    elif annotator_labels[i][j] == annotator_labels[i][k]:
                        A_eq[j, k] = -1
                    else:
                        A_eq[j, k] = 1
                        b_eq[j] += 1
            c = np.ones(n_annotators) * -1
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
            if not res.success:
                print("ERROR: Linear Programming not solvable")
                print(annotator_labels[i])
                print(annotator_mask[i])
                print(A_eq)
                print(b_eq)
                exit()
            annotator_weights.append(res.x)
        return np.array(annotator_weights)
    
    def get_annotator_model_weights(self, instances):
        if not torch.is_tensor(instances):
            instances = torch.tensor(instances)

        if len(instances.shape) == 1: # If only a single instance is passed,  make it a 2d tensor
            instances = instances.unsqueeze(0)

        with torch.no_grad():
            annotator_weights = self.model(instances.float().to(self.device)).cpu().squeeze()

        return annotator_weights
    
    def train(
            self,
            args, 
            train_x, 
            train_annotator_labels, 
            train_annotator_weights=None, 
            train_y=None, 
            train_annotator_mask=None,
            eval_x=None,
            eval_annotator_labels=None, 
            eval_annotator_weights=None, 
            eval_y=None, 
            eval_annotator_mask=None,
            from_scratch=True,
            plot_to_tensorboard=True,
            train_phase="boot" # For logging purposes
    ):
        
        if train_annotator_mask is None:
            train_annotator_mask = np.ones_like(train_annotator_labels)
        
        if train_annotator_weights is None:
            train_annotator_weights = self.get_annotator_lp_weights(train_annotator_labels, train_annotator_mask)

        train_dataset = TensorDataset(torch.tensor(train_x).float(), torch.tensor(train_annotator_labels), torch.tensor(train_annotator_weights), torch.tensor(train_annotator_mask), torch.tensor(train_y))
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True) # Need to change batch_size to args.batch_size
        
        if eval_x is not None and eval_annotator_labels is not None:
            evaluate = True
        else:
            evaluate = False

        if from_scratch:
            self.model = self.initialize_model()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=args["lr"]) # Change learning rate to args
        criterion_annotator = torch.nn.MSELoss()

        losses = []
        accuracy = []
        f1 = []
        for epoch in tqdm(range(args["n_epochs"])):
            batch_loss = []
            y_pred = []
            y_true = []
            for batch in train_dataloader:
                optimizer.zero_grad()

                x, annotator_labels, annotator_weights, annotator_mask, y = batch
                x = x.to(self.device)
                annotator_weights = annotator_weights.to(self.device)
                annotator_mask = annotator_mask.to(self.device)

                # annotator_mask = torch.tensor(np.random.randint(0, 2, annotator_labels.shape)).to(self.device)

                output = self.model(x)
                # print(output)
                # print(annotator_weights)
                # print(annotator_mask)
                # input()
                
                # loss = criterion_annotator(output, annotator_weights.float())
                loss = torch.sum(((output - annotator_weights) * annotator_mask)**2.0) / torch.sum(annotator_mask)
                batch_loss.append(loss.item())
                
                loss.backward()
                optimizer.step()

                y_true += list(y.numpy())
                if args["labeling_type"] == "weighted":
                    y_predicted, _ = get_weighted_labels(annotator_labels.numpy(), output.detach().cpu().numpy(), annotator_mask.cpu().numpy())
                elif args["labeling_type"] == "max":
                    y_predicted, _ = get_max_labels(annotator_labels.numpy(), output.detach().cpu().numpy(), annotator_mask.cpu().numpy())
                y_pred += y_predicted


            accuracy.append(accuracy_score(y_true=y_true, y_pred=y_pred))
            f1.append(f1_score(y_true=y_true, y_pred=y_pred))

            losses.append(sum(batch_loss) / len(batch_loss))
            # print(f"Epoch: {epoch+1} \t Loss: {losses[-1]} \t Accuracy: {accuracy[-1]} \t F1: {f1[-1]}")
            if (epoch + 1) % args["log_epochs"] == 0:
                tqdm.write(f"Epoch: {epoch+1} \t Loss: {losses[-1]} \t Accuracy: {accuracy[-1]} \t F1: {f1[-1]}")
                if self.writer and plot_to_tensorboard:
                    self.write_to_tensorboard(epoch=epoch, data={"loss": losses[-1], "accuracy": accuracy[-1], "f1": f1[-1]}, type=f"{train_phase}/train")
                if evaluate:
                    self.eval(epoch=epoch, eval_x=eval_x, eval_annotator_labels=eval_annotator_labels, eval_annotator_weights=eval_annotator_weights, eval_y=eval_y, eval_annotator_mask=eval_annotator_mask, 
                              labeling_type=args["labeling_type"], plot_to_tensorboard=plot_to_tensorboard, eval_phase=train_phase)
        self.writer.flush()

        return losses[-1], accuracy[-1], f1[-1]


    def eval(
            self,
            epoch,
            eval_x,
            eval_annotator_labels, 
            eval_annotator_weights=None, 
            eval_y=None, 
            eval_annotator_mask=None,
            labeling_type="max",
            plot_to_tensorboard=True,
            eval_phase="boot"
    ):
        if eval_annotator_mask is None:
            eval_annotator_mask = np.ones_like(eval_annotator_labels)
        
        if eval_annotator_weights is None:
            eval_annotator_weights = self.get_annotator_lp_weights(eval_annotator_labels, eval_annotator_mask)

        eval_dataset = TensorDataset(torch.tensor(eval_x).float(), torch.tensor(eval_annotator_labels), torch.tensor(eval_annotator_weights), torch.tensor(eval_annotator_mask), torch.tensor(eval_y))
        eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

        self.model.eval()
        
        criterion_annotator = torch.nn.MSELoss()

        eval_loss = []
        y_true = []
        y_pred = []
        for batch in tqdm(eval_dataloader):
            x, annotator_labels, annotator_weights, annotator_mask, y = batch
            x = x.to(self.device)
            annotator_weights = annotator_weights.to(self.device)
            annotator_mask = annotator_mask.to(self.device)

            with torch.no_grad():
                output = self.model(x)
            # print(output)
            # print(output.shape)
            
            # loss = criterion_annotator(output, annotator_weights.float())
            loss = torch.sum(((output - annotator_weights) * annotator_mask)**2.0) / torch.sum(annotator_mask)
            eval_loss.append(loss.item())

            y_true += list(y.numpy())
            if labeling_type == "weighted":
                y_predicted, _ = get_weighted_labels(annotator_labels.numpy(), output.detach().cpu().numpy(), annotator_mask.cpu().numpy())
            elif labeling_type == "max":
                y_predicted, _ = get_max_labels(annotator_labels.numpy(), output.detach().cpu().numpy(), annotator_mask.cpu().numpy())
            y_pred += y_predicted

        
        self.model.train()
        
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        eval_loss = sum(eval_loss) / len(eval_loss)
        tqdm.write(f"Eval Loss: {eval_loss} \t Eval Accuracy: {accuracy} \t Eval F1: {f1}")

        if self.writer:
            self.write_to_tensorboard(epoch=epoch, data={"loss": eval_loss, "accuracy": accuracy, "f1": f1}, type=f"{eval_phase}/val")


    def get_annotator(self, instance, annotator_mask, annotator_weights=None):
        '''For the given instance, select the annotator with the highest weight from the annotators whose mask is 0 (have not been queried).
            if annotator weight is none, the get weights from the model, else use the annotator weights.
        '''

        if annotator_weights == None:
            annotator_weights = self.get_annotator_model_weights(instances=instance)

        annotator_weights = annotator_weights * (1 - annotator_mask)

        annotator_idx = torch.argmax(annotator_weights).item()

        return annotator_idx