import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, restore_best_weights=True):
        """
        Initialize EarlyStopping object.

        Args:
        - patience (int): Number of epochs to wait after validation loss stops improving before stopping training.
        - delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        - verbose (bool): If True, prints a message for each validation loss improvement.
        - restore_best_weights (bool): If True, restores the model's weights to the best observed during training.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        """
        Check if the training should stop based on the validation loss.

        Args:
        - val_loss (float): Current value of the validation loss.
        - model: PyTorch model object.

        Returns:
        - early_stop (bool): True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss increased ({self.best_score:.6f} --> {val_loss:.6f}).')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if val_loss < self.best_score:
                if self.verbose:
                    print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).')
                self.best_score = val_loss
                self.best_model_state = model.state_dict()
            self.counter = 0

        return self.early_stop

    def best_weights(self):
        return self.best_model_state


def evaluate(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    acc = 100 * correct / total
    return acc