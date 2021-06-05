import logging
from config import config
import torch
import numpy as np

def train_loop(dataloader, model, loss_name, loss_fn, optimizer):
    """
    Performs one epoch of training the model through the dataset stored in dataloader, predicting one batch at a time 
    Using the given loss_fn and optimizer
    Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch 
    This function is called by BaseNet each epoch 
    """
    size = len(dataloader)
    num_datapoints = len(dataloader.dataset)
    training_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # Move tensors to GPU
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.view(-1,1))
        # Backpropagation and optimization 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Add up metrics
        training_loss += loss.item()
        if loss_name == 'bce':
            pred = (pred > 0.5).float()
            correct += (pred == y).float().sum()
        # Remove batch from gpu
        del X
        del y 
        torch.cuda.empty_cache()

    loss = training_loss / size 
    logging.info(f"Avg training loss: {loss:>7f}")
    if config['task'] == 'prosaccade-clf':  
        accuracy = correct / num_datapoints           
        logging.info(f"Avg training accuracy {accuracy:>8f}")
        return float(loss), float(accuracy) 
    return float(loss), -1 

def validation_loop(dataloader, model, loss_name, loss_fn):
    """
    Performs one prediction through the validation set set stored in the given dataloader
    Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch 
    This function is called by BaseNet each epoch, an early stopping is implemented on the returned validation loss 
    """
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    val_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Move tensors to GPU
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            # Predict 
            pred = model(X)
            # Compute metrics
            val_loss += loss_fn(pred, y.view(-1, 1)).item()
            if loss_name == 'bce':
                pred = (pred > 0.5).float()
                correct += (pred == y).float().sum() 
            # Remove batch from gpu
            del X
            del y 
            torch.cuda.empty_cache()
    
    loss = val_loss / num_batches
    logging.info(f"Avg validation loss: {loss:>8f}")
    if config['task'] == 'prosaccade-clf':
        accuracy = correct / size
        logging.info(f"Avg validation accuracy {accuracy:>8f}")
        return float(loss), float(accuracy)
    return float(loss), -1 # Can be used for early stopping


def test_loop(dataloader, model):
    """
    Performs one prediction through the validation set set stored in the given dataloader
    Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch
    This function is called by BaseNet each epoch, an early stopping is implemented on the returned validation loss
    """
    num_batches = len(dataloader)
    size = len(dataloader.dataset)

    val_loss, correct = 0, 0
    with torch.no_grad():
        all_pred = np.array([])
        for batch, (X, _) in enumerate(dataloader):
            # Move tensors to GPU
            if torch.cuda.is_available():
                X = X.cuda()
            # Predict
            pred = model(X)
            #print(pred.shape)
            #print(pred.detach().numpy().ravel().shape)
            all_pred = np.concatenate([all_pred, pred.detach().numpy().ravel()])
            del X
            torch.cuda.empty_cache()
    return np.array(all_pred)  # Can be used for early stopping
    