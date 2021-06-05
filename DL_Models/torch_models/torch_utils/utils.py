import torch 
import subprocess as sp
import os
import time 
from config import config 
from memory_profiler import profile 
from torch_models.torch_utils.custom_losses import angle_loss

def get_gpu_memory():
    """
    Returns and prints the available amount of GPU memory 
    """
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values

def timing_decorator(func):
    """
    Timing-Decorator for functions 
    """
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        delta = (t2 - t1) * 1000 * 1000  # seconds 
        print(f"{func.__name__}:{(delta):.4f}ms")
        return result

    return wrapper

#@profile 
def compute_loss(loss_fn, dataloader, pred_list, nb_models):
    """
    Computes the loss across all batches between the true labels in the dataloader and the batch predictions in pred_list
    ------------------
    Input:
    loss_fn: pytorch loss function
    dataloader: contains the validation dataset X, y
    pred_list: list of of tensors (one for each batch, size (batch, y))
    nb_models: number of models in the ensemble. divide by it to obtain averaged output
    ------------------
    Output:
    Scalar value of the mean loss over all batches of the validation set 
    """
    loss = 0
    size = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            y = y.cuda()
        # Average the predictions from the ensemble 
        pred = pred_list[batch]

        pred = torch.div(pred, nb_models).float() # is already on gpu 
        loss += loss_fn(pred, y)

    print(f"acc loss: {loss}, size: {size}, epoch loss: {loss/size}")
    return loss / size 

#@profile 
def compute_accuracy(dataloader, pred_list, nb_models):
    """
    Computes the accuracy across al batches between the true labels in the dataloader and the batch predictions in pred_list
    ------------------
    Input:
    dataloader: contains the validation dataset X,y
    pred_list: list of of tensors (one for each batch, size (batch, y))
    nb_models: number of models in the ensemble. divide by it to obtain averaged output
    ------------------
    Output:
    Scalar value of the mean accuracy over all batches of the validation set in the dataloader
    """
    correct = 0
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            y = y.cuda()
        # Average the predictions from the ensemble
        pred = pred_list[batch] 
        pred = torch.div(pred, nb_models).float() # tensor is already on gpu         
        pred = torch.round(pred) # majority decision for classification 
        pred = (pred > 0.5).float()
        correct += (pred == y).float().sum() 
    return correct / size 

def sum_predictions(dataloader, model, model_number, prediction_list):
    """
    Predict with the given model and add up the predictions in prediction_list to compute ensemble metrics 
    """
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            if torch.cuda.is_available():
                X.cuda()
                y.cuda()
            pred = model(X)
            if model_number == 0:
                prediction_list.append(pred) # append the predicted tensor for each batch 
            else:
                prediction_list[batch] += pred 
            # Remove batch from gpu
            del X
            del y 
            torch.cuda.empty_cache()
            
    return prediction_list