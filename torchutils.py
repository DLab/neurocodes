import torch
import torch.nn as nn

# import models
# import torchmodels

#----- Load and save models
def loadmodel(model, name):
    """ 
    Loads a PyTorch model.
        Inputs:
            model <obj>  : PyTorch Model, it needs to have the same number of parameters as the saved model.
            name  <str>  : Name of the model to load, it does have to include the extension.
        Outputs:
            none  <none> : Model is loaded and its parameters are updated.
    """
    device = model.device
    model.load_state_dict(torch.load(name, map_location=device))

def savemodel(model, name):
    """
    Saves a PyTorch model.
        Inputs:
            model <obj>  : PyTorch Model.
            name  <str>  : Name of the model to save, it does have to include the extension.
        Outputs:
            none  <none> : A file is created with the name plus the extension.
    """
    torch.save(model.state_dict(), name)

items = [loadmodel, savemodel]

def usage(verbose=True):
    for item in items:
        print(item.__name__, ":")
        if verbose:
            print(item.__doc__, "\n")