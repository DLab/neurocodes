#!usr/bin/env/ python3

import retinaldata

import torchmodels
import torchutils
import trainer
import datahandler

import torch
import torch.nn as nn
import torch.optim as optim

import click
import pickle
from os import path



@click.command()
@click.argument('directory')
@click.option('--epochs', default=300, show_default=True, 
             help='How many Epochs to train the network on.')
@click.option('--lr', default=0.0002, show_default=True, 
             help='Learning rate for the optimizer.')
@click.option('--cuda/--no-cuda', default=True, show_default=True, is_flag=True, 
             help='Whether to use CUDA to train the model or not.')
@click.option('--verbose/--no-verbose', default=True, show_default=True, is_flag=True,
             help='Verbose output (losses on certain epochs) or not.')
@click.option('--cell', type=click.Choice(["ganglionar", "bipolar"], case_sensitive=False), default="ganglionar",
             show_default=True, 
             help='Which kind of cell to work with.')
@click.option('--cell-type', default=4, show_default=True,
             help='Which cell to use (1 to 14 for bipolar) (1 to 39 for ganglionar)')
@click.option('--qi/--no-qi', default=False, show_default=True, is_flag=True,
             help='Filter by qi')
@click.option('--stimulus', default='whitenoise', type=click.Choice(["whitenoise", "chirp"], case_sensitive=False), 
             show_default=True,
             help='Which stimulus to train on.')
@click.option('--train-size', default=0.8, show_default=True,
             help='How much of the training data to use.')
@click.option('--batch-size', default=8, show_default=True,
             help='How many items per batch to use.')
@click.option('--white-full/--no-white-full', default=False, show_default=True, 
             help='Whether to use the full white noise stimulus to train.')
@click.option('--centered/--no-centered', default=False, show_default=True,
             help='Whether to center the whitenoise stimulus or not.')
@click.option('--save-model/--no-save-model', default=True, show_default=True,
             help='Save the resulting model.')
@click.option('--save-loss/--no-save-loss', default=True, show_default=True,
             help='Save the loss.')
def cli(directory, epochs, cuda, verbose, lr, cell, qi, 
        cell_type, stimulus, train_size, batch_size, 
        white_full, centered, save_model, save_loss):
    """
    Trains a neural network with data located in DIRECTORY
    """
    DIRECTORY = path.abspath(directory)
    cellData = retinaldata.Data(DIRECTORY + '/', cell=cell)
    
    if stimulus == 'chirp':
        raise NotImplementedError(f'{stimulus} not yet implemented')
    if cell == "bipolar" and cell_type not in range(1, 15):
        raise NameError(f'Cell {cell} does not have type {cell_type} (Choose 1..14)')
    if cell == "ganglionar" and cell_type not in range(1, 40):
        raise NameError(f'Cell {cell} does not have type {cell_type} (Choose 1..39)')
    
    CELLTYPE = cell_type
    
    if not centered:
        whiteStimulus = cellData.stimulus(stimulus, cell_type=CELLTYPE)
        whiteResponse = cellData.response(stimulus, cell_type=CELLTYPE)
        
    else:
        _, loc = cellData.rf(centered=True)
        whiteStimulus = cellData.stimulus(stimulus, cell_type=CELLTYPE, centered=True, loc=loc)
        whiteResponse = cellData.response(stimulus, cell_type=CELLTYPE)

    if not white_full:
        cutStimulus   = whiteStimulus[5000:12000] if not centered else whiteStimulus[:, 5000:12000]
        cutResponse   = whiteResponse[:, 5000:12000]

    else:
        cutStimulus = whiteStimulus[1000:16000] if not centered else whiteStimulus[:, 1000:16000]
        cutResponse = whiteResponse[:, 1000:16000]
            
    TRAINSIZE     = int(cutStimulus.shape[0]*train_size)

    trainStimulus = cutStimulus[:TRAINSIZE] if not centered else cutStimulus[:, :TRAINSIZE]
    validStimulus = cutStimulus[TRAINSIZE:] if not centered else cutStimulus[:, TRAINSIZE:] 

    trainResponse = cutResponse[:, :TRAINSIZE]
    validResponse = cutResponse[:, TRAINSIZE:]

    toTensor      = datahandler.ToTensor()
    
    if not centered:
        trainDataset = datahandler.WhiteNoiseDataset(trainStimulus, trainResponse, transform=toTensor)
        testDataset  = datahandler.WhiteNoiseDataset(validStimulus, validResponse, transform=toTensor)
        
    else:
        trainDataset = datahandler.WhiteNoiseDatasetCentered(trainStimulus, trainResponse, transform=toTensor)
        testDataset  = datahandler.WhiteNoiseDatasetCentered(validStimulus, validResponse, transform=toTensor)

        
    TRAINDICT = {
        "EPOCHS"    : epochs,
        "BATCHSIZE" : batch_size,
        "CUDA"      : cuda,
        "CRITERION" : (nn.MSELoss, {}),
        "OPTIM"     : (optim.Adam, {'lr':lr, 'betas':(0.5, 0.999)}),
        "VERBOSE"   : verbose,
                }
    
    device = torch.device("cuda:0") if cuda else torch.device("cpu")
    
    pytmodel = (torchmodels.Bati, {"lw":cutStimulus.shape[-2], "lh":cutStimulus.shape[-1], "device":device},)

    model, trloss, tsloss = trainer.customtrain(trainDataset, testDataset, 
                                       pytmodel, **TRAINDICT)
    
    if save_model:
        name = f"model_{pytmodel[0].__name__}_{pytmodel[1]['lw']}x{pytmodel[1]['lh']}_{cell}_type_{cell_type}_epochs_{epochs}_lr_{lr}_batch_size_{batch_size}.pt"
        
        torchutils.savemodel(model, name)
    
    if save_loss:
        nametr = f"trloss_{pytmodel[0].__name__}_{pytmodel[1]['lw']}x{pytmodel[1]['lh']}_{cell}_type_{cell_type}_epochs_{epochs}_lr_{lr}_batch_size_{batch_size}.pt"
        
        namets = f"tsloss_{pytmodel[0].__name__}_{pytmodel[1]['lw']}x{pytmodel[1]['lh']}_{cell}_type_{cell_type}_epochs_{epochs}_lr_{lr}_batch_size_{batch_size}.pt"
        
        with open(nametr + '.pkl', 'wb') as handle:
            pickle.dump(trloss, handle, protocol=pickle.HIGHEST_PROTOCOL)
           
        with open(namets + '.pkl', 'wb') as handle:
            pickle.dump(tsloss, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    cli()