import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

def train(dataset, pytmodel,
          TRAINSIZE=0.8,
          EPOCHS=100, BATCHSIZE=8, CUDA=True,
          CRITERION=(nn.MSELoss, {}),
          OPTIM=(optim.Adam, {'lr':0.0002, 'betas':(0.5, 0.999)}),
          VERBOSE=True):
    """
    Trains a MODEL for a dataset.
        Inputs:
          * args:
              dataset <pytorch dataset> : Dataset to train on and validate on according to a TRAINSIZE split.
              pytmodel          <tuple> : Tuple with two parameters, the first is the model, the second is a
                                          dictionary containing the keyword arguments to initialize the model.
          * kwargs:
              TRAINSIZE <float> : How much of the data to use for training (0.8->80%).
                                  ( default: 0.8 )
              EPOCHS      <int> : How many epochs to train on.
                                  ( default: 100 )
              BATCHSIZE:  <int> : Batch size to use in training.
                                  ( default: 8 )
              CUDA       <bool> : Whether to use cuda or not.
                                  ( default: True )
              CRITERION <tuple> : Tuple with two paremeters, the first is the criterion, the second is a dictionary
                                  containing the keyword arguments to initialize the model.
                                  ( default: (nn.MSELoss, {}) )
              OPTIM     <tuple> : Tuple with two paramenters, the first is the optimizer to use, the second is a
                                  dictionary containing the keywords arguments to initialize the model.
                                  ( default: (optim.Adam, {'lr':0.0002, 'betas':(0.5, 0.999)}) )
              VERBOSE    <bool> : Whether to print how the training is going or not.
                                  ( default: True )
         Outputs:
             model <pytorch model>        : The best model that had the best validation score among all the epochs.
             trainLoss      <list>        : The train losses for every epoch.
             testLoss       <list>        : The validation loss for every epoch.
             testDataset <pytorch dataset : The dataset that was used to validate on.
    """
    
    # Copy parameters from one model to another
    def copymodel(modelFrom, modelTo):
        params1 = modelFrom.named_parameters() 
        params2 = modelTo.named_parameters() 
        dict_params2 = dict(params2) 
        for name1, param1 in params1: 
            if name1 in dict_params2: 
                dict_params2[name1].data.copy_(param1.data)
    
    USECUDA = torch.cuda.is_available() and CUDA
    device_name = "cuda:0" if USECUDA else "cpu"
    device = torch.device(device_name)
    
    # Split the dataset
    train_size = int(TRAINSIZE * len(dataset))
    test_size = len(dataset) - train_size
    trainDataset, testDataset = torch.utils.data.random_split(dataset, [train_size, test_size])    
    
    # Make the iterators with Pytorch's dataloaders
    trainLoader = DataLoader(trainDataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True)
    testLoader = DataLoader(testDataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True)    
    
    # Train the network
    # Initialize the network
    model = pytmodel[0](**pytmodel[1]).to(device)
    bestmodel = pytmodel[0](**pytmodel[1]).to(device)
    
    best_test_loss = 0
    best_epochs = []
    
    trainLoss = []
    testLoss = []
    # Define the training loss function and the optimizers
    criterion = CRITERION[0](**CRITERION[1]).to(device)
    optimizer = OPTIM[0](model.parameters(), **OPTIM[1])
    
    for epoch in range(EPOCHS):
        if VERBOSE:
            print("\n\n###################\nEpoch {} out of {}\n###################\n\n".format(epoch + 1, EPOCHS))
        train_loss = 0
        for idx, (inp, target) in enumerate(trainLoader):
            
            inp = inp.to(device)
            target = target.to(device)
            
            model.train()
            optimizer.zero_grad()
            out = model(inp).to(device)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(trainLoader)
        trainLoss.append(train_loss)
        if VERBOSE:
            print("Train Loss: {}\n".format(train_loss))

        test_loss = 0
        for idx, (inp, target) in enumerate(testLoader):
            
            inp = inp.to(device)
            target = target.to(device)
            
            model.eval()
            out = model(inp).to(device)
            loss = criterion(out, target)
            test_loss += loss.item()
        test_loss /= len(testLoader)
        testLoss.append(test_loss)

        if VERBOSE:
            print("Test Loss: {}\n\n".format(test_loss))        
        
        if (test_loss < best_test_loss) or (epoch == 0):
            best_test_loss = test_loss
            copymodel(model, bestmodel)
            best_epochs.append(epoch)
            if VERBOSE:
                print("\n\n### MODEL SAVED ###\n\n")
    
    model = bestmodel

    return model, trainLoss, testLoss, testDataset

def customtrain(trainDataset, testDataset, pytmodel,
          EPOCHS=100, BATCHSIZE=8, CUDA=True,
          CRITERION=(nn.MSELoss, {}),
          OPTIM=(optim.Adam, {'lr':0.0002, 'betas':(0.5, 0.999)}),
          VERBOSE=True):
    """
    Trains a MODEL for a dataset but validates on a custom dataset.
        Inputs:
          * args:
              traindataset <pytorch dataset> : Dataset to train on.
              testdataset  <pytorch dataset> : Dataset to validate on.
              pytmodel               <tuple> : Tuple with two parameters, the first is the model, the second is a
                                               dictionary containing the keyword arguments to initialize the model.
          * kwargs:
              EPOCHS      <int> : How many epochs to train on.
                                  ( default: 100 )
              BATCHSIZE:  <int> : Batch size to use in training.
                                  ( default: 8 )
              CUDA       <bool> : Whether to use cuda or not.
                                  ( default: True )
              CRITERION <tuple> : Tuple with two paremeters, the first is the criterion, the second is a dictionary
                                  containing the keyword arguments to initialize the model.
                                  ( default: (nn.MSELoss, {}) )
              OPTIM     <tuple> : Tuple with two paramenters, the first is the optimizer to use, the second is a
                                  dictionary containing the keywords arguments to initialize the model.
                                  ( default: (optim.Adam, {'lr':0.0002, 'betas':(0.5, 0.999)}) )
              VERBOSE    <bool> : Whether to print how the training is going or not.
                                  ( default: True )
         Outputs:
             model <pytorch model>        : The best model that had the best validation score among all the epochs.
             trainLoss      <list>        : The train losses for every epoch.
             testLoss       <list>        : The validation loss for every epoch.
             testDataset <pytorch dataset : The dataset that was used to validate on.
    """
    
    # Copy parameters from one model to another
    def copymodel(modelFrom, modelTo):
        params1 = modelFrom.named_parameters() 
        params2 = modelTo.named_parameters() 
        dict_params2 = dict(params2) 
        for name1, param1 in params1: 
            if name1 in dict_params2: 
                dict_params2[name1].data.copy_(param1.data)
    
    USECUDA = torch.cuda.is_available() and CUDA
    device_name = "cuda:0" if USECUDA else "cpu"
    device = torch.device(device_name)
    
    # Make the iterators with Pytorch's dataloaders
    trainLoader = DataLoader(trainDataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True)
    testLoader = DataLoader(testDataset, batch_size=BATCHSIZE, shuffle=True, drop_last=True)    
    
    # Train the network
    # Initialize the network
    model = pytmodel[0](**pytmodel[1]).to(device)
    bestmodel = pytmodel[0](**pytmodel[1]).to(device)
    
    best_test_loss = 0
    best_epochs = []
    
    trainLoss = []
    testLoss = []
    # Define the training loss function and the optimizers
    criterion = CRITERION[0](**CRITERION[1]).to(device)
    optimizer = OPTIM[0](model.parameters(), **OPTIM[1])
    
    for epoch in range(EPOCHS):
        if VERBOSE:
            print("\n\n###################\nEpoch {} out of {}\n###################\n\n".format(epoch + 1, EPOCHS))
        train_loss = 0
        for idx, (inp, target) in enumerate(trainLoader):
            
            inp = inp.to(device)
            target = target.to(device)
            
            model.train()
            optimizer.zero_grad()
            out = model(inp).to(device)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(trainLoader)
        trainLoss.append(train_loss)
        if VERBOSE:
            print("Train Loss: {}\n".format(train_loss))

        test_loss = 0
        for idx, (inp, target) in enumerate(testLoader):
            
            inp = inp.to(device)
            target = target.to(device)
            
            model.eval()
            out = model(inp).to(device)
            loss = criterion(out, target)
            test_loss += loss.item()
        test_loss /= len(testLoader)
        testLoss.append(test_loss)

        if VERBOSE:
            print("Test Loss: {}\n\n".format(test_loss))        
        
        if (test_loss < best_test_loss) or (epoch == 0):
            best_test_loss = test_loss
            copymodel(model, bestmodel)
            best_epochs.append(epoch)
            if VERBOSE:
                print("\n\n### MODEL SAVED ###\n\n")
    
    model = bestmodel

    return model, trainLoss, testLoss, testDataset



items = [train, customtrain]

def usage(verbose=True):
    for item in items:
        print(item.__name__, ":")
        if verbose:
            print(item.__doc__, "\n")

if '__name__' == '__main__':
    usage()
    