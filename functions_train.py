# Functions to train a neuronal network

def information(architecture, learningrate, hiddenunits, epochs, filepath, gpu):
    
    '''
    Giving information about selected options for training
    '''
    statement = " Selection: Training a CNN \n Using a pretrained {} architecture \n With hyperparameters: learningrate {}, {} hidden units and {} epoch(s) \n The trained model is stored under {}checkpoint.pth \n The training is performed in {} mode".format(architecture, learningrate, hiddenunits, epochs, filepath, gpu)
    
    return print(statement)



def training_input():
    
    ''' 
    Input: None, automatically leading to filepaths
    Operation: Transforms training and validation data to input for model
    Output: testloader, validloader
    '''
    import torch
    from torchvision import datasets, transforms
    
    # Data directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Transform pictures
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # Using the image datasets and the transforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    
    return trainloader, validloader, train_data





def pretrained_model(architecture):
    
    ''' 
    Input: architecture
    Operation: Loads pretrained model and freezes its parameters
    Output: model
    '''
    import torch
    from torchvision import models
    
    possible_archs = {'vgg13': 25088, 'vgg16': 25088, 'vgg19': 25088, 'densenet121': 1024, 'densenet161': 2208, 'alexnet': 9216}
    
    if architecture in possible_archs:
        
        input_layer = possible_archs.get(architecture)
        
        # Remove string for further operations
        save_architecture = architecture
        architecture = architecture.replace("",'')
        model = getattr(models, architecture)(pretrained = True)
       
        for param in model.parameters():
            param.requires_grad = False
            
            
    else: 
        print( "Try again! Please give a valid architecture. \nValid architectures: vgg13, vgg16, vgg19, alexnet, densenet121, densenet161")
        return
    
    return model,  save_architecture, input_layer



def classifier(hiddenunits, learningrate, model, input_layer):
    
    ''' 
    Input: hidden_layers, learningrate, model
    Operation: Define the classifier network, setting optimizer and errorfunction
    Output: model.classifier, criterion, optimizer
    '''
    import torch
    from torch import nn, optim
    from collections import OrderedDict
    
    dropout_var = 0.5
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_layer, hiddenunits)),
                                       ('activation1', nn.ReLU()),
                                       ('dropout1', nn.Dropout(dropout_var)),
                                       ('fc2', nn.Linear(hiddenunits,102)),
                                       ('log_softmax', nn.LogSoftmax(dim = 1))]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learningrate)
    
    
    return model, criterion, optimizer





def training_network(epochs, device, model, trainloader, validloader, criterion, optimizer):
    '''
    Input: epochs, device, model, trainloader, validloader, criterion, optimizer
    Operation: Trains the classifier part of the CNN and validates the model every 20 steps
    Output: trained model with graphic
    '''
    
    import time
    import torch
    from torch import nn, optim
    from torchvision import datasets, transforms, models
   

    #Make a validation step after every 20 iterations
    step_every = 20

    model.to(device) 

    # Start measuring time
    start = time.time()
    print("Start training ...")

    # List values for plot at the end -> optional
    #map_train_loss = []
    #map_test_loss = []

    for epoch in range(epochs):
        
        steps = 0
        running_loss = 0
        print("Epoch", epoch +1)
        
    
        # Training the classifier
        for images, labels in trainloader:
        
            steps += 1 
            
            model.train()
        
            images, labels = images.to(device), labels.to(device) 
        
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            #Validation testing
        
            if steps % step_every == 0:
                model.eval()
                
                test_loss = 0
                accuracy = 0
                
                for imagesval, labelsval in validloader: 
                    
                    imagesval, labelsval = imagesval.to(device) , labelsval.to(device) 
                    
                    with torch.no_grad(): 
                        
                        outputs = model.forward(imagesval)
                        test_loss = criterion(outputs,labelsval)
                        ps = torch.exp(outputs).data
                        equal = (labelsval.data == ps.max(1)[1])
                        accuracy += equal.type_as(torch.FloatTensor()).mean()

                # Calculating the training loss, validation loss and accuracy
                test_loss /= len(validloader)
                accuracy /= len(validloader)
                train_loss = running_loss/step_every
            
            
                # Mapping training loss with plt.legend -> optional
                #map_train_loss.append(train_loss)
                #map_test_loss.append(test_loss)
                
                running_loss = 0

                print("Epoch {}/{}  Training Loss: {:.3f}  Validation Loss: {:.3f}  Accuracy: {:.2f}%".format((epoch + 1), 
                epochs, train_loss, test_loss, accuracy * 100))

    # Optional plotting training loss and test loss, to see if model tends to overfit
    #plt.plot(map_train_loss, label= "Training Loss")
    #plt.plot(map_test_loss, label= "Validation Loss")
    #plt.legend()
    
    
    end = time.time() 
    time = end - start
    print("Total training time : {}m {}s".format(int(time // 60), int(time % 60)))
    
    return model
   


    
    
def saving_model(filepath, train_data, learningrate, epochs, model, optimizer):
    
    '''
    Input : filepath, train_data, learningrate, epochs, model
    Operation : Saves trained model to checkpoint.pth
    Output : Statement where file was saved
    '''
    
    import torch
    import os
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': 25088,
                    'output_size': 102,
                    'classifier': model.classifier,
                    'learningrate': learningrate,
                    'epoch': epochs,
                    'class_to_idx': model.class_to_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
    
    # Saving the model and its hyperparameters to the filepath
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    os.path.join(filepath, 'checkpoint.pth')
    
    store = "{}checkpoint.pth".format(filepath)
    torch.save(checkpoint, store)
    
    print("saved model checkpoint under ", store)
    
    return store

 
    
    
        
    




    
    
    
    
    
    
    
    
