# Imports from here
import argparse
import functions_train 
import os

parser = argparse.ArgumentParser()

parser.add_argument('--arch', type = str, default = 'vgg19', help = 'which CNN Model should be used for pretraining, choose between vgg13, vgg16, vgg19, densenet121, densenet161, alexnet | (default = vgg19)')
parser.add_argument('--save_directory', type = str, default = 'SavedModel/', help = 'directory to save trained model | (default = SavedModel/)')
parser.add_argument('--learningrate', type = float, default = 0.001, help = 'give learningrate as a float | (default = 0.001)')
parser.add_argument('--hidden_units', type = int, default = 508, help = 'give number of hidden units as an integer | (default = 508)')
parser.add_argument('--epochs', type = int, default = 1, help = 'give number of epochs as an integer | (default = 1)')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'cuda or cpu | (default = cuda)')

args = parser.parse_args()

# Run functions from functions_train.py

functions_train.information(args.arch, args.learningrate, args.hidden_units, args.epochs, args.save_directory, args.gpu)

trainloader, validloader, train_data = functions_train.training_input()

model,  save_architecture, input_layer = functions_train.pretrained_model(args.arch)

model, criterion, optimizer = functions_train.classifier(args.hidden_units, args.learningrate, model, input_layer)

model = functions_train.training_network(args.epochs, args.gpu, model, trainloader, validloader, criterion, optimizer)

store = functions_train.saving_model(args.save_directory, train_data, args.learningrate, args.epochs, model, optimizer)


# Save architecture model and filepath

os.path.join(args.save_directory, 'save_progress.txt')
with open("save_progress.txt", "w") as output:
    output.write(str(store) + "\n" + str(args.arch))






