# Predict the flower class

import argparse
import functions_predict 


parser = argparse.ArgumentParser()

parser.add_argument('--top_k', type = int, default = 3, help = 'how many flower class predictions should be made for the picture | (default = 3)')
parser.add_argument('--category_names', type = str, default = 'cat_to_name.json' , help = 'file for flower name dictionary | (default = cat_to_name.json)')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'cuda or cpu | (default = cuda)')

args = parser.parse_args()


# Ask for image filepath to make predictions

image_filepath = str(input("\n Enter the image path, to predict its flower class \n e.g. flowers/test/81/image_00869.jpg \n Your input:"))


# Get information about saved architecture and checkpoint.pth filepath

with open ("save_progress.txt", "r") as file:
    data = file.readlines()
    saved_model_directory = data[0].strip()
    saved_architecture = data[1].strip()
    

# Run functions from functions_predict.py

functions_predict.information(image_filepath, args.top_k, args.category_names, args.gpu)

model = functions_predict.load_checkpoint(args.gpu, saved_model_directory, saved_architecture)

image = functions_predict.preprocess_image(image_filepath)

probs, classes = functions_predict.predict(image, args.gpu, model, args.top_k)

functions_predict.output(image_filepath, args.category_names, probs, classes)



