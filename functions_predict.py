# Functions to predict the flower class from an image

def information(image_filepath, top_k, category_names, gpu):
    
    '''
    Giving information about selected options for prediction
    '''
    statement = "\n Selection: Predicting the image {}: \n Giving a prediction of the {} most likely flower classes \n Using {} file as dictionary for flower classes \n The prediction is performed in {} mode \n".format(image_filepath, top_k, category_names, gpu)
    
    return print(statement)



def load_checkpoint(device, filepath, architecture):
    
    '''
    Input : device, filepath, architecture
    Operation : Loads previously saved model for further predictions
    Output : Saved model
    '''
    
    import torch
    from torchvision import models
    
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    
    
    checkpoint = torch.load(filepath, map_location = map_location)
    
    architecture = architecture.replace("",'')
    model = getattr(models, architecture)(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.to(device)
    
    return model


def preprocess_image(image_path):
    ''' 
    Input: image_path
    Operations: preprocesses the image to use as input for the model: crops, scales and normalizes the image
    Output: np.array of image    
    '''
    
    import numpy as np
    from PIL import Image
    import torch
    from torchvision import transforms
    
    # Open image from the image_path
    im = Image.open(image_path)
    
    # Preprocess using a the approach from the previous transformations
    
    preprocess = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    fin_im = np.array(preprocess(im))

    return fin_im



def predict(image, device, model, top_k):
    ''' 
    Input: Image_path, model, topk=5
    Predict the class (or classes) of an image using a trained deep learning model.
    Output: Pobablilty/Class Label/Index of 5 highest predicted classes 
    '''
    import torch
    import numpy as np
    
    # Process image to have a fitting input
    
    image_np = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(dim = 0)
    
    # Get probabilities for input
    
    model.to(device)
    image_predict = image_np.to(device)
                    
    with torch.no_grad():
        
        model.eval()
        
        logps = model.forward(image_predict)
        ps = torch.exp(logps)
        top_probs, top_class = ps.topk(top_k, dim = 1)
        
        probs = top_probs.cpu().numpy()[0]
        top_class = top_class.cpu().numpy()[0]
        class_to_idx = {lab: num for num, lab in model.class_to_idx.items()}
        classes = [class_to_idx[i] for i in top_class]
    
    return probs, classes





def output(image_path, dictionary, probs, classes):
    
    '''
    Input: image_path, dictionary, probs, classes
    Operation: Gives a prediction output of flower class and prediction percentage
    Output: none
    '''
    
    import json
    
        
    with open(dictionary, 'r') as f:
        
        cat_to_name = json.load(f)
    
        flower = [cat_to_name[f] for f in classes]
    
    ### Title optional -> showing, if prediction is correct | works only, if image is from the flowers/test filepath
    #number = image_path.split('/')[2]
    #flower_title = cat_to_name[number]
    #print("\n Real class is", flower_title.title())
    
    probs *= 100

    count = 1
    for title, prob in zip(flower, probs):
        print ("\n Prediction {}: {} {:.2f}%". format(count, title.title(), prob))
        count += 1
    
        
    return
        










    
    

