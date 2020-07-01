## Machine Learning Engineer (AI programming with python) Nanodegree at Udacity 

## Flower_Image_Classifier



### Project description
This code will allow you to train a selfbuild neuronal network on top of a pretrained model from torchvision.models and can predict the flower class of any flower image input. The project contains 2 subprojects:

**(1)** Flower_Image_Classifier for terminal use 

**(2)** Flower_Image_Classifier for Jupiter notebook 

<br/>

### Installation
To run the terminal code (1), you will need:

|**Python 3.x**|
|--------------|
|**Numpy**|
|**Matplotlib**|
|**Pytorch**|
|**Seaborn**|

**Optional:** Use a gpu to accelerate the model training and image prediction process <br/>
<br/>

If you want to execute the Jupiter notebook code (2):

**Jupiter notebook** + mentioned packages above

<br/>


### Usage

The command line project (1) consists of **2 parts**: `train.py` and `predict.py`

<br/>

**Training part of the project:**

Requires: `train.py` and `functions_train.py`\
Trains the complete model without additional input. For individual settings choose options.

  ```
  python ./train.py --help
  
  usage: train.py [--h] [--arch ARCH] [--save_directory SAVE_DIRECTORY]
                  [--learningrate LEARNINGRATE] [--hidden_units HIDDEN_UNITS]
                  [--epochs EPOCHS] [--gpu GPU]
                  
  optional arguments/default settings:
  --h, --help                        show this help message and exit
  
  --arch ARCH                        which CNN Model should be used for pretraining, choose between
                                     vgg13, vgg16, vgg19, densenet121, desnsenet161, alexnet
                                     (default = vgg19)
  
  --save directory SAVE_DIRECTORY    directory to save trained model
                                     (default = SavedModel/)
                                     
  --learningrate LEARNINGRATE        give learningrate as a float
                                     (default = 0.001)
  
  --hidden_units HIDDENUNITS         give number of hidden units as an integer
                                     (default = 508)
                                     
  --epochs EPOCHS                    give number of epochs as an integer
                                     (default = 1)
                                     
  --gpu GPU                          cuda or cpu
                                     (default = cuda)
```
  

<br/>

**Prediction part of the project:**

Requires: `predict.py` and `functions_predict.py` \
Predicts the flower class with the additional input of the image-filepath in the command line. For individual settings choose options.

```
  python ./predict.py --help
  
  usage: predict.py [--h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                    [gpu GPU]
  
  optional arguments/default settings:
  --h, --help                        show this help message and exit
  
  --top_k TOP_K                      how many flower class predictions sould be made for the picture
                                     (default = 3)
  
  --category_names CATEGORY_NAMES    file for flower name dictionary
                                     (default = cat_to_name.json)
  
  --gpu GPU                          cuda or cpu
                                     (default = cuda)
                                     
```
                                     

