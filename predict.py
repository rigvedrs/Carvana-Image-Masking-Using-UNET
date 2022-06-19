import torch
import numpy as np
from model import UNET

checkpoint = torch.load('C:/Users/lenovo/workspace/Carvana Image Segmentation UNET/Model/model.pth')
model = UNET() 
model.load_state_dict(checkpoint["model_state_dict"])

def predict(input_image):
    input_image = input_image.unsqueeze(0)
    output = model(input_image)
    output = torch.squeeze(output)
    
    output[output>0.0] = 1.0
    output[output<=0.0]=0
    #output = np.array(output)
    
    
    return output