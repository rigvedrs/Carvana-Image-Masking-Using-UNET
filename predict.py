# Comments for my reference
# import torch
# from model import UNET

# checkpoint = torch.load('C:/Users/lenovo/workspace/Carvana Image Segmentation UNET/Model/model.pth')
# model = UNET() 
# model.load_state_dict(checkpoint["model_state_dict"])

# def predict(input_image):
#     input_image = input_image.unsqueeze(0)
#     output = model(input_image)
#     output = torch.squeeze(output)
    
#     output[output>0.0] = 1.0
#     output[output<=0.0]=0
#     #output = np.array(output)
    
    
#     return output

import onnxruntime as onnxrt
from preprocess import preprocess
import numpy as np
import matplotlib.pyplot as plt


def predict(input_img):
    
    model_onnx = 'Model/model.onnx'
    session= onnxrt.InferenceSession(model_onnx, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_img})
    
    return result

if __name__ == '__main__':
    
    input_img = preprocess('Examples/train_1.jpg')
    input_img = input_img.unsqueeze(0)
    input_img = input_img.detach().numpy()
    result = predict(input_img)    
    result1 = np.array(result)   
    result1 = result1.reshape(572,572)
    plt.imshow(result1,cmap="gray")
    plt.title("Masked Image")


    # input_img = np.sum(input_img, axis=3)
    # input_img = input_img/3
    # input_img = input_img.reshape(572,572)
    # print(input_img.shape)

