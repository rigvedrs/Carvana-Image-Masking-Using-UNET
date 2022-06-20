import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess
from predict import predict


def inference(filepath):
#     input_batch = preprocess(filepath)
#     result = predict(input_batch)
#     pred_mask = np.array(result).astype(np.float32)
#     pred_mask = pred_mask * 255
#     pred_mask = pred_mask[0, 0, 0, :, :].astype(np.uint8)
#     plt.imshow(pred_mask)
#     plt.title("Predicted Tumor Mask")
    
#     print(data['image'].shape)

    # input_batch = preprocess(filepath)
    # result = predict(input_batch)    
    # result1 = np.array(result)
    # result1 = result1.reshape(572,572)
    # plt.imshow(result1,cmap="gray")
    # plt.title("Masked Image")
    
    input_img = preprocess(filepath)
    input_img = input_img.unsqueeze(0)
    input_img = input_img.detach().numpy()
    result = predict(input_img)    
    result1 = np.array(result)   
    result1 = result1.reshape(572,572)
    plt.imshow(result1,cmap="gray")
    plt.title("Masked Image")
    
    return plt



title = "Carvana Image Segmentation using PyTorch"
description = "Segmentation of cars from Carvana Dataset"
article = "<p style='text-align: center'><a href='https://www.kaggle.com/' target='_blank'>Kaggle Notebook: Image-Segmentation-PyTorch</a> | <a href='https://github.com/' target='_blank'>Github Repo</a></p>"
examples = [['Examples/train_1.jpg'], 
            ['Examples/train_2.jpg'], 
            ['Examples/train_3.jpg'], 
            ['Examples/train_4.jpg'],
            ['Examples/train_5.jpg']]  

outputs = gr.Plot()

demo = gr.Interface(inference, inputs=gr.inputs.Image(type="filepath"), outputs=outputs, title=title,
            description=description,
            article=article,
            examples=examples).launch(share=True,debug=False, enable_queue=True)

demo.lauch()