# Carvana-Image-Segmentation-Using-UNET
### Masking of Car Images using PyTorch and deployed using Heroku
The model architecture used is [UNET](https://arxiv.org/abs/1505.04597v1) which is trained using PyTorch, with the final model being converted to ONNX format and deployed using Heroku.




## Dataset 📂
Dataset used from Kaggle [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) which contains a large number of car images (as .jpg files). Each car has exactly 16 images, each one taken at different angles.



## Notebook 📒
View the notebook here: [](https://nbviewer.org/github/rigvedrs/Carvana-Image-Masking-Using-UNET/blob/main/Notebook/unet-image-segmentation.ipynb)



## Deployment 🚀
The model has been converted to ONNX format and deployed using Gradio & hosted on Heroku: [Car Image Masking](https://)



## Predictions 🔍
Predictions on unseen test data:


