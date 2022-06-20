import matplotlib.pyplot as plt
from preprocess import preprocess
from predict import predict



input_batch = preprocess("C:/Users/lenovo/workspace/Carvana Image Segmentation UNET/Examples/train_2.jpg")
result = predict(input_batch)
x = result.detach().numpy()
x = x.reshape(572,572)

plt.imshow(x, cmap='gray')
plt.title("Segmented Image")

#x = np.zeros( (106, 106, 3) )
#result = x[:, :, 0]