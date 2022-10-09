# Images-classification-model
Model was made as an exercise in the science club : AGH_RACING

# Model properties:
![Notebook](notebook_with_model/plot.png "Notebook")

## Model's basic documentation:
⚫ Dataset consists of 60000 photos. Training set consists of 50000 photos (83,33%) and testing set consists of 10000 photos (16,66%)

⚪ Model was based on AlexNet CNN . Some parameters were changed to smaller values due to small image size and lower class count, than net was originally meant to be trained on. In final version CIFAR-10 was used, however in the earlier version I used CIFAR-100, but due to poor performance (around 30% accuracy) I finally decided to change it to CIFAR-10 . You can read more about AlexNet on https://en.wikipedia.org/wiki/AlexNet

⚫ Model was trained by 30 epochs with batch consisted of 128 photos. The model predicts ten classes : ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] on the photos with accuracy around 70%. Images were resized to 64x64 pixels, because the original size of 32x32 pixel were to small to downsample through the passing layers and making them too large worsen the CNN accuracy.

### Dataset is available under the following link:
  - https://www.cs.toronto.edu/~kriz/cifar.html
### Code with full documentation of the model was saved in Google Colab notebook and is available at the following link :
  - https://colab.research.google.com/drive/1luOLOArFflxP6PwKvhxUoTh3TTtHCCu1?usp=sharing
