# Neural_Style_Transfer
Implementation of Neural Style Transfer algorithm with PyTorch library

Neural Style Transfer (NST) is one of the most fun techniques in deep learning, which generates a new image (G) by merging content image (C) and style image (S). <br />
Actually, the purpose of this algorithm is to find an image with the same content as the content image (C) and the same style as the style image (S). <br />

![Picture1](https://user-images.githubusercontent.com/85555218/132051238-e2acee10-4cd1-447b-a47a-418af5bfcc42.png)

![Picture2](https://user-images.githubusercontent.com/85555218/132051148-bc2cff37-bd2d-4c16-abed-18d3761c2eae.png)

## ConvNet
NST uses a previously trained convolutional network. (I've used VGG-19, which has been used in the original paper and has already been trained on the extensive ImageNet database) <br /> 
Using the transfer learning method is necessary for this task cause we want to extract appropriate features from images and not train the model again. The model parameters are fixed, and we change the generated image parameters (pixels) to optimize the loss functions. <br />  As seen below, the main idea is to extract features from multi layers of VGG-19: (Style features have been extracted from yellow blocks, and Content features have been extracted from the blue block)

![Screenshot (428)](https://user-images.githubusercontent.com/85555218/132066385-d1dab3c3-dfb3-479e-b6fb-8407dab5783f.png)

## loss functions
there are 2 loss (cost) functions: content-loss and style-loss

### content-loss: 
The purpose of this loss function is to ensure that the generated image G matches the content of the image C. <br />
The earlier (shallower) layers of a ConvNet tend to detect lower-level features such as edges and simple textures, and the later (deeper) layers tend to detect higher-level features such as more complex textures as well as object classes. you can choose any of the VGG-19 convolution layers, but you'll get the most visually pleasing results if you choose a layer in the middle of the network, neither too shallow nor too deep. <br />
the image below shows the definition of the content-loss function:

![Picture4](https://user-images.githubusercontent.com/85555218/132084126-42c7ecaa-cace-4386-857d-002f40c836d3.png)

### style-loss: 
The purpose of this loss function is to ensure that the generated image G has the same style as the style image S. <br />
I've extracted features from 5 layers to find the accurate style of the style image. the difference between content and style is that you should not match the style-image features to the generated-image features; I mean, you need to do some preprocessing to find the style matrix (Gram matrix). <br />
In linear algebra, the gram matrix G of a set of Vectors  (V1, ..., Vn)  is the matrix of dot products. In other words, G(ij) compares how similar V(i) is to V(j). If they are highly similar, you would expect them to have a large dot product, and thus for G(ij) to be large. <br />
Finding a gram matrix or correlation between channels for each layer's features is very simple; you can see the definition in the image below:

![Picture5](https://user-images.githubusercontent.com/85555218/132092473-ca71fd48-df8a-41da-ba46-e9e9c5f8ecc5.png)

The formula of style-loss for just one layer and the formula of the total style-loss (sum of each layer's style-loss) can be seen below:

![f1](https://user-images.githubusercontent.com/85555218/132092654-2315607b-dd60-480d-8a22-36387fd2a97f.png)
![Screenshot (438)](https://user-images.githubusercontent.com/85555218/132092657-cdeed26f-a685-4386-bafb-cee2b4b5e92e.png)

### total-loss: 
Finally, let's create a loss function that minimizes both the style and the content cost. The formula is:

![f3](https://user-images.githubusercontent.com/85555218/132093300-7e6a9d22-4e4e-43f1-9d57-f62dec63e73f.png)

## references
L. A. Gatys, A. S. Ecker, and M. Bethge. <br />
*"A Neural Algorithm of Artistic Style"* (arXiv-2015)
