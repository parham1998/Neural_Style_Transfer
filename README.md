# Neural_Style_Transfer
Implementation of Neural Style Transfer algorithm with PyTorch library

Neural Style Transfer (NST) is one of the most fun techniques in deep learning, which generates a new image (G) by merging content image (C) and style image (S). <br />
Actually, the purpose of this algorithm is to find an image with the same content as the content image (C) and the same style as the style image (S). <br />

![Picture1](https://user-images.githubusercontent.com/85555218/132051238-e2acee10-4cd1-447b-a47a-418af5bfcc42.png)

![Picture2](https://user-images.githubusercontent.com/85555218/132051148-bc2cff37-bd2d-4c16-abed-18d3761c2eae.png)

## ConvNet
NST uses a previously trained convolutional network. (I've used VGG-19 which has been used in the original paper and has already been trained on the very large ImageNet database) <br /> 
Actually, using the transfer learning method is necessary for this task, cause we want to extract appropriate features from images and don't train the model again. I mean the model parameters are fixed, and we change the generated image parameters (pixels) to optimize the loss functions. <br />  As seen below, the main idea is to extract features from multi layers of VGG-19: (Style features have been extracted from yellow blocks, and Content features have been extracted from the blue block)

![Screenshot (428)](https://user-images.githubusercontent.com/85555218/132066385-d1dab3c3-dfb3-479e-b6fb-8407dab5783f.png)

## loss functions

## references
L. A. Gatys, A. S. Ecker, and M. Bethge. <br />
*"A Neural Algorithm of Artistic Style"* (arXiv-2015)
