# =============================================================================
# Import required libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

# =============================================================================
# Check if CUDA is available
# =============================================================================
train_on_GPU = torch.cuda.is_available()
if not train_on_GPU:
    print('CUDA is not available. Training on CPU ...')
else:
    print('CUDA is available! Training on GPU ...')
    print(torch.cuda.get_device_properties('cuda'))

# =============================================================================
# Prepare data
# =============================================================================
def image_loader(path):
    image = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image).unsqueeze(0)
    if train_on_GPU:
        image = image.cuda()
    return image

# load images
content_image = image_loader('./images/content3.jpg')
style_image = image_loader('./images/style5.jpg')

# generate image    
generated_image = content_image.clone().requires_grad_(True)

def deprocess(tensor):
    image = tensor.cpu().clone()
    image = image.numpy()
    image = image.squeeze(0)
    image = image.transpose(1,2,0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = image.clip(0, 1)
    return image
    
plt.imshow(deprocess(content_image)) 
plt.imshow(deprocess(style_image)) 
plt.imshow(deprocess(generated_image.detach())) 

# =============================================================================
# Define loss functons
# =============================================================================
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, content_out, generated_out):
        loss = torch.sum((content_out - generated_out)**2)
        b, c, h, w = content_out.size()
        normalize = 1/(4 * c * h * w)
        return normalize * loss


class StyleLoss(nn.Module):
    def __init__(self, style_f):
        super(StyleLoss, self).__init__()
        self.style_weights = {
            'conv1_1': 0.2,
            'conv2_1': 0.2,
            'conv3_1': 0.2,
            'conv4_1': 0.2,
            'conv5_1': 0.2,
        }
        self.style_grams = {l : self.gram(style_f[l]) for l in style_f}
        
    def gram(self, tensor):
        b, c, h, w = tensor.size()
        tensor = tensor.view(c, h*w)
        return torch.mm(tensor, tensor.t())
    
    def forward(self, generated_feature):
        loss = 0
        for layer in self.style_weights:
            generated_f = generated_feature[layer]
            b, c, h, w = generated_f.size()
            generated_gram = self.gram(generated_f)
            style_gram = self.style_grams[layer]
            loss += self.style_weights[layer] * torch.sum((style_gram - generated_gram)**2)
            normalize = 1/(4 * (c * h * w)**2)
        return normalize * loss


class Totalloss(nn.Module):
    def __init__(self):
        super(Totalloss, self).__init__()
    
    def forward(self, c_loss, s_loss , A, B):    
        loss = A * c_loss + B * s_loss
        return loss

# =============================================================================
# CNN models
# =============================================================================  
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.layers = {
            '0': 'conv1_1', # style_feature
            '5': 'conv2_1', # style_feature
            '10': 'conv3_1', # style_feature
            '19': 'conv4_1', # style_feature
            '21': 'conv4_2', # content_feature
            '28': 'conv5_1' # style_feature
        } 
        self.net = torchvision.models.vgg19(pretrained=True).features[:29]
    
    def forward(self, img):
        features = {}
        for name, layer in self.net._modules.items():

            img = layer(img)

            if name in self.layers:
                features[self.layers[name]] = img
                
        return features
    
vgg19 = VGG19().eval()
for param in vgg19.parameters():
    param.requires_grad_(False)
    
if train_on_GPU:
    vgg19.cuda()
    print('\n net can be trained on gpu') 
    
content_f = vgg19(content_image)
style_f = vgg19(style_image)

# =============================================================================
# Specify loss function and optimizer
# =============================================================================
num_iterations = 3000

content_criterion = ContentLoss()
style_criterion = StyleLoss(style_f)
total_criterion = Totalloss()

# optimize generated_image parameters (pixels)
optimizer = optim.Adam([generated_image], lr=0.03)       

results = []

# =============================================================================
# training
# =============================================================================
print('==> Start Training ...')
for i in range(1, num_iterations+1):
    # forward pass: compute predicted image by passing noisy image to the model
    generated_f = vgg19(generated_image)     
        
    # calculate loss
    c_loss = content_criterion(content_f['conv4_2'], generated_f['conv4_2']) 
    s_loss = style_criterion(generated_f)
    loss = total_criterion(c_loss, s_loss, 1, 1e5)
    
    # zero the gradients parameter
    optimizer.zero_grad()
    
    # backward pass: compute gradient of the loss with respect to generated image parameters
    loss.backward()
    
    # parameters update
    optimizer.step()
    
    print('Iterations: {} \t c_loss: {:.3f} \t  s_Loss: {:.3f} \t Total_Loss: {:.3f}'.format(i, c_loss, s_loss, loss))
    if i % 500 == 0:
        results.append(deprocess(generated_image.detach()))
print('==> End of training ...')

for img in results:
    img = Image.fromarray(np.uint8(255*img))
    img = transforms.Resize((512, 512))(img)
    img.show()
