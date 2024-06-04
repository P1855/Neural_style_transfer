# Neural_style_transfer_Project
This project is based on fast style transfer using neural networks model from tensorflow hub and it takes content image and style image to produce a stylized content image. 
## Neural Style Tansfer 
Neural Style Transfer (NST) is a technique which combines two images (content image for the object of the image and the style image from which only the style is extracted) into a third target image.
This concept was first published in 2015 in this paper - [Link](https://arxiv.org/abs/1508.06576)

The fundamental idea involves minimizing two losses. The first loss quantifies the content difference between the Content image and the target image (Stylized image), while the second loss measures the style discrepancy between the style image and the input image’s style.

### Exmaple 


![download](https://github.com/P1855/Neural_style_transfer/assets/98693127/091ec87f-984c-4995-80c3-aab5de70b5e0)

 
### Losses


In neural style transfer, the overall loss function combines both content and style losses. Our objective is to minimize this joint loss on the target image:

$$
\text{Total Loss} = \alpha \cdot \text{Content Loss} + \beta \cdot \text{Style Loss}
$$

Here:
- $\alpha$ and $\beta$ represent the weighted factors for each loss component.
- The content loss measures the difference between the content image and the stylized image.
- The style loss quantifies the discrepancy in style between the style image and the input image.

### Content Loss


The **content loss**, one of the two components in neural style transfer (NST), is straightforward. Here's how it works:

1. We feed both the content image (*p*) and the target image (*x*) into the network.
2. The network produces feature representations from intermediate layers for both images.
3. The content loss is then computed as the squared error between these feature vectors:

$$
\text{Content Loss} = \frac{1}{2} \sum_{i,j} (F_{ij} - P_{ij})^2
$$

   - *F* represents the feature representation of the content image.
   - *P* represents the feature representation of the target image.
   - The summation is over all spatial dimensions (*i*, *j*) of the feature maps.

5. By minimizing this loss, we ensure that the target image captures the same content as the content image.

### Style Loss



To calculate the **style loss** in neural style transfer (NST), we need a way to capture the artistic style of an image and measure the correlations between features at each layer. Gram matrices come to our rescue. Specifically:

1. **Gram Matrix**:
   - The Gram matrix (*G*) is computed for each layer.
   - It encodes the correlations between feature maps.
   - While the original NST paper uses the Gram matrix, other algorithms that disregard feature position can also be effective.
   - In practice, the Gram matrix captures the essence of style by focusing on feature relationships.

2. **Style Loss Equation**:
   - The style loss is akin to computing the Maximum Mean Discrepancy (MMD) between two images.
   - As long as the measure aligns with the MMD algorithm, the loss is appropriately computed.
   - The total style loss combines mean-squared distances across layers (*E*) with weighted factors for each layer:

$$
\text{Style Loss} = \sum_{l} \frac{1}{4N_l^2 M_l^2} \sum_{i,j} (G_{ij}^l - A_{ij}^l)^2
$$

   - Here:
     - *N_l* represents the number of feature maps in layer *l*.
     - *M_l* denotes the spatial dimensions of the feature maps.
     - *G* is the Gram matrix for the style image.
     - *A* is the Gram matrix for the target image.
     - The summation is over spatial dimensions (*i*, *j*) of the Gram matrices.

By minimizing this style loss, we ensure that the target image captures the desired artistic style. 

### The model


In neural style transfer, we manipulate a pretrained model to extract features from images. We focus on convolutional layers and discard top layers. The process involves reconstructing target image features to match the desired style.


## Description of the project

### Requirements

1. Tensorflow
2. Tensorflow_Hub
3. Matplotlib
4. NumPy
5. Open-CV

In this project, first, all the environment for loading content image and style image were loaded. Then a [Tensorflow_hub model](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2) was used. The Content image and the syle image was feeded into the model which gave an output stylized image. Then multiple content images and style images were taken from the tensorflow tutorial and used to output some stylized images for demonstration purpose. 


![3](https://github.com/P1855/Neural_style_transfer/assets/98693127/2c219f9d-4200-4bba-97bb-86cd3703c2d5)


The model was fed with an **input video** which the model took each frame as a content image and stylized it to an output video using the style image.


Test video 


https://github.com/P1855/Neural_style_transfer/assets/98693127/37a85ed1-61f7-4791-8060-c6c42cf8aef5


Output Video



https://github.com/P1855/Neural_style_transfer/assets/98693127/fa574a92-1012-4839-84c3-b361d7a4c5c1



The **real time fast neural style transfer** file uses the same tensorflow hub model and uses **webcam** to stylize the video captured. It can be used as *filter* for making videos.
