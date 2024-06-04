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




