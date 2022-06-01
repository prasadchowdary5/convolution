Different Types of Convolution
If you’ve heard of different kinds of convolutions in Deep Learning (e.g. 2D / 3D / 1x1 / Transposed / Dilated (Atrous) / Spatially Separable / Depthwise Separable / Flattened / Grouped / Shuffled Grouped Convolution), and got confused what they actually mean, this article is written for you to understand how they actually work.

Here in this article, I summarize several types of convolution commonly used in Deep Learning, and try to explain them in a way that is accessible for everyone. Besides this article, there are several good articles from others on this topic. Please check them out (listed in the Reference).

Output shape
Use the formula [(W-K+2P)/S]+1[(W−K+2P)/S]+1.

1. Convolution v.s. Cross-correlation
Convolution is a widely used technique in signal processing, image processing, and other engineering / science fields. In Deep Learning, a kind of model architecture, Convolutional Neural Network (CNN), is named after this technique. However, convolution in deep learning is essentially the cross-correlation in signal / image processing. There is a subtle difference between these two operations.

Without diving too deep into details, here is the difference. In signal / image processing, convolution is defined as:



It is defined as the integral of the product of the two functions after one is reversed and shifted. The following visualization demonstrated the idea.

Convolution in signal processing. The filter g is reversed, and then slides along the horizontal axis. For every position, we calculate the area of the intersection between f and reversed g. The intersection area is the convolution value at that specific position. Image is adopted and edited from this link.

Here, function g is the filter. It’s reversed, and then slides along the horizontal axis. For every position, we calculate the area of the intersection between ff and reversed gg. That intersection area is the convolution value at that specific position.

On the other hand, cross-correlation is known as sliding dot product or sliding inner-product of two functions. The filter in cross-correlation is not reversed. It directly slides through the function f. The intersection area between ff and gg is the cross-correlation. The plot below demonstrates the difference between correlation and cross-correlation.

Difference between convolution and cross-correlation in signal processing. Image is adopted and edited from Wikipedia.

In Deep Learning, the filters in convolution are not reversed. Rigorously speaking, it’s cross-correlation. We essentially perform element-wise multiplication and addition. But it’s a convention to just call it convolution in deep learning. It is fine because the weights of filters are learned during training. If the reversed function g in the example above is the right function, then after training the learned filter would look like the reversed function g. Thus, there is no need to reverse the filter first before training as in true convolution.

2. Convolution in Deep Learning
The purpose of doing convolution is to extract useful features from the input. In image processing, there is a wide range of different filters one could choose for convolution. Each type of filters helps to extract different aspects or features from the input image, e.g. horizontal / vertical / diagonal edges. Similarly, in Convolutional Neural Network, different features are extracted through convolution using filters whose weights are automatically learned during training. All these extracted features then are ‘combined’ to make decisions.

There are a few advantages of doing convolution, such as weights sharing and translation invariant. Convolution also takes spatial relationship of pixels into considerations. These could be very helpful especially in many computer vision tasks, since those tasks often involve identifying objects where certain components have certain spatially relationship with other components (e.g. a dog’s body usually links to a head, four legs, and a tail).

2.1. Convolution: the single channel version
Convolution for a single channel. Image is adopted from this link.

In Deep Learning, convolution is the element-wise multiplication and addition. For an image with 1 channel, the convolution is demonstrated in the figure below. Here the filter is a 3 \times 33×3 matrix with element [[0, 1, 2], [2, 2, 0], [0, 1, 2]][[0,1,2],[2,2,0],[0,1,2]]. The filter is sliding through the input. At each position, it’s doing element-wise multiplication and addition. Each sliding position ends up with one number. The final output is then a 3 \times 33×3 matrix. (Notice that stride = 1 and padding = 0 in this example. These concepts will be described in the section of arithmetic below.

2.2. Convolution: the multi-channel version
In many applications, we are dealing with images with multiple channels. A typical example is the RGB image. Each RGB channel emphasizes different aspects of the original image, as illustrated in the following image.

Different channels emphasize different aspects of the raw image. The image was taken at Yuanyang, Yunnan, China.

Another example of multi-channel data is the layers in Convolutional Neural Network. A convolutional-net layer usually consists of multiple channels (typically hundreds of channels). Each channel describes different aspects of the previous layer. How do we make transition between layers with different depth? How do we transform a layer with depth nn to the following layer with depth mm?

Before describing the process, we would like to clarify a few terminologies: layers, channels, feature maps, filters, and kernels. From a hierarchical point of view, the concepts of layers and filters are at the same level, while channels and kernels are at one level below. Channels and feature maps are the same thing. A layer could have multiple channels (or feature maps): an input layer has 3 channels if the inputs are RGB images. “channel” is usually used to describe the structure of a “layer”. Similarly, “kernel” is used to describe the structure of a “filter”.

Difference between “layer” (“filter”) and “channel” (“kernel”).

The difference between filter and kernel is a bit tricky. Sometimes, they are used interchangeably, which could create confusions. Essentially, these two terms have subtle difference. A “Kernel” refers to a 2D array of weights. The term “filter” is for 3D structures of multiple kernels stacked together. For a 2D filter, filter is same as kernel. But for a 3D filter and most convolutions in deep learning, a filter is a collection of kernels. Each kernel is unique, emphasizing different aspects of the input channel.

With these concepts, the multi-channel convolution goes as the following. Each kernel is applied onto an input channel of the previous layer to generate one output channel. This is a kernel-wise process. We repeat such process for all kernels to generate multiple channels. Each of these channels are then summed together to form one single output channel. The following illustration should make the process clearer.

Here the input layer is a 5 \times 5 \times 35×5×3 matrix, with 3 channels. The filter is a 3 \times 3 \times 33×3×3 matrix. First, each of the kernels in the filter are applied to three channels in the input layer, separately. Three convolutions are performed, which result in 3 channels with size 3 \times 33×3.

The first step of 2D convolution for multi-channels: each of the kernels in the filter are applied to three channels in the input layer, separately. The image is adopted from this link.

Then these three channels are summed together (element-wise addition) to form one single channel (3 \times 3 \times 13×3×1). This channel is the result of convolution of the input layer (5 \times 5 \times 35×5×3 matrix) using a filter (3 \times 3 \times 33×3×3 matrix).



Equivalently, we can think of this process as sliding a 3D filter matrix through the input layer. Notice that the input layer and the filter have the same depth (channel number = kernel number). The 3D filter moves only in 2-direction, height & width of the image (That’s why such operation is called as 2D convolution although a 3D filter is used to process 3D volumetric data). At each sliding position, we perform element-wise multiplication and addition, which results in a single number. In the example shown below, the sliding is performed at 5 positions horizontally and 5 positions vertically. Overall, we get a single output channel.

Another way to think about 2D convolution: thinking of the process as sliding a 3D filter matrix through the input layer. Notice that the input layer and the filter have the same depth (channel number = kernel number). The 3D filter moves only in 2-direction, height & width of the image (That’s why such operation is called as 2D convolution although a 3D filter is used to process 3D volumetric data). The output is a one-layer matrix.

Now we can see how one can make transitions between layers with different depth. Let’s say the input layer has D_{in}D
in
  channels, and we want the output layer has D_{out}D
out
  channels. What we need to do is to just apply D_{out}D
out
  filters to the input layer. Each filter has D_{in}D
in
  kernels. Each filter provides one output channel. After applying D_{out}D
out
  filters, we have D_{out}D
out
  channels, which can then be stacked together to form the output layer.

Standard 2D convolution. Mapping one layer with depth Din to another layer with depth Dout, by using Dout filters.

3. 3D Convolution
In the last illustration of the previous section, we see that we were actually perform convolution to a 3D volume. But typically, we still call that operation as 2D convolution in Deep Learning. It’s a 2D convolution on a 3D volumetric data. The filter depth is same as the input layer depth. The 3D filter moves only in 2-direction (height & width of the image). The output of such operation is a 2D image (with 1 channel only).

Naturally, there are 3D convolutions. They are the generalization of the 2D convolution. Here in 3D convolution, the filter depth is smaller than the input layer depth (kernel size < channel size). As a result, the 3D filter can move in all 3-direction (height, width, channel of the image). At each position, the element-wise multiplication and addition provide one number. Since the filter slides through a 3D space, the output numbers are arranged in a 3D space as well. The output is then a 3D data.

_In 3D convolution, a 3D filter can move in all 3-direction (height, width, channel of the image)_. At each position, the element-wise multiplication and addition provide one number. Since the filter slides through a 3D space, the _output numbers are arranged in a 3D space as well. The output is then a 3D data._

Similar as 2D convolutions which encode spatial relationships of objects in a 2D domain, 3D convolutions can describe the spatial relationships of objects in the 3D space. Such 3D relationship is important for some applications, such as in 3D segmentations / reconstructions of biomedical imagining, e.g. CT and MRI where objects such as blood vessels meander around in the 3D space.

4. 1 x 1 Convolution
Since we talked about depth-wise operation in the previous section of 3D convolution, let’s look at another interesting operation, 1 \times 11×1 convolution.

You may wonder why this is helpful. Do we just multiply a number to every number in the input layer? Yes and No. The operation is trivial for layers with only one channel. There, we multiply every element by a number.

Things become interesting if the input layer has multiple channels. The following picture illustrates how 1 \times 11×1 convolution works for an input layer with dimension H \times W \times DH×W×D. After 1 \times 11×1 convolution with filter size 1 \times 1 \times D1×1×D, the output channel is with dimension H \times W \times 1H×W×1. If we apply N such 1 \times 11×1 convolutions and then concatenate results together, we could have a output layer with dimension H \times W \times NH×W×N.

1 x 1 convolution, where the filter size is 1 x 1 x D.

Initially, 1 x 1 convolutions were proposed in the Network-in-network paper. They were then highly used in the Google Inception paper..) A few advantages of 1 x 1 convolutions are:

Dimensionality reduction for efficient computations
Efficient low dimensional embedding, or feature pooling
Applying nonlinearity again after convolution
The first two advantages can be observed in the image above. After 1 \times 11×1 convolution, we significantly reduce the dimension depth-wise. Say if the original input has 200 channels, the 1 \times 11×1 convolution will embed these channels (features) into a single channel. The third advantage comes in as after the 1 \times 11×1 convolution, non-linear activation such as ReLU can be added. The non-linearity allows the network to learn more complex function.

These advantages were described in Google’s Inception paper as:

“One big problem with the above modules, at least in this naïve form, is that even a modest number of 5x5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters.

This leads to the second idea of the proposed architecture: judiciously applying dimension reductions and projections wherever the computational requirements would increase too much otherwise. This is based on the success of embeddings: even low dimensional embeddings might contain a lot of information about a relatively large image patch... That is, 1 x 1 convolutions are used to compute reductions before the expensive 3 x 3 and 5 x 5 convolutions. Besides being used as reductions, they also include the use of rectified linear activation which makes them dual-purpose.”

One interesting perspective regarding 1 x 1 convolution comes from Yann LeCun “In Convolutional Nets, there is no such thing as “fully-connected layers”. There are only convolution layers with 1x1 convolution kernels and a full connection table.”



5. Convolution Arithmetic
We now know how to deal with depth in convolution. Let’s move on to talk about how to handle the convolution in the other two directions (height & width), as well as important convolution arithmetic.

Here are a few terminologies:

Kernel size: kernel is discussed in the previous section. The kernel size defines the field of view of the convolution.
Stride: it defines the step size of the kernel when sliding through the image. Stride of 1 means that the kernel slides through the image pixel by pixel. Stride of 2 means that the kernel slides through image by moving 2 pixels per step (i.e., skipping 1 pixel). We can use stride (>= 2) for downsampling an image.
Padding: the padding defines how the border of an image is handled. A padded convolution (‘same’ padding in Tensorflow) will keep the spatial output dimensions equal to the input image, by padding 0 around the input boundaries if necessary. On the other hand, unpadded convolution (‘valid’ padding in Tensorflow) only perform convolution on the pixels of the input image, without adding 0 around the input boundaries. The output size is smaller than the input size.
This following illustration describes a 2D convolution using a kernel size of 3, stride of 1 and padding of 1.



There is an excellent article about detailed arithmetic (“A guide to convolution arithmetic for deep learning”)..) One may refer to it for detailed descriptions and examples for different combinations of kernel size, stride, and padding. Here I just summarize results for the most general case.

For an input image with size of i, kernel size of k, padding of p, and stride of s, the output image from convolution has size o:



6. Transposed Convolution (Deconvolution)
For many applications and in many network architectures, we often want to do transformations going in the opposite direction of a normal convolution, i.e. we’d like to perform up-sampling. A few examples include generating high-resolution images and mapping low dimensional feature map to high dimensional space such as in auto-encoder or semantic segmentation. (In the later example, semantic segmentation first extracts feature maps in the encoder and then restores the original image size in the decoder so that it can classify every pixel in the original image.)

Traditionally, one could achieve up-sampling by applying interpolation schemes or manually creating rules. Modern architectures such as neural networks, on the other hand, tend to let the network itself learn the proper transformation automatically, without human intervention. To achieve that, we can use the transposed convolution.

The transposed convolution is also known as deconvolution, or fractionally strided convolution in the literature. However, it’s worth noting that the name “deconvolution” is less appropriate, since transposed convolution is not the real deconvolution as defined in signal / image processing. Technically speaking, deconvolution in signal processing reverses the convolution operation. That is not the case here. Because of that, some authors are strongly against calling transposed convolution as deconvolution. People call it deconvolution mainly because of simplicity. Later, we will see why calling such operation as transposed convolution is natural and more appropriate.

It is always possible to implement a transposed convolution with a direct convolution. For an example in the image below, we apply transposed convolution with a 3 \times 33×3 kernel over a 2 \times 22×2 input padded with a 2 \times 22×2 border of zeros using unit strides. The up-sampled output is with size 4 \times 44×4.

Up-sampling a 2 x 2 input to a 4 x 4 output. Image is adopted from this [link](https://github.com/vdumoulin/conv\_arithmetic\).

Interestingly enough, one can map the same 2 x 2 input image to a different image size, by applying fancy padding & stride. Below, transposed convolution is applied over the same 2 x 2 input (with 1 zero inserted between inputs) padded with a 2 x 2 border of zeros using unit strides. Now the output is with size 5 x 5.

Up-sampling a 2 x 2 input to a 5 x 5 output. Image is adopted from this [link](https://github.com/vdumoulin/conv\_arithmetic\).

Viewing transposed convolution in the examples above could help us build up some intuitions. But to generalize its application, it is beneficial to look at how it is implemented through matrix multiplication in computer. From there, we can also see why “transposed convolution” is an appropriate name.

In convolution, let us define C as our kernel, Large as the input image, Small as the output image from convolution. After the convolution (matrix multiplication), we down-sample the large image into a small output image. The implementation of convolution in matrix multiplication follows as C_ x Large = SmallC
x
Large=Small.

The following example shows how such operation works. It flattens the input to a 16 x 1 matrix, and transforms the kernel into a sparse matrix (4 x 16). The matrix multiplication is then applied between sparse matrix and the flattened input. After that, the resulting matrix (4 x 1) is then transformed back to a 2 x 2 output.

Matrix multiplication for convolution: from a Large input image (4 x 4) to a Small output image (2 x 2).

Now, if we multiple the transpose of matrix CT on both sides of the equation, and use the property that multiplication of a matrix with its transposed matrix gives an Unit matrix, then we have the following formula CT x Small = Large, as demonstrated in the figure below.

Matrix multiplication for convolution: from a Small input image (2 x 2) to a Large output image (4 x 4).

As you can see here, we perform up-sampling from a small image to a large image. That is what we want to achieve. And now, you can also see where the name “transposed convolution” comes from.
