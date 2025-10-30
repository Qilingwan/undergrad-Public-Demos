# DCGAN for CIFAR-10 Image Generation

## 1. Project Description

(a) This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic images from the CIFAR-10 dataset<br>
(b) The dataset contains 50,000 32x32 color images across 10 classes<br>
(c) The model trains a generator to synthesize images from random noise and a discriminator to distinguish real from fake images using adversarial learning<br>

## 2. Tech Stack / Tools Used

(a) Python 3.11+<br>
(b) PyTorch<br>
(c) Torchvision<br>
(d) NumPy<br>
(e) Matplotlib<br>

## 3. Objectives / Tasks

(a) Load and preprocess the CIFAR-10 dataset from pickle files<br>
(b) Define the Generator network to synthesize 32x32x3 images from latent vectors<br>
(c) Define the Discriminator network to classify images as real or fake<br>
(d) Implement adversarial training with alternating updates to generator and discriminator<br>
(e) Visualize generated samples and training loss curves<br>

## 4. Implementation / Methods

### 4.1 Import Libraries

(a) Imported torch and torchvision for tensor operations and data handling<br>
(b) Used torch.nn and torch.optim for model definition and optimization<br>
(c) Imported matplotlib.pyplot for visualization and pathlib.Path for file system operations<br>
(d) Used pickle and numpy for loading and processing CIFAR-10 pickle batches<br>

### 4.2 Data Loading and Preprocessing

(a) Defined transform pipeline with ToTensor and normalization to scale pixel values to [-1, 1]<br>
(b) Implemented loadCifarData to read all data_batch_* files, stack images, and convert to tensors<br>
(c) Created TensorDataset and DataLoader with batch size 128, shuffling, 4 workers, and memory pinning<br>
(d) Set device to CUDA if available, otherwise CPU<br>

### 4.3 Define Generator Model

(a) Generator inherits from nn.Module with z_dim=100 input noise dimension<br>
(b) Uses sequential transposed convolutions to upsample from 1x1 to 32x32<br>
(c) Applies BatchNorm2d and ReLU after each transposed conv except the last<br>
(d) Final layer uses Tanh to output images in [-1, 1] range<br>
(e) Forward pass reshapes input z to (N, z_dim, 1, 1) before sequential processing<br>

### 4.4 Define Discriminator Model

(a) Discriminator inherits from nn.Module and processes 32x32x3 input<br>
(b) Uses sequential convolutions to downsample from 32x32 to 1x1<br>
(c) Applies LeakyReLU(0.2) after each conv and BatchNorm2d after intermediate layers<br>
(d) Final convolution outputs single value passed through Sigmoid for probability<br>
(e) Forward pass returns flattened scalar per image<br>

### 4.5 Design Training Function

(a) Initialized Generator and Discriminator on selected device<br>
(b) Used Adam optimizer with learning rate 2e-4 and betas=(0.5, 0.999) for both networks<br>
(c) Applied BCELoss as adversarial loss function<br>
(d) Generated fixed noise vector of 64 samples for consistent visualization<br>
(e) Training loop per epoch:<br>
    Loaded real batch and scaled to [-1, 1]<br>
    Trained Discriminator on real (label 1) and fake (label 0) samples<br>
    Trained Generator to fool Discriminator (target label 1)<br>
    Recorded final batch losses<br>
(f) Every 5 epochs: generated and saved sample images, saved generator checkpoint, printed losses<br>
(g) Returned loss history for plotting<br>

## 5. Results / Outputs
<p align="center">
  <img src="outputDemo/Training_Loss.png" alt="Training_Loss" width="50%">
</p>

**Epoch 10：**
<p align="center">
  <img src="outputDemo/epoch_10.png" alt="epoch_10">
</p>

**Epoch 30：**
<p align="center">
  <img src="outputDemo/epoch_30.png" alt="epoch_30">
</p>

**Epoch 50：**
<p align="center">
  <img src="outputDemo/epoch_50.png" alt="epoch_50">
</p>

## 6. Conclusion / Insights

(a) The DCGAN successfully learns to generate CIFAR-10-like images through adversarial training<br>
(b) Generator improves progressively while discriminator maintains pressure<br>
(c) Batch normalization and LeakyReLU contribute to training stability<br>
(d) Potential improvements include conditional generation or Wasserstein loss for better convergence<br>

## 7. Acknowledgements / References

(a) PyTorch Official Documentation: https://pytorch.org/docs/stable/index.html<br>
(b) CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html<br>
(c) Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks<br>





