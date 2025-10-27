# Convolutional Neural Networks (CNNs) Theory and Basics

---

## Introduction

Convolutional Neural Networks (CNNs) revolutionized how machines understand visual data, enabling breakthroughs in image recognition, video analysis, and more.
CNNs were first introduced in the late 1980s and early 1990s, with foundational work by Yann LeCun and colleagues, notably the LeNet model for handwritten digit recognition in 1989. However, CNNs gained widespread attention and practical success with advances in computational power and the ImageNet competition in 2012.

Before CNNs, dense (fully connected) neural networks were the primary approach for many tasks, but they had critical limitations for images:
- Dense networks treat each pixel independently, flattening spatial structure and losing locality.
- They require an enormous number of parameters when applied to high-dimensional images, leading to overfitting and inefficient training.
- Dense layers do not naturally exploit spatial hierarchies or local patterns in data.

CNNs solved these problems by introducing convolutional layers with learnable filters that slide across images to detect local patterns like edges, textures, and shapes. 
Weight sharing and local connectivity drastically reduce the number of parameters, making training feasible on large image datasets. Furthermore, CNNs naturally preserve spatial relationships and hierarchies, enabling progressively more abstract feature extraction through multiple layers.

Since their inception, many influential CNN architectures have been developed, including AlexNet (2012), VGGNet, GoogLeNet (Inception), ResNet, DenseNet, EfficientNet, and many others. 
These models have significantly advanced computer vision capabilities and inspired architectures in other domains such as natural language processing.

This README explains the core theory behind CNNs, focusing on convolution and pooling layers, their mathematical foundations, and practical considerations.

---

## Convolution Fundamentals

### What is a kernel/filter?

A kernel (also called a filter) is a small matrix of learnable weights that slides over the input. Each filter detects a specific pattern, such as an edge or texture. 
A layer can have many filters, producing multiple output feature maps (channels).

![kernel_filter.png](./images/kernel_filter_example.png)

### Stride and Padding

- Stride (S): How far the filter moves between positions.
  - Stride = 1 samples every location.
  - Stride > 1 downsamples more aggressively.
- Padding (P): How many zeros (or other values) are added around the input border.
  - Padding = 0 ("valid" convolution): Output is smaller.
  - Same padding: Output size matches input when stride = 1, with P roughly floor((F - 1) / 2) for odd F.
  - Zero-padding helps the kernel see borders and can influence receptive field.

### Output Size Formulas

For a 2D input with height H and width W, kernel size F, padding P, and stride S:

Output height = floor((H + 2P - F) / S) + 1

Output width = floor((W + 2P - F) / S) + 1

![output_size.png](./images/output_size_formula_diagram.png)

When using the same padding and stride = 1, the output remains the same height and width (subject to kernel size).

### Learnable Parameters and Initialization

- Kernel values are initialized to small random numbers (e.g., He or Xavier initialization).
- During training, gradient descent updates the kernel values to detect task-relevant patterns.
- Early layers often learn simple features (edges, blobs); deeper layers learn higher-level patterns.

### The 2D Convolution Equation

For an input X and a kernel K, the output Y at position (i, j) is:

Y[i, j] = sum over m=0 to F-1, n=0 to F-1 of K[m, n] * X[i + m, j + n]

Note: In practice, many frameworks implement cross-correlation (no kernel flip), which is equivalent for learning purposes.

### Concrete Example (No Padding, Stride 1)

Input 4x4, kernel 2x2, stride 1, padding 0:

Patch 1 (top-left patch of input):

[[1, 1],
 [1, 1]]

Applying kernel values (-1, 1; -1, 1) leads to result 0.

Patch 2 (slide right one):

[[1, 0],
 [1, 0]]

Result is -2.

Continue covering all patches to get a 3x3 output feature map.

### Clarifying Terminology

- Kernel and filter are interchangeable terms.
- A single filter produces one output channel (feature map).
- A layer with N filters yields N output channels.
- The filter spans all input channels (e.g., a 3x3x3 kernel for RGB image yields one feature map per filter).

---

## Pooling Basics

### What Pooling Does

Pooling reduces spatial dimensions of feature maps, lowering computation and introducing some translation invariance.

![pooling_layer.png](./images/pooling_layer_visual.png)

### Max vs Average Pooling

- Max pooling picks the maximum value in each pooling window.
- Average pooling computes the average value in each pooling window.
- Pooling layers have no learnable parameters.

### Output Size Formula for Pooling

For input size N, pooling window size F, padding P (usually 0), and stride S:

Output size = floor((N + 2P - F) / S) + 1

Example: From 4x4 input, pooling window 2, stride 2, output size is 2x2.

### Pros and Cons of Pooling

- Pros: Reduces computation, larger receptive field, some robustness to small shifts.
- Cons: Loses precise spatial detail; may hurt tasks needing exact localization.

---

## Data Flow and Filters in CNN Layers

Each layer receives a tensor with height, width, and channel dimensions. Convolutional layers apply learnable filters across spatial dimensions while spanning all input channels,
producing output feature maps equal to the number of filters. The depth dimension grows as filters increase.

Pooling layers do not use filters. Instead, they reduce spatial dimensions by summarizing local neighborhoods independently within each channel, preserving channel count.
The pooled output tensor passes directly to the next layer.

Flatten layers convert multi-dimensional feature maps into 1D vectors to feed fully connected layers, which combine all input features globally for final predictions.

---

## Layer Connectivity and Feature Preservation

Not all layers are fully connected. Only the final classification layers are fully connected layers. 
Convolutional and pooling layers connect sequentially, passing tensors from one to the next.

When filters are applied in convolutional layers, the generated set of feature maps is passed forward as input to the next layer.
Pooling layers then downsample these feature maps spatially but keep the number of channels intact.

Even though pooling reduces height and width (spatial dimensions), the features are preserved by selecting representative values (max or average) from each local neighborhood.
This spatial reduction is balanced by increased depth (number of filters) in subsequent layers, maintaining or increasing model expressiveness.

Flattening reshapes these multi-channel spatial features into a one-dimensional vector before feeding into fully connected layers, transforming spatially structured information into a global feature representation for final decisions.

---

## Shape Calculations and Layer Transitions (Example)

Input: (32, 32, 3) RGB image

1. Conv1: F=3, P=0, S=1, filters=32  
Output shape: (30, 30, 32)

2. MaxPool1: F=2, S=2, P=0  
Output shape: (15, 15, 32)

3. Conv2: F=3, P=0, S=1, filters=64  
Output shape: (13, 13, 64)

4. MaxPool2: F=2, S=2, P=0  
Output shape: (6, 6, 64)

5. Conv3: F=3, P=0, S=1, filters=64  
Output shape: (4, 4, 64)

6. Flatten → 4*4*64 = 1024 features

7. Dense Layer: output (64)

8. Output layer: output (10)

---

## TensorFlow Code for Example Model

```model = models.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax'),
])
```
backpropagation coming soon........
