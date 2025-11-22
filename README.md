# CNN-from-scratch-numpy-only-

## Convolutional Neural Networks (CNNs) Theory and Basics

### Introduction

Convolutional Neural Networks (CNNs) changed how machines understand visual data, enabling breakthroughs in image recognition and video analysis. CNNs were first introduced in the late 1980s (LeNet, 1989), but gained widespread practical success with the ImageNet competition in 2012.

**Why not just use Dense Networks?**
Before CNNs, dense (fully connected) neural networks were the primary approach, but they had critical limitations for images:
- **Loss of Locality:** Dense networks treat each pixel independently, flattening spatial structure.
- **Parameter Explosion:** Connecting every pixel to every neuron leads to overfitting and inefficient training.
- **No Spatial Hierarchy:** Dense layers do not naturally exploit local patterns like edges or shapes.

CNNs solved these problems by introducing **convolutional layers** with learnable filters. Weight sharing and local connectivity drastically reduce the number of parameters, preserving spatial relationships and enabling efficient training on large datasets.

This README explains the core theory behind CNNs, focusing on convolution and pooling layers and their mathematical foundations.

---

## Convolution Fundamentals

### What is a kernel/filter?

A kernel (also called a filter) is a small matrix of learnable weights that slides over the input. Each filter detects a specific pattern, such as an edge or texture. A layer can have many filters, producing multiple output feature maps (channels).

![kernel_filter.png](./images/kernel_filter.png)

### Clarifying Terminology
- **Kernel vs Filter:** Often used interchangeably. Technically, a "kernel" is the 2D matrix, and a "filter" is the collection of kernels for all input channels (e.g., a 3x3x3 filter).
- **Output Channels:** A single filter produces one output channel (feature map). If a layer has $N$ filters, it will produce $N$ output channels.
- **Color (RGB):** The filter depth always matches the input depth. For an RGB image (3 channels), the filter is $3 \times 3 \times 3$.

### Stride and Padding

- **Stride (S):** How far the filter moves between positions.
  - Stride = 1 samples every location.
  - Stride > 1 downsamples the output.
- **Padding (P):** Adding zeros around the input border.
  - **Valid (P=0):** Output is smaller than input.
  - **Same:** Output size matches input size (requires specific padding).

### Output Size Formulas

For a 2D input with height $H$ and width $W$, kernel size $F$, padding $P$, and stride $S$:

$$H_{out} = \lfloor \frac{H + 2P - F}{S} \rfloor + 1$$
$$W_{out} = \lfloor \frac{W + 2P - F}{S} \rfloor + 1$$

![output_size.png](./images/output_size.png)

### The 2D Convolution Equation

For an input $X$ and a kernel $K$, the output $Y$ at position $(i, j)$ is:

$$Y[i, j] = \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} K[m, n] \cdot X[i + m, j + n]$$

*(Note: In practice, deep learning frameworks implement cross-correlation, which is equivalent for learning purposes.)*

---

### Concrete Example: The Sliding Window

**Settings:**
- **Input:** 4x4 Matrix
- **Kernel:** 2x2 Matrix
- **Stride:** 1
- **Padding:** 0

**The Input Matrix:**
```text
[[1, 2, 3, 0],
 [0, 1, 2, 3],
 [3, 0, 1, 2],
 [2, 3, 0, 1]]
```

**The Kernel (Filter):**
```text
[[ 1,  0],
 [-1,  1]]
```

#### Step 1: Top-Left Patch
We overlay the kernel on the first 2x2 patch of the input.

**Input Patch:**
```text
[[1, 2],
 [0, 1]]
```
**Calculation:**
$(1 \cdot 1) + (2 \cdot 0) + (0 \cdot -1) + (1 \cdot 1)$
$= 1 + 0 + 0 + 1$
$= 2$

**Output Map:** `[[2, ?, ?], ...]`

#### Step 2: Slide Right (Stride 1)
Move the kernel one pixel to the right.

**Input Patch:**
```text
[[2, 3],
 [1, 2]]
```
**Calculation:**
$(2 \cdot 1) + (3 \cdot 0) + (1 \cdot -1) + (2 \cdot 1)$
$= 2 + 0 - 1 + 2$
$= 3$

**Output Map:** `[[2, 3, ?], ...]`

#### Step 3: Slide Right Again
Move the kernel one more pixel to the right.

**Input Patch:**
```text
[[3, 0],
 [2, 3]]
```
**Calculation:**
$(3 \cdot 1) + (0 \cdot 0) + (2 \cdot -1) + (3 \cdot 1)$
$= 3 + 0 - 2 + 3$
$= 4$

**Output Map:** `[[2, 3, 4], ...]`

*...and so on, sliding down to the next row once the width is covered.*

---

## Pooling Basics

### What Pooling Does

Pooling reduces spatial dimensions of feature maps, lowering computation and introducing translation invariance.

![pooling_layer.png](./images/pooling_layer.png)

### Max vs Average Pooling

- **Max pooling:** Picks the maximum value in the window.
- **Average pooling:** Computes the average value.
- *Note: Pooling layers have zero learnable parameters.*

### Pros and Cons of Pooling
- **Pros:** Reduces computation (fewer pixels), creates larger receptive fields, adds robustness to shifts.
- **Cons:** Loses precise spatial detail and pixel-perfect localization.

### Output Size Formula

$$\text{Output Size} = \lfloor \frac{N + 2P - F}{S} \rfloor + 1$$

---

## Layer Connectivity and Data Flow

1.  **Convolution Layers:** Apply learnable filters. Input depth = Input Channels. Output depth = Number of Filters.
2.  **Pooling Layers:** Downsample spatially. Depth remains constant.
3.  **Flatten:** Reshapes the 3D tensor (Height, Width, Depth) into a 1D vector.
4.  **Dense Layers:** Fully connected layers for final classification.

---

## Shape Calculations (Example Architecture)

Input: (32, 32, 3) RGB image

1.  **Conv1:** F=3, P=0, S=1, filters=32
    - Output: `(30, 30, 32)`
2.  **MaxPool1:** F=2, S=2
    - Output: `(15, 15, 32)`
3.  **Conv2:** F=3, P=0, S=1, filters=64
    - Output: `(13, 13, 64)`
4.  **MaxPool2:** F=2, S=2
    - Output: `(6, 6, 64)`
5.  **Conv3:** F=3, P=0, S=1, filters=64
    - Output: `(4, 4, 64)`
6.  **Flatten:** $4 \times 4 \times 64$
    - Output: `1024` features
7.  **Dense:** Output 10

---

## Filter Selection and Redundancy

Why do we choose 32 or 64 filters?

- **Spatial Reduction vs. Depth Increase:** As the image passes through the network, $H$ and $W$ shrink (via Pooling), but we typically increase the number of filters. Early layers (16-32 filters) capture simple lines. Deep layers (64-128 filters) capture complex object parts.
- **Small Input Redundancy:** Applying 32 filters to a tiny **raw input** (e.g., 4x4 pixels) is inefficient because the input lacks enough information (16 pixels) to generate 32 unique, meaningful features. This leads to duplicate filters or noise.

---

## TensorFlow Reference Code

```python
model = models.Sequential([
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
