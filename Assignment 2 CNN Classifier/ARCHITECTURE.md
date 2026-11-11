# Required architecture

The assignment describes input in height-width-channel order: **64 x 64 x 3**. PyTorch uses channel-first batched tensors: **(N, 3, 64, 64)**. In the table, shapes omit the batch dimension and use **C x H x W**.

| Layer | Configuration | Output | Trainable parameters |
| --- | --- | --- | ---: |
| Input | RGB, 64 x 64 | 3 x 64 x 64 | 0 |
| Conv2d | 3 input channels, 16 filters, kernel 3, stride 1, padding 1 | 16 x 64 x 64 | 448 |
| ReLU | Elementwise activation | 16 x 64 x 64 | 0 |
| MaxPool2d | Kernel 2, stride 2 | 16 x 32 x 32 | 0 |
| Conv2d | 16 input channels, 32 filters, kernel 3, stride 1, padding 1 | 32 x 32 x 32 | 4,640 |
| ReLU | Elementwise activation | 32 x 32 x 32 | 0 |
| MaxPool2d | Kernel 2, stride 2 | 32 x 16 x 16 | 0 |
| Flatten | 32 x 16 x 16 | 8,192 | 0 |
| Linear | 8,192 inputs, 100 outputs | 100 | 819,300 |
| ReLU | Elementwise activation | 100 | 0 |
| Linear | 100 inputs, 10 outputs | 10 logits | 1,010 |
| Total | | | **825,398** |

Convolution parameters, including bias: `C_out x (C_in x K_h x K_w + 1)`.

- First convolution: `16 x (3 x 3 x 3 + 1) = 448`.
- Second convolution: `32 x (16 x 3 x 3 + 1) = 4,640`.
- First fully connected layer: `8,192 x 100 + 100 = 819,300`.
- Final layer: `100 x 10 + 10 = 1,010`.
- Total: `448 + 4,640 + 819,300 + 1,010 = 825,398`.

The implemented model has exactly these layers. A softmax layer is not added to the model: training uses raw logits with CrossEntropyLoss, and inference converts those logits to probabilities.
