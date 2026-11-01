# Assignment 3: Building Blocks of GAN

## Question 1

For dilation 1, the output size along either spatial dimension is

\[
O=(I-1)S-2P+K+OP.
\]

Substituting the given values:

\[
O=(8-1)\times2-2\times1+4+1=14-2+4+1=17.
\]

The height and width use the same parameters, so the output is **17 x 17**.

## Question 2

Holding the input and all other parameters from Question 1 fixed:

\[
O_{S=3}=(8-1)\times3-2\times1+4+1=21-2+4+1=24.
\]

The output changes from **17 x 17 to 24 x 24**. In general,

\[
\Delta O=(I-1)(S_{new}-S_{old})=I-1
\]

when the stride increases from 2 to 3. Here, each spatial dimension grows by 7.
The size does not simply scale by 3/2 because the padding and kernel terms
remain fixed.

## Question 3

Assuming dilation 1:

\[
H_{out}=(H_{in}-1)S_H-2P_H+K_H+OP_H,
\]

\[
W_{out}=(W_{in}-1)S_W-2P_W+K_W+OP_W.
\]

For equal parameters in both dimensions, write

\[
\boxed{O=(I-1)S-2P+K+OP}.
\]

For completeness, with dilation D, the per-dimension formula is

\[
O=(I-1)S-2P+D(K-1)+OP+1.
\]

## Question 4

Assume zero padding, zero output padding, and dilation 1. Then

\[
32=(16-1)S+K=15S+K.
\]

Choosing **kernel size 2 x 2 and stride 2** gives

\[
O=(16-1)\times2+2=30+2=32.
\]

Thus, `ConvTranspose2d(in_channels, out_channels, kernel_size=2,
stride=2, padding=0, output_padding=0)` is one valid configuration.

## Question 5

The batch is [6, 8, 10, 6]. The batch mean is

\[
\mu_B=\frac{6+8+10+6}{4}=7.5.
\]

BatchNorm uses the batch variance with denominator 4 for this training-time
normalization:

\[
\sigma_B^2=\frac{(6-7.5)^2+(8-7.5)^2+(10-7.5)^2+(6-7.5)^2}{4}
=\frac{2.25+0.25+6.25+2.25}{4}=2.75.
\]

With no affine transformation, each normalized value is

\[
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}.
\]

Ignoring epsilon for the requested zero-mean, unit-variance calculation,

\[
\hat{x}=\frac{[-1.5,0.5,2.5,-1.5]}{\sqrt{2.75}}
=\frac{[-3,1,5,-3]}{\sqrt{11}}
\]

\[
\boxed{\hat{x}\approx[-0.9045,\ 0.3015,\ 1.5076,\ -0.9045]}.
\]

Check: the exact values sum to zero, and their mean square is
(9 + 1 + 25 + 9)/(11 x 4) = 1. A small positive epsilon makes the variance
slightly below 1. Epsilon is not specified in the question.

## Question 6

\[
\operatorname{ReLU}(x)=\max(0,x)=
\begin{cases}x,&x\geq0,\\0,&x<0.\end{cases}
\]

\[
\operatorname{LeakyReLU}(x)=
\begin{cases}x,&x\geq0,\\\alpha x,&x<0,\end{cases}
\quad 0<\alpha<1.
\]

The key difference is the negative-region slope: 0 for ReLU versus alpha
for LeakyReLU. In this assignment alpha = 0.2; for example, an input of -5
produces 0 with ReLU and -1 with LeakyReLU.

## Question 7

For a negative preactivation, ReLU has zero derivative, so that example
provides no gradient through the activation. If a unit remains negative for
all training examples, it can stop learning, often called a dying ReLU.
LeakyReLU has derivative alpha in the negative region, allowing gradients
to pass and reducing this risk. This can help optimization in a deep
network or GAN discriminator. It does not guarantee better performance
or eliminate every source of vanishing gradients; alpha is a design choice.

## References

- Assignment3.pdf, supplied assignment instructions and questions.
- PyTorch, ConvTranspose2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
- PyTorch, BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
- PyTorch, ReLU and LeakyReLU API documentation.
