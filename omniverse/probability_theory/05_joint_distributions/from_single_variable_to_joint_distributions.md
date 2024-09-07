# From Single Variable to Joint Distributions


In the simplest sense, joint distributions are extensions of the PDFs and PMFs we studied in the previous chapters. We summarize them as follows.

```{prf:remark} What is a joint distribution?
:label: remark-what-is-a-joint-distribution

Joint distributions are high-dimensional PDFs (or PMFs or CDFs).
```

What do we mean by a high-dimensional PDF? We know that a single random variable is characterized by a 1-dimensional PDF $f_X(x)$. If we have a pair of random variables, then we use a 2-dimensional function $f_{X, Y}(x, y)$, and if we have a triplet of random variables, we use a 3-dimensional function $f_{X, Y, Z}(x, y, z)$. In general, the dimensionality of the PDF grows as the number of variables:

$$
\underbrace{f_X(x)}_{\text {one variable }} \Longrightarrow \underbrace{f_{X_1, X_2}\left(x_1, x_2\right)}_{\text {two variables }} \Longrightarrow \cdots \Longrightarrow \underbrace{f_{X_1, \ldots, X_N}\left(x_1, \ldots, x_N\right)}_{N \text { variables }} .
$$

$f_{X_1, \ldots, X_N}\left(x_1, \ldots, x_N\right)$ is not a friendly notation. A more concise way to write $f_{X_1, \ldots, X_N}\left(x_1, \ldots, x_N\right)$ is to define a vector of random variables $\boldsymbol{X}=$ $\left[X_1, X_2, \ldots, X_N\right]^T$ with a vector of states $\boldsymbol{x}=\left[x_1, x_2, \ldots, x_N\right]^T$, and to define the PDF as

$$
f_{\boldsymbol{X}}(\boldsymbol{x})=f_{X_1, \ldots, X_N}\left(x_1, \ldots, x_N\right) .
$$

(imagenet)=
## ImageNet

For example, an image in ImageNet is a drawn from a latent distribution. Each sample
is $x \in \mathbb{R}^{3 \times 224 \times 224}$, where $3$ is the number of channels, $224$ is the height, and $224$ is the width.
So, if we flatten the image, we get a vector of $x \in \mathbb{R}^{150528}$, then the probability of drawing an image is
determined by the joint distribution $\pdfjoint(x_1, x_2, \ldots, x_{150528})$. For example, 
we can imagine that for the car class, the probability of drawing a Ferrari is lower than
the probability of drawing a Toyota, just because a Ferrari is more expensive than a Toyota,
and hence rarer {cite}`chan_2021`.

