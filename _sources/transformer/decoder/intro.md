# Generative Pre-trained Transformers

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)

The realm of **natural language processing** (NLP) has been transformed by the
advent of **decoder-based transformers**, a cornerstone in the latest
breakthroughs in AI language models. These architectures, which diverge from the
traditional **encoder-decoder** framework of the pioneering
[_Attention Is All You Need_](https://arxiv.org/abs/1706.03762) paper, are
exclusively focused on the generative aspects of language processing. This
singular focus has propelled them to the forefront of tasks like **text
completion**, **creative writing**, **language synthesis**, and more critically,
in developing advanced conversational AI.

At the core of these models is the **masked self-attention** mechanism, a unique
feature that enables the model to generate language by considering only the
previous context, a technique known as unidirectional or causal attention. This
approach is radically different from the bidirectional context seen in full
transformers, making it particularly suited for sequential data generation,
where the future context is unknown.

To this end, we would discuss the architecture of decoder-based transformers,
through the family of Generative Pre-trained Transformers (GPT). We would review
mainly the GPT-2 paper, discuss some concepts, and then implement from scratch a
simplified version of the GPT-2 model.

## Table of Contents

```{tableofcontents}

```

## Citations

-   [1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
    Ł. Kaiser, and I. Polosukhin.
    ["Attention is all you need"](https://arxiv.org/abs/1706.03762). In Advances
    in Neural Information Processing Systems, pp. 5998–6008, 2017.
-   [2] I. Loshchilov and F. Hutter,
    ["Decoupled weight decay regularization"](https://arxiv.org/abs/1711.05101),
    arXiv preprint arXiv:1711.05101, [Submitted on 14 Nov 2017 (v1), last
    revised 4 Jan 2019 (this version, v3)].
-   [3] D. P. Kingma and J. Ba,
    ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980),
    arXiv preprint arXiv:1412.6980, [Submitted on 22 Dec 2014 (v1), last revised
    30 Jan 2017 (this version, v9)].
-   [4] L. Liu, H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and J. Han,
    ["On the Variance of the Adaptive Learning Rate and Beyond"](https://arxiv.org/abs/1908.03265),
    arXiv preprint arXiv:1908.03265, [Submitted on 8 Aug 2019 (v1), last revised
    26 Oct 2021 (this version, v4)].
-   A. Zhang, Z. C. Lipton, M. Li, and A. J. Smola,
    ["Chapter 9: Recurrent Neural Networks"](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
    in Dive into Deep Learning, Cambridge University Press, 2023.
-   A. Zhang, Z. C. Lipton, M. Li, and A. J. Smola,
    ["Chapter 11. Attention Mechanisms and Transformers"](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
    in Dive into Deep Learning, Cambridge University Press, 2023.
-   D. Jurafsky and J. H. Martin,
    ["Chapter 3. N-gram Language Models"](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
    in Speech and Language Processing, 3rd ed., Pearson, 2023. pp. 32-59.
-   D. Jurafsky and J. H. Martin,
    ["Chapter 10. Transformers and Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf)
    in Speech and Language Processing, 3rd ed., Pearson, 2023. pp. 213-241.
