# Generative Pre-trained Transformers

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)

In modern years, **Natural language processing** (NLP) has been transformed by
the rise of **decoder-based transformers**, which diverge from the traditional
**encoder-decoder** framework of the pioneering
[_Attention Is All You Need_](https://arxiv.org/abs/1706.03762) paper. The
_auto-regressive_ and _masked self-attention_ mechanisms are key features that
enable these models to do text generation effectively.

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
-   [5] A. Zhang, Z. C. Lipton, M. Li, and A. J. Smola,
    ["Chapter 9. Recurrent Neural Networks"](https://d2l.ai/chapter_recurrent-neural-networks/index.html)
    in Dive into Deep Learning, Cambridge University Press, 2023.
-   [6] A. Zhang, Z. C. Lipton, M. Li, and A. J. Smola,
    ["Chapter 11. Attention Mechanisms and Transformers"](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
    in Dive into Deep Learning, Cambridge University Press, 2023.
-   [7] D. Jurafsky and J. H. Martin,
    ["Chapter 3. N-gram Language Models"](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
    in Speech and Language Processing, 3rd ed., Pearson, 2023. pp. 32-59.
-   [8] D. Jurafsky and J. H. Martin,
    ["Chapter 10. Transformers and Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf)
    in Speech and Language Processing, 3rd ed., Pearson, 2023. pp. 213-241.
-   [9] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever,
    ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).
-   [10] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever,
    ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
