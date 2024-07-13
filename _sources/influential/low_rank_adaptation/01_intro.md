---
jupytext:
    cell_metadata_filter: -all
    formats: md:myst
    text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
        jupytext_version: 1.11.5
mystnb:
    number_source_lines: true
kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Low-Rank Adaptation Of Large Language Models

[![Twitter Handle](https://img.shields.io/badge/Twitter-@gaohongnan-blue?style=social&logo=twitter)](https://twitter.com/gaohongnan)
[![LinkedIn Profile](https://img.shields.io/badge/@gaohongnan-blue?style=social&logo=linkedin)](https://linkedin.com/in/gao-hongnan)
[![GitHub Profile](https://img.shields.io/badge/GitHub-gao--hongnan-lightgrey?style=social&logo=github)](https://github.com/gao-hongnan)
![Tag](https://img.shields.io/badge/Tag-Organized_Chaos-orange)

```{contents}
:local:
```

## Table of Contents

```{tableofcontents}

```

## Citations

-   https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html
-   https://en.wikipedia.org/wiki/Low-rank_approximation
-   https://en.wikipedia.org/wiki/Rank_(linear_algebra)
-   [1] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and
    W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," _arXiv
    preprint arXiv:2106.09685_, submitted Jun. 17, 2021, revised Oct. 16, 2021.
    [Online]. Available: https://arxiv.org/abs/2106.09685
-   [2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
    Ł. Kaiser, and I. Polosukhin.
    ["Attention is all you need"](https://arxiv.org/abs/1706.03762). In Advances
    in Neural Information Processing Systems, pp. 5998–6008, 2017.
-   https://www.ethanepperly.com/index.php/2021/10/26/big-ideas-in-applied-math-low-rank-matrices/
-   https://pyproximal.readthedocs.io/en/stable/tutorials/matrixfactorization.html
-   https://rendazhang.medium.com/matrix-decomposition-series-6-low-rank-matrix-factorization-5a3b96832bad
