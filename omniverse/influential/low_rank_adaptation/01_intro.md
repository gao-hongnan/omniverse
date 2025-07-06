---
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
```

This blog post provides a brief overview on the concept of low-rank adaptation
of large language models (LLMs) and its implementation in PyTorch.

```{tableofcontents}

```

## Citations

-   [1] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and
    W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," _arXiv
    preprint arXiv:2106.09685_, submitted Jun. 17, 2021, revised Oct. 16, 2021.
    [Online]. Available: https://arxiv.org/abs/2106.09685
-   [2] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
    Ł. Kaiser, and I. Polosukhin.
    ["Attention is all you need"](https://arxiv.org/abs/1706.03762). In Advances
    in Neural Information Processing Systems, pp. 5998–6008, 2017.

## References

-   [LoRA Fine-Tuning Tutorial - PyTorch](https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html)
-   [Low-Rank Approximation - Wikipedia](https://en.wikipedia.org/wiki/Low-rank_approximation)
-   [Rank (Linear Algebra) - Wikipedia](<https://en.wikipedia.org/wiki/Rank_(linear_algebra)>)
-   [Big Ideas in Applied Math: Low-Rank Matrices - Ethan Epperly](https://www.ethanepperly.com/index.php/2021/10/26/big-ideas-in-applied-math-low-rank-matrices/)
-   [Matrix Factorization Tutorial - PyProximal](https://pyproximal.readthedocs.io/en/stable/tutorials/matrixfactorization.html)
-   [Matrix Decomposition Series: Low-Rank Matrix Factorization - Renda Zhang](https://rendazhang.medium.com/matrix-decomposition-series-6-low-rank-matrix-factorization-5a3b96832bad)
-   [Fine-Tuning Using LoRA - GitHub](https://github.com/AviSoori1x/Tuning-the-Finetuning/blob/main/Step%202%20Fine%20tuning%20using%20%20LoRA.py)
-   [LoRA for Sequence Classification with Roberta, Llama, and Mistral - Hugging Face Blog](https://github.com/huggingface/blog/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md)
-   [Implementing LoRA from Scratch for Fine-Tuning LLMs - Daily Dose of DS](https://www.dailydoseofds.com/implementing-lora-from-scratch-for-fine-tuning-llms/)
-   [LoRA for Sequence Classification with Roberta, Llama, and Mistral - Mehdi Ir](https://github.com/mehdiir/Roberta-Llama-Mistral/blob/main/Lora-for-sequence-classification-with-Roberta-Llama-Mistral.md)
-   [LoRA on MNIST - Sunil Kumar](https://github.com/sunildkumar/lora_from_scratch/blob/main/lora_on_mnist.ipynb)
-   [Efficient Fine-Tuning with LoRA - Databricks](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms)
-   [LoRA Fine-Tuning Tutorial - PyTorch](https://pytorch.org/torchtune/stable/tutorials/lora_finetune.html)
-   [LoRA: Low-Rank Adaptation of Large Language Models - Arxiv](https://arxiv.org/pdf/2106.09685)
-   [Practical Tips for Fine-Tuning LLMs - Sebastian Raschka](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
-   [What LoRA Alpha Actually Does (In Theory) - CivitAI](https://civitai.com/articles/2125/what-lora-alpha-actually-does-in-theory)
-   [Fine-Tuning with LoRA on 8-bit - Serp AI](https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py)
-   [Eternal Question: What Rank R and Alpha to Use in LoRA - Reddit](https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/eternal_question_what_rank_r_and_alpha_to_use_in/)
-   [LLMs from Scratch - Rasbt](https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-E/01_main-chapter-code/appendix-E.ipynb)
-   [Initialize perceptron weights with zero](https://datascience.stackexchange.com/questions/26134/initialize-perceptron-weights-with-zero)
