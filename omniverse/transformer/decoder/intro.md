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
