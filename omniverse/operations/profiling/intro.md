# Profiling

## Table of Contents

```{tableofcontents}

```

In this series, we look at how to profile deep learning code with PyTorch. These
profiling tools are a means for us to pry open the black box of CUDA mechanics
and understanding how to interpret the performance metrics that are output by
these tools will be sufficient for most deep learning practitioners. We will
leave the hardcore CUDA optimization to the experts!

Of course, we will touch on PyTorch's own
[profiler](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/),
while our posts are all very basic and introductory, we would still want
advanced users to know _how_ to incorporate these tools into their workflows
(especially training deep learning models). What better to just read
[torchtune](https://github.com/pytorch/torchtune), another native library
developed by the PyTorch team for training/finetuning LLMs. Inside you can see
how they do profiling in practice for training.

## References and Further Readings

-   [PyTorch Benchmarking Tutorial](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)
-   [CUDA Mode Notes - Lecture 001 by Christian J. Mills](https://christianjmills.com/posts/cuda-mode-notes/lecture-001/)
-   [GitHub Repository: Spring2024 Assignment2 Systems by Marcel Roed](https://github.com/marcelroed/spring2024-assignment2-systems/tree/master)
-   [PyTorch Profiler Recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
-   [PyTorch Torchtune GitHub Repository](https://github.com/pytorch/torchtune)
-   [Understanding GPU Memory - Part 1](https://pytorch.org/blog/understanding-gpu-memory-1/)
-   [Understanding GPU Memory - Part 2](https://pytorch.org/blog/understanding-gpu-memory-2/)
