# Introduction

**Transformers** have revolutionized the field of **natural language
processing** (NLP), offering a distinct approach to handling sequential data.
The groundbreaking paper,
[_"Attention Is All You Need"_](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>),
introduced in 2017 by Vaswani et al., marked a paradigm shift away from
traditional **recurrent neural network**
([RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network)) models, like
[**LSTMs**](https://en.wikipedia.org/wiki/Long_short-term_memory) and
[**GRUs**](https://en.wikipedia.org/wiki/Gated_recurrent_unit), which were the
standard for tasks like **machine translation**, **text summarization**, and
more.

At the heart of the transformer architecture is the **self-attention
mechanism**, a novel component that allows the model to weigh the significance
of different parts of the input data independently of their sequential order.
This is a fundamental departure from the RNN-based models, where the processing
of input sequences is inherently sequential, often leading to challenges like
[**vanishing gradients**](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
and difficulty in capturing long-range dependencies.

The transformer model eschews recurrence entirely, relying instead on a fully
**attention-based** approach, coupled with **feed-forward networks**. This
design not only addresses the limitations of RNNs but also allows for
significantly more parallelization during training, reducing training times and
enabling the model to scale in a way that RNNs could not.

Furthermore, the **encoder-decoder** structure of the transformer provides a
versatile framework for various NLP tasks. The encoder maps an input sequence of
symbol representations to a sequence of continuous representations, which the
decoder then converts into an output sequence. This process is facilitated by
the attention mechanism, which dynamically focuses on different parts of the
input sequence, providing a more context-aware representation.

The implications of the transformer model have been profound, leading to the
development of highly influential models like **BERT**
([_Bidirectional Encoder Representations from Transformers_](<https://en.wikipedia.org/wiki/BERT_(language_model)>)),
**GPT**
([_Generative Pre-trained Transformer_](https://en.wikipedia.org/wiki/GPT-3)),
and others. These models have set new benchmarks across a variety of NLP tasks,
showcasing the versatility and power of the transformer architecture.

In this blog post, we delve into the intricacies of the transformer model,
exploring how it functions, its advantages over previous models, and its vast
array of applications in the modern NLP landscape.

## References and Further Readings

1. Stanford CS224N Course: [link](https://web.stanford.edu/class/cs224n/)
2. Aman AI Transformers Primer:
   [link](https://aman.ai/primers/ai/transformers/#transformer-core)
3. Implementing a Transformer from Scratch in PyTorch:
   [link](https://www.lesswrong.com/posts/2kyzD5NddfZZ8iuA7/implementing-a-transformer-from-scratch-in-pytorch-a-write)
4. Illustrated Transformer by Jay Alammar:
   [link](http://jalammar.github.io/illustrated-transformer/)
5. Mislav Juric's Transformer from Scratch:
   [link](https://github.com/MislavJuric/transformer-from-scratch/blob/main/layers/MultiHeadAttention.py)
6. Peter Bloem's Blog on Transformers:
   [link](https://peterbloem.nl/blog/transformers)
7. Transformer Family Explained by Lilian Weng:
   [link](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
8. D2L AI Book - Multihead Attention:
   [link](https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html)
9. NLP Course at NTU:
   [link](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/self_v7.pdf)
10. Attention Is All You Need (Original Transformer Paper):
    [link](https://arxiv.org/pdf/1706.03762.pdf)
11. Harvard NLP - Attention in Transformers:
    [link](https://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking)
12. Annotated Transformer:
    [link](http://nlp.seas.harvard.edu/annotated-transformer/)
13. LabML AI - MultiHead Attention:
    [link](https://nn.labml.ai/transformers/mha.html)
14. NTU Speech and Language Processing Course:
    [link](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)
15. Google Colab - Self-Attention Example:
    [link](https://colab.research.google.com/drive/1u-610KA-urqfJjDH5O0pecwfP--V9DQs?usp=sharing#scrollTo=iXZ5B0EKJGs8)
16. Self-Attention from Scratch by Sebastian Raschka:
    [link](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
17. UvA DL Course - Transformers and MHAttention:
    [link](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)
18. Simple Attention-based Text Prediction Model:
    [link](https://datascience.stackexchange.com/questions/94205/a-simple-attention-based-text-prediction-model-from-scratch-using-pytorch)
19. The AI Summer - Self-Attention Explanation:
    [link](https://theaisummer.com/self-attention/#how-multi-head-attention-works-in-detail)
