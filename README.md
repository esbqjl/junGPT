# junGPT


This project is a PyTorch-based re-implementation of the GPT model, encompassing both training and inference processes. Titled junGPT, it serves as a personal exploration into the mechanics of the GPT architecture.

In essence, the model processes a sequence of indices through a Transformer network, outputting a probability distribution for the subsequent index. The core challenge lies in optimizing the batching techniques, both across examples and sequence lengths, to enhance efficiency.

While this project may not be groundbreaking, it offers a foundational demonstration for beginners aiming to understand the basics of GPT models. Much of its structure borrows from mingpt (https://github.com/karpathy/minGPT.git), which itself is an excellent starter demonstration for constructing a GPT model.

Contrasting with mingpt, junGPT primarily leverages transformer concepts to build upon the GPT model, focusing on expanding the decoder-based architecture rather than integrating all components within a singular class.

The implementation is divided across three main files:

gpt/model.py: Defines the actual Transformer model.
gpt/utils.py: Provides utilities for sampling data from a trained GPT model, with methods adopted from minGPT.
gpt/trainer.py: Contains generic PyTorch boilerplate code for training the model, along with a demo that employs the library to train a novel text generator.
Additionally, the generate.ipynb notebook demonstrates how to load a locally trained (non-official) GPT-2 model and generate text based on a given prompt.

Due to time and efficiency constraints, extensive training was not feasible; however, the provided demo includes a test using text from one of JinYong's novels—a popular method for evaluating GPT models in China.

I hope you find this implementation engaging and educational.

### Library Installation

If you want to `import mingpt` into your project:

```
git clone https://github.com/karpathy/minGPT.git
cd minGPT
pip install -e .
```

### Usage

just like what I have done in my demo.ipynb file, let's say if you want to train your own text generator.
1. Setup your own file segment method or the way you deal with your corpus, there are various ways to do so, so I didn't put this method to train.py. In the demo, I use one of the minGpt demo to do so.
2. Setup the model config and train config to initialize the model and trainer, you can check out the parameters to see what you need to change based on your demand
3. Train and enjoy it.

References
Code:
karpathy's mingpt. https://github.com/karpathy/minGPT.git

huggingface/transformers https://github.com/huggingface/transformers.git
