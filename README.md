# Neural-Probabilistic-Language-Model
**Python implementation of "A Neural Probabilistic Language Model" (Bengio et al.)**

Read the paper [here!](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (My annotated version
is also in this repo.)

(Taken from Andrej Karpathy's [makemore](https://karpathy.ai/zero-to-hero.html) series.)


## Paper implementation overview

**Embedding:**

The paper embeds words in C, an embeddings matrix. Then passes those embeddings through a function
(mult by weights and softmax, in this case), that calculates the probability distribution over possible
next characters.

$$f(i, w_{i-1}, \cdots, w_{i-n+1}) = g(i, C(w_{i-1}), \cdots, C(w_{i-n+1}))$$

Creating $C$ (embedding our data)
```Python
self.embedding_dim = embedding_dim
self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
self.W = torch.randn((embedding_dim, self.vocab_size), requires_grad=True)
```

Finding $g$ of our embeddings
```Python
logits = x_embed @ self.W
probs = F.softmax(logits, dim=1)
```

**Sampling the next character:**

In the paper, this sampling is simply described as

$$\text{i-th output} = P(w_t = i \mid \textit{context})$$

In the code, just used `torch.multinomial` to sample what would be our $i$'th output.

```Python
x_embed = self.embedding(torch.tensor([ix]))  # (1, embedding_dim)
logits = x_embed @ self.W  # (1, vocab_size)
probs = F.softmax(logits, dim=1)  # (1, vocab_size)
ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
```

**Loss:**
As described in the paper, the loss is calculated as:

$$L = \frac{1}{T} \sum_t \log f(w_t, w_{t-1}, \cdots, w_{t-n+1}; \theta) + R(\theta),$$

```Python
loss = -probs[torch.arange(self.num), self.ys].log().mean()
loss += regularization * (self.W**2).mean() + regularization * (self.embedding.weight**2).mean()
```
(With regularization baked in to account for softmax bounds)
