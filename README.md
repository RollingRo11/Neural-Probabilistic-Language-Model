# Neural-Probabilistic-Language-Model
**Python implementation of "A Neural Probabilistic Language Model" (Bengio et. al.)**

(Taken from Andrej Karpathy's [makemore](https://karpathy.ai/zero-to-hero.html) series.)


## Paper implementation overview

**Embedding:**

The paper embeds words in C, an embeddings matrix. Then passes those embeddings through a function
(mult by weights and softmax, in this case), that calculates the probability distribution over possible
next characters.

$$f(i, w_{i-1}, \cdots, w_{i-n+1}) = g(i, C(w_{i-1}), \cdots, C(w_{i-n+1}))$$

```Python
self.embedding_dim = embedding_dim
self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
self.W = torch.randn((embedding_dim, self.vocab_size), requires_grad=True)
```

Finding $g(\text{our embeddings})$
```Python
logits = x_embed @ self.W
probs = F.softmax(logits, dim=1)
```
