import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BigramModel:
    def __init__(self, words_file='names.txt', embedding_dim=27):
        self.words = open(words_file, 'r').read().splitlines()
        chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s: i+1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        self.xs, self.ys = self.build_dataset()
        self.num = len(self.xs)

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.W = torch.randn((embedding_dim, self.vocab_size), requires_grad=True)

    def build_dataset(self):
        xs, ys = [], []
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                xs.append(ix1)
                ys.append(ix2)
        return torch.tensor(xs), torch.tensor(ys)

    def train(self, num_iterations=100, learning_rate=0.5, regularization=0.01):
        for k in range(num_iterations):
            # Forward pass
            x_embed = self.embedding(self.xs)  # (num, embedding_dim)
            logits = x_embed @ self.W  # (num, vocab_size)
            probs = F.softmax(logits, dim=1)  # softmax over vocab

            # Loss: negative log likelihood
            loss = -probs[torch.arange(self.num), self.ys].log().mean()
            loss += regularization * (self.W**2).mean() + regularization * (self.embedding.weight**2).mean()

            # Backward pass
            for param in [self.W, self.embedding.weight]:
                if param.grad is not None:
                    param.grad.zero_()
            loss.backward()

            # Gradient descent
            with torch.no_grad():
                self.W -= learning_rate * self.W.grad
                self.embedding.weight -= learning_rate * self.embedding.weight.grad

            if k % 10 == 0:
                print(f"Iteration {k}: loss = {loss.item():.4f}")

    def generate_names(self, num_names=5):
        g = torch.Generator()
        generated_names = []

        for _ in range(num_names):
            out = []
            ix = 0  # Start with '.'
            while True:
                x_embed = self.embedding(torch.tensor([ix]))  # (1, embedding_dim)
                logits = x_embed @ self.W  # (1, vocab_size)
                probs = F.softmax(logits, dim=1)  # (1, vocab_size)
                ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
                if ix == 0:
                    break
                out.append(self.itos[ix])
            generated_names.append(''.join(out))

        return generated_names

    def visualize_bigram_matrix(self):
        N = torch.zeros(self.vocab_size, self.vocab_size, dtype=torch.int32)
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                N[ix1, ix2] += 1

        plt.figure(figsize=(16, 16))
        plt.imshow(N, cmap='Blues')
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                chstr = self.itos[i] + self.itos[j]
                plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
                plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
        plt.axis('off')
        plt.show()
