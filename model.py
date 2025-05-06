import torch
import torch.nn.functional as F

class BigramModel:
    def __init__(self, words_file='names.txt'):
        self.words = open(words_file, 'r').read().splitlines()
        
        chars = sorted(list(set(''.join(self.words))))
        self.stoi = {s: i+1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0  # Special start/end token
        self.itos = {i: s for s, i in self.stoi.items()}
        
        self.xs, self.ys = self.build_dataset()
        self.num = len(self.xs)
        
        self.W = None
        
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
    
    def build_bigram_matrix(self):
        N = torch.zeros(27, 27, dtype=torch.int32)
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]
                N[ix1, ix2] += 1
        return N
    
    def train(self, num_iterations=100, learning_rate=50, regularization=0.01):
        g = torch.Generator()
        self.W = torch.randn((27, 27), generator=g, requires_grad=True)
        
        # Gradient descent
        for k in range(num_iterations):
            xenc = F.one_hot(self.xs, num_classes=27).float()  # one-hot encoding
            logits = xenc @ self.W  # predict log-counts
            counts = logits.exp()  # counts, equivalent to N
            probs = counts / counts.sum(1, keepdim=True)  # probabilities for next character
            
            loss = -probs[torch.arange(self.num), self.ys].log().mean() + regularization * (self.W**2).mean()
            
            self.W.grad = None  
            loss.backward()
            
            self.W.data += -learning_rate * self.W.grad
            
            if k % 10 == 0:
                print(f'iteration {k} | loss = {loss.item():.4f}')
    
    def generate_names(self, num_names=5):
        g = torch.Generator()
        generated_names = []
        
        for _ in range(num_names):
            out = []
            ix = 0  
            
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
                
                logits = xenc @ self.W  # predict log-counts
                counts = logits.exp()  # softmax / squishification
                p = counts / counts.sum(1, keepdim=True)  # probabilities for next character
                
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                
                if ix == 0:
                    break
                    
                out.append(self.itos[ix])
                
            generated_names.append(''.join(out))
            
        return generated_names

