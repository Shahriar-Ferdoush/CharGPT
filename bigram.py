import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200
n_embed = 32
head_count = 4


# Torch Seed for reproducibility
torch.manual_seed(1337)

# Read data from file
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create character to index mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder and Decoder to convert between string and digits
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda x: "".join([itos[i] for i in x])


# Train Test Split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data, val_data = data[:n], data[n:]


# Create a dataloader
def get_batch(split):
    # Generate small batch of data of input x and target y
    data = train_data if split == "train" else val_data

    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


class Head(nn.Module):
    """
    One head self attention block
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)  # (C, H)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # Compute the key, query and value
        k = self.key(x)  # (B, T, C) @ (C, H) --> (B, T, H)
        q = self.query(x)  # (B, T, H)

        # Compute the attention score
        wei = (
            q @ k.transpose(-2, -1) * C ** (-0.5)
        )  # (B, T, H) @ (B, H, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Compute the weighted aggregation of the value
        v = self.value(x)  # (B, T, H)
        out = wei @ v  # (B, T, T) @ (B, T, H) --> (B, T, H)

        return out


class MultiHeadAttention(nn.Module):
    """
    Multi head self attention block
    """

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class FeedForward(nn.Module):
    """
    Feed forward block
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: Communication followed by computation
    """

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_head = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = self.sa_head(x)
        x = self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Each token directly reads off the logits for the next tokens from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = MultiHeadAttention(
            head_count, n_embed // head_count
        )  # 4 communication channel
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and target are both (B, T) dimentional tensor of integers
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        position_embeddings = self.position_embedding_table(
            torch.arange(T).to(device)
        )  # (T, C)

        x = token_embeddings + position_embeddings  # (B, T, C)
        x = self.sa_head(x)
        x = self.ffwd(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = targets.view(B * T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_context = idx[:, -block_size:]  # (B, T)
            # Get the predictions
            logits, loss = self(idx_context)
            # Focus on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # After every eval_interval evaluate the loss on train and val data
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Iter: {iter}, Train Loss: {losses['train']:0.4f}, Val Loss: {losses['val']:0.4f}"
        )

    # Get a batch of data
    X, Y = get_batch("train")

    # Compute the loss
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(context, 200).squeeze().tolist()))
