import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eval_iters = 200


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


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # Each token directly reads off the logits for the next tokens from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and target are both (B, T) dimentional tensor of integers
        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = targets.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
    

model = BigramLanguageModel(vocab_size).to(device)
m = model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # After every eval_interval evaluate the loss on train and val data
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Iter: {iter}, Train Loss: {losses['train']:0.4f}, Val Loss: {losses['val']:0.4f}")
    
    # Get a batch of data
    X, Y = get_batch("train")
    
    # Compute the loss
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(m.generate(context, 400).squeeze().tolist()))