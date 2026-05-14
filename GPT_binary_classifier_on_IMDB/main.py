import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import os
import torch
from datasets import load_dataset
import tiktoken # https://github.com/openai/tiktoken/tree/main
from GPTBinaryClassifier import GPTBinaryClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# hyperparameters
token_seq_len = 256
batch_size = 32
learning_rate = 2e-4
epochs = 3
evals_per_epoch = 2
eval_percentage = 0.2
n_embd = 256
n_head = 8
n_layer = 3
dropout = 0.2
# ------------
torch.manual_seed(42)

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.max_token_value # <|endoftext|>

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        tokens = self.tokenizer.encode(text, allowed_special='all')
        if len(tokens) > token_seq_len:
            tokens = tokens[:token_seq_len]
        else:
            padding_length = token_seq_len - len(tokens)
            tokens = tokens + [self.pad_token_id] * padding_length
        x_tensor = torch.tensor(tokens, dtype=torch.long)
        # torch.nn.functional.binary_cross_entropy_with_logits requires float
        y_tensor = torch.tensor([label], dtype=torch.float)
        return x_tensor, y_tensor

def print_stats(texts, tokenizer, name):
    counts = []
    for text in texts:
        counts.append(len(tokenizer.encode(text)))
    print(f'{name}: min: {min(counts)}, max: {max(counts)}', end=', ')
    print(f'median: {np.median(counts)}, average: {np.average(counts)}')

@torch.no_grad()
def estimate_loss(device, model, loader, eval_iters):
    model.eval()
    metrics = {}
    losses = torch.zeros(eval_iters)
    accuracies = torch.zeros(eval_iters)
    data_iter = iter(loader)
    for i in range(eval_iters):
        xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        losses[i] = loss.item()
        preds = (logits.view(-1) > 0.0).float()
        acc = (preds == yb.view(-1)).float().mean()
        accuracies[i] = acc.item()
    metrics['loss'] = losses.mean().item()
    metrics['acc'] = accuracies.mean().item()
    model.train()
    return metrics

def save_and_print_losses(history, training_duration):
    est_start_time = time.perf_counter()
    for loader, type in [(train_loader, 'train'), (test_loader, 'val')]:
        metrics = estimate_loss(device, model, loader, int(eval_percentage * len(loader)))
        history[f'{type}_loss'].append(metrics['loss'])
        history[f'{type}_acc'].append(metrics['acc'])
        print(f"{type}: loss: {metrics['loss']:.4f}, accuracy: {metrics[f'acc']*100:.1f}%")
    est_duration = time.perf_counter() - est_start_time
    print(f'time train: {training_duration:.2f} s; time estimate_loss(): {est_duration:.2f} s\n')

if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    tokenizer = tiktoken.get_encoding('gpt2')
    print(f'last token: ({tokenizer.max_token_value}, {tokenizer.decode([tokenizer.max_token_value])})')
    raw_datasets = load_dataset('imdb', cache_dir=data_dir)
    print('data token count stats:')
    print_stats([i['text'] for i in raw_datasets['train']], tokenizer, 'train')
    print_stats([i['text'] for i in raw_datasets['test']], tokenizer, 'test')
    train_dataset = IMDBDataset(raw_datasets['train'], tokenizer)
    test_dataset = IMDBDataset(raw_datasets['test'], tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(f'batches count: train: {len(train_loader)}, test: {len(test_loader)}')
    x_batch, y_batch = next(iter(train_loader))
    print(f'inputs: {x_batch.shape}')
    print(f'targets: {y_batch.shape}')
    print([(t, tokenizer.decode([t])) for t in x_batch[0, :20].tolist()])

    model = GPTBinaryClassifier(
        device = device,
        vocab_size = tokenizer.max_token_value + 1,
        token_seq_len = token_seq_len,
        n_embd = n_embd,
        n_head = n_head,
        n_layer = n_layer,
        dropout = dropout
    )
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    eval_interval = max(1, len(train_loader) // evals_per_epoch)
    save_and_print_losses(history, 0)
    train_start_time = time.perf_counter()
    for epoch in range(epochs):
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (i + 1) % eval_interval == 0:
                training_duration = time.perf_counter() - train_start_time
                print(f'epoch {epoch + 1}/{epochs}, iteration {i + 1}/{len(train_loader)}')
                save_and_print_losses(history, training_duration)
                train_start_time = time.perf_counter()
    
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='validation')
    plt.title('learning curve - loss function (bce)')
    plt.xlabel(f'evaluations ({evals_per_epoch} times per epoch on {eval_percentage*100}% of the whole dataset)')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train')
    plt.plot(history['val_acc'], label='validation')
    plt.title('learning curve - classification accuracy')
    plt.xlabel(f'evaluations ({evals_per_epoch} times per epoch on {eval_percentage*100}% of the whole dataset)')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()
