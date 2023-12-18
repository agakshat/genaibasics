import os
import time
import urllib

import torch
import torch.nn as nn

from utils import EarlyStopping
from gpt_model import GPT, ModelConfig
from tokenizer import CharacterTokenizer

#### TODO List ####
# 2. Gradient clipping and otherwise model stability.
# 3. Better tokenizer.
# 4. Optimize multi-head implementation.

def download_dataset_if_not_exists(url):
    # Download dataset if not already downloaded.
    filepath = "input.txt"
    if not os.path.exists(filepath):
        print("Downloading Shakespeare dataset...")
        data = urllib.request.urlopen(url)
        with open(filepath, 'w') as f:
            f.write(data.read().decode('utf-8'))
    return open(filepath, 'r').read()

class DataConfig:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    train_frac = 0.9
    sentence_separator = "\n\n"
    cap_sequence_length = 100
    batch_size = 256
    pad_token = 0
    tokenizer = CharacterTokenizer

class TrainerConfig:
    num_epochs = 1000

    validate_every = 100
    save_every = 50

    early_stopping_epsilon = 1e-3
    early_stopping_patience = 5

    lr_max = 5e-4
    lr_factor = 0.2
    lr_patience = 3
    lr_threshold = 5e-3


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data = download_dataset_if_not_exists(hparams.url)
        # get tokenizer. TODO: Better tokenizer.
        self.tokenizer = hparams.tokenizer(self.data, hparams.pad_token, "<EOS>")
        # split into sentences.
        self.sentences = self.data.split(hparams.sentence_separator)
        # tokenize the sentences
        self.tokens_list = [self.tokenizer.tokenize(sentence) for sentence in self.sentences]
        # truncate the tokens to a fixed length.
        self.tokens_list = [tokens[:hparams.cap_sequence_length] for tokens in self.tokens_list]
        # pad the tokens to the max sequence length.
        self.tokens_list = [tokens + [hparams.pad_token] * (hparams.cap_sequence_length - len(tokens)) for tokens in self.tokens_list]

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        phrase = self.tokens_list[idx]
        return torch.tensor(phrase[:-1], dtype=torch.long), torch.tensor(phrase[1:], dtype=torch.long) # input, target



def main(unused_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the dataset and dataloaders.
    data_config = DataConfig()
    dataset = CustomDataset(data_config)
    vocab_size = dataset.tokenizer.vocab_size()
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [data_config.train_frac, 1.-data_config.train_frac], torch.Generator().manual_seed(7))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=data_config.batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=data_config.batch_size, shuffle=False, num_workers=0)

    # Build the model.
    model_config = ModelConfig()
    model = GPT(model_config, data_config.cap_sequence_length - 1, vocab_size)
    model.to(device)

    # Build the trainer.
    train_config = TrainerConfig()
    loss_fn = nn.CrossEntropyLoss(ignore_index=data_config.pad_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr_max)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_config.lr_factor, patience=train_config.lr_patience, threshold=train_config.lr_threshold, verbose=True)
    early_stopping = EarlyStopping(train_config.early_stopping_patience, train_config.early_stopping_epsilon)

    print(f"Model Summary: \n{model}")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    for epoch in range(train_config.num_epochs):
        print(f"Starting epoch {epoch}.")
        epoch_start = time.time()
        model.train()
        for _, batch in enumerate(train_dataloader):
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            logits = model.forward(features)
            # labels is batch_size x seq_length, logits is batch_size x seq_length x vocab_size
            loss = loss_fn(torch.reshape(logits, (-1, logits.shape[-1])), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = 0
        model.eval()
        for _, test_batch in enumerate(test_dataloader):
            features, labels = test_batch
            features, labels = features.to(device), labels.to(device)
            logits = model.forward(features)
            loss = loss_fn(torch.reshape(logits, (-1, logits.shape[-1])), labels.view(-1))
            val_loss += loss.item()
        scheduler.step(val_loss)

        print(f"Validation loss after epoch {epoch}: {val_loss}, took {time.time() - epoch_start} seconds.")
        early_stopping(val_loss)

        if early_stopping.save_model:
            # torch.save(model.state_dict(), f"experimental/akshat/learning/gpt_best_checkpoint.pt")
            print(f"Saved model checkpoint at epoch {epoch}.")

        if early_stopping.early_stop:
            print(f"Loss has converged, breaking out of training loop.")
            break

if __name__ == "__main__":
    main(None)
