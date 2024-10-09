import os
import re
import sys
import string
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)


class SentDataset(Dataset):
    def __init__(self, train_path, label_path=None, vocab=None):
        self.word2idx = vocab if vocab else {}
        self.texts = []
        self.labels = []
        self.bigram2idx = {}
        
        # Read texts and create bigrams
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
                
                # Build bigram vocabulary if not provided
                if not vocab:
                    for bigram in bigrams:
                        if bigram not in self.bigram2idx:
                            self.bigram2idx[bigram] = len(self.bigram2idx)
                
                # Convert bigrams to indices
                bigram_indices = [self.bigram2idx.get(bigram, self.bigram2idx.get('<UNK>', 0)) 
                                 for bigram in bigrams]
                self.texts.append(bigram_indices)
        
        # Load labels if provided
        if label_path:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.labels.append(int(line.strip()))

    def vocab_size(self):
        return len(self.bigram2idx)
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = torch.tensor(self.texts[i], dtype=torch.long)
        if self.labels:
            label = torch.tensor(self.labels[i], dtype=torch.float)
            return text, label
        return text, torch.tensor(0, dtype=torch.float)


class Model(nn.Module):
    def __init__(self, num_vocab, embedding_dim=100, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_vocab, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Get embeddings for each bigram in the sequence
        embedded = self.embedding(x)  # shape: batch_size x seq_len x embedding_dim
        
        # Average the bigram embeddings
        h0 = torch.mean(embedded, dim=1)  # shape: batch_size x embedding_dim
        
        # Apply first linear layer with ReLU activation
        h1 = F.relu(self.fc1(h0))
        
        # Apply dropout
        h1 = self.dropout(h1)
        
        # Apply second linear layer and sigmoid
        out = torch.sigmoid(self.fc2(h1)).squeeze(1)
        
        return out


def collator(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad sequences
    max_len = max([len(text) for text in texts])
    padded_texts = [F.pad(text, (0, max_len - len(text))) for text in texts]
    
    return torch.stack(padded_texts), torch.stack(labels)


def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            texts = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            outputs = model(texts)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()

            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bigram_vocab': dataset.bigram2idx,
    }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))


def test(model, dataset, thres=0.5, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    predictions = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts)
            predictions.extend((outputs > thres).int().tolist())
    
    return [str(pred) for pred in predictions]


def main(args):
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    
    assert args.train or args.test, "Please specify --train or --test"
    if args.train:
        assert args.label_path is not None, "Please provide the labels for training using --label_path argument"
        dataset = SentDataset(args.text_path, args.label_path)
        num_vocab = dataset.vocab_size()
        model = Model(num_vocab).to(device)
        
        batch_size = 32
        learning_rate = 0.001
        num_epochs = 5

        train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        checkpoint = torch.load(args.model_path)
        
        dataset = SentDataset(args.text_path, vocab=checkpoint['bigram_vocab'])
        
        model = Model(dataset.vocab_size()).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        preds = test(model, dataset, 0.5, device)

        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(preds))
    print('\n==== All done ====')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the model file')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)