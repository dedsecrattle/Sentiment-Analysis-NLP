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
        self.texts = None
        self.labels = None
        self.vocab = vocab
        
        with open(train_path, 'r', encoding='utf-8') as f:
            self.texts = [re.sub(r'<br\s*/>', ' ',l.rstrip().lower()) for l in f.readlines()]
        
        if label_path is not None:
            with open(label_path, 'r', encoding='utf-8') as f:
                self.labels = [l.rstrip() for l in f.readlines()]
        
        if vocab is None:
            self.vocab = {}
            idx = 1  # Reserve 0 for padding
            for text in self.texts:
                tokens = text.split()
                for i in range(len(tokens)-1):
                    bigram = f"{tokens[i]} {tokens[i+1]}"
                    if bigram not in self.vocab:
                        self.vocab[bigram] = idx
                        idx += 1
    
    def vocab_size(self):
        return len(self.vocab)
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        tokens = self.texts[i].split()
        bigrams = []
        for j in range(len(tokens)-1):
            bigram = f"{tokens[j]} {tokens[j+1]}"
            bigrams.append(self.vocab.get(bigram, 0))
        
        label = None
        if self.labels:
            label = int(self.labels[i])
        
        return bigrams, label

def collator(batch):
    texts, labels = [], []
    max_len = max([len(item[0]) for item in batch])
    
    for item in batch:
        text = torch.tensor(item[0])
        padded = F.pad(text, (0, max_len - len(text)))
        texts.append(padded)
        
        if item[1] is not None:
            labels.append(item[1])
    
    texts = torch.stack(texts)
    labels = torch.tensor(labels) if labels else None
    
    return texts, labels

class Model(nn.Module):
    def __init__(self, num_vocab):
        super().__init__()
        self.embedding_dim = 50
        self.hidden_dim = 256
        
        self.embedding = nn.Embedding(num_embeddings=num_vocab+1, 
                                     embedding_dim=self.embedding_dim, 
                                     padding_idx=0)
        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.mean(x, dim=1)
        h1 = self.relu(self.fc1(h0))
        h1 = self.dropout(h1)
        output = self.fc2(h1)
        return self.sigmoid(output).squeeze()

def train(model, dataset, batch_size, learning_rate, num_epoch, device='cpu', model_path=None):
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            texts = data[0].to(device)
            labels = torch.tensor([float(l) for l in data[1]]).to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(texts)

            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # Save the model and vocabulary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': dataset.vocab
    }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))

def test(model, dataset, thres=0.5, device='cpu'):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, collate_fn=collator, shuffle=False)
    labels = []
    with torch.no_grad():
        for data in data_loader:
            texts = data[0].to(device)
            outputs = model(texts)
            predictions = (outputs > thres).int()
            labels.extend([str(p.item()) for p in predictions])
    return labels

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
        
        batch_size = 64
        learning_rate = 0.001
        num_epochs = 5

        train(model, dataset, batch_size, learning_rate, num_epochs, device, args.model_path)
    
    if args.test:
        assert args.model_path is not None, "Please provide the model to test using --model_path argument"
        
        # Load the checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        vocab = checkpoint['vocab']
        
        # Create the test dataset object using SentDataset class
        dataset = SentDataset(args.text_path, vocab=vocab)
        
        # Initialize and load the model
        num_vocab = dataset.vocab_size()
        model = Model(num_vocab).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Run the prediction
        preds = test(model, dataset, 0.5, device)
        
        # Write the output
        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(preds))
    
    print('\n==== All done ====')

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', help='path to the text file')
    parser.add_argument('--label_path', default=None, help='path to the label file')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--model_path', required=True, help='path to the model file during testing')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)