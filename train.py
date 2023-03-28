import torch
import torch.nn as nn 
from src import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

import logging

# set up a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a console handler and set its level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a formatter and add it to the console handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add the console handler to the logger
logger.addHandler(console_handler)

def train_epoch(model, dataloader, optimizer, device, e, epoch):
    model.train()
    whole_loss = list()
    for batch in tqdm(dataloader):
        input = batch["input"].to(device)
        label = batch["label"].to(device)
        _, loss = model(
            input_ids=input,
            label=label
        )
        whole_loss.append(loss.cpu().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logger.info('Epoch [{}/{}], Train Loss: {:.4f}'.format(e+1, epoch, np.mean(whole_loss)))


def validate_epoch(model, dataloader, device, e, epoch):
    whole_loss = list()
    whole_acc = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input = batch["input"].to(device)
            label = batch["label"].to(device)
            logits, loss = model(
                input_ids=input,
                label=label
            )
            pred = torch.argmax(logits, dim=1)
            whole_acc += torch.sum(pred == label).cpu().detach().item()
            whole_loss.append(loss.cpu().item())
    logger.info('Epoch [{}/{}], Valid Loss: {:.4f} Valid acc: {:.4f}'.format(e+1, epoch, np.mean(whole_loss), whole_acc / dataloader.batch_size / len(dataloader)))


def main():
    tokenizer, embedding, id2word, word2id = prepare_tokenizer_embedding('glove/glove.6B.100d.txt')
    train = prepare_datasets('train')
    dev = prepare_datasets('dev')
    test = prepare_datasets('test')
    
    def collate_fn(example): # examples are list of sentences with label
        label = list()
        ex = list()
        for e in example:
            ex.append(torch.as_tensor(tokenizer.tokenize_sentence(e["input"]), dtype=torch.long))
            label.append(e["label"])
        return {
            "input": pad_sequence(ex, batch_first=True),
            "label": torch.as_tensor(label, dtype=torch.long)
        }
        
    train_loader = DataLoader(
        train,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    dev_loader = DataLoader(
        dev,
        batch_size=32,
        collate_fn=collate_fn
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RNNModel(
        hidden_dim=256,
        hidden_layer=4,
        embedding_layer=embedding,
        sequence_length=100,
        vocab_size=len(id2word) + 1,
        embedding_dim=100,
        dropout=0.2,
        bidirectional=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=[0.9, 0.999],
        weight_decay=1e-6
    )
    
    epoch = 40
    
    # for e in range(epoch):
    #     whole_loss = list()
    #     for batch in tqdm(train_loader):
    #         input = batch["input"].to(device)
    #         label = batch["label"].to(device)
    #         _, loss = model(
    #             input_ids=input,
    #             label=label
    #         )
    #         whole_loss.append(loss.cpu().item())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     logger.info('Epoch [{}/{}], Train Loss: {:.4f}'.format(e+1, epoch, np.mean(whole_loss)))
    for e in range(epoch):
        train_epoch(model, train_loader, optimizer, device, e, epoch)
        validate_epoch(model, dev_loader, device, e, epoch)
    
    
if __name__ == '__main__':
    main()