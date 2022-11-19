# Input
source_folder = 'def_data'

# Libraries
import time
import csv
import torch
import torch.nn as nn
import pandas as pd
import os
from torchtext.data import Field, TabularDataset, Iterator
from transformers import BertTokenizer, BertForSequenceClassification

# Prepare
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained classifier
best_model = BERT().to(device)
load_checkpoint(source_folder + '/Model/model.pt', best_model)

# Fields
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]

# BERT Module
class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

# Load function
def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

# Predict function
def predict(model, pred_loader):
    y_pred = []
    y_score = []

    model.eval()
    with torch.no_grad():
        for (labels, title, text, titletext), _ in pred_loader:

            labels = labels.type(torch.LongTensor)           
            labels = labels.to(device)
            titletext = titletext.type(torch.LongTensor)  
            titletext = titletext.to(device)
            output = model(titletext, labels)
            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_score.extend(output[:,1].tolist())
    return y_pred, y_score


if __name__ == "__main__":
    # Load keyword list
    data=pd.read_csv('Keywords-Springer-83K-20210405.csv')
    kw_list = []
    for kw in data['keyword']:
        kw_list.append(kw)
    kw_list = kw_list[:83131]

    # Get start and end indices for prediction
    startind = int(input('start: '))
    endind = int(input('end: '))

    for kw in kw_list[startind:endind]:
        # print(kw)
        if os.path.exists("has_kw_sent/{}.txt".format(kw)):
            # Start counting time
            time_start = time.time()

            # Load candidate sentences
            with open("has_kw_sent/{}.txt".format(kw), 'r') as f:
                sent_list = f.readlines()

            # Reformate and save to file
            with open(source_folder+'/temp_for_pred.csv', 'w') as csvfile:
                fieldnames = ['label', 'title', 'text', 'titletext']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for sent in sent_list:
                    writer.writerow({'label': '0', 'title':kw, 'text':sent, 'titletext': '{} [DEF] {}'.format(kw, sent)})
            
            # Prepare TabularDataset for prediction
            pred = TabularDataset( path=source_folder+'/temp_for_pred.csv', format='CSV', fields=fields, skip_header=True)
            pred_iter = Iterator(pred, batch_size=16, device=device, train=False, shuffle=False, sort=False)

            # Predict
            y_pred, y_score = predict(best_model, pred_iter)

            # Save predicted result to ./predout/
            with open('predout/{}.csv'.format(kw), 'w') as csvfile:
                fieldnames = ['keyword', 'sentence', 'class', 'confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for sent, cls, conf in zip(sent_list, y_pred, y_score):
                    writer.writerow({'keyword': kw, 'sentence': sent, 'class': cls, 'confidence': conf})

            # Print time interval for prediting current keyword
            print("pred time: {} -- {} s".format(kw, time.time() - time_start))