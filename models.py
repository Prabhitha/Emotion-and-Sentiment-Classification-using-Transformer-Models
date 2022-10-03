# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:19:30 2020

@author: Jiang Yuxin
"""

import torch
from torch import nn
from transformers import BertTokenizer, TFBertModel
from transformers import (
    BertForSequenceClassification, 
    AlbertForSequenceClassification, 
    XLNetForSequenceClassification, 
    RobertaForSequenceClassification, 
    DistilBertForSequenceClassification,
    GPT2ForSequenceClassification,
    ElectraForSequenceClassification,
    DebertaForSequenceClassification,
    AutoTokenizer,
    XLNetTokenizer,
    RobertaTokenizer,
    DistilBertTokenizer,
    AlbertTokenizer,
    BertTokenizer,
    ElectraTokenizer,
    DebertaTokenizer,
    GPT2Tokenizer
)


class AlbertModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(AlbertModel, self).__init__()
        self.albert = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels = 28)
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
        #self.albert = AlbertForSequenceClassification.from_pretrained('albert-xxlarge-v2', num_labels = 6)
        #self.tokenizer = AutoTokenizer.from_pretrained('albert-xxlarge-v2', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.albert.parameters():
            param.requires_grad = True  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.albert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                            token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        
     
        
class BertModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = 28)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        #self.bert = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-SST-2',num_labels = 2)
        #self.tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-SST-2', do_lower_case=True)
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.bert = TFBertModel.from_pretrained("bert-base-uncased")
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        
        
        
class RobertModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(RobertModel, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels = 28)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
        
    
    
class XlnetModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(XlnetModel, self).__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 28)
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.xlnet.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

class distilBertModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(distilBertModel, self).__init__()
        self.distilBert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 28)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.distilBert.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.distilBert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

class GPT2Model(nn.Module):
    def __init__(self, requires_grad = True):
        super(GPT2Model, self).__init__()
        self.GPT2 = GPT2ForSequenceClassification.from_pretrained('microsoft/DialogRPT-updown', num_labels = 28,ignore_mismatched_sizes=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialogRPT-updown', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.GPT2.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.GPT2(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

class ElectraModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(ElectraModel, self).__init__()
        self.Electra = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator', num_labels = 28)
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.Electra.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.Electra(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

class DebertaModel(nn.Module):
    def __init__(self, requires_grad = True):
        super(DebertaModel, self).__init__()
        self.Deberta = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base', num_labels = 28)
        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = torch.device("cuda")
        for param in self.Deberta.parameters():
            param.requires_grad = requires_grad  # Each parameter requires gradient

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.Deberta(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
