import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel#, AdamW
import copy
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn as nn
from os.path import exists, join, isdir
import os
import json
class BaseNet():
    def __init__(self, pretrained_model_path,dataset_config,base,Author):
        raise NotImplementedError()
    def forward(self,batch):
        raise NotImplementedError()
    def train(train_dataset, valid_dataset, num_epochs, batch_size, save_path):
        raise NotImplementedError()
    def predict():
        raise NotImplementedError()


                    
                                
        
class Bert(BaseNet):
    def __init__(self, pretrained_model_path,lr=1e-5):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
        self.model = BertModel.from_pretrained(pretrained_model_path).to(self.device)
        
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 2).to(self.device)
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) +
            list(self.classifier.parameters()),
            lr=lr
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self,batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
      
        logits = self.classifier(pooled_output)
        return logits

   
    def train(self, train_dataset, valid_dataset, num_epochs=3, batch_size=8):
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

        train_loss_values = []
        valid_accuracy_values = []

        for epoch in range(num_epochs):
            self.model.train()
            self.classifier.train()
            tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            epoch_train_loss = 0.0
            nan_num=0
            for batch in tqdm_train_loader:
                labels = batch['labels'].squeeze()
                if labels.dim() == 0:
                    labels= labels.unsqueeze(0)
                labels = labels.to(self.device)
                
                
                logits=self.forward(batch)
                 
                    
                loss = self.loss_fn(logits, labels)
                
                if torch.isnan(loss):
                    loss =torch.finfo(torch.float32).max - 10
                    
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
                tqdm_train_loader.set_postfix({'loss': loss.item()})  # Update the progress bar with the current loss

            train_loss = epoch_train_loss / len(train_loader)
            train_loss_values.append(train_loss)

            self.model.eval()
            self.classifier.eval()
            total_correct=0
            total_samples=0
            for batch in valid_loader:

                labels = batch['labels'].squeeze()

                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                    
                with torch.no_grad():
                    labels = labels.to(self.device)
                    logits=self.forward(batch)
                    logits=F.softmax(logits, dim=1)            
                    predicted = torch.argmax(logits, dim=1)
                    correct=predicted==labels
                    total_correct += correct.sum().item()
                    total_samples += labels.size(0)

            valid_accuracy = total_correct / total_samples
            print("valid_Acc: ", valid_accuracy)
            valid_accuracy_values.append(valid_accuracy)


        return valid_accuracy_values,train_loss_values
   
    def predict(self, test_dataset, batch_size=8):
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        self.model.eval()
        self.classifier.eval()
        result=[]
        total_correct=0
        total_samples=0
        with torch.no_grad():
            for batch in test_loader:
                labels = batch['labels'].squeeze()
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                labels= labels.to(self.device)


                logits=self.forward(batch)
                logits=F.softmax(logits, dim=1)                
                                
                
                predicted = torch.argmax(logits, dim=1)
                result.append(predicted)
                total_correct += (predicted==labels).sum().item()
                total_samples += labels.size(0)
                
        result=torch.cat(result,dim=0)
        accuracy = total_correct / total_samples
        return result
    

    def save_model(self,path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer': self.tokenizer
        }
        torch.save(checkpoint, path)
  

    @classmethod
    def load_model(cls, pretrained_model_path,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        model = cls(pretrained_model_path=pretrained_model_path)
        new_state_dict = checkpoint['model_state_dict']
        new_state_dict_classifier = checkpoint['classifier_state_dict']

        
        model.model.load_state_dict(new_state_dict)
        model.classifier.load_state_dict(new_state_dict_classifier)
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.tokenizer = checkpoint['tokenizer']
        return model
       
                                
                          
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, text,labels):

        self.encodings =text
        self.labels =list(labels)
 
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item
 
    def __len__(self):
        return len(self.labels)
