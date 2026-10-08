#!/usr/bin/env python3

"""
The main script for running experiments
"""
##=============================
#The goal for this is to learn how language models and multimodal models work by using two datasets: the Jokes dataset and the Meme dataset. 
# In the first part, I trained a Transformer-based language model from scratch, start to finish.
#In the second part, I built three different models to understand how memes can be classified using images, text, or both together.
##=============================
    

import pandas as pd
import numpy as np
from transformers import (
    pipeline,
    set_seed,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from datasets import load_dataset
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import TrainerCallback
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
import torchvision.transforms as T
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from sklearn.metrics import recall_score


dataset_directory = "Dataset"  # Path to the dataset directory
SEED = 7563  # RANDOM seed as the Last four digits of my student ID

#load dataset jokes
def get_dataset_jokes():  # Function to load jokes from the dataset
    path = f"Dataset/jokes.csv"
    df = pd.read_csv(path)
    jokes = df["Joke"].tolist()
    return jokes

#load meme images
def get_memeImages(): 
    excel_path = "Dataset/meme/labels.xlsx"
    image_root = "Dataset/meme/images/"
    
    df = pd.read_excel(excel_path)
    
    images = df["image_name"].tolist()

    image_paths = [os.path.join(image_root, img) for img in images]
    return image_paths


# Main function to demonstrate loading datasets
def main():
    jokes = get_dataset_jokes()
    image_paths = get_memeImages()
    print(
        f"Here are {len(jokes)} jokes from the dataset."
    )  # for the main function to print number of jokes make sure the get_dataset_jokes function is working properly
    print(f"Here are {len(image_paths)} meme images from the dataset.") # for the main function to print number of meme images make sure the get_memeImages function is working properly
    
def q1_1():
    print("Q1-- 1. Language Modelling ")
    print(
        "Q1.1: Train a language model from scratch on the jokes dataset using a Transformer Decoder architecture."
    )

    SAVE_PATH = "./jokes_model"
    
    label_path = "Dataset/jokes.csv"
    df = pd.read_csv(label_path)
    jokes = df["Joke"]
    
    print(f"Number of jokes: {len(jokes)}")
    
    BLOCK_SIZE = 128  # larger block size = faster, but needs more GPU memory
    VOCAB_SIZE = 20000  # Smaller vocab = faster
    
    TOKENIZER_PATH = "./jokes_tokenizer"
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True)
    tokenizer.train_from_iterator(jokes.astype(str).tolist(), trainer)
    
    hf_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.add_special_tokens({
        "eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>",
        "pad_token": "<pad>", "mask_token": "<mask>"
    })
    
    #save the tokenizer
    hf_tokenizer.save_pretrained(TOKENIZER_PATH)
    print("Tokenizer saved and trained.")
    
    dataset = Dataset.from_dict({"text": jokes.astype(str).tolist()})
    tokenized = dataset.map(lambda x: hf_tokenizer(x["text"], max_length=BLOCK_SIZE,truncation=True, padding="max_length"),batched=True,)
    
    lm_dataset = tokenized
    data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)
    
    
    #=== Model Config (smaller = faster) ===
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=BLOCK_SIZE,
        n_ctx=BLOCK_SIZE,
        n_embd=256,
        n_layer=4,
        n_head=4,
    )
    
    model = GPT2LMHeadModel(config)
    print("GPT2 Model initialized (random weights).")
    
    
    # # === Training Arguments ===
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=2000,
        save_total_limit=2,
        seed=SEED, #set seed as the last four digits of my student ID
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_dataset,
    )
    
    # # === Generate Jokes ===
    print ("Training the model...")
    trainer.train()
    print("Model trained and saved.")
    #trainer.save_pretrained(SAVE_PATH)



#train over 10 epochs for this and plot.

def q1_2(): 
   print("Q1.2: Train a language model from scratch on the jokes dataset using a Transformer Decoder architecture in the 10 epoch.")
   
   TOKENIZER_PATH = "./jokes_tokenizer"
   SAVE_PATH = "./jokes_model2"
   
   
   label_path = "Dataset/jokes.csv"
   
   df = pd.read_csv(label_path)
   #df = df.sample(231657, random_state=SEED) #sampling to reduce the dataset size for faster training for testing purpose
   jokes = df["Joke"]
   
   print(f"Number of jokes: {len(jokes)}")
   
   BLOCK_SIZE = 128  # larger block size = faster, but needs more GPU memory
   VOCAB_SIZE = 20000  # Smaller vocab = faster
   
   tokenizer = Tokenizer(BPE())
   tokenizer.pre_tokenizer = ByteLevel()
   tokenizer.decoder = ByteLevelDecoder()
   
   trainer = BpeTrainer(vocab_size=VOCAB_SIZE, show_progress=True)
   tokenizer.train_from_iterator(jokes.astype(str).tolist(), trainer)
   hf_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
   hf_tokenizer.add_special_tokens({ "eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>","pad_token": "<pad>", "mask_token": "<mask>"})
   
   #save the tokenizer
   hf_tokenizer.save_pretrained(TOKENIZER_PATH)
   print("Tokenizer saved and trained.")
   
   dataset = Dataset.from_dict({"text": jokes.astype(str).tolist()})
   tokenized = dataset.map(lambda x: hf_tokenizer(x["text"], max_length=BLOCK_SIZE,truncation=True, padding="max_length"),batched=True,)
   
   lm_dataset = tokenized
   data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=False)
    
    
    #=== Model Config (smaller = faster) ===
   config = GPT2Config(
       vocab_size=VOCAB_SIZE,
       n_positions=BLOCK_SIZE,
       n_ctx=BLOCK_SIZE,
       n_embd=128,
       n_layer=2,
       n_head=2,
       )
    
   model = GPT2LMHeadModel(config)
   print("GPT2 Model initialized (random weights).")
    
    # # === Training Arguments ===
   training_args = TrainingArguments(
       output_dir=SAVE_PATH,
       overwrite_output_dir=True,
       num_train_epochs=10,
       per_device_train_batch_size=2,
       save_steps=2000,
       save_total_limit=2,
       seed=SEED,) #set seed as the last four digits of my student ID
   
   trainer = Trainer(
       model=model,
       args=training_args,
       data_collator=data_collator,
       train_dataset=lm_dataset,)
   
   # # === Generate Jokes ===
   print ("Training the model for the 10 epoch...")
   trainer.train()
   trainer.save_model(SAVE_PATH)
   
   print("Model trained with 10 epoch complet.")
   
   # Plotting the loss curve
   def plot_loss(trainer, save_path="loss_plot_q1_2.png"):
       logs = trainer.state.log_history
       losses = []
       epochs = []
       
       for entry in logs:
           if "loss" in entry and "epoch" in entry:
               losses.append(entry["loss"])
               epochs.append(entry["epoch"])
        
    # Plotting       
       plt.figure(figsize=(8, 5))
       plt.plot(epochs, losses, marker="o")
       plt.xlabel("Epoch")
       plt.ylabel("Training Loss")
       plt.title("Training Loss Over 10 Epochs")
       plt.grid(True)
       plt.savefig(save_path)
       print(f"Loss plot saved to {save_path}")
    
   plot_loss(trainer)

def q1_3(): 
    print("Q1.3: Trained model to generate at least 5 new jokes.")
    MODEL_PATH = "./jokes_model2"
    TOKENIZER_PATH = "./jokes_tokenizer"
    
    tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    
    #text generation pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    prompt = ["Why did the chicken", "who is under", "what do you call", "I have to go", "knock Knock"] #prompts to generate jokes
    set_seed(SEED)
    
    print("---------------------------------")
    # Generate jokes
    print ("Generating 5 new jokes:..........")
    for p in prompt:
        
        outputs = generator(
            p,
            max_new_tokens=25,      # generate ONLY 25 new tokens
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.8,
            no_repeat_ngram_size=2
        )
    
        print(f"Prompt: {p}")
        for output in outputs:
            print(f"Generated Joke: {output['generated_text']}")
            print("_"*50)



print("Q2-- Multimodal Learning")

def q2_1(): 
    print("Q2.1: Convolutional Neural Network (CNN) using PyTorch")
    
    dataset = "meme_images"   # same style as voting/volcanoes/spam
    
    excel_path = "Dataset/meme/labels.xlsx"
    image_root = "Dataset/meme/images/"   # required for image dataset
    
    df = pd.read_excel(excel_path)
    
    # Equivalent to schema, X, y = get_dataset(...)
    label_map = {label: idx for idx, label in enumerate(df["sentiment"].unique())} 
    df["label_id"] = df["sentiment"].map(label_map)
    
    X = df["image_name"]
    y = df["label_id"]  # assuming label_id is the column with labels
    
    # 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    
    # 20% val, 20% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
    
    print(f"Dataset: {dataset}") # print dataset name
    
    #validating the splits
    n = len(X)
    print(f"Test ratio={len(X_test)/n:.4f}, Validation ratio={len(X_val)/n:.4f}, Train ratio={len(X_train)/n:.4f}")
    print(f"Total={n}, Train={len(X_train)}, validate={len(X_val)}, Test={len(X_test)}")
    
    
    #transform for image 
    transform = T.Compose([T.Resize((128, 128)),T.ToTensor(),])
    
     #pytorch dataset class
    class memeImageData(Dataset):
        def __init__(self, image_names, labels, image_root, transform=None):
            self.image_names = image_names.reset_index(drop=True)
            self.labels = labels.reset_index(drop=True)
            self.image_root = image_root
            self.transform = transform  
        
        def __len__(self):return len(self.image_names)
        
        
        def __getitem__(self, idx):
            img_name = self.image_names.iloc[idx]
            label = self.labels.iloc[idx] 
            
            img_path = os.path.join(self.image_root, img_name)
            
            
            
        #image 5119 is missing that is why it is handled here this is very important to avoid error during training
            
            if not os.path.exists(img_path):
                image = Image.new("RGB", (128, 128), color=(255, 255, 255))  # create a blank white image
            else:
                image = Image.open(img_path).convert("RGB")
                
            if self.transform:
                image = self.transform(image)
                    
            return image, label
        
    train_dataset = memeImageData(X_train, y_train, image_root, transform=transform)
    val_dataset = memeImageData(X_val, y_val, image_root, transform=transform)
    test_dataset = memeImageData(X_test, y_test, image_root, transform=transform)
        
    #dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        #Cnn model as myCNN as described in the question
    class MyCNN(nn.Module):
        def __init__(self, num_classes=3):
            super(MyCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(2, 2)
            
            self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool4 = nn.MaxPool2d(2, 2)
            
            self.fc1 = nn.Linear(128 * 8 * 8 , 256)
            self.fc2 = nn.Linear(256, num_classes)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.pool1(self.relu(self.conv1(x)))
            x = self.pool2(self.relu(self.conv2(x)))
            x = self.pool3(self.relu(self.conv3(x)))
            x = self.pool4(self.relu(self.conv4(x)))
            
            x = x.view(x.size(0), -1) # flatten
            
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use gpu if available not use cpu
    model = MyCNN(num_classes=5).to(device) #assuming 5 classes for sentiment
    
    criterion = nn.CrossEntropyLoss() #loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #optimizer
    
    #Now train in the 10 epochs
    for epoc in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoc+1}, Loss: {epoch_loss:.4f}")
        
        
    def evaluate_model(model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            "Micro Accuracy": accuracy_score(all_labels, all_preds),
            "Macro Accuracy": recall_score(all_labels, all_preds, average="macro"),
            "Micro F1": f1_score(all_labels, all_preds, average="micro"),
            "Macro F1": f1_score(all_labels, all_preds, average="macro"),
        }
        return metrics
        
    print ("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader)
    
    print("Test Set Metrics:")
    print(test_metrics)


    
def q2_2(): 
    print("Q2.2: Recurrent Neural network using PyTorch and name is MyRNN") 
    dataset = "meme_text"   # same style as voting/volcanoes/spam
    df = pd.read_excel("Dataset/meme/labels.xlsx")
    
    # Equivalent to schema, X, y = get_dataset(...)
    label_map = {label: idx for idx, label in enumerate(df["sentiment"].unique())} 
    df["label_id"] = df["sentiment"].map(label_map)
    
    X = df["text"].fillna("").astype(str)
    y = df["label_id"]  # assuming label_id is the column with labels
    
    # 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    
    # 20% val, 20% test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)
    
    print(f"Dataset: {dataset}")
    n = len(X)
    print(f"Test ratio={len(X_test)/n:.4f}, Validation ratio={len(X_val)/n:.4f}, Train ratio={len(X_train)/n:.4f}")
    print(f"Total={n}, Train={len(X_train)}, validate={len(X_val)}, Test={len(X_test)}")    
    
    tokenizer = get_tokenizer("basic_english") # basic english tokenizer
    
    def yield_tokens(text_list): 
        for text in text_list:
            yield tokenizer(text)
            
    vocab = build_vocab_from_iterator(yield_tokens(X_train))   #build vocab from training data
    vocab._default_index = vocab["<unk>"] #set default index for unknown tokens


  
    #pytorch dataset class
    class memeTextData(Dataset):
        def __init__(self, texts, labels, transform=None):
            self.texts = texts.reset_index(drop=True)
            self.labels = labels.reset_index(drop=True)
        
        def __len__(self):return len(self.texts)
        
        
        def __getitem__(self, idx):
            tokens = tokenizer(self.texts.iloc[idx])
            indices = [vocab[token] for token in tokens]
            token_ids = torch.tensor(indices, dtype=torch.long)
            label = torch.tensor(self.labels.iloc[idx], dtype=torch.long) #convert to tensor
            return token_ids, label
            
        
    #create datasets
    train_dataset = memeTextData(X_train, y_train)
    val_dataset = memeTextData(X_val, y_val)
    test_dataset = memeTextData(X_test, y_test)
    
    def collate_batch(batch):
            text_list, label_list = [], []
            for (_text, _label) in batch:
                text_list.append(_text)
                label_list.append(_label)
            text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=0)
            label_list = torch.tensor(label_list, dtype=torch.long)
            return text_list, label_list
        
    #dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
        
        #define CNN model
    class MyRNN(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
            super(MyRNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        
        def forward(self, x):
            x = self.embedding(x)
            output, hidden = self.rnn(x)
            hidden = hidden[-1]
            out = self.fc(hidden)
            return out
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyRNN(vocab_size=len(vocab),embed_size=64,hidden_size=128,num_classes=len(label_map)).to(device)

    #loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    #Now train in the 10 epochs
    for epoc in range(10):
        model.train()
        running_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * texts.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset) #calculate epoch loss
        print(f"Epoch {epoc+1}, Loss: {epoch_loss:.4f}")
        
        #evaluation function
    def evaluate_model(model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for texts, labels in dataloader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {
            "Micro Accuracy": accuracy_score(all_labels, all_preds),
            "Macro Accuracy": recall_score(all_labels, all_preds, average="macro"),
            "Micro F1": f1_score(all_labels, all_preds, average="micro"),
            "Macro F1": f1_score(all_labels, all_preds, average="macro"), 
        }
        return metrics #return metrics
    
    # Evaluate on test set
    print ("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader)
    
    print("Test Set Metrics:")
    print(test_metrics)


def q2_3():
    print("Q2.3: Multimodal model in PyTorch MyCNN and MyRNN; ")
    excel_path = "Dataset/meme/labels.xlsx"
    image_root = "Dataset/meme/images/"
    df = pd.read_excel(excel_path) #load excel file

    # label encoding
    label_map = {label: idx for idx, label in enumerate(df["sentiment"].unique())}
    df["label_id"] = df["sentiment"].map(label_map)
    
    #split dataset
    X_img = df["image_name"] #image column
    X_text = df["text"].fillna("").astype(str)
    y = df["label_id"] #assuming label_id is the column with labels
    
    # 60% train, 40% temp
    X_img_train, X_img_temp, text_train, text_temp, y_train, y_temp = train_test_split(
        X_img, X_text, y, test_size=0.4, random_state=SEED, stratify=y
    )
    
    # 20% val, 20% test
    X_img_val, X_img_test, text_val, text_test, y_val, y_test = train_test_split(
        X_img_temp, text_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    
    #validating the splits
    n = len(X_img)
    print(f"Train={len(X_img_train)}, Val={len(X_img_val)}, Test={len(X_img_test)}")
    print(f"Ratios Train={len(X_img_train)/n:.4f}, Val={len(X_img_val)/n:.4f}, Test={len(X_img_test)/n:.4f}")


    tokenizer = get_tokenizer("basic_english")
    
    def yield_tokens(text_lists):
        for text in text_lists:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(yield_tokens(text_train))
    
   
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor()
    ])
    
    #models described usined for the multimodal dataset
    class MultimodalDataset(Dataset):
        def __init__(self, imgs, texts, labels, image_root, transform=None):
            self.imgs = imgs.reset_index(drop=True)
            self.texts = texts.reset_index(drop=True)
            self.labels = labels.reset_index(drop=True)
            self.image_root = image_root
            self.transform = transform
            
        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            img_name = self.imgs.iloc[idx]
            img_path = os.path.join(self.image_root, img_name)

            if not os.path.exists(img_path):
                image = Image.new("RGB", (128, 128), (255, 255, 255))
            else:
                image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            tokens = tokenizer(self.texts.iloc[idx]) #tokenize text 
            ids = [vocab[tok] for tok in tokens]  # unk tokens go to default index
            ids = torch.tensor(ids, dtype=torch.long) #convert to tensor

            label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

            return image, ids, label #return image, ids, label

    # Collate function for variable-length text
    def collate_batch(batch):
        images, texts, labels = [], [], [] #initialize lists

        for img, txt, lbl in batch:
            images.append(img)
            texts.append(txt)
            labels.append(lbl)

        images = torch.stack(images)
        texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=vocab["<unk>"])
        labels = torch.tensor(labels, dtype=torch.long)

        return images, texts, labels #return images, texts, labels

 
    train_ds = MultimodalDataset(X_img_train, text_train, y_train, image_root, transform) #create train dataset
    val_ds   = MultimodalDataset(X_img_val, text_val, y_val, image_root, transform) #create val dataset
    test_ds  = MultimodalDataset(X_img_test, text_test, y_test, image_root, transform) #create test dataset

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batch) #create train dataloader
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_batch) #create val dataloader
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_batch) #create test dataloader
    
    #define the multimodal model both the cnn and rnn
    class MyCNN(nn.Module):
        def __init__(self, out_dim=128):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            )
            self.fc = nn.Linear(128 * 8 * 8, out_dim)

        def forward(self, x):
            x = self.feature(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    class MyRNN(nn.Module):
        def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

        def forward(self, x):
            x = self.embedding(x)
            _, hidden = self.rnn(x)
            return hidden[-1]

    class MyMultimodalModel(nn.Module):
        def __init__(self, vocab_size, num_classes):
            super().__init__()
            self.cnn = MyCNN(out_dim=128)
            self.rnn = MyRNN(vocab_size=vocab_size)
            self.fc = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, image, text):
            img_feat = self.cnn(image)
            txt_feat = self.rnn(text)
            combined = torch.cat([img_feat, txt_feat], dim=1)
            return self.fc(combined)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyMultimodalModel(vocab_size=len(vocab), num_classes=len(label_map)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Traing the model
    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for images, texts, labels in train_loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images, texts)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # Validation
    def evaluate_model(loader):
        model.eval()
        all_preds = []
        all_labels = []    
        with torch.no_grad():
            for images, texts, labels in loader:
                images, texts = images.to(device), texts.to(device)
                outputs = model(images, texts)
                _, preds = torch.max(outputs, 1)
                        
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        metrics = {
            "Micro Accuracy": accuracy_score(all_labels, all_preds),
            "Macro Accuracy": recall_score(all_labels, all_preds, average="macro"),
            "Micro F1": f1_score(all_labels, all_preds, average="micro"),
            "Macro F1": f1_score(all_labels, all_preds, average="macro"),}
        return metrics   
    
    # Evaluate on test set      
    print ("Evaluating on test set...")
    test_metrics = evaluate_model(test_loader)
    print("Test Set Metrics:")
    print(test_metrics)


    
    
if __name__ == "__main__":
    main()
    # q1_1()
    # q1_2()
    q1_3()
    q2_1()
    q2_2()
    q2_3()
    
  
