import config
import dataset
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from model import BERTBaseUncased 

# from torch.optim import AdamW
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import engine

def run():

    df = pd.read_csv(config.TRIAINING_FILE, nrows=100)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    df_train, df_valid = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment'].values 
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        review=df['review'].values,
        target=df['sentiment'].values
        )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        review=df['review'].values,
        target=df['sentiment'].values
        )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    device = torch.device(config.DEVICE)

    model = BERTBaseUncased()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0},
        
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE* config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    # model = nn.DataParallel(model)

    best_accuracy = 0
    for ecpoh in range(config.EPOCHS):
        engine.train_func(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_func(valid_data_loader, model, device)
        
        outputs = np.array(outputs) > 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f'Accuracy Score = {accuracy}')

        if accuracy > best_accuracy:
            
            path_to_save = '/'.join(config.MODEL_PATH.split('/')[:-1])
            # model.config.to_json_file(path_to_save + "/config.json")            
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy


    print(model.state_dict())
    
if __name__ == '__main__':
    run()