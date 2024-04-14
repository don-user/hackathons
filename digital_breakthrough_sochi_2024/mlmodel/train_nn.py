import pandas as pd
import numpy as np
import random
import torch

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(13)

def train_loop_fn(train_dataloader, model, device, optimizer, scheduler, train_losses):
    model.train()
    tqdm_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
    for _, batch in enumerate(tqdm_bar):
        features = batch["features"] # (input_ids)
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        features = features.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad() # зануляет градиенты перед backward

        outputs = model(input_ids=features,
                        attention_mask=attention_mask,
                        labels=targets,
                        return_dict=True)

        loss = outputs['loss']
        loss.backward() # вычисляет градиенты
        optimizer.step() # обнолвяет weights

        if scheduler is not None:
            scheduler.step() # изменяет lr, если loss почти не улучшается
        tqdm_bar.desc = "Training loss: {:.2e}; lr: {:.2e}".format(loss.item(), scheduler.get_last_lr()[0])

        if (_ % 9 == 0):
            train_losses.append(loss.item())


def eval_loop_fn(val_dataloader, model, device, val_losses):

    model.eval()
    final_targets = []
    final_logits = []
    tqdm_bar = tqdm(val_dataloader, desc="Validating", position=0, leave=True)

    for _, batch in enumerate(tqdm_bar):
        features = batch["features"] # (input_ids)
        attention_mask = batch["attention_mask"]
        targets = batch["targets"]

        features = features.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids=features,
                            attention_mask=attention_mask,
                            labels=targets,
                            return_dict=True)

        eval_loss = outputs['loss']
        val_losses.append(eval_loss.item())

        # перемещает logits и labels на CPU
        final_targets.append(targets.detach().cpu().numpy())
        final_logits.append(outputs['logits'].detach().cpu().numpy())

    # объеденяет результаты всех batches.
    flat_predictions = np.concatenate(final_logits, axis=0)
    flat_true_labels = np.concatenate(final_targets, axis=0)

    predicted_labels = np.argmax(flat_predictions, axis=1).flatten() # argmax для каждого образца, чтобы вывести метки, а не оценки

    return predicted_labels, flat_true_labels

def run_inference(model, new_document, device):

    # Создаем dataloader
    unlabeled_dataset = TextClassificationDataset(
        texts=unlabeled_df['title'].values.tolist(),
        labels=None,
        max_seq_length=cfg['model']['max_seq_length'],
        model_name=cfg['model']['model_name'],
    )

    unlabeled_loader = DataLoader(
        dataset=unlabeled_dataset,
        sampler=SequentialSampler(unlabeled_dataset),
        batch_size=cfg['training']['batch_size'],
    )


    # Разница между eval_loop_fn в том, что у нас нет true_labels.
    model.eval()
    final_logits = []
    tqdm_bar = tqdm(unlabeled_dataloader, desc="Inference", position=0, leave=True)
    for _, batch in enumerate(tqdm_bar):
        features = batch["features"] # (input_ids)
        attention_mask = batch["attention_mask"]

        features = features.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids=features,
                            attention_mask=attention_mask,
                            return_dict=True)

        final_logits.append(outputs['logits'].detach().cpu().numpy())

    # Объедините результаты всех batches.
    flat_predictions = np.concatenate(final_logits, axis=0)
    predicted_labels = np.argmax(flat_predictions, axis=1).flatten()

    return predicted_labels


def save_checkpoint(state, filename="saved_weights.pth"):

    print(f"=> Saving checkpoint at epoch {state['epoch']}")
    torch.save(state,filename)


def load_checkpoint(checkpoint):

    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def choose(predictions):
    """
    голосование по предиктам
    """

    number_of_classes = 11
    voices = Counter()

    for prediction in predictions:
        voices[prediction] = voices.get(prediction, 0) + 1

    return voices.most_common(number_of_classes)


def vote_prediction(df, id_column="id", prediction_column="predicted_label"):
    # Новое голосование
    voted_preds = {}
    id_data = []
    choice_data = []

    for doc_id in test_sentences[id_column].unique():
        id_data.append(doc_id)
        choice_data.append(choose(df[df[id_column] == doc_id].to_list()))

    voted_preds = pd.DataFrame({"id": id_data, "choice": choice_data})

    return voted_preds


def run(model, cfg, train_dataloader, val_dataloader, unlabeled_dataloader, test_sentences, y_test, train_losses, val_losses, accuracy_scores, F1_scores, from_checkpoint=False):

    device = torch.device("cuda")

    # перевод модели на GPU.
    desc = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg['training']['learning_rate']),
        eps=float(cfg['training']['adam_epsilon']), # adam_epsilon - по default 1e-8.
    )

    num_epochs = cfg['training']['num_epochs']
    total_steps = len(train_dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # default parameter
        num_training_steps = total_steps,
    )

    if from_checkpoint:
        best_checkpoint = torch.load('best_f1_state.pth')
        model.load_state_dict(best_checkpoint["state_dict"])
        last_max_F1 = best_checkpoint['F1_score']
    else:
        last_max_F1 = 0

    for epoch in tqdm(range(num_epochs)):
        print()
        print("====================================")
        print(f"Epoch = {epoch}")
        print("====================================\n")
        train_loop_fn(train_dataloader, model, device, optimizer, scheduler, train_losses)
        predicted_labels, targets = eval_loop_fn(val_dataloader, model, device, val_losses)
        # объединить в соответствии с test_sentences и предсказать нормальный confusion_matrix
        test_sentences['predicted_labels'] = predicted_labels

        voted_preds = {}
        id_data = []
        choice_data = []

        for doc_id in test_sentences["id"].unique():
             id_data.append(doc_id)
             #print(doc_id)
             #print(test_sentences[test_sentences["id"] == doc_id]['predicted_labels'].to_list())
             choice_data.append(choose(test_sentences[test_sentences["id"] == doc_id]['predicted_labels'].to_list())[0][0])
             #print(choice_data)
             #print()
        print('y_test -', y_test)
        print('choice_data -', choice_data)
        # len(choice_data) == 22
        accuracy = accuracy_score(y_test, choice_data)
        f1 = f1_score(y_test, choice_data, average='weighted')
        print(f"Validation acc = {accuracy}; F1 = {f1}")

        if (f1 > last_max_F1):
            print(classification_report(y_test, choice_data))
            print(confusion_matrix(y_test, choice_data))
            print(sns.heatmap(confusion_matrix(y_test, choice_data), annot=True, cmap='Blues', fmt='3g'))
            plt.show()
            last_max_F1 = f1

            checkpoint = {
                'epoch' : epoch,
                'accuracy' : accuracy,
                'F1_score' : f1,
                'state_dict' : deepcopy(model.state_dict()),
            }
            save_checkpoint(checkpoint, filename="best_f1_state.pth")
