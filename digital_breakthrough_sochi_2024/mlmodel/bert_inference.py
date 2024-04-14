import re
from collections import Counter
from typing import List, Mapping
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm

from transformers import AutoTokenizer

torch.manual_seed(13)

WORDS_KEYS_SET = {
    'доверенность',
    'договор',
    'акт',
    'заявление',
    'приказ',
    'счет',
    'приложение',
    'соглашение',
    'договор'
    'оферты',
    'устав',
    'решение'
}

class TextClassificationDataset(Dataset):

    def __init__(
            self,
            texts: List[str],
            labels: List[int] = None,
            labels_dict: Mapping[str, str] = None,
            max_seq_length: int = 128,
            model_name: str = None,
            add_spec_tokens: bool = False,
    ):
        """
        Args:
            texts: a list of text to classify
            labels: a list with classification labels
            labels_dict: a dictionary mapping class names to class ids
            max_seq_length: maximum sequence length in tokens
            model_name: transformer model name
            add_spec_tokens: if we want to add special tokens to the tokenizer
        """

        self.texts = texts
        self.labels = labels
        self.labels_dict = labels_dict
        self.max_seq_length = max_seq_length

        if (self.labels_dict is None and labels is not None):
            self.labels_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        """
        Gets encoded representation of a single element (text) in the dataset
        Args:
            index (int): index of the element in dataset
        Returns:
            Single element by index
        """
        text = self.texts[index]

        # Словарь с ключами 'input_ids', 'token_type_ids', 'attention_masks'
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        encoded_dict["features"] = encoded_dict["input_ids"].squeeze(0)
        del encoded_dict["input_ids"]
        del encoded_dict["token_type_ids"]  #в принципе не нужны тут, предсказывают классификацию пар

        # Кодирование target label с помощью labels_dict
        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.labels_dict.get(y, -1)]).long().squeeze(0)
            encoded_dict["targets"] = y_encoded

        return encoded_dict



def del_process_text(text):
    """
    удаляем в тексте ненужные подчеркивания и данные в скобках
    """
    text = re.sub("_+", "", text)
    text = re.sub("\([^)]*\)", "", text)
    text = text.lower()

    return text


def del_clauses_process_text(sequences):
    """
    удаляем пункты договора
    """
    res = [re.sub("\n", "", x) for x in sequences]
    res = [re.sub(r'\s+', ' ', x).strip() for x in sequences]

    return res

def splitting_text_regex(text: str,
                            splitter='[\t\n]\s*\d+[0-9\.]*\.\s'):
    """
        Стандартный regex сплитит по пунктам договора
    """
    points = re.findall(splitter, text)
    result = re.split(splitter, text)
    splitted_text = []
    for text in result:
        if len(text.split(' ')) > 600:
            # > 600 слов => разбиение по \n
            splitted_text.extend(text.split('\n'))
        else:
            splitted_text.append(text)

    return splitted_text, points


def choose(predictions, number_of_classes=11):
    """
    Посчитывает голоса для каждого класса
    """

    voices = Counter()

    for prediction in predictions:
        voices[prediction] = voices.get(prediction, 0) + 1

    return voices.most_common(number_of_classes)[0][0], voices


def run_inference(new_document_text, device, model, sentence_length=5):
    # Читаем кфг
    with open("./config.yml", "r") as yamlfile:
        cfg = yaml.safe_load(yamlfile)

    # Препроцесс
    splitted_text, splitting_points = splitting_text_regex(del_process_text(new_document_text))
    processed_splitted_text = del_clauses_process_text(splitted_text)

    splitting_points = list(range(len(splitted_text)))
    # Удаляем sequences длины которых <= 5 (заголовки, которые одинаковые для многих документов)
    filtered_splitted_texts = []
    filtered_splitting_points = []
    for text, splitter in zip(processed_splitted_text, splitting_points):
        if (len(text.split(" ")) >= sentence_length) | (len(set(text.split(" ")).intersection(set(WORDS_KEYS_SET))) > 0):
            filtered_splitted_texts.append(text)
            filtered_splitting_points.append(splitter)

    # Образуем unlabeled_dataloader
    unlabeled_dataset = TextClassificationDataset(
        texts=processed_splitted_text,
        labels=None,
        max_seq_length=cfg['model']['max_seq_length'],
        model_name=cfg['model']['model_name'],
    )

    unlabeled_loader = DataLoader(
        dataset=unlabeled_dataset,
        sampler=SequentialSampler(unlabeled_dataset),
        batch_size=cfg['training']['batch_size'],
    )

    # Разница между eval_loop_fn в том, что сейчас у нас нет true_labels.
    model.eval()
    final_logits = []
    tqdm_bar = tqdm(unlabeled_loader, desc="Inference", position=0, leave=True)
    for _, batch in enumerate(tqdm_bar):
        features = batch["features"]  # (input_ids)
        attention_mask = batch["attention_mask"]

        features = features.to(device, dtype=torch.long)
        attention_mask = attention_mask.to(device, dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids=features,
                            attention_mask=attention_mask,
                            return_dict=True)
        final_logits.append(outputs['logits'].detach().cpu().numpy())

    # объеденяем результаты всех batches.
    flat_predictions = np.concatenate(final_logits, axis=0)
    predicted_labels = np.argmax(
        flat_predictions, axis=1
        ).flatten()  # argmax для каждого образца, чтобы вывести метки, а не оценки

    # возвращяем наиболее распространенные labels Counter
    most_confident_labels = choose(predicted_labels)

    probabilities = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0, 10 : 0}
    for label in probabilities.keys():
        probabilities[label] = most_confident_labels[1][label] / sum(most_confident_labels[1].values())
    most_confident_label = most_confident_labels[0]
    # Most confidence для каждого класса
    getting_confidences_args = (flat_predictions, predicted_labels, most_confident_labels)

    return most_confident_label, most_confident_labels, getting_confidences_args, probabilities
