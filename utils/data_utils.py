import os
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.utils.data import Dataset
from transformers import DataProcessor, logging

logger = logging.get_logger(__name__)


@dataclass
class InputExample:
    guid: str
    words: List[str]
    intent: Optional[str]
    slots: Optional[List[str]]


class NluProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train/intent_seq.in")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train/intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "train/intent_seq.out")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev/intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "dev/intent_seq.out")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "predict")

    def get_labels(self, data_dir):
        """See base class."""
        slot_labels_list = self._read_tsv(os.path.join(data_dir, "vocab/slot_vocab"))
        slot_labels = [label[0] for label in slot_labels_list]
        intent_labels_list = self._read_tsv(os.path.join(data_dir, "vocab/intent_vocab"))
        intent_labels = [label[0] for label in intent_labels_list]
        labels = {'intent_labels': intent_labels, 'slot_labels': slot_labels}
        return labels

    def _create_examples(self, lines_in, lines_out, set_type):
        """Creates examples for the training, dev and test sets."""

        examples = []
        for i, (line, out) in enumerate(zip(lines_in, lines_out)):
            label_split = out[0].strip().split()
            guid = "%s-%s" % (set_type, i)
            words = line[0][4:].strip()
            slots = None if set_type == "predict" else label_split[1:]
            intent = None if set_type == "predict" else label_split[0]
            examples.append(InputExample(guid=guid, words=words, slots=slots, intent=intent))

        return examples


class TrainingInstance:
    def __init__(self, example, max_seq_len):
        self.words = example.words.split()
        self.slots = example.slots
        self.intent = example.intent
        self.max_seq_len = max_seq_len

    def make_instance(self, tokenizer, intent_label_map, slot_label_map, pad_label_id=-100):
        tokens = []
        slot_ids = []
        if self.slots:
            for word, label in zip(self.words, self.slots):
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    slot_ids.extend([slot_label_map[label]] + [pad_label_id] * (len(word_tokens) - 1))
        else:
            # 预测时，把需要预测的位置置为１
            for word in self.words:
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    slot_ids.extend([1] + [0] * (len(word_tokens) - 1))

        # TODO 判断长度越界

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        self.slot_ids = [pad_label_id] + slot_ids + [pad_label_id]
        self.input_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.segment_id = [0] * len(self.input_ids)
        self.input_mask = [1] * len(self.input_ids)
        padding_length = self.max_seq_len - len(self.input_ids)
        if padding_length > 0:
            self.input_ids = self.input_ids + [0] * padding_length
            self.segment_id = self.segment_id + [0] * padding_length
            self.input_mask = self.input_mask + [0] * padding_length
            self.slot_ids = self.slot_ids + [pad_label_id] * padding_length
        self.intent_id = intent_label_map[self.intent] if self.intent else None


class NluDataset(Dataset):
    def __init__(self, data, annotated=True):
        self.data = data
        self.len = len(data)
        self.annotated = annotated

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_ids for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        slot_ids = torch.tensor([f.slot_ids for f in batch], dtype=torch.long)
        intent_id = torch.tensor([f.intent_id for f in batch], dtype=torch.long) if self.annotated else None
        return input_ids, segment_ids, input_mask, slot_ids, intent_id


def prepare_data(examples, max_seq_len, tokenizer, labels):
    slot_label_map = {label: idx for idx, label in enumerate(labels['slot_labels'])}
    intent_label_map = {label: idx for idx, label in enumerate(labels['intent_labels'])}
    data = []

    for example in examples:
        instance = TrainingInstance(example, max_seq_len)
        instance.make_instance(tokenizer, intent_label_map, slot_label_map)
        data.append(instance)

    return data


glue_processor = {
    'nlu': NluProcessor()
}
