import argparse
import os

import numpy as np
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig

from model import Model
from utils.data_utils import NluDataset, glue_processor, prepare_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, data_raw, id_to_label, mode='dev'):
    model.eval()
    test_data = NluDataset(data_raw)
    test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=test_data.collate_fn)
    slot_labels = id_to_label['slot_labels']

    joint_all = 0
    joint_correct = 0
    s_preds = []
    s_labels = []
    i_preds = []
    i_labels = []
    epoch_pbar = tqdm(test_dataloader, desc="Evaluation", disable=False)
    for step, batch in enumerate(test_dataloader):
        batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
        input_ids, segment_ids, input_mask, slot_ids, intent_id = batch
        with torch.no_grad():
            intent_output, slot_output = model(input_ids, segment_ids, input_mask)
        # intent_evaluate
        intent_output = intent_output.argmax(dim=1)
        intent_output = intent_output.tolist()
        intent_label = intent_id.tolist()
        i_preds = i_preds + intent_output
        i_labels = i_labels + intent_label
        # slot_evaluate
        slot_output = slot_output.argmax(dim=2)
        slot_output = slot_output.tolist()
        slot_ids = slot_ids.tolist()
        for idx, (p, l) in enumerate(zip(slot_output, slot_ids)):
            p_text, l_text = align_predictions(p, l, slot_labels)
            joint_all += 1
            if p_text == l_text and intent_label[idx] == intent_output[idx]:
                joint_correct += 1

            s_preds.append(p_text)
            s_labels.append(l_text)
        epoch_pbar.update(1)
    epoch_pbar.close()

    res = {
        "joint_acc": joint_correct / joint_all,
        "intent_accuracy": cal_acc(i_preds, i_labels),
        "slot_accuracy_score": accuracy_score(s_labels, s_preds),
        "slot_precision": precision_score(s_labels, s_preds),
        "slot_recall": recall_score(s_labels, s_preds),
        "slot_f1": f1_score(s_labels, s_preds),
    }
    print('Evaluation on ', mode, ' dataset: ', res)
    return res


def cal_acc(preds, labels):
    acc = sum([1 if p == l else 0 for p, l in zip(preds, labels)]) / len(labels)
    return acc


def align_predictions(preds, slot_ids, id_to_label):
    aligned_labels = []
    aligned_preds = []
    for p, l in zip(preds, slot_ids):
        if l != -100:
            aligned_preds.append(id_to_label[p])
            aligned_labels.append(id_to_label[l])
    return aligned_preds, aligned_labels


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    # Data
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)
    dev_data_raw = prepare_data(dev_examples, args.max_seq_len, tokenizer, labels)
    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels)

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.num_intent = len(labels['intent_labels'])
    model_config.num_slot = len(labels['slot_labels'])
    model = Model(model_config)
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    evaluate(model, dev_data_raw, labels, 'dev')
    evaluate(model, test_data_raw, labels, 'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='nlu', type=str)
    parser.add_argument("--data_dir", default='data/atis/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--model_ckpt_path", default='outputs/model_best.bin', type=str)
    parser.add_argument("--max_seq_len", default=60, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)
