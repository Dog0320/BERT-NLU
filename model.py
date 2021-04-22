import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config=config)

        self.dropout = nn.Dropout(config.dropout)
        self.intent_classifier = nn.Linear(config.hidden_size, config.num_intent)
        self.slot_classifier = nn.Linear(config.hidden_size, config.num_slot)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)



        return intent_logits, slot_logits
