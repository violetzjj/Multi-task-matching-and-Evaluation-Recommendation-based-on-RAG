# from transformers.modeling_bert import *
from transformers.models.bert.modeling_bert import *
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import config as myconfig

class BertMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiTask, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.match_classifier = nn.Linear(config.hidden_size * 101, 2)
        self.linear_score_layer = nn.Linear(config.hidden_size * 101, 1)
        self.sigmoid = nn.Sigmoid()
        # self.hidden_size = config.hidden_size
        self.init_weights()

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        # 得到判别值
        logits = self.classifier(padded_sequence_output)
        sequence_length = len(input_ids[0]) - 1
        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        # match_classifier = nn.Linear(sequence_output.shape[1], 2).to(myconfig.device)
        match_logits = self.match_classifier(sequence_output)
        score_logits = self.linear_score_layer(sequence_output)
        score_logits = self.sigmoid(score_logits)
        score_logits = score_logits * 5
        # loss_match = torch.nn.BCEWithLogitsLoss()
        # loss_match = CrossEntropyLoss()
        score_loss = torch.nn.MSELoss()
        outputs = (logits, match_logits, score_logits, sequence_output)
        if labels is not None:
            loss_mask = labels[0].gt(-1)
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                # 只留下label存在的位置计算loss
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels[0].view(-1)[active_loss]
                loss1 = loss_fct(active_logits, active_labels)
            else:
                loss1 = loss_fct(logits.view(-1, self.num_labels), labels[0].view(-1))
            loss2 = F.cross_entropy(match_logits, labels[1])
            loss3 = score_loss(score_logits.view(labels[2].shape), labels[2])
            loss = 0.1 * loss1 + 0.1 * loss2 + 0.8 * loss3
            outputs = (loss,) + outputs

        # contain: (loss), scores
        return outputs
