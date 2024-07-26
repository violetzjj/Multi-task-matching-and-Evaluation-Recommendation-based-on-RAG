import torch
import logging
import torch.nn as nn
from tqdm import tqdm

import config
import numpy as np
from model import BertMultiTask
from metrics import f1_score, bad_case, f1_score_match
from transformers import BertTokenizer
from torchmetrics import R2Score
from torch.utils.tensorboard import SummaryWriter

step_index = 0

def train_epoch(train_loader, model, optimizer, scheduler, epoch, writer):
    global step_index
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels, batch_labels_match, batch__labels_score = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks,
                     labels=[batch_labels, batch_labels_match, batch__labels_score])[0]
        train_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        # write the loss to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), step_index)
        step_index += 1
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertMultiTask.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_f1_ner = 0.0
    best_val_f1_match = 0.0
    best_val_r2 = 0
    patience_counter = 0
    
    writer = SummaryWriter()
    
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch, writer)
        val_metrics = evaluate(dev_loader, model)
        val_f1_ner = val_metrics['f1']
        val_f1_match = val_metrics['f1_match']
        val_MSE = val_metrics['MSE']
        val_RMSE = val_metrics['RMSE']
        val_MAE = val_metrics['MAE']
        val_r2 = val_metrics['r2']
        logging.info("Epoch: {}, dev loss: {}, NER f1 score: {}, Match f1 score: {}, spatial similarity score MSE: {}, spatial similarity score RMSE: {},spatial similarity score MAE: {}, spatial similarity score r2: {}".format(epoch, val_metrics['loss'],
                                                                                            val_f1_ner, val_f1_match, val_MSE, val_RMSE, val_MAE, val_r2))
        improve_f1_ner = val_f1_ner - best_val_f1_ner
        improve_f1_match = val_f1_match - best_val_f1_match
        improve_r2 = val_r2 - best_val_r2
        improve_f1 = 0.1 * improve_f1_ner + 0.1 * improve_f1_match + 0.8 * improve_r2
        if improve_f1 > 1e-5:
            best_val_f1_ner = val_f1_ner
            best_val_f1_match = val_f1_match
            best_val_r2 = val_r2
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
            if improve_f1_match < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best ner val f1: {}".format(best_val_f1_ner))
            logging.info("Best match val f1: {}".format(best_val_f1_match))
            break
    logging.info("Training Finished!")
    # close the SummaryWriter object
    writer.close()


def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(config.macbert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    true_match_tags = []
    pred_match_tags = []
    true_score_tags = []
    pred_score_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(tqdm(dev_loader)):
            batch_data, batch_token_starts, batch_tags, batch_labels_match, batch_labels_score = batch_samples
            sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                               if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks,
                         labels=[batch_tags, batch_labels_match, batch_labels_score])[0]
            dev_losses += loss.item()
            # shape: (batch_size, max_len, num_labels)
            batch_output, barch_output_match, batch_output_score, _ = model((batch_data, batch_token_starts),
                                                     token_type_ids=None, attention_mask=batch_masks)

            batch_output = batch_output.detach().cpu().numpy()
            barch_output_match = barch_output_match.detach().cpu().numpy()
            batch_output_score = batch_output_score.detach().cpu().numpy()
            batch_tags = batch_tags.to('cpu').numpy()
            batch_tags_match = batch_labels_match.to('cpu').numpy()
            batch_tags_score = batch_labels_score.to('cpu').numpy()

            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in np.argmax(batch_output, axis=2)])
            true_tags.extend([[id2label.get(idx) if idx != -1 else 'O' for idx in indices] for indices in batch_tags])

            for idx in np.argmax(barch_output_match, axis=1):
                pred_match_tags.append([idx])
            for idx in batch_tags_match:
                true_match_tags.append([idx])
            pred_score_tags.append([batch_output_score])
            true_score_tags.append([batch_tags_score])
    if len(pred_score_tags[-1]) != config.batch_size:
        pred_score_tags.pop()
    if len(true_score_tags[-1]) != config.batch_size:
        true_score_tags.pop()
    pred_score_tags = torch.tensor(pred_score_tags)
    true_score_tags = torch.tensor(true_score_tags)

    assert len(pred_tags) == len(true_tags)
    assert len(sent_data) == len(true_tags)

    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        precision_match, recall_match, f1_match = f1_score_match(true_match_tags, pred_match_tags, mode)
        metrics['f1'] = f1
        metrics['f1_match'] = f1_match
        # 计算 MSE
        loss = torch.nn.MSELoss()
        mse = loss(pred_score_tags, true_score_tags)
        metrics['MSE'] = mse
        # 计算 RMSE
        rmse = torch.sqrt(mse)
        metrics['RMSE'] = rmse
        # 计算 MAE
        loss = torch.nn.L1Loss()
        mae = loss(pred_score_tags, true_score_tags)
        metrics['MAE'] = mae
        r2score = R2Score()
        r2 = r2score(true_score_tags.view(-1), pred_score_tags.view(-1))
        metrics['r2'] = r2
    else:
        # bad_case(true_tags, pred_tags, sent_data)
        bad_case(true_match_tags, pred_match_tags, sent_data)
        # f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        # print(f1_score(true_tags, pred_tags, mode))
        f1_labels, f1, precision_labels, precision, recall_labels, recall = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
        metrics['precision_labels'] = precision_labels
        metrics['precision'] = precision
        metrics['recall_labels'] = recall_labels
        metrics['recall'] = recall

        # print(f1_score_match(true_match_tags, pred_match_tags, mode))
        precision_match, recall_match, f1_match = f1_score_match(true_match_tags, pred_match_tags, mode)
        metrics['f1_match'] = f1_match
        metrics['precision_match'] = precision_match
        metrics['recall_match'] = recall_match

        # 计算 MSE
        loss = torch.nn.MSELoss()
        mse = loss(pred_score_tags, true_score_tags)
        metrics['MSE'] = mse
        # 计算 RMSE
        rmse = torch.sqrt(mse)
        metrics['RMSE'] = rmse
        # 计算 MAE
        loss = torch.nn.L1Loss()
        mae = loss(pred_score_tags, true_score_tags)
        metrics['MAE'] = mae
        r2score = R2Score()
        r2 = r2score(true_score_tags.view(-1), pred_score_tags.view(-1))
        metrics['r2'] = r2

    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics

def predict_score(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(config.macbert_model, do_lower_case=True, skip_special_tokens=True)
    sent_data = []
    dev_losses = 0
    scores = []
    match = []
    with torch.no_grad():
        for idx, batch_samples in enumerate(tqdm(dev_loader)):
            batch_data, batch_token_starts, batch_tags, batch_labels_match, batch_labels_score = batch_samples
            sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                               if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks,
                         labels=[batch_tags, batch_labels_match, batch_labels_score])[0]
            dev_losses += loss.item()
            # shape: (batch_size, max_len, num_labels)
            batch_output, barch_output_match, batch_output_score, _ = model((batch_data, batch_token_starts),
                                                     token_type_ids=None, attention_mask=batch_masks)

            batch_output_score = batch_output_score.detach().cpu().numpy()
            barch_output_match = barch_output_match.detach().cpu().numpy()
            scores.append(batch_output_score)
            match_label = []
            for i in range(len(barch_output_match)):
                if barch_output_match[i][0] < barch_output_match[i][1]:
                    match_label.append(1)
                else:
                    match_label.append(0)
            match.append(match_label)
    return scores, match

def predict_vector(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(config.macbert_model, do_lower_case=True, skip_special_tokens=True)
    sent_data = []
    dev_losses = 0
    vectors = []
    with torch.no_grad():
        for idx, batch_samples in enumerate(tqdm(dev_loader)):
            batch_data, batch_token_starts, batch_tags, batch_labels_match, batch_labels_score = batch_samples
            sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                               if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks,
                         labels=[batch_tags, batch_labels_match, batch_labels_score])[0]
            dev_losses += loss.item()
            # shape: (batch_size, max_len, num_labels)
            batch_output, barch_output_match, batch_output_score, sequence_output = model((batch_data, batch_token_starts),
                                                     token_type_ids=None, attention_mask=batch_masks)

            sequence_output = sequence_output.detach().cpu().numpy()
            vectors.append(sequence_output)
    return vectors



def predict_label(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(tqdm(dev_loader)):
            # print('batch_samples:', batch_samples)
            batch_data, batch_token_starts = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks)[0]
            # dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output)
            # (batch_size, max_len)
            # batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            # (batch_size, max_len - padding_label_len)

    return pred_tags


if __name__ == "__main__":
    a = [101, 679, 6814, 8024, 517, 2208, 3360, 2208, 1957, 518, 7027, 4638,
         1957, 4028, 1447, 3683, 6772, 4023, 778, 8024, 6844, 1394, 3173, 4495,
         807, 4638, 6225, 830, 5408, 8024, 5445, 3300, 1126, 1767, 3289, 3471,
         4413, 4638, 2767, 738, 976, 4638, 3683, 6772, 1962, 511, 0, 0,
         0, 0, 0]
    t = torch.tensor(a, dtype=torch.long)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    word = tokenizer.convert_ids_to_tokens(t[1].item())
    sent = tokenizer.decode(t.tolist())
    print(word)
    print(sent)
