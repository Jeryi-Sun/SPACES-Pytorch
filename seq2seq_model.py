import datetime
import os, json
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
import argparse
import torch
from transformers import AdamW
import torch.nn as nn
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
import logging
from snippets import *
from bert_seq2seq import Tokenizer, load_chinese_base_vocab
from bert_seq2seq import load_bert
# 基本参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--each_test_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='nezha', help='matching model')
parser.add_argument('--checkpoint', type=str, default="./checkpoint/", help='checkpoint path')
parser.add_argument('--bert_maxlen', type=int, default=512, help='max length of each case')
parser.add_argument('--maxlen', type=int, default=1024, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=384)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--k_sparse', type=int, default=10)
parser.add_argument('--cuda_pos', type=str, default='0', help='which GPU to use')
parser.add_argument('--seed', type=int, default=42, help='max length of each case')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')
log_name = "log_train"
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='./logs/{}.log'.format(log_name),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )



data_seq2seq_json = data_json[:-5] + '_seq2seq.json'
seq2seq_config_json = data_json[:-10] + 'seq2seq_config.json'

if len(sys.argv) == 1:
    fold = 0
else:
    fold = int(sys.argv[1])


def load_data(filename):
    """加载数据
    返回：[{...}]
    """
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D




def generate_copy_labels(source, target):
    """构建copy机制对应的label
    """
    mapping = longest_common_subsequence(source, target)[1]
    source_labels = [0] * len(source)
    target_labels = [0] * len(target)
    i0, j0 = -2, -2
    for i, j in mapping:
        if i == i0 + 1 and j == j0 + 1:
            source_labels[i] = 2
            target_labels[j] = 2
        else:
            source_labels[i] = 1
            target_labels[j] = 1
        i0, j0 = i, j
    return source_labels, target_labels


def random_masking(token_ids_all):
    """对输入进行随机mask，增加泛化能力
    """
    result = []
    for token_ids in token_ids_all:
        rands = np.random.random(len(token_ids))
        result.append([
            t if r > 0.15 else np.random.choice(token_ids)
            for r, t in zip(rands, token_ids)
        ])
    return result


class DataGenerator(Dataset):
    def __init__(self, input_data, random=True):
        super(DataGenerator, self).__init__()
        self.input_data = input_data
        self.random = random

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):

        i = np.random.choice(2) + 1 if self.random else 1
        source, target = self.input_data[idx]['source_%s' % i], self.input_data[idx]['target']
        return [source, target]


class Collate:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)

        self.max_seq_len = args.maxlen

    def __call__(self, batch):
        # assert len(A_batch) == 1
        # print("A_batch: ", A_batch)
        dic_data = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True,
                                                      max_length=self.max_seq_len)
        mask_dic_data = copy.deepcopy(dic_data)

        token_ids = dic_data["input_ids"]

        masked_token_ids = random_masking(token_ids)
        mask_dic_data['input_ids'] = masked_token_ids
        labels = []
        for item_masked_token_ids, item_token_ids in zip(masked_token_ids, token_ids):
            idx = item_token_ids.index(self.tokenizer.sep_token_id) + 1
            source_labels, target_labels = generate_copy_labels(
                item_masked_token_ids[:idx], item_token_ids[idx:]
            )
            """
            [CLS]...[SEP]   ... [SEP]
            """
            labels.append(source_labels[1:] + target_labels)  # 因为是预测所以第一位后移


        return torch.tensor(dic_data["input_ids"]), torch.tensor(dic_data["token_type_ids"]), torch.tensor(labels)



def build_pretrain_dataloader(data, batch_size, shuffle=True, num_workers=0,):
    data_generator =DataGenerator(data, random=True)
    collate = Collate()
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )


def compute_seq2seq_loss(predictions, token_type_id, input_ids, vocab_size):

    predictions = predictions[:, :-1].contiguous()
    target_mask = token_type_id[:, 1:].contiguous()
    """
       target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
    """
    predictions = predictions.view(-1, vocab_size)
    labels = input_ids[:, 1:].contiguous()
    labels = labels.view(-1)
    target_mask = target_mask.view(-1).float()
    # 正loss
    pos_loss = predictions[list(range(predictions.shape[0])), labels]
    # 负loss
    y_pred = torch.topk(predictions, k=args.k_sparse)[0]
    neg_loss = torch.logsumexp(y_pred, dim=-1)

    loss = neg_loss - pos_loss
    return (loss * target_mask).sum() / target_mask.sum()  ## 通过mask 取消 pad 和句子a部分预测的影响


def compute_copy_loss(predictions, token_type_id, labels):
    predictions = predictions[:, :-1].contiguous()
    target_mask = token_type_id[:, 1:].contiguous()
    """
       target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
    """
    predictions = predictions.view(-1, 3)
    labels = labels.view(-1)
    target_mask = target_mask.view(-1).float()
    loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    return (loss(predictions, labels) * target_mask).sum() / target_mask.sum()  ## 通过mask 取消 pad 和句子a部分预测的影响

class GenerateModel(nn.Module):
    def __init__(self):
        super(GenerateModel, self).__init__()
        self.word2idx = load_chinese_base_vocab(pretrained_nezha_fold+"vocab.txt", simplfied=False)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_nezha_fold)
        self.bert_model = load_bert(self.word2idx, model_name=args.model_name, model_class="seq2seq")
        ## 加载预训练的模型参数～
        self.bert_model.load_pretrain_params(pretrained_nezha_fold+"pytorch_model.bin")
        self.bert_model.set_device(device)
        self.configuration = self.bert_model.config
        self.linear = nn.Linear(self.configuration.hidden_size, 3)

    def forward(self, token_ids, token_type_ids):
        seq2seq_predictions,  hidden_state = self.bert_model(token_ids, token_type_ids)
        copy_predictions = self.linear(nn.GELU()(hidden_state))

        return seq2seq_predictions, copy_predictions


def load_checkpoint(model, optimizer, trained_epoch):
    filename = args.checkpoint + '/' + f"seq2seq-{trained_epoch}.pkl"
    save_params = torch.load(filename)
    model.load_state_dict(save_params["model"])
    optimizer.load_state_dict(save_params["optimizer"])


def save_checkpoint(model, optimizer, trained_epoch):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    if not os.path.exists(args.checkpoint):
        # 判断文件夹是否存在，不存在则创建文件夹
        os.mkdir(args.checkpoint)
    filename = args.checkpoint + '/' + f"seq2seq-{trained_epoch}.pkl"
    torch.save(save_params, filename)


def train_valid(train_data, valid_data, model):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ema = EMA(model, 0.9999)
    # ema.register()
    for epoch in range(args.epochs):
        epoch_loss = 0.
        current_step = 0
        model.train()
        # for batch_data in tqdm(train_data_loader, ncols=0):
        pbar = tqdm(train_data, desc="Iteration", postfix='train')
        for batch_data in pbar:
            input_ids, token_type_ids, labels = batch_data
            input_ids, token_type_ids, labels = input_ids.to(device), token_type_ids.to(device), labels.to(device)
            seq2seq_predictions, copy_predictions = model(input_ids, token_type_ids)

            seq2seq_loss = compute_seq2seq_loss(seq2seq_predictions, token_type_ids, input_ids,
                                                model.configuration.vocab_size)
            copy_loss = compute_copy_loss(copy_predictions, token_type_ids, labels)
            loss = seq2seq_loss + 2 * copy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.update()
            loss_item = loss.cpu().detach().item()
            epoch_loss += loss_item
            current_step += 1
            pbar.set_description("train loss {}".format(epoch_loss / current_step))
            if current_step % 100 == 0:
                logging.info("train step {} loss {}".format(current_step, epoch_loss / current_step))

        epoch_loss = epoch_loss / current_step
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} train epoch {} loss: {:.4f}'.format(time_str, epoch, epoch_loss))
        logging.info('train epoch {} loss: {:.4f}'.format(epoch, epoch_loss))
        # todo 看一下 EMA是否会让模型准确率提升，如果可以的话在保存模型前加入 ema
        save_checkpoint(model, optimizer, epoch)
        with torch.no_grad():
            model.eval()
            # ema.apply_shadow()
            evaluate(valid_data, model, filename=r'./result/abstract.txt')
            # ema.restore()
        model.train()

class AutoSummary(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='logits', use_states=True)
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        seq2seq_predictions, copy_predictions = self.model(torch.tensor(token_ids, device=device), torch.tensor(segment_ids, device=device))
        prediction = [seq2seq_predictions[:, -1].cpu().numpy(), torch.softmax(copy_predictions[:, -1], dim=-1).cpu().numpy()]  # 返回最后一个字符的预测结果，（1， vocab_size）,(1, 3) todo 我这里需要加一个softmax 前面的生成模型给也需要
        # states用来缓存ngram的n值
        if states is None:
            states = [0]
        elif len(states) == 1 and len(token_ids) > 1:
            states = states * len(token_ids)
        # 根据copy标签来调整概率分布
        probas = np.zeros_like(prediction[0]) - 1000  # 最终要返回的概率分布
        for i, token_ids in enumerate(inputs[0]):
            if states[i] == 0:
                prediction[1][i, 2] *= -1  # 0不能接2
            label = prediction[1][i].argmax()  # 当前label
            if label < 2:
                states[i] = label
            else:
                states[i] += 1  # 2后面接什么都行
            if states[i] > 0:
                ngrams = self.get_ngram_set(token_ids, states[i])
                prefix = tuple(output_ids[i, 1 - states[i]:])
                if prefix in ngrams:  # 如果确实是适合的ngram
                    candidates = ngrams[prefix]
                else:  # 没有的话就退回1gram
                    ngrams = self.get_ngram_set(token_ids, 1)
                    candidates = ngrams[tuple()]
                    states[i] = 1
                candidates = list(candidates)
                probas[i, candidates] = prediction[0][i, candidates]
            else:
                probas[i] = prediction[0][i]
            idxs = probas[i].argpartition(-args.k_sparse)
            probas[i, idxs[:-args.k_sparse]] = -1000
        return probas, states

    def generate(self, text, topk=1):
        max_c_len = args.maxlen - self.maxlen
        encode_text = self.model.tokenizer(text, padding=True, truncation=True,
                                         max_length=max_c_len)
        token_ids, segment_ids = encode_text['input_ids'], encode_text['token_type_ids']
        output_ids = self.beam_search([token_ids, segment_ids],
                                       topk)  # 基于beam search
        return ''.join(self.model.tokenizer.convert_ids_to_tokens(output_ids))





def evaluate(data, model, topk=1, filename=None):
    """验证集评估
    """
    autosummary = AutoSummary(
        start_id=model.tokenizer.cls_token_id,
        end_id=model.tokenizer.sep_token_id,
        maxlen=args.maxlen // 2,
        model=model
    )
    if filename is not None:
        F = open(filename, 'w', encoding='utf-8')
    total_metrics = {k: 0.0 for k in metric_keys}
    for d in tqdm(data, desc=u'评估中'):
        pred_summary = autosummary.generate(d['source_1'], topk)
        metrics = compute_metrics(pred_summary, d['target'])
        for k, v in metrics.items():
            total_metrics[k] += v
        if filename is not None:
            F.write(d['target'] + '\t' + pred_summary + '\n')
            F.flush()
    if filename is not None:
        F.close()
    print(total_metrics)
    return {k: v / len(data) for k, v in total_metrics.items()}

if __name__ == '__main__':
    # 加载数据
    data = load_data(data_seq2seq_json)
    train_data = data_split(data, fold, num_folds, 'train')
    valid_data = data_split(data, fold, num_folds, 'valid')
    train_data_loader = build_pretrain_dataloader(train_data, args.batch_size)
    G_model = GenerateModel()
    print(G_model)
    G_model = G_model.to(device)
    train_valid(train_data_loader, valid_data, G_model)




