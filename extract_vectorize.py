#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 抽取式：句向量化
# 科学空间：https://kexue.fm

import json
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoModel, AutoTokenizer
from snippets import *
import torch.nn as nn
import torch



class GlobalAveragePooling1D(nn.Module):
    """自定义全局池化
    对一个句子的pooler取平均，一个长句子用短句的pooler平均代替
    """
    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()


    def forward(self, inputs, mask=None):
        if mask is not None:
            mask = mask.to(torch.float)[:, :, None]
            return torch.sum(inputs * mask, dim=1) / torch.sum(mask, dim=1)
        else:
            return torch.mean(inputs, dim=1)


class Selector_1(nn.Module):
    def __init__(self):
        super(Selector_1, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_fold, mirror='tuna', do_lower_case=True)
        self.Pooling = GlobalAveragePooling1D()
        self.encoder = BertModel.from_pretrained(pretrained_bert_fold)
        self.max_seq_len = 512


    def predict(self, texts):
        """句子列表转换为句向量
        """
        with torch.no_grad():
            bert_output = self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True, max_length=self.max_seq_len, return_tensors="pt")
            output_1 = self.encoder(**bert_output)["last_hidden_state"]
            outputs = self.Pooling(output_1)
        return outputs



def load_data(filename):
    """加载数据
    返回：[texts]
    """
    D = []
    with open(filename) as f:
        for l in f:
            texts = json.loads(l)[0]
            D.append(texts)
    return D




def convert(data):
    """转换所有样本
    """
    embeddings = []
    model = Selector_1()
    for texts in tqdm(data, desc=u'向量化'):
        outputs = model.predict(texts)
        embeddings.append(outputs)
    embeddings = sequence_padding(embeddings)
    return embeddings


if __name__ == '__main__':

    data_extract_json = data_json[:-5] + '_extract.json'
    data_extract_npy = data_json[:-5] + '_extract'

    data = load_data(data_extract_json)
    embeddings = convert(data)
    np.save(data_extract_npy, embeddings)
    print(u'输出路径：%s.npy' % data_extract_npy)
