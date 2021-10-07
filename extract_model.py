import datetime
import json

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from snippets import *
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--epoch_num', type=int, default=20, help='number of epochs')
parser.add_argument('--each_test_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='decay weight of optimizer')
parser.add_argument('--model_name', type=str, default='bert', help='matching model')
parser.add_argument('--checkpoint', type=str, default="./checkpoint/", help='checkpoint path')
parser.add_argument('--max_length', type=int, default=512, help='max length of each case')
parser.add_argument('--input_size', type=int, default=768)
parser.add_argument('--hidden_size', type=int, default=384)
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.3)
parser.add_argument('--cuda_pos', type=str, default='1', help='which GPU to use')
parser.add_argument('--seed', type=int, default=42, help='max length of each case')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

log_name = "log_train"
logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='./logs/{}.log'.format(log_name),
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )


# 配置信息

data_extract_json = data_json[:-5] + '_extract.json'
data_extract_npy = data_json[:-5] + '_extract.npy'

device = torch.device('cuda:'+args.cuda_pos) if torch.cuda.is_available() else torch.device('cpu')


if len(sys.argv) == 1:
    fold = 0
else:
    fold = int(sys.argv[1])


def load_checkpoint(model, optimizer, trained_epoch):
    filename = args.checkpoint + '/' + f"extract-{trained_epoch}.pkl"
    save_params = torch.load(filename)
    model.load_state_dict(save_params["model"])
    #optimizer.load_state_dict(save_params["optimizer"])

def save_checkpoint(model, optimizer, trained_epoch):
    save_params = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
    }
    if not os.path.exists(args.checkpoint):
        # 判断文件夹是否存在，不存在则创建文件夹
        os.mkdir(args.checkpoint)
    filename = args.checkpoint + '/' + f"extract-{trained_epoch}.pkl"
    torch.save(save_params, filename)

def load_data(filename):
    """加载数据
    返回：[(texts, labels, summary)]
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            D.append(json.loads(l))
    return D


class ResidualGatedConv1D(nn.Module):
    """门控卷积
    """
    def __init__(self, filters, kernel_size, dilation_rate=1):
        super(ResidualGatedConv1D, self).__init__()
        self.filters = filters  # 输出维度
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.supports_masking = True
        self.padding = self.dilation_rate*(self.kernel_size - 1)//2
        self.conv1d = nn.Conv1d(filters, 2*filters, self.kernel_size, padding=self.padding, dilation=self.dilation_rate)
        self.layernorm = nn.LayerNorm(self.filters)
        self.alpha = nn.Parameter(torch.zeros(1))


    def forward(self, inputs):
        input_cov1d = inputs.permute([0, 2, 1])
        outputs = self.conv1d(input_cov1d)
        outputs = outputs.permute([0, 2, 1])
        gate = torch.sigmoid(outputs[..., self.filters:])
        outputs = outputs[..., :self.filters] * gate
        outputs = self.layernorm(outputs)

        if hasattr(self, 'dense'):
            inputs = self.dense(inputs)

        return inputs + self.alpha * outputs


class Selector2(nn.Module):
    def __init__(self, input_size, filters, kernel_size, dilation_rate):
        """
        :param feature_size:每个词向量的长度
        """
        super(Selector2, self).__init__()
        self.dense1 = nn.Linear(input_size, filters, bias=False)
        self.ResidualGatedConv1D_1 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[0])
        self.ResidualGatedConv1D_2 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[1])
        self.ResidualGatedConv1D_3 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[2])
        self.ResidualGatedConv1D_4 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[3])
        self.ResidualGatedConv1D_5 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[4])
        self.ResidualGatedConv1D_6 = ResidualGatedConv1D(filters, kernel_size, dilation_rate=dilation_rate[5])
        self.dense2 = nn.Linear(filters, 1)


    def forward(self, inputs):
        mask = inputs.ge(0.00001)
        mask = torch.sum(mask, axis=-1).bool()
        x1 = self.dense1(nn.Dropout(0.1)(inputs))
        x2 = self.ResidualGatedConv1D_1(nn.Dropout(0.1)(x1))
        x3 = self.ResidualGatedConv1D_2(nn.Dropout(0.1)(x2))
        x4 = self.ResidualGatedConv1D_3(nn.Dropout(0.1)(x3))
        x5 = self.ResidualGatedConv1D_4(nn.Dropout(0.1)(x4))
        x6 = self.ResidualGatedConv1D_5(nn.Dropout(0.1)(x5))
        x7 = self.ResidualGatedConv1D_6(nn.Dropout(0.1)(x6))
        output = nn.Sigmoid()(self.dense2(nn.Dropout(0.1)(x7)))
        return output, mask



class Selector_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        super(Selector_Dataset, self).__init__()
        self.data_x_tensor = torch.from_numpy(data_x)
        self.data_y_tensor = torch.from_numpy(data_y)
    def __len__(self):
        return len(self.data_x_tensor)
    def __getitem__(self, idx):
        return self.data_x_tensor[idx], self.data_y_tensor[idx]




def train(model, train_dataloader, valid_dataloader):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss(reduction='none')
    for epoch in range(args.epoch_num):
        epoch_loss = 0.0
        current_step = 0
        model.train()
        pbar = tqdm(train_dataloader, desc="Iteration", postfix='train')
        for batch_data in pbar:
            x_batch, label_batch = batch_data
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            output_batch, batch_mask = model(x_batch)
            output_batch = output_batch.permute([0, 2, 1])
            loss = criterion(output_batch.squeeze(), label_batch.squeeze())
            loss = torch.div(torch.sum(loss*batch_mask), torch.sum(batch_mask))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        save_checkpoint(model, optimizer, epoch)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            current_step = 0
            pbar = tqdm(valid_dataloader, desc="Iteration", postfix='valid')
            for batch_data in pbar:
                x_batch, label_batch = batch_data
                x_batch = x_batch.to(device)
                label_batch = label_batch.to(device).long()
                output_batch, batch_mask = model(x_batch)
                label_batch = label_batch.to(device)
                total += torch.sum(batch_mask)
                vec_correct = ((output_batch.squeeze()>args.threshold).long() == label_batch.squeeze().long())*batch_mask
                correct += torch.sum(vec_correct).cpu().item()
                pbar.set_description("valid acc {}".format(correct / total))
                current_step += 1
                if current_step % 100 == 0:
                    logging.info('valid epoch {} acc {}/{}={:.4f}'.format(epoch, correct, total, correct / total))
            time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('{} valid epoch {} acc {}/{}={:.4f}'.format(time_str, epoch, correct, total, correct / total))
            logging.info('valid epoch {} acc {}/{}={:.4f}'.format(epoch, correct, total, correct / total))


if __name__ == '__main__':

    # 加载数据
    data = load_data(data_extract_json)
    data_x = np.load(data_extract_npy)
    data_y = np.zeros_like(data_x[..., :1])

    for i, d in enumerate(data):
        for j in d[1]:
            data_y[i, j] = 1

    train_data = data_split(data, fold, num_folds, 'train')
    valid_data = data_split(data, fold, num_folds, 'valid')
    train_x = data_split(data_x, fold, num_folds, 'train')
    valid_x = data_split(data_x, fold, num_folds, 'valid')
    train_y = data_split(data_y, fold, num_folds, 'train')
    valid_y = data_split(data_y, fold, num_folds, 'valid')

    train_dataloader = DataLoader(Selector_Dataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(Selector_Dataset(valid_x, valid_y), batch_size=len(valid_x), shuffle=False)

    model = Selector2(args.input_size, args.hidden_size, kernel_size=args.kernel_size, dilation_rate=[1, 2, 4, 8, 1, 1])

    train(model, train_dataloader, valid_dataloader)



