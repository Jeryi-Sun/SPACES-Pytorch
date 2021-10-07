import numpy as np
import json
from rouge import Rouge
import os, sys
import jieba
import six
from collections import defaultdict



# 自定义词典
user_dict_path = '/new_disk2/zhongxiang_sun/code/explanation_project/SPACES_torch/datasets/user_dict.txt'
user_dict_path_2 = '/new_disk2/zhongxiang_sun/code/explanation_project/SPACES_torch/datasets/user_dict_2.txt'
jieba.load_userdict(user_dict_path)
jieba.initialize()

# 设置递归深度
sys.setrecursionlimit(1000000)

# 标注数据
data_json = '/new_disk2/zhongxiang_sun/code/explanation_project/SPACES_torch/datasets/train.json'

# 保存权重的文件夹
if not os.path.exists('weights'):
    os.mkdir('weights')

# bert配置
pretrained_bert_fold = "/new_disk2/zhongxiang_sun/code/pretrain_model/bert_legal/"
pretrained_nezha_fold = "/new_disk2/zhongxiang_sun/code/pretrain_model/NEZHA/"
# nezha配置
nezha_config_path = '/root/kg/bert/nezha_base/bert_config.json'
nezha_checkpoint_path = '/root/kg/bert/nezha_base/model.ckpt-900000'
nezha_dict_path = '/root/kg/bert/nezha_base/vocab.txt'

# 将数据划分N份，一份作为验证集
num_folds = 15

# 指标名
metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

# 计算rouge用
rouge = Rouge()

def softmax(x, axis=-1):
    """numpy版softmax
    """
    x = x - x.max(axis=axis, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=axis, keepdims=True)

class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略
    """
    def __init__(self, start_id, end_id, maxlen,minlen=1, model=None, tokenizer=None):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.model = model
        self.tokenizer = tokenizer
        if start_id is None:
            self.first_output_ids = np.empty((1, 0), dtype=int)
        else:
            self.first_output_ids = np.array([[self.start_id]])

    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数
        目前包含：1. 设置rtype参数，并做相应处理；
                  2. 确定states的使用，并做相应处理；
                  3. 设置温度参数，并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(
                self,
                inputs,
                output_ids,
                states,
                temperature=1,
                rtype=default_rtype
            ):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)

                if not use_states:
                    prediction = (prediction, None)

                if default_rtype == 'logits':
                    prediction = (
                        softmax(prediction[0] / temperature), prediction[1]
                    )
                elif temperature != 1:
                    probas = np.power(prediction[0], 1.0 / temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == 'probas':
                    return prediction
                else:
                    return np.log(prediction[0] + 1e-12), prediction[1]

            return new_predict

        return actual_decorator



    def predict(self, inputs, output_ids, states=None):
        """用户需自定义递归预测函数
        说明：定义的时候，需要用wraps方法进行装饰，传入default_rtype和use_states，
             其中default_rtype为字符串logits或probas，probas时返回归一化的概率，
             rtype=logits时则返回softmax前的结果或者概率对数。
        返回：二元组 (得分或概率, states)
        """
        raise NotImplementedError

    def beam_search(self, inputs, topk, states=None, temperature=1, min_ends=1):
        """beam search解码
        说明：这里的topk即beam size；
        返回：最优解码序列。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids, output_scores = self.first_output_ids, np.zeros(1)
        for step in range(self.maxlen):
            scores, states = self.predict(
                inputs, output_ids, states, temperature, 'logits'
            )  # 计算当前得分
            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [np.repeat(i, topk, axis=0) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
            indices = scores.argpartition(-topk, axis=None)[-topk:]  # 仅保留topk
            indices_1 = indices // scores.shape[1]  # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = np.concatenate([output_ids[indices_1], indices_2],
                                        1)  # 更新输出
            output_scores = np.take_along_axis(
                scores, indices, axis=None
            )  # 更新得分
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best = output_scores.argmax()  # 得分最大的那个
                if is_end[best] and end_counts[best] >= min_ends:  # 如果已经终止
                    return output_ids[best]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        return output_ids[output_scores.argmax()]

    def random_sample(
        self,
        inputs,
        n,
        topk=None,
        topp=None,
        states=None,
        temperature=1,
        min_ends=1
    ):
        """随机采样n个结果
        说明：非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp
             表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        返回：n个解码序列组成的list。
        """
        inputs = [np.array([i]) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            probas, states = self.predict(
                inputs, output_ids, states, temperature, 'probas'
            )  # 计算当前概率
            probas /= probas.sum(axis=1, keepdims=True)  # 确保归一化
            if step == 0:  # 第1步预测后将结果重复n次
                probas = np.repeat(probas, n, axis=0)
                inputs = [np.repeat(i, n, axis=0) for i in inputs]
                output_ids = np.repeat(output_ids, n, axis=0)
            if topk is not None:
                k_indices = probas.argpartition(-topk,
                                                axis=1)[:, -topk:]  # 仅保留topk
                probas = np.take_along_axis(probas, k_indices, axis=1)  # topk概率
                probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
            if topp is not None:
                p_indices = probas.argsort(axis=1)[:, ::-1]  # 从高到低排序
                probas = np.take_along_axis(probas, p_indices, axis=1)  # 排序概率
                cumsum_probas = np.cumsum(probas, axis=1)  # 累积概率
                flag = np.roll(cumsum_probas >= topp, 1, axis=1)  # 标记超过topp的部分
                flag[:, 0] = False  # 结合上面的np.roll，实现平移一位的效果
                probas[flag] = 0  # 后面的全部置零
                probas /= probas.sum(axis=1, keepdims=True)  # 重新归一化
            sample_func = lambda p: np.random.choice(len(p), p=p)  # 按概率采样函数
            sample_ids = np.apply_along_axis(sample_func, 1, probas)  # 执行采样
            sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
            if topp is not None:
                sample_ids = np.take_along_axis(
                    p_indices, sample_ids, axis=1
                )  # 对齐原id
            if topk is not None:
                sample_ids = np.take_along_axis(
                    k_indices, sample_ids, axis=1
                )  # 对齐原id
            output_ids = np.concatenate([output_ids, sample_ids], 1)  # 更新输出
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                flag = is_end & (end_counts >= min_ends)  # 标记已完成序列
                if flag.any():  # 如果有已完成的
                    for ids in output_ids[flag]:  # 存好已完成序列
                        results.append(ids)
                    flag = (flag == False)  # 标记未完成序列
                    inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
                    output_ids = output_ids[flag]  # 只保留未完成部分候选集
                    end_counts = end_counts[flag]  # 只保留未完成部分end计数
                    if len(output_ids) == 0:
                        break
        # 如果还有未完成序列，直接放入结果
        for ids in output_ids:
            results.append(ids)
        # 返回结果
        return results

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """字符串转换为unicode格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, str):
            text = text.decode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


def convert_to_str(text, encoding='utf-8', errors='ignore'):
    """字符串转换为str格式（假设输入为utf-8格式）
    """

    if isinstance(text, bytes):
        text = text.decode(encoding, errors=errors)
    return text


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)

_open_ = open
is_py2 = six.PY2
class open:
    """模仿python自带的open函数
    作用：1.主要是为了同时兼容py2和py3；2.增加了索引功能，方便读取大文件。
    """
    def __init__(
        self, name, mode='r', encoding=None, errors='strict', indexable=False
    ):
        self.name = name
        if is_py2:
            self.file = _open_(name, mode)
        else:
            self.file = _open_(name, mode, encoding=encoding, errors=errors)
        self.encoding = encoding
        self.errors = errors
        self.iterator = None
        if indexable:
            if is_string(indexable) and os.path.exists(indexable):
                self.offsets = json.load(_open_(indexable))
            else:
                self.create_indexes()
                if is_string(indexable):
                    json.dump(self.offsets, _open_(indexable, 'w'))

    def create_indexes(self):
        print('creating indexes ...')
        self.offsets, offset = [], 0

        while self.readline():
            self.offsets.append(offset)
            offset = self.tell()
        self.seek(0)
        print('indexes created.')

    def __getitem__(self, key):
        self.seek(self.offsets[key])
        l = self.readline()
        if self.encoding:
            l = convert_to_unicode(l, self.encoding, self.errors)
        return l

    def __len__(self):
        return len(self.offsets)

    def __iter__(self):
        if hasattr(self, 'offsets'):
            for i in range(len(self)):
                yield self[i]
        else:
            for l in self.file:
                if self.encoding:
                    l = convert_to_unicode(l, self.encoding, self.errors)
                yield l

    def next(self):
        if self.iterator is None:
            self.iterator = self.__iter__()
        return next(self.iterator)

    def __next__(self):
        return self.next()

    def read(self):
        text = self.file.read()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def readline(self):
        text = self.file.readline()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def readlines(self):
        if self.encoding:
            return [
                convert_to_unicode(text, self.encoding, self.errors)
                for text in self.file.readlines()
            ]
        else:
            return self.file.readlines()

    def write(self, text):
        if self.encoding:
            text = convert_to_str(text, self.encoding, self.errors)
        self.file.write(text)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def tell(self):
        return self.file.tell()

    def seek(self, offset=0):
        return self.file.seek(offset)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()


def parallel_apply(
    func,
    iterable,
    workers,
    max_queue_size,
    callback=None,
    dummy=False,
    random_seeds=True,
    unordered=True
):
    """多进程或多线程地将func应用到iterable的每个元素中。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是
    输出可能是func(c), func(a), func(b)。
    参数：
        callback: 处理单个输出的回调函数；
        dummy: False是多进程/线性，True则是多线程/线性；
        random_seeds: 每个进程的随机种子；
        unordered: 若为False，则按照输入顺序返回，仅当callback为None时生效。
    """
    generator = parallel_apply_generator(
        func, iterable, workers, max_queue_size, dummy, random_seeds
    )

    if callback is None:
        if unordered:
            return [d for i, d in generator]
        else:
            results = sorted(generator, key=lambda d: d[0])
            return [d for i, d in results]
    else:
        for d in generator:
            callback(d)

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)

def parallel_apply_generator(
    func, iterable, workers, max_queue_size, dummy=False, random_seeds=True
):
    """多进程或多线程地将func应用到iterable的每个元素中。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是
    输出可能是func(c), func(a), func(b)。结果将作为一个
    generator返回，其中每个item是输入的序号以及该输入对应的
    处理结果。
    参数：
        dummy: False是多进程/线性，True则是多线程/线性；
        random_seeds: 每个进程的随机种子。
    """
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue

    in_queue, out_queue, seed_queue = Queue(max_queue_size), Queue(), Queue()
    if random_seeds is True:
        random_seeds = [None] * workers
    elif random_seeds is None or random_seeds is False:
        random_seeds = []
    for seed in random_seeds:
        seed_queue.put(seed)

    def worker_step(in_queue, out_queue):
        """单步函数包装成循环执行
        """
        if not seed_queue.empty():
            np.random.seed(seed_queue.get())
        while True:
            i, d = in_queue.get()
            r = func(d)
            out_queue.put((i, r))

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for i, d in enumerate(iterable):
        in_count += 1
        while True:
            try:
                in_queue.put((i, d), block=False)
                break
            except six.moves.queue.Full:
                for _ in range(out_queue.qsize()):
                    yield out_queue.get()
                    out_count += 1
        if in_count % max_queue_size == 0:
            for _ in range(out_queue.qsize()):
                yield out_queue.get()
                out_count += 1

    while out_count != in_count:
        for _ in range(out_queue.qsize()):
            yield out_queue.get()
            out_count += 1

    pool.terminate()

def text_segmentate(text, maxlen, seps='\n', strips=None):
    """将文本按照标点符号划分为若干个短句
    """
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def load_user_dict(filename):
    """加载用户词典
    """
    user_dict = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            w = l.split()[0]
            user_dict.append(w)
    return user_dict


def data_split(data, fold, num_folds, mode):
    """划分训练集和验证集
    """
    if mode == 'train':
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]

    if isinstance(data, np.ndarray):
        return np.array(D)
    else:
        return D


def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l
    """
    # if unit == 'word':
    #     source = jieba.cut(source, HMM=False)
    #     target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_metrics(source, target, unit='word'):
    """计算所有metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics


def compute_main_metric(source, target, unit='word'):
    """计算主要metric
    """
    return compute_metrics(source, target, unit)['main']


def longest_common_subsequence(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]