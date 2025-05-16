import torch
import jieba
from collections import Counter


def tokenize(text):
    """使用jieba进行中文分词"""
    return list(jieba.cut(text))


def build_vocab(data, min_freq=5):
    """
    构建词汇表（合并问题与答案的词汇）
    
    遍历数据集，统计每个词出现的频率，并过滤掉出现次数小于min_freq的词
    为词汇表添加特殊符号：<PAD>, <SOS>, <EOS>，分别表示填充、序列开始和序列结束
    
    参数:
    data: 二维列表，包含问题和答案的数据集
    min_freq: int，词汇最小出现频率，默认为5
    
    返回:
    vocab: 字典，词到索引的映射
    """
    # 初始化计数器
    counter = Counter()
    # 遍历数据集，更新词频
    for line in data:
        counter.update(line)
    # 过滤低频词，保留出现次数大于等于min_freq的词
    filtered_words = [word for word, cnt in counter.items() if cnt >= min_freq]
    # 构建词到索引的映射（保留<PAD>/<SOS>/<EOS>位置）
    vocab = {word: idx+1 for idx, word in enumerate(filtered_words)}
    # 添加填充符号，索引为0
    vocab["<PAD>"] = 0
    # 添加序列开始符号，索引为当前词汇表长度
    vocab["<SOS>"] = len(vocab)
    # 添加序列结束符号，索引为当前词汇表长度
    vocab["<EOS>"] = len(vocab)
    # 返回词汇表
    return vocab


def text_to_seq(text, vocab, max_len=50):
    """
    将文本转换为索引序列（带填充/截断）。

    参数:
    text (list of str): 分词后的文本列表。
    vocab (dict): 词汇表，将单词映射到其索引。
    max_len (int): 最大序列长度，默认为50。

    返回:
    list of int: 转换后的索引序列，包括填充或截断。
    """
    # 将文本中的每个词转换为词汇表中的索引，未登录词用vocab长度表示
    seq = [vocab.get(word, len(vocab)) for word in text]
    
    # 根据max_len进行截断或填充
    if len(seq) > max_len:
        # 如果序列长度大于max_len，则截断到max_len
        return seq[:max_len]
    else:
        # 如果序列长度小于max_len，则用<PAD>的索引填充到max_len
        return seq + [vocab["<PAD>"]] * (max_len - len(seq))


class QADataset(torch.utils.data.Dataset):
    """自定义问答数据集类"""
    def __init__(self, q_path, a_path, vocab, max_len=50):
        """
        初始化函数
        
        读取问题和答案文件，并对每行文本进行分词处理。同时，存储词汇表和最大长度参数。
        
        参数:
        - q_path: 问题文件的路径，文件中的每行文本代表一个问题
        - a_path: 答案文件的路径，文件中的每行文本代表一个答案
        - vocab: 词汇表，用于后续处理（具体用途未在代码片段中展示）
        - max_len: 最大长度，默认值为50，用于限制文本序列的最大长度
        """
        # 读取并处理问题文件
        with open(q_path, 'r', encoding='utf-8') as f:
            self.questions = [tokenize(line.strip()) for line in f.readlines()]
        # 读取并处理答案文件
        with open(a_path, 'r', encoding='utf-8') as f:
            self.answers = [tokenize(line.strip()) for line in f.readlines()]
        # 存储词汇表
        self.vocab = vocab
        # 存储最大长度参数
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q_seq = text_to_seq(self.questions[idx], self.vocab, self.max_len)
        a_seq = text_to_seq(["<SOS>"] + self.answers[idx] + ["<EOS>"], self.vocab, self.max_len)
        return torch.tensor(q_seq), torch.tensor(a_seq)