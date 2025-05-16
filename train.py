import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import build_vocab, QADataset, tokenize
from model import Encoder, Decoder, Seq2Seq


# 超参数设置
VOCAB_MIN_FREQ = 1
MAX_LEN = 100
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 32
EPOCHS = 70  # 调整训练轮次以观察损失变化趋势
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5


# 数据路径
Q_PATH = 'data/Q.txt'
A_PATH = 'data/A.txt'


def main():
    # 1. 加载并预处理数据
    # 从问题文件中读取数据，每行作为一个问题
    with open(Q_PATH, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f.readlines()]
    # 从答案文件中读取数据，每行作为一个答案
    with open(A_PATH, 'r', encoding='utf-8') as f:
        answers = [line.strip() for line in f.readlines()]

    # 构建词汇表（合并问题与答案的分词结果）
    # 对所有问题和答案进行分词，然后构建词汇表
    all_tokens = [tokenize(q) for q in questions] + [tokenize(a) for a in answers]
    vocab = build_vocab(all_tokens, min_freq=VOCAB_MIN_FREQ)
    vocab_size = len(vocab) + 1  # 包含未登录词（索引为len(vocab)）

    # 创建数据集和数据加载器
    # 使用自定义的数据集类和数据加载器来处理数据
    dataset = QADataset(Q_PATH, A_PATH, vocab, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 初始化模型
    # 初始化编码器、解码器和序列到序列模型
    encoder = Encoder(vocab_size, EMBED_DIM, HIDDEN_DIM)
    decoder = Decoder(vocab_size, EMBED_DIM, HIDDEN_DIM)
    model = Seq2Seq(encoder, decoder).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. 定义损失函数和优化器
    # 使用交叉熵损失函数，并忽略填充词的损失计算
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练循环
    # 开始训练模型，包括前向传播、计算损失、反向传播和参数更新
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (src, trg) in enumerate(train_loader):
            src = src.to(model.device)
            trg = trg.to(model.device)

            optimizer.zero_grad()
            output = model(src, trg, TEACHER_FORCING_RATIO)

            # 计算损失（忽略trg的最后一个token，因为output长度是trg_len-1）
            loss = criterion(output.view(-1, vocab_size), trg[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}')

    # 保存词汇表和模型
    # 保存训练好的词汇表和模型参数，以便后续使用
    torch.save(vocab, 'vocab.pth')
    # 将词汇表写入txt文件（word: index格式）
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        for word, idx in vocab.items():
            f.write(f'{word}: {idx}')
    torch.save(model.state_dict(), 'seq2seq_model.pth')
    print('词汇表（pth/txt格式）和模型已保存')


if __name__ == '__main__':
    main()