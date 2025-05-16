import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        # 定义词嵌入层，将词汇索引映射为密集向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 定义LSTM层，接收嵌入后的输入并输出隐藏状态
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell  # outputs: (batch_size, seq_len, hidden_dim)


class Attention(nn.Module):
    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch_size, hidden_dim) -> 调整维度为(batch_size, 1, hidden_dim)
        hidden = hidden.permute(1, 0, 2)
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        energy = torch.matmul(hidden, encoder_outputs.transpose(1, 2))  # (batch_size, 1, seq_len)
        attention_weights = F.softmax(energy, dim=2)  # (batch_size, 1, seq_len)
        context = torch.bmm(attention_weights, encoder_outputs)  # (batch_size, 1, hidden_dim)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention()
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: (batch_size, 1)  当前时间步输入
        embedded = self.embedding(x)  # (batch_size, 1, embed_dim)
        context, _ = self.attention(hidden, encoder_outputs)  # context: (batch_size, 1, hidden_dim)
        input_with_context = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed_dim+hidden_dim)
        output, (hidden, cell) = self.lstm(input_with_context, (hidden, cell))  # output: (batch_size, 1, hidden_dim)
        prediction = self.fc(output)  # (batch_size, 1, vocab_size)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 动态获取设备（优先GPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features

        # 初始化输出存储
        outputs = torch.zeros(batch_size, trg_len-1, vocab_size).to(src.device)

        # 编码器前向传播
        encoder_outputs, hidden, cell = self.encoder(src)

        # 初始输入为<SOS>（即trg的第一个token）
        input = trg[:, 0:1]  # (batch_size, 1)

        for t in range(1, trg_len):
            # 解码器前向传播
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t-1, :] = output.squeeze(1)

            # Teacher Forcing机制
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = trg[:, t:t+1] if teacher_force else top1

        return outputs

    def generate(self, src, vocab, max_len=50, beam_size=3):
        src = src.to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        sos_idx = vocab.get("<SOS>", 1)
        eos_idx = vocab.get("<EOS>", 2)
        # 初始化beam：(sequence, score, hidden, cell)
        beams = [(torch.LongTensor([[sos_idx]]).to(self.device), 0.0, hidden, cell)]
        final_sequences = []

        for _ in range(max_len):
            new_beams = []
            for seq, score, h, c in beams:
                if seq[0, -1] == eos_idx:
                    final_sequences.append((seq, score))
                    continue
                # 解码当前token
                output, new_h, new_c = self.decoder(seq[:, -1:], h, c, encoder_outputs)
                probs = F.log_softmax(output, dim=2).squeeze(1)
                top_probs, top_indices = probs.topk(beam_size)
                # 扩展候选序列
                for i in range(beam_size):
                    next_token = top_indices[0, i].item()
                    new_score = score + top_probs[0, i].item()
                    new_seq = torch.cat([seq, torch.LongTensor([[next_token]]).to(self.device)], dim=1)
                    new_beams.append((new_seq, new_score, new_h, new_c))
            # 按分数排序并保留前beam_size个候选
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

        # 选择分数最高的完整序列
        if final_sequences:
            final_sequences.sort(key=lambda x: x[1], reverse=True)
            return final_sequences[0][0].tolist()[0]
        # 若没有完整序列，返回最长候选
        return beams[0][0].tolist()[0]

if __name__ == '__main__':
    # 测试模型结构
    vocab_size = 1000
    embed_dim = 256
    hidden_dim = 512
    encoder = Encoder(vocab_size, embed_dim, hidden_dim)
    decoder = Decoder(vocab_size, embed_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder)

    # 模拟输入
    src = torch.randint(0, vocab_size, (32, 20))  # (batch_size=32, src_len=20)
    trg = torch.randint(0, vocab_size, (32, 25))  # (batch_size=32, trg_len=25)
    output = model(src, trg)
    print(f'输出形状: {output.shape}')  # 应输出(32, 24, 1000)（trg_len-1）