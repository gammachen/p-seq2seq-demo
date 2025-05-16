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
        """
        定义前向传播函数，处理输入数据通过LSTM网络
    
        参数:
        x: 输入的数据张量，形状为(batch_size, seq_len)，其中batch_size是批次大小，seq_len是序列长度
    
        返回:
        outputs: LSTM的输出，形状为(batch_size, seq_len, hidden_dim)，其中hidden_dim是隐藏层维度
        hidden: 最后一层LSTM的隐藏状态，用于后续处理或解码
        cell: 最后一层LSTM的细胞状态，同样用于后续处理或解码
        """
        # 使用嵌入层将输入x转换为嵌入表示
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
    
        # 将嵌入表示输入到LSTM中，获取输出、隐藏状态和细胞状态
        outputs, (hidden, cell) = self.lstm(embedded)
    
        # 返回LSTM的输出、最后的隐藏状态和细胞状态
        return outputs, hidden, cell  # outputs: (batch_size, seq_len, hidden_dim)


class Attention(nn.Module):
    def forward(self, hidden, encoder_outputs):
        """
        前向传播函数，计算注意力权重和上下文向量
        
        参数:
        hidden: 解码器隐藏状态, 形状为 (1, batch_size, hidden_dim)
        encoder_outputs: 编码器所有时间步的输出, 形状为 (batch_size, seq_len, hidden_dim)
        
        返回:
        context: 上下文向量, 形状为 (batch_size, 1, hidden_dim)
        attention_weights: 注意力权重, 形状为 (batch_size, 1, seq_len)
        """
        # 调整隐藏状态的维度，以便进行矩阵操作
        hidden = hidden.permute(1, 0, 2)
        
        # 计算能量值，用于确定注意力的焦点
        energy = torch.matmul(hidden, encoder_outputs.transpose(1, 2))
        
        # 应用softmax函数，将能量值转换为注意力权重
        attention_weights = F.softmax(energy, dim=2)
        
        # 计算上下文向量，作为注意力机制的输出之一
        context = torch.bmm(attention_weights, encoder_outputs)
        
        # 返回上下文向量和注意力权重
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        """
        初始化Decoder类的实例。
    
        参数:
        - vocab_size (int): 词汇表大小，用于确定输出维度。
        - embed_dim (int): 词嵌入的维度。
        - hidden_dim (int): LSTM隐藏层的维度。
        """
        # 初始化父类
        super(Decoder, self).__init__()
        
        # 词嵌入层，将词汇表大小的索引映射到词嵌入维度的向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM层，接受词嵌入和隐藏状态的拼接输入，输出隐藏状态
        # 输入维度为词嵌入维度加上隐藏层维度，因为我们将使用注意力机制合并它们
        # 输出维度为隐藏层维度
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        
        # 注意力机制，用于计算当前隐藏状态的加权输入
        self.attention = Attention()
        
        # 全连接层，将LSTM的输出映射到词汇表大小的维度，为每个词汇分配一个分数
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        """
        前向传播函数，用于解码器端的信号处理。
        
        参数:
        - x: (batch_size, 1) 当前时间步输入
        - hidden: (batch_size, 1, hidden_dim) 隐藏状态
        - cell: (batch_size, 1, hidden_dim) LSTM的细胞状态
        - encoder_outputs: (batch_size, seq_len, hidden_dim) 编码器的全部输出
        
        返回:
        - prediction: (batch_size, 1, vocab_size) 对输出词汇的预测分布
        - hidden: (batch_size, 1, hidden_dim) 更新后的隐藏状态
        - cell: (batch_size, 1, hidden_dim) 更新后的细胞状态
        """
        # 使用嵌入层将输入x转换为嵌入表示
        embedded = self.embedding(x)  # (batch_size, 1, embed_dim)
        
        # 使用注意力机制计算当前隐藏状态与编码器全部输出的上下文向量
        context, _ = self.attention(hidden, encoder_outputs)  # context: (batch_size, 1, hidden_dim)
        
        # 将嵌入表示和上下文向量在第三个维度（特征维度）上拼接
        input_with_context = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed_dim+hidden_dim)
        
        # 将拼接后的输入与当前的隐藏状态和细胞状态一起传入LSTM，得到输出、新的隐藏状态和细胞状态
        output, (hidden, cell) = self.lstm(input_with_context, (hidden, cell))  # output: (batch_size, 1, hidden_dim)
        
        # 使用全连接层对LSTM的输出进行变换，得到最终的预测分布
        prediction = self.fc(output)  # (batch_size, 1, vocab_size)
        
        # 返回预测分布、新的隐藏状态和细胞状态
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    class Seq2Seq(nn.Module):
        """
        初始化Seq2Seq模型。
    
        参数:
        - encoder: 编码器对象，用于对输入序列进行编码。
        - decoder: 解码器对象，用于对编码后的信息进行解码，生成输出序列。
    
        该构造函数主要完成以下任务：
        1. 调用父类的构造方法以初始化nn.Module类。
        2. 初始化编码器和解码器。
        3. 根据设备可用性（CPU或CUDA），设置模型运行的设备。
        """
        def __init__(self, encoder, decoder):
            # 调用父类的构造方法，初始化nn.Module类
            super(Seq2Seq, self).__init__()
            # 初始化编码器和解码器
            self.encoder = encoder
            self.decoder = decoder
            # 动态获取设备（优先GPU）
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        前向传播函数
        :param src: 输入源序列 (batch_size, src_len)
        :param trg: 目标序列 (batch_size, trg_len)，用于计算损失和Teacher Forcing
        :param teacher_forcing_ratio: 使用Teacher Forcing的概率，默认为0.5
        :return: 网络的输出序列 (batch_size, trg_len-1, vocab_size)
        """
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

    def generate(self, src, vocab, max_len=50):
        """
        使用编码器-解码器架构生成序列。
    
        参数:
        src (Tensor): 输入序列的张量。
        vocab (dict): 词汇表，键是词元，值是它们的索引。
        max_len (int): 生成序列的最大长度，默认为50。
    
        返回:
        list: 生成的序列，由词元索引组成。
        """
        # 将输入序列转移到适当的设备
        src = src.to(self.device)
        # 通过编码器获取输入序列的编码输出、隐藏状态和单元状态
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # 获取开始符和结束符的索引
        sos_idx = vocab.get("<SOS>", 1)
        eos_idx = vocab.get("<EOS>", 2)
        
        # 初始化输入为开始符
        input = torch.LongTensor([[sos_idx]]).to(self.device)
        generated = []
        
        # 循环生成序列，直到达到最大长度或生成结束符
        for _ in range(max_len):
            # 通过解码器生成下一个词元的输出、更新隐藏状态和单元状态
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            
            # 获取最可能的下一个词元的索引
            top1 = output.argmax(2)  # (1,1)
            generated_idx = top1.item()
            generated.append(generated_idx)
            
            # 如果生成了结束符，则停止生成
            if generated_idx == eos_idx:
                break
            
            # 将当前生成的词元作为下一步的输入
            input = top1
        
        # 返回生成的序列
        return generated

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