import json
import torch
from flask import Flask, request, jsonify
import jieba
from model import Encoder, Decoder, Seq2Seq

# 初始化Flask应用
app = Flask(__name__)

# 加载预训练模型和词汇表（示例路径，需根据实际训练路径调整）
# 注意：实际使用时需替换为真实的模型和词汇表加载逻辑
vocab = torch.load('vocab.pth')  # 假设词汇表已保存
vocab_size = len(vocab) + 1  # 与训练时一致，包含未登录词（索引为len(vocab)）
embed_dim = 256  # 需与训练时的参数一致
hidden_dim = 512  # 需与训练时的参数一致
# 初始化模型结构
encoder = Encoder(vocab_size, embed_dim, hidden_dim)
decoder = Decoder(vocab_size, embed_dim, hidden_dim)
model = Seq2Seq(encoder, decoder)
# 加载预训练参数
model.load_state_dict(torch.load('seq2seq_model.pth'))
model.eval()

# 数据预处理函数
def tokenize(text):
    return list(jieba.cut(text))

def text_to_seq(text, vocab, max_len=50):
    seq = [vocab.get(word, vocab.get("<UNK>", 0)) for word in text]
    return seq[:max_len] + [vocab["<PAD>"]] * (max_len - len(seq))

@app.route('/qa', methods=['POST'])
def qa():
    # 获取用户输入的问题
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'error': '问题不能为空'}), 400

    # 预处理问题
    tokens = tokenize(question)
    input_seq = text_to_seq(tokens, vocab)
    input_tensor = torch.LongTensor([input_seq])

    # 模型推理生成答案
    with torch.no_grad():
        # 假设model.generate为自定义的生成函数（需根据实际模型实现调整）
        answer_indices = model.generate(input_tensor, vocab)

    # 将索引转换为文本
    idx2word = {v: k for k, v in vocab.items()}
    answer_tokens = [idx2word[idx] for idx in answer_indices if idx in idx2word and idx not in [vocab["<PAD>"], vocab["<EOS>"]]]
    answer = ''.join(answer_tokens)

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 生产环境应关闭debug模式