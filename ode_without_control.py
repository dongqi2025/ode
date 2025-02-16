import json
import os
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn, optim
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchdiffeq import odeint
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 导入 tqdm

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 指定本地模型路径
local_model_path = "/Users/dongqi/PycharmProjects/dq/数据质量/all-MiniLM-L12-v2"
config = AutoConfig.from_pretrained(local_model_path, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True, clean_up_tokenization_spaces=True)
# 使用配置加载模型
model = AutoModel.from_pretrained(local_model_path, config=config, local_files_only=True)

def get_bert_embedding(texts, max_len=128, layer_index=-1):
    """
    提取DistilBERT第 layer_index 层的embedding (seq维度mean).
    """
    batch_size_bert = 32
    all_emb = []
    for start_i in tqdm(range(0, len(texts), batch_size_bert), desc="BERT Embedding Extraction"):
        sub_text = texts[start_i: start_i + batch_size_bert]
        inputs = tokenizer(sub_text, return_tensors='pt',
                           padding=True, truncation=True,
                           max_length=max_len).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple
            chosen_layer = hidden_states[layer_index]  # [batch, seq_len, dim]
            emb = chosen_layer.mean(dim=1)  # mean => [batch, dim]
            all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)

questions = []
answers = []
# 定义文件路径
# file_path = "/Users/dongqi/PycharmProjects/dq/paper/data/commensense_qa/train-00000-of-00001.parquet"

# 读取 Parquet 文件
# print("开始读取 Parquet 文件...")
# data = pd.read_parquet(file_path)
# unique_questions_df = data.drop_duplicates(subset='question')
# print("完成读取 Parquet 文件。")
#
# # 查看数据的基本信息
# print(unique_questions_df.info())
# print(unique_questions_df.shape)
# # 查看数据集行数和列数
# rows, columns = data.shape

file_path="/Users/dongqi/PycharmProjects/dq/paper/data/trivia_qa/train-00000-of-00001 .parquet"
data = pd.read_parquet(file_path)
unique_questions_df = data.drop_duplicates(subset='question')
# questions = []
# answers = []
# corrections = []
# file_path = "/Users/dongqi/PycharmProjects/dq/paper/data/GammaCorpus-fact-qa-fact-qa-450k/gammacorpus-fact-qa-450k.jsonl"
# with open(file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         data = json.loads(line)
#         questions.append(data["question"])
#         answers.append(data["answer"])
questions = unique_questions_df.iloc[:, 0].tolist()
answers = unique_questions_df.iloc[:, -1].map(lambda d: d.get('value')).tolist()
# options_column = unique_questions_df.iloc[:, -2]
# answer_labels = unique_questions_df.iloc[:, -1].tolist()


# 定义提取答案文本的函数
def extract_answer_text(options_dict, answer_label):
    """
    从选项字典中提取与答案标签对应的文本内容。
    """
    labels = options_dict.get('label', [])
    texts = options_dict.get('text', [])
    # 将 numpy array 转换为列表
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    # 创建标签到文本的映射
    label_to_text = dict(zip(labels, texts))
    return label_to_text.get(answer_label, None)

# 应用函数提取答案文本列表
# answers = unique_questions_df.apply(
#     lambda row: extract_answer_text(row.iloc[-2], row.iloc[-1]),
#     axis=1
# ).tolist()
# answers = unique_questions_df.iloc[:, 3].tolist()

# 仅处理5000个数据样本
sample_size = 5000
questions_raw = questions[:sample_size]
answers_raw = answers[:sample_size]
print("开始拆分训练集和测试集...")
q_train_text, q_test_text, a_train_text, a_test_text = train_test_split(
    questions_raw, answers_raw, test_size=0.2, random_state=42
)
print("完成拆分训练集和测试集。")
print("开始提取训练集和测试集的BERT嵌入...")
trainQ_emb = get_bert_embedding(q_train_text, layer_index=-1)
trainA_emb = get_bert_embedding(a_train_text, layer_index=-1)
testQ_emb = get_bert_embedding(q_test_text, layer_index=-1)
testA_emb = get_bert_embedding(a_test_text, layer_index=-1)
print("完成提取BERT嵌入。")
trainQ, trainA = trainQ_emb.to(device), trainA_emb.to(device)
testQ, testA = testQ_emb.to(device), testA_emb.to(device)
print("TrainQ:", trainQ.shape)
print("TestQ: ", testQ.shape)

class ODEFunc(nn.Module):
    def __init__(self, dim=384):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, dim)
        )
    def forward(self, t, y):
        return self.net(y)

def mini_batches(q_tensor, a_tensor, batch_size=32):
    idxs = torch.randperm(len(q_tensor)).to(device)
    for start_i in range(0, len(q_tensor), batch_size):
        excerpt = idxs[start_i:start_i + batch_size]
        yield q_tensor[excerpt], a_tensor[excerpt]

def train_ode(model, trainQ, trainA, testQ, testA,
              epochs=10, steps_t=10, lr=1e-4, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    t_span = torch.linspace(0., 1., steps_t).to(device)
    train_losses, val_losses = [], []
    for ep in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_train_loss = 0.0
        batch_iter = tqdm(mini_batches(trainQ, trainA, batch_size), desc=f"Epoch {ep + 1} Training", leave=False)
        for qbatch, abatch in batch_iter:
            optimizer.zero_grad()
            z_traj = odeint(model, qbatch, t_span)  # [steps_t, batch, dim]
            pred_final = z_traj[-1]
            loss = criterion(pred_final, abatch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            batch_iter.set_postfix(loss=loss.item())
        avg_train_loss = total_train_loss * batch_size / len(trainQ)
        model.eval()
        total_val_loss = 0.0
        val_batch_iter = tqdm(mini_batches(testQ, testA, batch_size), desc=f"Epoch {ep + 1} Validation", leave=False)
        with torch.no_grad():
            for qbatch, abatch in val_batch_iter:
                z_traj = odeint(model, qbatch, t_span)
                pred_final = z_traj[-1]
                vloss = criterion(pred_final, abatch)
                total_val_loss += vloss.item()
                val_batch_iter.set_postfix(loss=vloss.item())
        avg_val_loss = total_val_loss * batch_size / len(testQ)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {ep + 1}/{epochs}, Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
    return train_losses, val_losses

# 初始化ODE模型
ode_model = ODEFunc(dim=384).to(device)
print("开始训练ODE模型...")
train_losses, val_losses = train_ode(ode_model, trainQ, trainA, testQ, testA,
                                     epochs=30, steps_t=10, lr=1e-3, batch_size=32)
print("完成训练ODE模型。")
# 训练曲线
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Neural ODE Training")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
# plt.show()
save_path = '/Users/dongqi/PycharmProjects/dq/paper/new_res/'+file_path.split("/")[-2]
# 如果路径不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)
full_path = os.path.join(save_path, 'Learning Curves: Training and Validation Loss.png')
plt.savefig(full_path)
print(f"保存训练曲线到 {full_path}。")

loss_txt_path=os.path.join(save_path, 'loss.txt')
with open(loss_txt_path,"a+") as f:
    f.write("train_lose:"+str(train_losses[-1])+";"+"val_losse:"+str(val_losses[-1]))

def get_all_trajectories(model, testQ, steps_t=10, batch_size=32, num_samples=200):
    """
    一次性收集所有测试样本在 t=0..1 (steps_t steps) 的隐状态
    返回 arr_3d: shape [N, steps_t, dim]
    """
    model.eval()
    t_span = torch.linspace(0., 1., steps_t).to(device)
    arr_list = []
    # 只选取前 num_samples 个测试样本
    testQ = testQ[:num_samples]
    with torch.no_grad():
        for start_i in tqdm(range(0, len(testQ), batch_size), desc="Collecting Trajectories"):
            batch_inp = testQ[start_i: start_i + batch_size]  # [bs, dim]
            z_traj = odeint(model, batch_inp, t_span)  # [steps_t, bs, dim]
            # 转为 [bs, steps_t, dim]
            z_traj = z_traj.permute(1, 0, 2).cpu().numpy()
            arr_list.append(z_traj)
    # 拼接 => [N, steps_t, dim]
    return np.concatenate(arr_list, axis=0)

def visualize_all_samples_pca(trajectories_3d, steps_t=10, alpha=0.2):
    """
    trajectories_3d: [N, steps_t, dim]
    在一张图上绘制每个样本的多时刻PCA轨迹
    alpha: 轨迹不透明度, 避免图像过于拥挤
    """
    N = trajectories_3d.shape[0]
    dim = trajectories_3d.shape[2]
    # reshape => [N*steps_t, dim]
    arr_2d = trajectories_3d.reshape(N * steps_t, dim)
    # PCA
    print("进行PCA降维...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(arr_2d)  # shape [N*steps_t, 2]

    # 分别画N条折线
    idx_offset = 0
    plt.figure(figsize=(10, 6))
    for i in tqdm(range(N), desc="Plotting Trajectories"):
        sub_c = coords[idx_offset: idx_offset + steps_t]
        idx_offset += steps_t
        # plot
        # 大量样本 => alpha=0.2
        plt.plot(sub_c[:, 0], sub_c[:, 1], '-', color='blue', alpha=alpha)
        # 标注起点/终点
        plt.scatter(sub_c[0, 0], sub_c[0, 1], c='yellow', edgecolors='k', marker='o', s=30, alpha=alpha)
        plt.scatter(sub_c[-1, 0], sub_c[-1, 1], c='cyan', edgecolors='k', marker='X', s=30, alpha=alpha)
    plt.title("Input-to-Output Transformation Diagram")
    plt.grid(True)
    # plt.show()
    full_path = os.path.join(save_path, 'Input-to-Output Transformation Diagram.png')
    plt.savefig(full_path)
    print(f"保存轨迹图到 {full_path}。")

# 生成全部测试样本轨迹（随机抽取200个样本点绘制）
print("开始收集测试集轨迹...")
test_traj_3d = get_all_trajectories(ode_model, testQ, steps_t=10, batch_size=32, num_samples=150)
print("test_traj_3d shape:", test_traj_3d.shape)  # e.g. [N, steps_t, dim]

# 画图
print("开始绘制PCA轨迹图...")
visualize_all_samples_pca(test_traj_3d, steps_t=10, alpha=0.3)
print("完成绘制PCA轨迹图。")