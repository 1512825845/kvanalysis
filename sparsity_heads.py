import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型和tokenizer
model_name = "/home/tempuser/models/meta-llama/Llama-3.1-8B-Instruct"  # 可替换为Llama-2/Phi等
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
model.set_attn_implementation('eager')
model.eval()

# 2. 运行推理，获取注意力权重
datasets = ["narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
    "qasper_e",
    "multifieldqa_en_e",
    "hotpotqa_e",
    "2wikimqa_e",
    "gov_report_e",
    "multi_news_e",
    "trec_e",
    "triviaqa_e",
    "samsum_e",
    "passage_count_e",
    "passage_retrieval_en_e",
    "lcc_e",
    "repobench-p_e"]
subset = datasets[1]  # 选择第一个数据集进行测试
dataset = load_dataset("/home/tempuser/Dataset/THUDM/LongBench", subset, split="test")
sample = dataset[0]
text = sample["input"]  # LongBench样本的输入文本
inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attn_data = [attn.detach().cpu().numpy() for attn in outputs.attentions]

# 5. 计算稀疏性指标
def calculate_sparsity(attn_weights, threshold=1e-2):
    """
    计算注意力矩阵的稀疏性指标
    :param attn_weights: 注意力权重 [num_heads, seq_len, seq_len]
    :param threshold: 判定为"零"的阈值
    :return: 每个头的指标列表
    """
    seq_len = attn_weights.shape[1]
    # 生成causal mask: 下三角为有效（1），上三角为mask（0）
    mask = np.tril(np.ones((seq_len, seq_len)))
    
    heads_data = []
    
    for head_idx, head in enumerate(attn_weights):  # 遍历每个注意力头
        head_sparsity = []
        head_entropy = []
        for row_idx, row in enumerate(head):  # each row is a distribution over keys
            row = np.nan_to_num(row, nan=0, posinf=0, neginf=0)
            row = np.maximum(row, 0)  # ensure non-negative
            
            # 获取该行的有效位置（非mask）
            mask_row = mask[row_idx]
            valid_positions = mask_row == 1
            valid_count = np.sum(valid_positions)
            
            # 1. 稀疏度：有效位置中低于阈值的元素占比
            if valid_count > 0:
                zero_count = np.sum((row < threshold) & valid_positions)
                sparsity = zero_count / valid_count
            else:
                sparsity = 0
            head_sparsity.append(sparsity)
            
            # 2. 熵值（归一化到0-1）
            if len(row) <= 1:
                entropy = 0.0
            else:
                row_prob = row + 1e-10  # 避免log(0)
                row_prob = row_prob / np.sum(row_prob)
                entropy = -np.sum(row_prob * np.log2(row_prob)) / np.log2(len(row))  # 归一化
                if not np.isfinite(entropy):
                    entropy = 0.0
            head_entropy.append(entropy)
        
        # 3. Top10占比 for whole head (排除mask位置)
        flat_head = head.flatten()
        flat_mask = mask.flatten()
        valid_head_data = flat_head[flat_mask == 1]  # 只保留有效位置
        if len(valid_head_data) > 0:
            top10 = np.sort(valid_head_data)[-10:]
            top10_ratio = np.sum(top10) / np.sum(valid_head_data)
        else:
            top10_ratio = 0.0
        
        heads_data.append({
            "head_idx": head_idx,
            "avg_sparsity": float(np.mean(head_sparsity)),
            "avg_entropy": float(np.mean(head_entropy)),
            "top10_ratio": float(top10_ratio)
        })
    
    return heads_data

# 遍历各层计算指标
layer_data = {}
for layer_idx, weights in enumerate(attn_data):
    # weights shape: [batch, num_heads, seq_len, seq_len] → 取第一个batch
    layer_data[layer_idx] = calculate_sparsity(weights[0])

# 保存每层的信息到文件
with open("/home/tempuser/sqm/attention_sparsity_per_head.json", "w") as f:
    json.dump(layer_data, f, indent=4)

print("数据已保存到 /home/tempuser/sqm/attention_sparsity_per_head.json")

# 6. 可视化注意力矩阵（以第一层第一个头为例）
first_layer_weights = attn_data[15][0][0]  # layer 0, batch 0, head 0
plt.figure(figsize=(8, 6))
plt.imshow(first_layer_weights, cmap="hot", interpolation="nearest")
plt.colorbar(label="Attention Score")
plt.title("Attention Matrix (Layer 16, Head 1)")
plt.xlabel("Key Token")
plt.ylabel("Query Token")
plt.xticks(range(len(inputs.input_ids[0])), tokenizer.convert_ids_to_tokens(inputs.input_ids[0]), rotation=45)
plt.yticks(range(len(inputs.input_ids[0])), tokenizer.convert_ids_to_tokens(inputs.input_ids[0]), rotation=45)
plt.tight_layout()
plt.savefig("/home/tempuser/sqm/attention_matrix_layer16_head1.png")