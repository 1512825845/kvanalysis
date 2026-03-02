import torch
import numpy as np
from typing import List, Set, Dict, Tuple
import re
import json

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class AttentionHeadKVManager:
    """
    注意力头KV Cache管理器：按流式头/检索头（多sinks/chunk/full cache）差异化管理KV Cache
    """
    def __init__(
        self,
        num_heads: int,
        streaming_heads: List[int],       # 流式头索引列表
        multi_sinks_heads: List[int],     # 多sinks头索引列表
        chunk_heads: List[int],           # chunk头索引列表
        stream_window_size: int = 4,   # 流式头窗口大小（StreamingLLM）
        top_k: int = 4,                  # 注意力分数选Top-K token作为c1/c2
        punctuation: str = r'[.！？；,，。!?;]',  # 分割窗口的标点正则
        debug: bool = False  # 调试模式：打印窗口分割和逐出信息
    ):
        self.num_heads = num_heads
        self.stream_window_size = stream_window_size
        self.top_k = top_k
        self.punctuation = punctuation
        self.debug = debug
        
        # 1. 校验注意力头分类（无重叠、无遗漏）
        all_heads = set(range(num_heads))
        streaming_set = set(streaming_heads)
        multi_sinks_set = set(multi_sinks_heads)
        chunk_set = set(chunk_heads)
        full_cache_set = all_heads - streaming_set - multi_sinks_set - chunk_set
        
        # 检查重叠
        overlap = streaming_set & multi_sinks_set | streaming_set & chunk_set | multi_sinks_set & chunk_set
        if overlap:
            raise ValueError(f"注意力头分类重叠：{overlap}")
        # 检查遗漏
        if streaming_set | multi_sinks_set | chunk_set | full_cache_set != all_heads:
            raise ValueError("注意力头分类遗漏，请检查索引")
        
        # 2. 保存各类头的索引
        self.head_type: Dict[str, Set[int]] = {
            "streaming": streaming_set,
            "multi_sinks": multi_sinks_set,
            "chunk": chunk_set,
            "full_cache": full_cache_set
        }
        
        # 3. 初始化KV Cache（shape: [num_heads, seq_len, hidden_dim]）
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0  # 当前缓存的token长度
        self.first_token = True  # 是否是首token（所有头保留）
        
        # 4. 多sinks/chunk头的窗口状态
        self.window_history: Dict = {
            "prev_window": [],    # 上一个可变窗口的token索引
            "prev_c1": set(),     # 上一个窗口选的token集合c1
            "current_window": []  # 当前可变窗口的token索引
        }

    def _split_tokens_by_punctuation(self, tokens: List[int]) -> List[List[int]]:
        """
        按标点符号分割token序列为可变窗口（sentenceKV方式）
        Args:
            tokens: 输入token序列（索引形式）
        Returns:
            分割后的窗口列表，每个窗口是[起始索引, 结束索引]
        """
        # 模拟token转文本（实际场景需替换为真实tokenizer的decode）
        # 这里简化：假设token 100+是标点，仅用于演示分割逻辑
        token_text = [str(t) if t < 100 else "。" for t in tokens]
        text = "".join(token_text)
        
        # 按标点分割位置
        split_positions = [0]
        for match in re.finditer(self.punctuation, text):
            split_positions.append(match.end())
        split_positions.append(len(text))
        
        # 生成窗口（token索引范围）
        windows = []
        for i in range(len(split_positions)-1):
            start = split_positions[i]
            end = split_positions[i+1]
            if start < end:  # 跳过空窗口
                windows.append([start, end])
        return windows

    def _compute_jaccard_similarity(self, set1: Set[int], set2: Set[int]) -> float:
        """计算两个集合的Jaccard相似度"""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0.0

    def _select_topk_tokens(self, attn_scores: torch.Tensor, window: List[int]) -> Set[int]:
        """
        从指定窗口内选择注意力分数Top-K的token集合
        Args:
            attn_scores: 注意力分数，shape [num_heads, seq_len, seq_len]
            window: 窗口范围 [start, end]
        Returns:
            Top-K token的索引集合
        """
        start, end = window
        # 取窗口内的注意力分数（均值作为token的最终分数）
        window_scores = attn_scores[:, start:end].mean(dim=0).flatten()  # Flatten to 1D
        # 选Top-K
        topk_indices = torch.topk(window_scores, min(self.top_k, window_scores.shape[0])).indices
        # 转换为全局索引
        return set(topk_indices.cpu().numpy().tolist())

    def update_kv_cache(
        self,
        k: torch.Tensor,       # 新输入K，shape [num_heads, 1, hidden_dim]（单步推理）
        v: torch.Tensor,       # 新输入V，shape [num_heads, 1, hidden_dim]
        attn_scores: torch.Tensor  # 注意力分数，shape [num_heads, seq_len+1, seq_len+1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV Cache，按不同头类型执行差异化策略
        Args:
            k/v: 单步推理的新K/V（batch=1，仅新增1个token）
            attn_scores: 包含新token的注意力分数矩阵
        Returns:
            更新后的K/V Cache，shape [num_heads, new_seq_len, hidden_dim]
        """
        # 1. 初始化/扩展KV Cache
        if self.k_cache is None:
            self.k_cache = k  # 首token
            self.v_cache = v
            self.first_token = False
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=1)
            self.v_cache = torch.cat([self.v_cache, v], dim=1)
        self.seq_len = self.k_cache.shape[1]

        # 2. 按头类型处理KV Cache
        # 2.1 流式头：StreamingLLM逻辑（保留最近N个token，首token保留）
        streaming_mask = torch.zeros(self.num_heads, self.seq_len, dtype=torch.bool)
        for head_idx in self.head_type["streaming"]:
            # 保留首token + 最近stream_window_size个token
            keep_indices = [0] + list(range(max(1, self.seq_len - self.stream_window_size), self.seq_len))
            streaming_mask[head_idx, keep_indices] = True

        # 2.2 Full Cache头：保留所有token
        full_cache_mask = torch.ones(self.num_heads, self.seq_len, dtype=torch.bool)
        for head_idx in self.head_type["full_cache"]:
            full_cache_mask[head_idx] = True

        # 2.3 多sinks/chunk头：动态窗口+Jaccard相似度逐出
        dynamic_mask = torch.ones(self.num_heads, self.seq_len, dtype=torch.bool)
        # 分割当前token序列为可变窗口
        token_indices = list(range(self.seq_len))
        windows = self._split_tokens_by_punctuation(token_indices)
        if windows:
            self.window_history["current_window"] = windows[-1]  # 取最后一个窗口作为当前窗口
            
            if self.debug and len(windows) > 1:
                print(f"  [DEBUG] 窗口分割: {len(windows)} 个窗口，当前窗口: {self.window_history['current_window']}")
            
            # 有上一个窗口时计算Jaccard相似度
            if self.window_history["prev_window"]:
                # 选择c1（上一窗口Top-K）和c2（当前窗口Top-K）
                c1 = self.window_history["prev_c1"]
                c2 = self._select_topk_tokens(attn_scores, self.window_history["current_window"])
                
                # 计算阈值
                len_c1 = len(c1)
                len_current_win = self.window_history["current_window"][1] - self.window_history["current_window"][0]
                threshold1 = len_c1 / (len_c1 + len_current_win) if (len_c1 + len_current_win) != 0 else 0.0
                threshold2 = 1 / (len_c1 + len_current_win) if (len_c1 + len_current_win) != 0 else 0.0
                
                # 计算Jaccard相似度
                jaccard = self._compute_jaccard_similarity(c1, c2)
                
                if self.debug:
                    print(f"  [DEBUG] Jaccard相似度: {jaccard:.3f}, 阈值: [{threshold2:.3f}, {threshold1:.3f}]")
                
                # 执行逐出逻辑
                evict_count = 0
                for head_idx in self.head_type["multi_sinks"] | self.head_type["chunk"]:
                    if jaccard >= threshold1:
                        # 逐出上一窗口中不在c1的token
                        evict_indices = [idx for idx in range(*self.window_history["prev_window"]) if idx not in c1 and idx < self.seq_len]
                        if evict_indices:
                            dynamic_mask[head_idx, evict_indices] = False
                            evict_count += len(evict_indices)
                    elif jaccard <= threshold2:
                        # 逐出上一窗口所有token
                        evict_indices = [idx for idx in range(*self.window_history["prev_window"]) if idx < self.seq_len]
                        if evict_indices:
                            dynamic_mask[head_idx, evict_indices] = False
                            evict_count += len(evict_indices)
                    # 其他情况：不逐出
                
                if self.debug and evict_count > 0:
                    print(f"  [DEBUG] 逐出token数: {evict_count}")
                
            # 更新窗口历史
            self.window_history["prev_window"] = self.window_history["current_window"]
            self.window_history["prev_c1"] = self._select_topk_tokens(attn_scores, self.window_history["current_window"])

        # 3. 合并所有头的mask（保留首token）
        final_mask = torch.zeros(self.num_heads, self.seq_len, dtype=torch.bool)
        # 流式头mask
        final_mask[list(self.head_type["streaming"])] = streaming_mask[list(self.head_type["streaming"])]
        # Full Cache头mask
        final_mask[list(self.head_type["full_cache"])] = full_cache_mask[list(self.head_type["full_cache"])]
        # 多sinks/chunk头mask - 这些头的mask可以进行压缩
        final_mask[list(self.head_type["multi_sinks"] | self.head_type["chunk"])] = dynamic_mask[list(self.head_type["multi_sinks"] | self.head_type["chunk"])]
        # 强制保留首token（所有头）
        final_mask[:, 0] = True

        # 4. 应用mask裁剪KV Cache - 按头类型分别裁剪
        # 关键改进：计算每个头类型实际保留的tokens，按最严格的mask（non-full-cache头）进行全局裁剪
        self.k_cache = self.k_cache * final_mask.unsqueeze(-1)
        self.v_cache = self.v_cache * final_mask.unsqueeze(-1)
        
        # 只删除那些所有non-full-cache头都不需要的位置
        # 这样full-cache头可能会得到fewer tokens（不会为了它们而额外保留）
        non_full_cache_heads = list(self.head_type["streaming"] | self.head_type["multi_sinks"] | self.head_type["chunk"])
        if non_full_cache_heads:
            # 检查non-full-cache头的需求
            non_full_cache_mask = final_mask[non_full_cache_heads]  # [non_full_cache_heads, seq_len]
            positions_needed_by_non_full = non_full_cache_mask.any(dim=0)  # 任何non-full-cache头需要
            
            # 删除所有non-full-cache头都不需要的位置
            self.k_cache = self.k_cache[:, positions_needed_by_non_full]
            self.v_cache = self.v_cache[:, positions_needed_by_non_full]
        else:
            # 如果没有non-full-cache头，使用原来的逻辑
            self.k_cache = self.k_cache[:, final_mask.any(dim=0)]
            self.v_cache = self.v_cache[:, final_mask.any(dim=0)]
        
        # 保存各头类型的mask，用于统计（保存在裁剪前的原始mask）
        self._last_masks = {}
        if self.head_type["streaming"]:
            self._last_masks["streaming"] = streaming_mask[list(self.head_type["streaming"])]
        if self.head_type["multi_sinks"]:
            self._last_masks["multi_sinks"] = dynamic_mask[list(self.head_type["multi_sinks"])]
        if self.head_type["chunk"]:
            self._last_masks["chunk"] = dynamic_mask[list(self.head_type["chunk"])]
        if self.head_type["full_cache"]:
            self._last_masks["full_cache"] = full_cache_mask[list(self.head_type["full_cache"])]

        return self.k_cache, self.v_cache

    def get_head_cache_stats(self) -> Dict[str, Dict[str, any]]:
        """
        获取不同头类型的KV Cache统计信息（按头类型分别管理）
        Returns:
            Dict containing cache stats for each head type
        """
        if self.k_cache is None:
            return {}
        
        stats = {}
        total_cache_size = 0
        
        # 获取最后一次的mask（用于计算每个头类型实际保留的tokens）
        # 注意：这里需要在update_kv_cache中保存mask
        
        for head_type_name, head_indices in self.head_type.items():
            if not head_indices:
                continue
                
            head_list = list(head_indices)
            k_shape = self.k_cache[head_list].shape
            v_shape = self.v_cache[head_list].shape
            cache_size = k_shape[0] * k_shape[1] * k_shape[2] + v_shape[0] * v_shape[1] * v_shape[2]
            total_cache_size += cache_size
            
            stats[head_type_name] = {
                "k_shape": str(k_shape),
                "v_shape": str(v_shape),
                "num_heads": len(head_list),
                "cache_elements": k_shape[1]  # token个数
            }
        
        stats["_total_cache_elements"] = self.k_cache.shape[1]
        return stats
    
    def get_per_head_type_tokens(self) -> Dict[str, int]:
        """
        获取每个头类型在当前mask下实际需要保留的token数量
        这反映了如果按头类型分别存储，会有多少tokens
        """
        if hasattr(self, '_last_masks'):
            masks = self._last_masks
            stats = {}
            for head_type_name, head_indices in self.head_type.items():
                if not head_indices or head_type_name == "full_cache":
                    continue
                if head_type_name in masks and masks[head_type_name].numel() > 0:
                    mask = masks[head_type_name]  # [num_heads_in_type, seq_len]
                    tokens_kept = mask.any(dim=0).sum().item()  # 任何头需要的位置
                    stats[head_type_name] = tokens_kept
            return stats
        return {}

# ------------------------------ 测试示例 ------------------------------
if __name__ == "__main__":
    # 选择使用真实文本或随机数据
    use_real_text = HAS_TRANSFORMERS
    
    if use_real_text:
        print("使用真实文本模拟推理...")
        # 加载tokenizer和模型（使用小型模型以节省内存）
        try:
            model_name = "/home/tempuser/models/meta-llama/Llama-3.1-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, output_attentions=True)
            model.eval()
            device = "cuda:0" #if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            # 真实文本示例
            sample_text = """
            The quick brown fox jumps over the lazy dog. This is a test sentence for modeling attention patterns.
            Attention mechanisms help the model focus on relevant parts of the input sequence.
            Understanding how different attention heads work is crucial for interpreting transformer models.
            """
            
            # Tokenize文本
            tokens = tokenizer.encode(sample_text, return_tensors="pt").to(device)
            
            # 初始化管理器
            num_heads = model.config.num_attention_heads
            num_layers = model.config.num_hidden_layers
            kv_manager = AttentionHeadKVManager(
                num_heads=num_heads,
                streaming_heads=list(range(num_heads//4)),  # 前1/4的头为streaming
                multi_sinks_heads=list(range(num_heads//4, num_heads//2)),  # 中间1/4为multi_sinks
                chunk_heads=list(range(num_heads//2, 3*num_heads//4)),  # 后1/4中前1/2为chunk
                stream_window_size=4,
                top_k=4,
                debug=True  # 启用调试输出
            )
            
            print(f"模型: {model_name}")
            print(f"Attention heads: {num_heads}")
            print(f"Token数量: {tokens.shape[1]}")
            print(f"Token序列: {tokenizer.decode(tokens[0])[:100]}...")
            
            # 打印头的分配
            print(f"\n头类型分配:")
            print(f"  Streaming heads: {list(kv_manager.head_type['streaming'])}")
            print(f"  Multi-sinks heads: {list(kv_manager.head_type['multi_sinks'])}")
            print(f"  Chunk heads: {list(kv_manager.head_type['chunk'])}")
            print(f"  Full-cache heads: {list(kv_manager.head_type['full_cache'])}")
            
            # 逐token推理（模拟单步解码）
            hidden_dim = model.config.hidden_size
            for step in range(min(tokens.shape[1], 128)):  # 限制在20tokens以节省时间
                # 取当前和之前的所有tokens
                input_ids = tokens[:, :step+1]
                
                # 前向传播获取注意力
                with torch.no_grad():
                    outputs = model(input_ids, output_attentions=True)
                    # 取最后一层的注意力
                    attentions = outputs.attentions[-1]  # [batch, num_heads, seq_len, seq_len]
                
                # 模拟该步的K、V（使用隐层状态的一部分）
                hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
                # 模拟K、V（实际应从attention层获取，这里简化处理）
                k_step = hidden_states[:, step:step+1, :].view(1, 1, hidden_dim)  # [1, 1, hidden_dim]
                v_step = hidden_states[:, step:step+1, :].view(1, 1, hidden_dim)
                
                # 复制到所有头 [num_heads, 1, hidden_dim/num_heads]
                hd_per_head = hidden_dim // num_heads
                k_step = k_step.repeat(num_heads, 1, 1)[:, :, :hd_per_head].cpu()  # 移到CPU
                v_step = v_step.repeat(num_heads, 1, 1)[:, :, :hd_per_head].cpu()  # 移到CPU
                
                # 提取注意力分数（当前step的注意力）
                attn_scores = attentions[0].cpu()  # 移到CPU [num_heads, seq_len, seq_len]
                
                # 更新KV Cache
                k_cache, v_cache = kv_manager.update_kv_cache(k_step, v_step, attn_scores)
                
                # 显示信息
                if (step + 1) % 5 == 0:
                    token_str = tokenizer.decode(tokens[0, :step+1])
                    print(f"\n=== Step {step+1} - Token: '{tokenizer.decode(tokens[0, step])}' ===")
                    print(f"文本片段: ...{token_str[-50:]}")
                    stats = kv_manager.get_head_cache_stats()
                    total_tokens = stats.pop("_total_cache_elements", 0)
                    compression_rate = 1 - (total_tokens / (step + 1)) if step > 0 else 0
                    
                    # 获取每个头类型实际保留的tokens（假设分别存储）
                    per_type_tokens = kv_manager.get_per_head_type_tokens()
                    
                    print(f"  各头类型缓存 (假设分别存储):")
                    for head_type_name, cache_info in stats.items():
                        if head_type_name in per_type_tokens:
                            # 显示该头类型实际保留的tokens
                            kept_tokens = per_type_tokens[head_type_name]
                            ratio = (kept_tokens / (step + 1)) * 100 if step > 0 else 100
                            print(f"    {head_type_name}: {kept_tokens}/{step+1} tokens (保留{ratio:.1f}%)")
                        elif head_type_name == "full_cache":
                            # Full-cache总是保留所有
                            print(f"    {head_type_name}: {step+1}/{step+1} tokens (保留100%)")
                    
                    print(f"  共享缓存: K{k_cache.shape}, V{v_cache.shape} (压缩率: {compression_rate*100:.1f}%)")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            use_real_text = False
    
    # 如果没有transformers库，使用随机数据
    if not use_real_text:
        print("使用随机数据模拟推理...\n")
        kv_manager = AttentionHeadKVManager(
            num_heads=8,
            streaming_heads=[0,1],
            multi_sinks_heads=[2,3],
            chunk_heads=[4,5],
            stream_window_size=4,
            top_k=4
        )

        hidden_dim = 64
        for step in range(64):
            k_step = torch.randn(8, 1, hidden_dim)
            v_step = torch.randn(8, 1, hidden_dim)
            seq_len = step + 1
            attn_scores = torch.randn(8, seq_len, seq_len)
            
            k_cache, v_cache = kv_manager.update_kv_cache(k_step, v_step, attn_scores)
            
            if (step + 1) % 10 == 0:
                print(f"\n=== Step {step+1} - Head Type Cache Shapes ===")
                stats = kv_manager.get_head_cache_stats()
                for head_type_name, cache_info in stats.items():
                    print(f"{head_type_name}: K{cache_info['k_shape']}, V{cache_info['v_shape']} ({cache_info['num_heads']} heads)")
                print(f"Full K Cache shape: {k_cache.shape}, V Cache shape: {v_cache.shape}")