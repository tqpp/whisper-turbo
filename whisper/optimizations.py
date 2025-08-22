"""
Whisper Turbo 优化模块
包含各种性能优化技术的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import warnings

try:
    import torch._dynamo as dynamo
    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class OptimizedMultiHeadAttention(nn.Module):
    """优化的多头注意力机制"""
    
    def __init__(self, n_state: int, n_head: int, use_flash_attn: bool = True):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        
        # KV缓存优化
        self.kv_cache = {}
        self.cache_enabled = False
        
    def enable_kv_cache(self):
        """启用KV缓存以加速推理"""
        self.cache_enabled = True
        self.kv_cache.clear()
        
    def disable_kv_cache(self):
        """禁用KV缓存"""
        self.cache_enabled = False
        self.kv_cache.clear()
    
    def forward(
        self,
        x: torch.Tensor,
        xa: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kv_cache_key: Optional[str] = None,
    ):
        q = self.query(x)
        
        if kv_cache_key and self.cache_enabled and kv_cache_key in self.kv_cache:
            # 使用缓存的KV
            k, v = self.kv_cache[kv_cache_key]
        else:
            if xa is None:
                # 自注意力
                k = self.key(x)
                v = self.value(x)
            else:
                # 交叉注意力
                k = self.key(xa)
                v = self.value(xa)
            
            # 缓存KV
            if kv_cache_key and self.cache_enabled:
                self.kv_cache[kv_cache_key] = (k, v)
        
        if self.use_flash_attn and x.dtype == torch.float16:
            # 使用Flash Attention
            return self._flash_attention(q, k, v, mask)
        else:
            # 使用优化的标准注意力
            return self._optimized_attention(q, k, v, mask)
    
    def _flash_attention(self, q, k, v, mask):
        """Flash Attention实现"""
        batch_size, seq_len, _ = q.shape
        
        q = q.view(batch_size, seq_len, self.n_head, -1)
        k = k.view(batch_size, -1, self.n_head, q.size(-1))
        v = v.view(batch_size, -1, self.n_head, q.size(-1))
        
        # Flash Attention要求特定的tensor布局
        q = q.transpose(1, 2)  # [batch, n_head, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = flash_attn_func(q, k, v, causal=mask is not None)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, -1)
        
        return self.out(out)
    
    def _optimized_attention(self, q, k, v, mask):
        """优化的标准注意力实现"""
        batch_size, seq_len, _ = q.shape
        head_dim = self.n_state // self.n_head
        
        q = q.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, head_dim).transpose(1, 2)
        
        # 使用PyTorch的优化SDPA
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask,
                is_causal=mask is not None,
                enable_gqa=True  # 启用分组查询注意力优化
            )
        else:
            # 回退到手动实现
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if mask is not None:
                scores = scores + mask
            
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out(out)


class OptimizedFFN(nn.Module):
    """优化的前馈网络"""
    
    def __init__(self, n_state: int, use_gelu_approx: bool = True):
        super().__init__()
        self.use_gelu_approx = use_gelu_approx
        
        # 使用更高效的线性层
        self.c_fc = nn.Linear(n_state, 4 * n_state)
        self.c_proj = nn.Linear(4 * n_state, n_state)
        
        # 预计算GELU近似的常数
        if use_gelu_approx:
            self.gelu_coeff = math.sqrt(2.0 / math.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        
        if self.use_gelu_approx:
            # 使用快速GELU近似
            x = 0.5 * x * (1.0 + torch.tanh(self.gelu_coeff * (x + 0.044715 * x * x * x)))
        else:
            x = F.gelu(x)
        
        return self.c_proj(x)


class BatchProcessor:
    """批处理优化器"""
    
    def __init__(self, max_batch_size: int = 16):
        self.max_batch_size = max_batch_size
    
    def create_batches(self, audio_list, batch_size: Optional[int] = None):
        """创建优化的批次"""
        if batch_size is None:
            batch_size = min(self.max_batch_size, len(audio_list))
        
        # 按音频长度排序以优化批处理效率
        sorted_audio = sorted(enumerate(audio_list), key=lambda x: len(x[1]))
        
        batches = []
        for i in range(0, len(sorted_audio), batch_size):
            batch_indices = [idx for idx, _ in sorted_audio[i:i+batch_size]]
            batch_audio = [audio for _, audio in sorted_audio[i:i+batch_size]]
            batches.append((batch_indices, batch_audio))
        
        return batches
    
    def pad_batch(self, audio_batch):
        """智能填充批次"""
        max_len = max(len(audio) for audio in audio_batch)
        
        # 使用最接近的2的幂次长度以优化GPU计算
        padded_len = 2 ** math.ceil(math.log2(max_len))
        
        padded_batch = []
        for audio in audio_batch:
            if len(audio) < padded_len:
                padding = padded_len - len(audio)
                padded_audio = F.pad(audio, (0, padding))
            else:
                padded_audio = audio[:padded_len]
            padded_batch.append(padded_audio)
        
        return torch.stack(padded_batch)


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        self.gradient_checkpointing = False
        self.mixed_precision = False
    
    def enable_gradient_checkpointing(self, model):
        """启用梯度检查点以节省内存"""
        self.gradient_checkpointing = True
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    def enable_mixed_precision(self):
        """启用混合精度训练"""
        self.mixed_precision = True
    
    @staticmethod
    def optimize_memory_layout(tensor):
        """优化tensor内存布局"""
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor
    
    @staticmethod
    def clear_cache():
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelCompiler:
    """模型编译优化器"""
    
    def __init__(self):
        self.compiled_models = {}
    
    def compile_model(self, model, mode: str = "default"):
        """编译模型以获得更好的性能"""
        if not DYNAMO_AVAILABLE:
            warnings.warn("torch.compile不可用，跳过模型编译")
            return model
        
        model_id = id(model)
        if model_id not in self.compiled_models:
            if mode == "max-autotune":
                compiled_model = torch.compile(model, mode="max-autotune")
            elif mode == "reduce-overhead":
                compiled_model = torch.compile(model, mode="reduce-overhead")
            else:
                compiled_model = torch.compile(model)
            
            self.compiled_models[model_id] = compiled_model
        
        return self.compiled_models[model_id]


# 全局优化器实例
batch_processor = BatchProcessor()
memory_optimizer = MemoryOptimizer()
model_compiler = ModelCompiler()