import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM, GPT2Model

def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps: float = 1e-6, add_unit_offset: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output
    
class ParamEmbedding(nn.Module):
    def __init__(self, patch_len, d_model) -> None:
        super().__init__()
        
        self.weight = nn.Parameter(
            torch.empty((patch_len, d_model)),
            requires_grad=True,
        )
        
    def forward(self, x):
        weight = self.weight
        output = F.embedding(x, weight)
        return output
    
class LinearEmbedding(nn.Module):
    def __init__(self, patch_len, d_model, bias=True):
        super().__init__()
        
        self.fc = nn.Linear(patch_len, d_model, bias=bias)
    
    def forward(self, x):
        output = self.fc(x)
        return output
    
class EncoderMLP(nn.Module):
    def __init__(self, d_model, inter_dim, dropout=0.1) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, inter_dim)
        self.up_proj = nn.Linear(d_model, inter_dim)
        self.down_proj = nn.Linear(inter_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        fuse = self.dropout(fuse)
        outputs = self.down_proj(fuse)
        return outputs
    
class Attention(nn.Module):
    def __init__(self, nheads, head_dim, hidden_size, dropout=0.1):
        super().__init__()
        
        self.nheads = nheads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        
        self.qkv_size = nheads * head_dim
        
        self.scaling = self.head_dim**-0.5
        
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            3 * self.qkv_size
        )
        self.o_proj = nn.Linear(
            self.qkv_size,
            hidden_size
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, freqs_cis=None, mask=None):
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3
        
        batch_size, input_len, _ = hidden_states_shape
        
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.qkv_size, self.qkv_size, self.qkv_size],
                               dim=-1)
        
        xq = xq.view(batch_size, -1, self.nheads, self.head_dim)
        xk = xk.view(batch_size, -1, self.nheads, self.head_dim)
        xv = xv.view(batch_size, -1, self.nheads, self.head_dim)
        
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        scores = self.dropout(scores)
        
        if mask:
            scores = scores + mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        
        output = torch.matmul(scores, v)
        
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output
    
class Patching(nn.Module):
    def __init__(self, patch_len, stride) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
    def forward(self, x: torch.Tensor):
        #  bs x num_patch x patch_len
        return x.unfold(dimension=1, size=self.patch_len, step=self.stride)

class EncoderLayer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.attn = Attention(
            nheads=configs.nheads,
            head_dim=configs.head_dim,
            hidden_size=configs.d_model
        )
        
        self.mlp = EncoderMLP(
            d_model=configs.d_model,
            inter_dim=configs.ff_dim,
            dropout=configs.dropout
        )
        
        self.input_ln = RMSNorm(configs.d_model)
        self.post_att_ln = RMSNorm(configs.d_model)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, mask=None):
        residual = hidden_states
        hidden_states = self.input_ln(hidden_states)
        hidden_states = self.attn(
            hidden_states, freqs_cis, mask
        )
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        hidden_states = self.post_att_ln(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
    
class Encoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(configs.num_hidden_layers):
            self.layers.append(EncoderLayer(configs))
        self.norm = RMSNorm(configs.d_model)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, mask=None):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class FlattenHead(nn.Module):
    def __init__(self, configs, concat_size) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.fc = nn.Linear(concat_size, configs.pred_len)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
class FFAug(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        self.configs = configs
        
        # build matrix subject to norm
        self.ffmatrix = nn.Parameter(torch.randn(configs.d_model, configs.d_model) * 0.1, requires_grad=configs.requires_grad)
        self.down_proj = nn.Linear(2 * configs.d_model, configs.d_model)
        
    def forward(self, x):
        x_proj = (2.*torch.pi*x) @ self.ffmatrix
        x_con = torch.concat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        x = self.down_proj(x_con)
        
        return x
    
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        self.configs = configs
        
        self.concat_patch = (configs.seq_len - configs.patch_len) // configs.stride + 1
        
        if configs.rope is True:
            freqs_cis = precompute_freqs_cis(configs.head_dim,
                                            self.concat_patch,
                                            )
            self.register_buffer('freqs_cis', freqs_cis)
        
        else:
            self.freqs_cis = None
        
        self.patch = Patching(configs.patch_len, configs.stride)
        self.embedder = LinearEmbedding(configs.patch_len, configs.d_model, bias=True)
        
        self.encoder = Encoder(configs)
        
        ''' ====== pre-train LLM alignment ====== '''
        
        requires_grad = configs.requires_grad
        layer_idx = configs.layer_idx
        self.llm_config = configs.llm_config
            
        if configs.llm_config == "gemma-2b" or configs.llm_config == "gemma-2b-before": # 18 layers
            # gemma 2B
            self.llm_layer = AutoModelForCausalLM.from_pretrained("google/gemma-2b").model.layers[layer_idx]
            for i, (name, param) in enumerate(self.llm_layer.named_parameters()):
                param.requires_grad = requires_grad
            
            self.llm_d_model = 2048
            
        
        elif configs.llm_config == "None":
            self.llm_layer = None
            
        elif configs.llm_config == "freq_after":
            self.freq_layer = FFAug(configs)
            self.llm_d_model = 0
            
        elif configs.llm_config == "freq_before":
            self.freq_layer = FFAug(configs)
            self.llm_d_model = 0
            
        # LLM adapter
        if configs.llm_config != "None" and configs.llm_config[:4] != "freq":
            self.pre_adapter = nn.Linear(configs.d_model, self.llm_d_model)
            self.post_adapter = nn.Linear(self.llm_d_model, configs.d_model)
        
        ''' ===== End ===== '''
        
        self.pred_head = FlattenHead(configs, self.concat_patch * configs.d_model)
        
    def forward(self, x, mask=None):
        # Normalization from Non-stationary Transformer
        
        # x: bs x seq_len
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        
        x = self.patch(x)
        # x : bs x num_patch x patch_len
        
        x = self.embedder(x)
        # x : bs x num_patch x d_model
        
        if self.llm_config == "freq_before":
            x = self.freq_layer(x)
            
        if self.llm_config == "gemma-2b-before":
            x = self.pre_adapter(x)
            # x : bs x num_patch x llm_d_model
            
            pos_ids = torch.arange(x.shape[1]).to(x.device)
            pos_ids = pos_ids.repeat(x.shape[0], 1)
            
            x = self.llm_layer(hidden_states=x, position_ids=pos_ids)
            
            x = self.post_adapter(*x)
        
        x = self.encoder(x, self.freqs_cis)
        
        if self.llm_config != "None" and self.llm_config != "gpt2" and self.llm_config[:4] != "freq" and self.llm_config != "gemma-2b-before":
            x = self.pre_adapter(x)
            # x : bs x num_patch x llm_d_model
            
            pos_ids = torch.arange(x.shape[1]).to(x.device)
            pos_ids = pos_ids.repeat(x.shape[0], 1)
            
            x = self.llm_layer(hidden_states=x, position_ids=pos_ids)
            
            x = self.post_adapter(*x)
            # x : bs x num_patch x d_model
            
        elif self.llm_config == "gpt2":
            x = self.pre_adapter(x)
            # x : bs x num_patch x llm_d_model
            
            x = self.llm_layer(hidden_states=x)
            
            x = self.post_adapter(x[0])
            # x : bs x num_patch x d_model
            
        elif self.llm_config == "freq_after":
            x = self.freq_layer(x)
        
        x = self.pred_head(x)
        # x : bs x pred_len
        
        # De-Normalization from Non-stationary Transformer
        x = x * stdev
        x = x + means
        
        return x