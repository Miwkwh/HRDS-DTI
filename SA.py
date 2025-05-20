import torch
import numbers
import torch.nn as nn
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):

    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        elif x.dim() == 3:
            return self.body(x)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")


class SA(nn.Module):

    def __init__(self, dim, num_heads, bias=True):
        super(SA, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv_linear = nn.Linear(dim, dim * 3, bias=bias)
        self.project_out_linear = nn.Linear(dim, dim, bias=bias)


        self.attn_drop = nn.Dropout(0.)


        self.attn1 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        if x.dim() == 4:

            b, c, h, w = x.shape
            qkv = self.qkv_dwconv(self.qkv_conv(x))  #
            q, k, v = qkv.chunk(3, dim=1)


            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)


            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)

            _, _, C, _ = q.shape


            mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
            mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
            mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
            mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)


            attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b, h, C, C]


            index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
            mask1.scatter_(-1, index, 1.)
            attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

            index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
            mask2.scatter_(-1, index, 1.)
            attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

            index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
            mask3.scatter_(-1, index, 1.)
            attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

            index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
            mask4.scatter_(-1, index, 1.)
            attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))


            attn1 = attn1.softmax(dim=-1)
            attn2 = attn2.softmax(dim=-1)
            attn3 = attn3.softmax(dim=-1)
            attn4 = attn4.softmax(dim=-1)


            out1 = attn1 @ v  # [b, h, C, d]
            out2 = attn2 @ v
            out3 = attn3 @ v
            out4 = attn4 @ v


            out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4


            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out_conv(out)  # [b, dim, h, w]
            return out

        elif x.dim() == 3:

            qkv = self.qkv_linear(x)  # [b, s, 3*dim]
            q, k, v = qkv.chunk(3, dim=-1)


            batch_size, seq_length, embed_dim = q.shape
            head_dim = embed_dim // self.num_heads
            q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
            k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
            v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)


            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)


            attn = (q @ k.transpose(-2, -1)) * self.temperature  # [b, h, s, s]


            mask1 = torch.zeros(batch_size, self.num_heads, seq_length, seq_length, device=x.device,
                                requires_grad=False)
            mask2 = torch.zeros(batch_size, self.num_heads, seq_length, seq_length, device=x.device,
                                requires_grad=False)
            mask3 = torch.zeros(batch_size, self.num_heads, seq_length, seq_length, device=x.device,
                                requires_grad=False)
            mask4 = torch.zeros(batch_size, self.num_heads, seq_length, seq_length, device=x.device,
                                requires_grad=False)

            index = torch.topk(attn, k=int(seq_length / 2), dim=-1, largest=True)[1]
            mask1.scatter_(-1, index, 1.)
            attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

            index = torch.topk(attn, k=int(seq_length * 2 / 3), dim=-1, largest=True)[1]
            mask2.scatter_(-1, index, 1.)
            attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

            index = torch.topk(attn, k=int(seq_length * 3 / 4), dim=-1, largest=True)[1]
            mask3.scatter_(-1, index, 1.)
            attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

            index = torch.topk(attn, k=int(seq_length * 4 / 5), dim=-1, largest=True)[1]
            mask4.scatter_(-1, index, 1.)
            attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

            attn1 = attn1.softmax(dim=-1)
            attn2 = attn2.softmax(dim=-1)
            attn3 = attn3.softmax(dim=-1)
            attn4 = attn4.softmax(dim=-1)

            out1 = attn1 @ v  # [b, h, s, d]
            out2 = attn2 @ v
            out3 = attn3 @ v
            out4 = attn4 @ v


            out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4


            out = rearrange(out, 'b h s d -> b s (h d)', h=self.num_heads)

            out = self.project_out_linear(out)  # [b, s, dim]
            return out
        else:
            raise ValueError(f"No Agreement: {x.dim()}")


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SA(dim, num_heads, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x
