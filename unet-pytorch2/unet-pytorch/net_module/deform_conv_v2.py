import torch
from torch import nn


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
            调制(bool，可选):如果为True，则为Modulated Defomable Convolution (Deformable ConvNets v2)。
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # 定义了self.conv（最终输出的卷积层，设置输入通道数和输出通道数）
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # self.p_conv（偏置层，学习之前公式(2)中说的偏移量）
        # self.p_conv输出通道为2*kernel_size*kernel_size代表了卷积核中所有元素的偏移坐标(因为同时存在x和y的偏移，所以要乘以2)
        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            # self.m_conv(权重学习层，这个是后来提出的第二个版本的卷积也就是公式(3)描述的卷积)
            # self.m_conv(kernel_size*kernel_size)代表了卷积核每个元素的权重。
            # 他们的kernel_size为3，stride可以由我们自己设置(这里涉及之前公式(1,2)对于 [公式] 的查找)stride默认值为1
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # 首先我们的数据先经过self.p_conv学习出offset(坐标偏移量)，
        # 如果modulation设置为true的话就同时学习出偏置
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        # 接下来通过self._get_p()这个函数获取所有卷积核中心坐标 [公式] 具体操作如下
        p = self._get_p(offset, dtype)

        # 我们学习出的量是float类型的，而像素坐标都是整数类型的，所以我们还要用双线性插值的方法去推算相应的值

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 现在只获取了坐标值，我们最终木的是获取相应坐标上的值，这里我们通过self._get_x_q()获取相应值
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 由于(h,w)被压缩成了(h*w)所以在这个维度上，每过w个元素，就代表了一行，所以我们的坐标index=offset_x*w+offset_y
        # (这样就能在h*w上找到(h,w)的相应坐标)
        # 同时再把偏移expand()到每一个通道最后返回x_offset(b,c,h,w,N)。
        # (最后输出x_offset的h,w指的是x的h,w而不是q的)
        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        # 在获取所有值后我们计算出x_offset，但是x_offset的size是(b,c,h,w,N)，
        # 我们的目的是将最终的输出结果的size变成和x一致即(b,c,h,w)，所以在最后用了一个reshape的操作。
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 比如self.kernel_size为3，通过torch.meshgrid生成从（-1，-1）到（1，1）9个坐标
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        # 将坐标的x和y分别存储，然后再将x，y 以(1,2N,1,1)的形式返回，这样我们就获取了一个卷积核的所有相对坐标。
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    # 接下来获取卷积核在feature map上对应的中心坐标，也就是 Po，代码实现如下
    # 输入参数的h,w 就是通过p_conv后的feature map的尺寸信息
    def _get_p_0(self, h, w, N, dtype):
        # 通过torch.meshgrid生成所有中心坐标，通过kernel_size推断初始坐标通过self.stride推断所有中心坐标，
        # (这里注意一下，代码默认torch.arange从1开始，实际上这是kernel_size为3时的情况。作者推论)
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 比如p(在N=0时)p_0就是中心坐标，而p_n=(-1,-1)，所以此时的p就是卷积核中心坐标加上(-1,-1)(即红色块左上方的块)再加上offset。
    # 同理可得N=1,N=2...分别代表了一个卷积核上各个元素。
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        # 首先通过函数get_p_n()生成了卷积的相对坐标
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        # 接下来获取卷积核在feature map上对应的中心坐标，也就是 Po
        p_0 = self._get_p_0(h, w, N, dtype)
        # 再将我们获取的相对坐标信息与中心坐标相加就获得了我们卷积核的所有坐标
        # 卷积坐标加上之前学习出的offset后就是论文提出的公式(2)也就是加上了偏置后的卷积操作
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 输入x是我们最早输入的数据x，q则是我们的坐标信息。
        # 首先我们获取q的相关尺寸信息(b,h,w,2N)，
        b, h, w, _ = q.size()
        # 再获取x的 w 保存在 padding_w中
        padded_w = x.size(3)
        c = x.size(1)
        # 将 x(b,c,h,w) 通过view变成 (b,c,h*w)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # 这样子就把x的坐标信息压缩在了最后一个维度(h*w)，这样做的目的是为了使用tensor.gather()通过坐标来获取相应值。
        # (这里注意下q的h,w和x的h,w不一定相同，比如stride不为1的时候)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # 函数首先获取了x_offset的所有size信息，然后以kernel_size为单位进行reshape，因为N=kernel_size*kernel_size，
        # 所以我们分两次进行reshape，
        # 第一次先把输入view成(b,c,h,ks*w,ks)，
        # 第二次再view将size变成(b,c,h*ks,w*ks)
        # (这部分我实在不知道咋描述了，图也画的比较捉急，大家康康，自己悟一下)
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
