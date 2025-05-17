import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class SpatialTransformer2D(nn.Module):
    """
    2D Spatial Transformer cho từng lát cắt
    """
    def __init__(self, size, mode='bilinear', get_unnormed=False):
        super().__init__()
        self.mode = mode

        # Tạo lưới lấy mẫu
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)

        self.register_buffer('grid', grid)
        self.get_unnormed = get_unnormed

    def forward(self, src, flow):
        # Vị trí mới
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        unnormed_locs = torch.clone(new_locs)

        # Chuẩn hóa giá trị lưới về [-1, 1] để sử dụng grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Chuyển đổi cho lát cắt 2D
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # x, y cho lát cắt 2D

        # Trả về lưới không chuẩn hóa nếu cần thiết (cho mục đích hiển thị)
        if self.get_unnormed:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode), unnormed_locs
        else:
            return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt2D(nn.Module):
    """
    Tích hợp trường vectơ qua kỹ thuật scaling and squaring (dành cho 2D)
    """
    def __init__(self, inshape, nsteps):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer2D(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform2D(nn.Module):
    """
    Thay đổi kích thước biến đổi, bao gồm việc thay đổi kích thước trường vectơ và điều chỉnh tỷ lệ
    """
    def __init__(self, vel_resize):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'bilinear'  # Luôn là 2D

    def forward(self, x):
        if self.factor < 1:
            # Thay đổi kích thước trước để tiết kiệm bộ nhớ
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            # Nhân trước để tiết kiệm bộ nhớ
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        # Không làm gì nếu tỷ lệ là 1
        return x


class ConvBlock2D(nn.Module):
    """
    Khối tích chập 2D theo sau là LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)
        self.norm = nn.InstanceNorm2d(out_channels)  # Thêm InstanceNorm cho hiệu quả tốt hơn

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


def default_unet_features_2d():
    """
    Cấu hình mặc định cho các đặc trưng của Unet.
    """
    nb_features = [
        [16, 32, 32, 32],  # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features


class Unet2D(nn.Module):
    """
    Kiến trúc Unet cho 2D. Các đặc trưng của layer có thể được chỉ định trực tiếp như một danh sách
    của đặc trưng encoder và decoder hoặc như một số nguyên duy nhất cùng với số lượng cấp độ unet.
    """
    def __init__(self, inshape, c_dim, nb_features=None, nb_levels=None, feat_mult=1, in_channels=1):
        super().__init__()
        """
        Tham số:
            inshape: Kích thước đầu vào. ví dụ: (192, 192)
            c_dim: Số chiều của điều kiện (condition dimension)
            nb_features: Đặc trưng tích chập Unet. Có thể được chỉ định thông qua danh sách các danh sách
                với dạng [[đặc trưng encoder], [đặc trưng decoder]], hoặc như một số nguyên duy nhất.
            nb_levels: Số cấp độ trong unet. Chỉ được sử dụng khi nb_features là một số nguyên.
            feat_mult: Bội số của đặc trưng trên mỗi cấp độ. Chỉ được sử dụng khi nb_features là một số nguyên.
            in_channels: Số kênh đầu vào.
        """
        # Đảm bảo kích thước chính xác
        self.c_dim = c_dim
        ndims = len(inshape)
        assert ndims == 2, 'ndims should be 2 for Unet2D. found: %d' % ndims

        # Đặc trưng mặc định của encoder và decoder nếu không có gì được cung cấp
        if nb_features is None:
            nb_features = default_unet_features_2d()

        # Xây dựng danh sách đặc trưng tự động
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Cấu hình encoder (đường dẫn lấy mẫu xuống)
        prev_nf = in_channels + c_dim
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock2D(prev_nf, nf, stride=2))
            prev_nf = nf

        self.nf_encoder = prev_nf

        # Cấu hình decoder (đường dẫn lấy mẫu lên)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock2D(channels, nf, stride=1))
            prev_nf = nf

        self.decoder_nf = prev_nf

        # Cấu hình các lớp tích chập bổ sung của decoder (không lấy mẫu lên)
        prev_nf += in_channels + c_dim
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock2D(prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        # Lấy kích hoạt của encoder
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # Tích chập, lấy mẫu lên, nối chuỗi
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # Các tích chập bổ sung ở độ phân giải đầy đủ
        for layer in self.extras:
            x = layer(x)

        return x


class DiffeoGenerator2D(nn.Module):
    """
    Mạng VoxelMorph cho đăng ký phi tuyến (không giám sát) giữa hai hình ảnh 2D.
    """
    def __init__(self, inshape, c_dim, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1, int_steps=7,
                 int_downsize=2, bidir=False, use_probs=True, in_channels=1):
        """
        Tham số:
            inshape: Kích thước đầu vào. ví dụ: (192, 192)
            c_dim: Số chiều của điều kiện (condition dimension)
            nb_unet_features: Đặc trưng tích chập Unet. Có thể được chỉ định thông qua danh sách các danh sách
                với dạng [[đặc trưng encoder], [đặc trưng decoder]], hoặc như một số nguyên duy nhất.
            nb_unet_levels: Số cấp độ trong unet. Chỉ được sử dụng khi nb_features là một số nguyên.
            unet_feat_mult: Bội số của đặc trưng trên mỗi cấp độ. Chỉ được sử dụng khi nb_features là một số nguyên.
            int_steps: Số bước tích hợp dòng chảy. Biến đổi không phải là diffeomorphic khi giá trị này là 0.
            int_downsize: Số nguyên chỉ định hệ số lấy mẫu xuống của trường dòng chảy cho tích hợp vectơ.
                Trường dòng chảy không được lấy mẫu xuống khi giá trị này là 1.
            bidir: Bật hàm chi phí hai chiều. Mặc định là False.
            use_probs: Sử dụng xác suất trong trường dòng chảy. Mặc định là False.
            in_channels: Số kênh đầu vào.
        """
        super().__init__()

        # Cờ nội bộ cho biết có nên trả về dòng chảy hay warp tích hợp trong quá trình suy luận
        self.training = True
        self.use_probs = use_probs
        self.c_dim = c_dim

        # Đảm bảo kích thước chính xác
        ndims = len(inshape)
        assert ndims == 2, 'ndims should be 2 for DiffeoGenerator2D. found: %d' % ndims

        # Cấu hình mô hình unet cốt lõi
        self.unet_model = Unet2D(
            inshape,
            c_dim,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            in_channels=in_channels
        )

        # Cấu hình lớp trường dòng chảy unet
        self.flow = nn.Conv2d(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # Khởi tạo lớp dòng chảy với trọng số nhỏ và độ lệch
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if use_probs:
            self.flow_logsigma = nn.Conv2d(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)
            # Khởi tạo lớp dòng chảy với trọng số nhỏ và độ lệch
            self.flow_logsigma.weight = nn.Parameter(Normal(0, 1e-10).sample(self.flow.weight.shape))
            self.flow_logsigma.bias = nn.Parameter(torch.ones(self.flow.bias.shape) * (-10))

        # Cấu hình các lớp thay đổi kích thước tùy chọn
        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform2D(int_downsize) if resize else None
        self.fullsize = ResizeTransform2D(1 / int_downsize) if resize else None

        # Cấu hình đào tạo hai chiều
        self.bidir = bidir

        # Cấu hình lớp tích hợp tùy chọn cho warp diffeomorphic
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt2D(down_shape, int_steps) if int_steps > 0 else None

        # Cấu hình transformer
        self.transformer = SpatialTransformer2D(inshape)

    def forward(self, source, target, registration=False):
        '''
        Tham số:
            source: Tensor hình ảnh nguồn.
            target: Nhãn mục tiêu để chuyển đổi hình ảnh.
            registration: Trả về hình ảnh đã biến đổi và dòng chảy. Mặc định là False.
        '''
        # Điều chỉnh target để phù hợp với kích thước của source
        target = target.view(target.size(0), self.c_dim, 1, 1)
        target = target.repeat(1, 1, source.size(2), source.size(3))

        # Nối các đầu vào và truyền qua unet
        to_unet = torch.cat((source, target), 1)
        shape = self.unet_model(to_unet)

        # Biến đổi thành trường dòng chảy
        flow_field = self.flow(shape)

        # Thay đổi kích thước dòng chảy để tích hợp
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # Tích hợp để tạo ra warp diffeomorphic
        if self.integrate:
            pos_flow = self.integrate(pos_flow)

            # Thay đổi kích thước về độ phân giải cuối cùng
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)

        # Warp hình ảnh với trường dòng chảy
        y_source = self.transformer(source, pos_flow)

        # Trả về trường dòng chảy không tích hợp nếu đang huấn luyện
        if not registration:
            return y_source, preint_flow
        else:
            return y_source, pos_flow