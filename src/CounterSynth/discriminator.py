import torch
import torch.nn as nn

class CounterSynthDiscriminator(nn.Module):
    """
    Discriminator của CounterSynth
    
    Là một mạng tích chập hoàn toàn, phân biệt ảnh thật/giả và phân loại điều kiện (tuổi, giới tính)
    """
    def __init__(self, img_size=130, condition_dim=2):
        super(CounterSynthDiscriminator, self).__init__()
        
        self.condition_dim = condition_dim
        
        # Các lớp tích chập
        self.conv_layers = nn.Sequential(
            # Input: [batch, 1, img_size, img_size]
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # [batch, 16, img_size/2, img_size/2]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [batch, 32, img_size/4, img_size/4]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [batch, 64, img_size/8, img_size/8]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch, 128, img_size/16, img_size/16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch, 256, img_size/32, img_size/32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Tính toán kích thước feature map sau các lớp conv
        feature_size = img_size // 32
        flattened_size = 256 * feature_size * feature_size
        
        # Dự đoán Real/Fake (nhị phân)
        self.adv_layer = nn.Sequential(
            nn.Linear(flattened_size, 1),
            nn.Sigmoid()  # Sigmoid cho đầu ra binary
        )
        
        # Dự đoán phân phối điều kiện
        # Đối với tuổi (liên tục): một giá trị duy nhất
        # Đối với giới tính (nhị phân): một giá trị với sigmoid
        self.condition_layer = nn.Sequential(
            nn.Linear(flattened_size, condition_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Hình ảnh đầu vào [batch, 1, H, W]
            
        Returns:
            validity: Xác suất hình ảnh là thật [batch, 1]
            pred_condition: Dự đoán về điều kiện [batch, condition_dim]
        """
        batch_size = x.size(0)
        
        # Trích xuất đặc trưng
        features = self.conv_layers(x)
        features_flat = features.view(batch_size, -1)
        
        # Dự đoán real/fake
        validity = self.adv_layer(features_flat)
        
        # Dự đoán điều kiện
        pred_condition = self.condition_layer(features_flat)
        
        return validity, pred_condition