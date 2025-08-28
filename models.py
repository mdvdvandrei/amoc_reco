import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm.models.vision_transformer import VisionTransformer
import math



class CustomSqueezeNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.net = models.squeezenet1_0(pretrained=False)

        # adjust first conv if grayscale or other channel count
        if input_channels != 3:
            self.net.features[0] = nn.Conv2d(
                input_channels, 96, kernel_size=7, stride=2, padding=3, bias=False
            )

        # replace the classifier conv to output `num_classes`
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )

    def forward(self, x):
        x = self.net(x)            # → shape (B, num_classes, 1, 1)
        return torch.flatten(x, 1) # → shape (B, num_classes)


        
class CustomResNet(nn.Module):
    def __init__(self, input_channels, output_dim, input_height=144, input_width=108):
        super(CustomResNet, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # Initialize ResNet-18 without pretrained weights.
        self.resnet = models.squeezenet1_0(pretrained=False)
        
        # Adjust the first convolutional layer if input_channels != 3.
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace the final fully connected layer to output the desired number of classes.
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)
        
        # Linear connection from the flattened input to the output.
        self.input_linear = nn.Linear(input_channels * input_height * input_width, output_dim)
    
    def forward(self, x):
        # Flatten the input for the linear skip connection.
        input_flat = x.view(x.size(0), -1)
        
        # Forward pass through ResNet.
        resnet_out = self.resnet(x)
        
        # Compute the linear input connection.
        input_connection = self.input_linear(input_flat)
        
        # Combine the outputs.
        return resnet_outxw





    # ---------- Fourier Block for Large-Scale Patterns ----------
class FourierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes_height, modes_width):
        """
        A simple Fourier block that:
        - Computes the 2D FFT of the input,
        - Applies a learned complex multiplication on a limited set of Fourier modes,
        - Returns the inverse FFT result.
        """
        super(FourierBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_height = modes_height  # number of Fourier modes to keep along height
        self.modes_width = modes_width    # number of Fourier modes to keep along width
        
        # Initialize learnable weights for the Fourier coefficients.
        # We parameterize them as separate real and imaginary parts.
        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes_height, modes_width))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes_height, modes_width))
    
    def forward(self, x):
        """
        x: Tensor of shape (batch, in_channels, height, width)
        """
        batchsize, _, height, width = x.shape
        # Compute FFT on the last two dimensions
        x_ft = torch.fft.rfft2(x)  # shape: (batch, in_channels, height, width//2 + 1)
        
        # Create an output tensor in the Fourier domain
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1),
                             dtype=torch.cfloat, device=x.device)
        
        # Only update a subset of Fourier modes (top-left corner frequencies)
        # x_ft[:, :, :modes_height, :modes_width] shape: (batch, in_channels, modes_height, modes_width)
        # weights shape: (in_channels, out_channels, modes_height, modes_width)
        # The einsum multiplies over the in_channels dimension.
        weight = self.weights_real + 1j * self.weights_imag
        out_ft[:, :, :self.modes_height, :self.modes_width] = \
            torch.einsum("bixy, ioxy -> boxy", 
                         x_ft[:, :, :self.modes_height, :self.modes_width], weight)
        
        # Inverse FFT to return to the spatial domain
        x_out = torch.fft.irfft2(out_ft, s=(height, width))
        return x_out

# ---------- Two-Branch Model Combining Fourier and CNN ----------
class TwoBranchModel(nn.Module):
    def __init__(self, input_channels, output_dim):
        """
        input_channels: Number of channels in the input image.
        output_dim: Dimension of the regression output.
        """
        super(TwoBranchModel, self).__init__()
        
        # Branch 1: Fourier Neural Operator (for large-scale patterns)
        # Here we choose arbitrary numbers for modes and output channels;
        # these can be tuned according to your application.
        self.fourier = FourierBlock(in_channels=input_channels, 
                                    out_channels=64, 
                                    modes_height=16, 
                                    modes_width=16)
        
        # Branch 2: Shallow CNN (for small-scale patterns)
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.BatchNorm2d(16),
            nn.MaxPool2d(2),  # reduces spatial dims by factor of 2
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
        
            nn.MaxPool2d(2)
        )
        
        # To obtain fixed-size representations regardless of input dims,
        # use adaptive pooling on each branch.
        self.adapt_fourier = nn.AdaptiveAvgPool2d((16, 16))
        self.adapt_cnn = nn.AdaptiveAvgPool2d((16, 16))
        
        # Dense part: Combine features from both branches.
        # The Fourier branch outputs 64 channels and the CNN branch outputs 32 channels,
        # both with spatial dimensions 8x8.
        combined_features = (64 + 64) * 16 * 16
        self.fc = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )
        
        # Skip connection from flattened input.
        # Adjust the dimensions here if your input size is different.
        # For example, assume input images are of size 144x108.
        self.input_linear = nn.Linear(input_channels * 144 * 108, output_dim)
    
    def forward(self, x):
        """
        x: Tensor of shape (batch, input_channels, height, width)
        """
        # Save the original input for the skip (linear) connection.
        input_flat = x.view(x.size(0), -1)
        
        # --- Branch 1: Fourier ---
        x_fourier = self.fourier(x)
        x_fourier = self.adapt_fourier(x_fourier)
        
        # --- Branch 2: Shallow CNN ---
        x_cnn = self.cnn_branch(x)
        x_cnn = self.adapt_cnn(x_cnn)
        
        # Flatten branch outputs
        x_fourier_flat = x_fourier.view(x_fourier.size(0), -1)
        x_cnn_flat = x_cnn.view(x_cnn.size(0), -1)
        
        # Concatenate features from both branches
        features = torch.cat([x_fourier_flat, x_cnn_flat], dim=1)
        
        # Pass through the dense (fully-connected) layers
        out = self.fc(features)
        
        # Skip connection: add a linear projection of the flattened input
        skip = self.input_linear(input_flat)
        out = out + skip
        
        return out
    
import torch
import torch.nn as nn

# Define a residual block which will be used repeatedly.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Use a 1x1 conv if dimensions do not match (for skip connection).
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# A new CNN that integrates residual blocks.
class ResidualCNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(ResidualCNN, self).__init__()
        
        # An initial convolution before residual blocks.
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # Create a series of residual blocks.
        self.layer1 = ResidualBlock(16, 32, stride=2, dropout=0.3)
        self.layer2 = ResidualBlock(32, 64, stride=2, dropout=0.3)
        self.layer3 = ResidualBlock(64, 128, stride=2, dropout=0.3)
        self.layer4 = ResidualBlock(128, 256, stride=2, dropout=0.3)
        
        # Global average pooling to handle spatial dimensions dynamically.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers after pooling.
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )
        
        # An additional skip connection directly from the input.
        # This expects a fixed input of dimensions: (input_channels, 144, 108).
        flat_input_dim = input_channels * 144 * 108 #144 * 108  * 48 * 36
        self.input_skip = nn.Linear(flat_input_dim, output_dim)
        
    def forward(self, x):
        # Skip connection: flatten the raw input.
        input_skip = x.view(x.size(0), -1)
        
        # Initial convolution.
        out = self.initial(x)
        
        # Pass through residual layers.
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global pooling.
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        
        # Fully connected layers for the main branch.
        out = self.fc(out)
        
        # Merge with the flattened raw input connection.
        out = out + self.input_skip(input_skip)
        return out




class ResidualBlock_new(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, p_drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.LeakyReLU(0.01, inplace=True)
        self.drop  = nn.Dropout(p_drop)
        self.short = (
            nn.Identity() if stride == 1 and in_channels == out_channels else
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return self.relu(out)

class ResidualCNNHet(nn.Module):
    """Residual CNN with two output heads: mean (mu) and log-variance (logvar)."""
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.layer1 = ResidualBlock_new(16, 32, 2)
        self.layer2 = ResidualBlock_new(32, 64, 2)
        self.layer3 = ResidualBlock_new(64, 128, 2)
        self.layer4 = ResidualBlock_new(128, 256, 2)
        self.pool   = nn.AdaptiveAvgPool2d(1)

        self.mu_head = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.01, inplace=True), nn.Dropout(0.5),
            nn.Linear(64, out_dim)
        )
        self.log_head = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.01, inplace=True), nn.Dropout(0.5),
            nn.Linear(64, out_dim)
        )

        flat_dim = in_channels * 144 * 108  # adjust if your spatial dim differs
        self.skip_mu  = nn.Linear(flat_dim, out_dim)
        self.skip_log = nn.Linear(flat_dim, out_dim)

    def forward(self, x):
        flat = x.view(x.size(0), -1)
        out  = self.stem(x)
        out  = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out  = self.pool(out).view(x.size(0), -1)
        mu     = self.mu_head(out)  + self.skip_mu(flat)
        logvar = self.log_head(out) + self.skip_log(flat)
        logvar = 8.0 * torch.tanh(logvar / 8.0)
        return mu, logvar



class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(SimpleCNN, self).__init__()
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.drop1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(negative_slope=0.01)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Calculate the number of features for the fully connected layer.
        self._calculate_fc_in_features()
        
        # Fully connected layers
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.fc_in_features, 128)
        self.relu5 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(128, output_dim)
        
        # Linear connection from input to output (flattening assumed input dims 144x108)
        self.input_linear = nn.Linear(input_channels * 144 * 108, output_dim)
    
    def _calculate_fc_in_features(self):
        dummy_input = torch.zeros(1, self.input_channels, 144, 108)
        x = self.conv1(dummy_input)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        
        self.fc_in_features = x.numel()
    
    def forward(self, x):
        input_flat = x.reshape(x.size(0), -1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.drop2(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        input_connection = self.input_linear(input_flat)
        x = x + input_connection
        return x
    



class SimpleViT(nn.Module):
    def __init__(self, input_channels, output_dim, patch_size=8, embedding_dim=128, num_heads=8, num_layers=6, ff_dim=1024):
        super(SimpleViT, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim

        # Vision Transformer configuration
        self.vit = VisionTransformer(
            img_size=(144, 108),
            patch_size=patch_size,
            in_chans=input_channels,
            num_classes=output_dim,
            embed_dim=embedding_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.3,
            attn_drop_rate=0.3,
            drop_path_rate=0.3,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x):
        return self.vit(x)
'''
# Example usage
model = SimpleViT(input_channels=1, output_dim=1)  # 1-channel input for grayscale images
input_tensor = torch.randn(32, 1, 144, 108)  # Batch size of 32
output = model(input_tensor)
print(output.shape)  # Should be [32, 1]
print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
'''








# ----------------------------------------------------------------------
#  1.  Little CNN that converts (C, H, W) → d_model
# ----------------------------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),      # → (B, 128, 1, 1)
            nn.Flatten(),                 # → (B, 128)
            nn.Linear(128, d_model),
        )

    def forward(self, x):                 # x: (B, C, H, W)
        return self.net(x)                # (B, d_model)


# ----------------------------------------------------------------------
#  2.  Positional encoding (sinusoidal – no extra params)
# ----------------------------------------------------------------------
def positional_encoding(max_len: int, d_model: int, device: torch.device):
    """Return (max_len, d_model) table."""
    pe = torch.zeros(max_len, d_model, device=device)
    pos = torch.arange(0, max_len, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device) *
                    (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ----------------------------------------------------------------------
#  3.  Complete model
# ----------------------------------------------------------------------
class TemporalTransformer(nn.Module):
    """
    (B, T, C, H, W)  →  predict y (vector or scalar)
    """
    def __init__(
        self,
        in_channels: int,
        output_dim: int = 1,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 512,
        max_seq_len: int = 120,      # plenty for monthly data
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False,       # (T, B, d)
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.register_buffer(
            "pos_embed",
            positional_encoding(max_seq_len, d_model, device="cpu"),
            persistent=False,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x):                    # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)           # merge batch & time
        feats = self.encoder(x)              # (B*T, d)
        feats = feats.view(B, T, -1)         # (B, T, d)
        feats = feats + self.pos_embed[:T]   # add PE  (broadcast on B)

        feats = feats.permute(1, 0, 2)       # → (T, B, d) for transformer
        feats = self.temporal(feats)         # (T, B, d)
        last = feats[-1]                     # last timestep, shape (B, d)
        return self.head(last)               # (B, output_dim)





