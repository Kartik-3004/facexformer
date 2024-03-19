import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Any, Optional, Tuple, Type
from torchvision.models import swin_b
from .transformer import TwoWayTransformer, LayerNorm2d

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
class FaceDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: 256,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.landmarks_token = nn.Embedding(1, transformer_dim)
        self.pose_token = nn.Embedding(1, transformer_dim)
        self.attribute_token = nn.Embedding(1, transformer_dim)
        self.visibility_token = nn.Embedding(1, transformer_dim)
        self.age_token = nn.Embedding(1, transformer_dim)
        self.gender_token = nn.Embedding(1, transformer_dim)
        self.race_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(11, transformer_dim)
        

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        self.output_hypernetwork_mlps = MLP(
            transformer_dim, transformer_dim, transformer_dim // 8, 3
            )
                
        self.landmarks_prediction_head = MLP(
            transformer_dim, transformer_dim, 136, 3
        )
        self.pose_prediction_head = MLP(
            transformer_dim, transformer_dim, 3, 3
        )
        self.attribute_prediction_head = MLP(
            transformer_dim, transformer_dim, 40, 3
        )
        self.visibility_prediction_head = MLP(
            transformer_dim, transformer_dim, 29, 3
        )
        self.age_prediction_head = MLP(
            transformer_dim, transformer_dim, 8, 3
        )
        self.gender_prediction_head = MLP(
            transformer_dim, transformer_dim, 2, 3
        )
        self.race_prediction_head = MLP(
            transformer_dim, transformer_dim, 5, 3
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.landmarks_token.weight, self.pose_token.weight, self.attribute_token.weight, self.visibility_token.weight, self.age_token.weight, self.gender_token.weight, self.race_token.weight,self.mask_tokens.weight], dim=0) 
        tokens = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)

        src = image_embeddings
        pos_src = image_pe.expand(image_embeddings.size(0), -1, -1, -1)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
    
        landmarks_token_out = hs[:, 0, :]
        pose_token_out =  hs[:, 1, :]
        attribute_token_out = hs[:, 2, :]
        visibility_token_out = hs[:, 3, :]
        age_token_out = hs[:, 4, :]
        gender_token_out = hs[:, 5, :]
        race_token_out = hs[:, 6, :]
        mask_token_out =  hs[:, 7:, :]
        
        
        landmark_output = self.landmarks_prediction_head(landmarks_token_out)
        headpose_output = self.pose_prediction_head(pose_token_out)
        attribute_output = self.attribute_prediction_head(attribute_token_out)
        visibility_output = self.visibility_prediction_head(visibility_token_out)
        age_output = self.age_prediction_head(age_token_out)
        gender_output = self.gender_prediction_head(gender_token_out)
        race_output = self.race_prediction_head(race_token_out)
        
        src = src.transpose(1, 2).view(b, c, h, w) 
        upscaled_embedding = self.output_upscaling(src)  
        hyper_in = self.output_hypernetwork_mlps(mask_token_out)
        b, c, h, w = upscaled_embedding.shape
        seg_output = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        
        return landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class FaceXFormerMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states

class FaceXFormer(nn.Module):
    def __init__(self):
        super(FaceXFormer, self).__init__()

        swin_v2 = swin_b(weights='IMAGENET1K_V1')
        self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
        self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
        self.multi_scale_features = []
        

        embed_dim = 1024
        out_chans = 256
        
        self.pe_layer = PositionEmbeddingRandom(out_chans // 2)   

        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name))
        
        self.face_decoder = FaceDecoder(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ))    
        
        num_encoder_blocks = 4
        hidden_sizes = [128, 256, 512, 1024]
        decoder_hidden_size = 256
        
        mlps = []
        for i in range(num_encoder_blocks):
            mlp = FaceXFormerMLP(input_dim=hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_size * num_encoder_blocks,
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
    
    def save_features_hook(self, name):
        def hook(module, input, output):
            self.multi_scale_features.append(output.permute(0,3,1,2).contiguous()) 
        return hook

    def forward(self, x, labels, tasks):
        self.multi_scale_features.clear()
        
        _,_,h,w = x.shape
        features = self.backbone(x).squeeze()
        
        
        batch_size = self.multi_scale_features[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
        
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=self.multi_scale_features[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
        
        fused_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        image_pe = self.pe_layer((fused_states.shape[2], fused_states.shape[3])).unsqueeze(0)
        
        landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output = self.face_decoder(
                image_embeddings=fused_states,
                image_pe=image_pe
            )
        
        segmentation_indices = (tasks == 0)
        seg_output = seg_output[segmentation_indices]
        
        landmarks_indices = (tasks == 1)
        landmark_output = landmark_output[landmarks_indices]

        headpose_indices = (tasks == 2)
        headpose_output = headpose_output[headpose_indices]
        
        attribute_indices = (tasks == 3)
        attribute_output = attribute_output[attribute_indices]

        age_indices = (tasks == 4)
        age_output = age_output[age_indices]
        gender_output = gender_output[age_indices]
        race_output = race_output[age_indices]
        
        visibility_indices = (tasks == 5)
        visibility_output = visibility_output[visibility_indices]
    
        return landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output