import torch
import torch.nn as nn

class MultiModalVLAPolicy(nn.Module):
    """
    Vision-Language-Action Policy conditioned on visual state,
    joint observations, and language embeddings. Outputs action chunks of size k=50.
    """
    def __init__(self, obs_dim=78, lang_dim=32, action_dim=14, chunk_size=50):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Multimodal fusion backbone
        self.fusion = nn.Sequential(
            nn.Linear(obs_dim + lang_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU()
        )

        # Action chunk temporal decoder
        self.action_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, chunk_size * action_dim)
        )

    def forward(self, state_obs: torch.Tensor, lang_goal: torch.Tensor) -> torch.Tensor:
        """
        state_obs: [B, 78] (14 joints + 64 vision embedding)
        lang_goal: [B, 32] (text embedding)
        Returns action chunk: [B, chunk_size, action_dim]
        """
        x = torch.cat([state_obs, lang_goal], dim=-1)
        feat = self.fusion(x)
        actions = self.action_head(feat)
        return actions.view(-1, self.chunk_size, self.action_dim)