import torch
import numpy as np

class DiffusionUtils:
    """
    Utilities for Gaussian Diffusion (DDPM style but adapted for score matching).
    """
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02, schedule='linear', device='cpu'):
        self.n_steps = n_steps
        self.device = device
        
        if schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, n_steps).to(device)
        elif schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(n_steps).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def _cosine_beta_schedule(self, n_steps, s=0.008):
        steps = n_steps + 1
        x = torch.linspace(0, n_steps, steps)
        alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: sample x_t from x_0 at time step t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def get_score_target(self, noise, t):
        """
        The target for score matching is -noise / sqrt(1 - alpha_cumprod).
        """
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return -noise / sqrt_one_minus_alphas_cumprod_t

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond_c):
        """
        Reverse diffusion sampling.
        """
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.n_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, cond_c)
            
        return img

    @torch.no_grad()
    def p_sample(self, model, x, t, cond_c):
        """
        Single step of reverse diffusion.
        Using the score model to derive the mean of the reverse distribution.
        """
        alpha_t = self.alphas[t].view(-1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
        beta_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # Model predicts the score function s(x, t, c)
        score = model(x, t, cond_c)
        
        # Relation between predicted noise and score: noise = -score * sqrt(1 - alpha_cumprod)
        # mean = 1/sqrt(alpha_t) * (x + beta_t * score)
        
        model_mean = (1.0 / torch.sqrt(alpha_t)) * (x + beta_t * score)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance = beta_t * (1.0 - self.alphas_cumprod_prev[t].view(-1, 1)) / (1.0 - alpha_cumprod_t)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * noise
