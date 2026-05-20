"""
ddpm_mnist.py — Minimal MNIST DDPM  (Lecture 4, Q3)

Setup : MNIST 28×28 grayscale, normalised to [-1, 1]
        T = 200, linear β-schedule
        Simple time-conditioned U-Net (residual blocks + sinusoidal time emb.)

Usage : python ddpm_mnist.py
Output: outputs/  ← forward_noise.png, reverse_traj.png, generated.png, loss_curve.png
"""

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
T          = 200
BETA_START = 1e-4
BETA_END   = 0.02
EPOCHS     = 15
BATCH      = 128
LR         = 1e-3
DEVICE     = ("cuda" if torch.cuda.is_available()
              else "mps"  if torch.backends.mps.is_available()
              else "cpu")
OUT        = "outputs"
os.makedirs(OUT, exist_ok=True)
print(f"Device: {DEVICE}")

# ── Noise schedule ─────────────────────────────────────────────────────────────
betas     = torch.linspace(BETA_START, BETA_END, T, device=DEVICE)  # β_1 … β_T
alphas    = 1.0 - betas                                               # α_t
alpha_bar = torch.cumprod(alphas, dim=0)                              # ᾱ_t
sqrt_ab   = alpha_bar.sqrt()                                          # √ᾱ_t
sqrt_1mab = (1.0 - alpha_bar).sqrt()                                  # √(1-ᾱ_t)

# ── Forward process  q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I) ─────────────────
def q_sample(x0, t, eps=None):
    if eps is None:
        eps = torch.randn_like(x0)
    a = sqrt_ab[t].view(-1, 1, 1, 1)
    b = sqrt_1mab[t].view(-1, 1, 1, 1)
    return a * x0 + b * eps, eps

# ── Sinusoidal time embedding ──────────────────────────────────────────────────
class SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) *
                      torch.arange(half, device=t.device) / (half - 1))
        e = t.float()[:, None] * f[None, :]
        return torch.cat([e.sin(), e.cos()], dim=1)

# ── Residual block with time conditioning ─────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, out_ch)
        self.n2 = nn.GroupNorm(8, out_ch)
        self.tp = nn.Linear(t_dim, out_ch)        # time → channel scale
        self.sk = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.tp(t_emb).view(-1, h.shape[1], 1, 1)
        h = F.silu(self.n2(self.c2(h)))
        return h + self.sk(x)

# ── UNet (encoder–bottleneck–decoder with skip connections) ────────────────────
class UNet(nn.Module):
    """
    Input : (B, 1, 28, 28)  +  timestep t ∈ {0,…,T-1}
    Output: (B, 1, 28, 28)  — predicted noise ε_θ(x_t, t)
    """
    def __init__(self, t_dim=128):
        super().__init__()
        self.temb = nn.Sequential(
            SinEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        # Encoder  28→14→7
        self.e1 = ResBlock(1,   32,  t_dim)
        self.e2 = ResBlock(32,  64,  t_dim)
        self.e3 = ResBlock(64,  128, t_dim)
        # Bottleneck
        self.bt = ResBlock(128, 128, t_dim)
        # Decoder  7→14→28  (skip connections from encoder)
        self.d2 = ResBlock(128 + 64, 64, t_dim)   # cat(up(b), e2)
        self.d1 = ResBlock(64  + 32, 32, t_dim)   # cat(up(d2), e1)
        self.out = nn.Conv2d(32, 1, 1)
        self.dn  = nn.MaxPool2d(2)
        self.up  = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        te = self.temb(t)
        e1 = self.e1(x,            te)              # (B,32,28,28)
        e2 = self.e2(self.dn(e1),  te)              # (B,64,14,14)
        e3 = self.e3(self.dn(e2),  te)              # (B,128, 7, 7)
        b  = self.bt(e3,           te)              # (B,128, 7, 7)
        d2 = self.d2(torch.cat([self.up(b),  e2], 1), te)  # (B,64,14,14)
        d1 = self.d1(torch.cat([self.up(d2), e1], 1), te)  # (B,32,28,28)
        return self.out(d1)

# ── Reverse sampling  p_θ(x_{t-1} | x_t) ─────────────────────────────────────
@torch.no_grad()
def p_sample(model, x, t_idx):
    """One DDPM reverse step."""
    t      = torch.full((x.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
    ep     = model(x, t)
    beta   = betas[t_idx]
    ab     = alpha_bar[t_idx]
    # μ_θ = (1/√α_t)(x_t − β_t/√(1-ᾱ_t) · ε_θ)
    mean   = (x - beta / (1 - ab).sqrt() * ep) / alphas[t_idx].sqrt()
    if t_idx == 0:
        return mean
    return mean + beta.sqrt() * torch.randn_like(x)

@torch.no_grad()
def sample(model, n, return_traj=False):
    """Full reverse denoising from x_T ~ N(0,I) to x_0."""
    x = torch.randn(n, 1, 28, 28, device=DEVICE)
    snap = {T - 1: x.clone()}
    for t in range(T - 1, -1, -1):
        x = p_sample(model, x, t)
        if return_traj and t in {150, 100, 50, 20, 0}:
            snap[t] = x.clone()
    return (x, snap) if return_traj else x

# ── Visualisation helpers ──────────────────────────────────────────────────────
def save_grid(tensor, path, nrow=8):
    g = make_grid(tensor.clamp(-1, 1), nrow=nrow,
                  normalize=True, value_range=(-1, 1))
    save_image(g, path)
    print(f"Saved {path}")

def make_forward_noise_grid(x0):
    """One MNIST image progressively corrupted at various timesteps."""
    steps = [0, 25, 50, 75, 100, 125, 150, 175, 199]
    imgs  = []
    for s in steps:
        t    = torch.tensor([s], device=DEVICE)
        xt,_ = q_sample(x0[:1], t)
        imgs.append(xt)
    imgs = torch.cat(imgs, dim=0)
    save_grid(imgs, f"{OUT}/forward_noise.png", nrow=len(steps))

# ── Data (MNIST, normalised to [-1, 1]) ───────────────────────────────────────
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                      num_workers=0, pin_memory=False)

# ── Training loop  L = ||ε − ε_θ(x_t, t)||² ──────────────────────────────────
model = UNet().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

losses = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for x0, _ in train_dl:
        x0        = x0.to(DEVICE)
        t         = torch.randint(0, T, (x0.shape[0],), device=DEVICE)
        xt, eps   = q_sample(x0, t)
        loss      = F.mse_loss(model(xt, t), eps)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x0.shape[0]
    sched.step()
    avg = total / len(train_ds)
    losses.append(avg)
    print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}")

torch.save(model.state_dict(), f"{OUT}/model.pt")
print(f"Model saved to {OUT}/model.pt")

# ── Loss curve ────────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 3))
plt.plot(range(1, EPOCHS + 1), losses, marker="o", ms=4, color="steelblue")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Training Loss (MNIST DDPM)")
plt.tight_layout()
plt.savefig(f"{OUT}/loss_curve.png", dpi=150)
plt.close()
print(f"Saved {OUT}/loss_curve.png")

# ── Forward noising grid ───────────────────────────────────────────────────────
x0_demo, _ = next(iter(train_dl))
x0_demo = x0_demo.to(DEVICE)
make_forward_noise_grid(x0_demo)

# ── Reverse denoising trajectory ──────────────────────────────────────────────
model.eval()
_, snap = sample(model, 4, return_traj=True)
ordered   = sorted(snap.keys(), reverse=True)           # T-1 → … → 0
traj_imgs = torch.cat([snap[k] for k in ordered], dim=0)
save_grid(traj_imgs, f"{OUT}/reverse_traj.png", nrow=len(ordered))

# ── Generated samples (8×8 grid) ─────────────────────────────────────────────
gen = sample(model, 64)
save_grid(gen, f"{OUT}/generated.png", nrow=8)

print("\nAll outputs saved to", OUT)
