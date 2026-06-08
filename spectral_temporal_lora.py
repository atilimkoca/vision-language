"""
spectral_temporal_lora.py — Spectral-Temporal LoRA (ST-LoRA) çekirdeği
=======================================================================
Standart LoRA:        ΔW   = (α/r) · B A            (tüm timestep'ler için sabit)
ST-LoRA (öneri):      ΔW(t) = (α/r) · B diag(s(t)) A

  • s(t) ∈ R^r : timestep'e bağlı per-rank kapı (küçük MLP, sinüzoidal t-gömme).
  • s(t) ≡ 1 başlangıcı → standart LoRA'ya KESİN indirgenir (B=0 ile ΔW(0 anı)=0).
  • Difüzyon-özgü: aynı düşük-rank bütçesini gürültü seviyesi boyunca uzmanlaştırır
    (yüksek t = global yapı, düşük t = ince doku).

ΔW(t) merge EDİLEMEZ (t'ye bağlı) → PEFT yerine custom katman. UNet forward'ına
o anki timestep'i bir forward-pre-hook ile iletiriz; her ST-LoRA katmanı kendi
gate'ini hesaplar.

Hem eğitimde (train_stlora.py) hem de değerlendirmede (base SD'ye enjekte +
state yükle + hook) kullanılır.
"""

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

UNET_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
TEXT_ENCODER_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")
TIME_EMB_DIM = 128


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int = TIME_EMB_DIM) -> torch.Tensor:
    """t: [B] (long/float) → [B, dim] sinüzoidal gömme (difüzyon standardı)."""
    t = t.float().view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb   # [B, dim]


class STLoRALinear(nn.Module):
    """Donuk base Linear + düşük-rank A,B + (opsiyonel) timestep-modülasyonlu gate.

    out = base(x) + (α/r) · B diag(s(t)) A x
    timestep_gate=False ise s≡1 → standart LoRA (timestep-bağımsız bileşenler için).
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float,
                 timestep_gate: bool = True, gate_hidden: int = 32):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        d_in = base_linear.in_features
        d_out = base_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.timestep_gate = timestep_gate

        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)   # B=0 → ΔW başlangıçta 0 (LoRA standardı)

        if timestep_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(TIME_EMB_DIM, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, rank),
            )
            # Son katman sıfır → gate çıkışı 0 → s = 1 + 0 = 1 (başlangıçta standart LoRA)
            nn.init.zeros_(self.gate_net[-1].weight)
            nn.init.zeros_(self.gate_net[-1].bias)

        self._t = None   # forward-pre-hook tarafından her adımda set edilir

    def set_timestep(self, t):
        self._t = t

    def _gate(self, batch: int, dtype, device):
        if not self.timestep_gate or self._t is None:
            return None
        t = self._t
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device)
        if t.numel() == 1 and batch > 1:
            t = t.expand(batch)
        emb = sinusoidal_timestep_embedding(t.to(device), TIME_EMB_DIM)
        s = 1.0 + self.gate_net(emb)          # [B, rank], init ≈ 1
        return s.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        ax = F.linear(x, self.lora_A.to(x.dtype))          # [..., rank]
        s = self._gate(x.shape[0], x.dtype, x.device)
        if s is not None:
            view = [x.shape[0]] + [1] * (x.dim() - 2) + [self.rank]
            ax = ax * s.view(view)                          # per-sample, per-rank modülasyon
        delta = F.linear(ax, self.lora_B.to(x.dtype))      # [..., d_out]
        return out + self.scaling * delta


def _iter_target_modules(model, targets):
    """(parent_module, attr_name, full_name) — adı targets ile biten Linear'lar."""
    for name, module in list(model.named_modules()):
        for tgt in targets:
            if name.endswith(tgt) and isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                parent = model.get_submodule(parent_name) if parent_name else model
                attr = name.rsplit(".", 1)[-1]
                yield parent, attr, name
                break


def inject_stlora(model, targets, rank, alpha, timestep_gate) -> Dict[str, STLoRALinear]:
    """Hedef Linear'ları STLoRALinear ile değiştirir. {full_name: layer} döndürür."""
    injected = {}
    for parent, attr, full_name in _iter_target_modules(model, targets):
        base = getattr(parent, attr)
        new = STLoRALinear(base, rank=rank, alpha=alpha, timestep_gate=timestep_gate)
        setattr(parent, attr, new)
        injected[full_name] = new
    return injected


def attach_timestep_hook(unet, layers: List[STLoRALinear]):
    """UNet forward'ından önce o anki timestep'i tüm ST-LoRA katmanlarına stash eder.

    UNet forward imzası: forward(sample, timestep, encoder_hidden_states, ...).
    Hem pozisyonel hem kwargs çağrısını destekler (eğitim + diffusers pipeline).
    """
    def _pre_hook(module, args, kwargs):
        t = None
        if len(args) >= 2:
            t = args[1]
        elif "timestep" in kwargs:
            t = kwargs["timestep"]
        if t is not None:
            for layer in layers:
                layer.set_timestep(t)
    try:
        return unet.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    except TypeError:
        # eski torch: with_kwargs yok → sadece pozisyonel
        def _pre_hook_posonly(module, args):
            if len(args) >= 2:
                for layer in layers:
                    layer.set_timestep(args[1])
        return unet.register_forward_pre_hook(_pre_hook_posonly)


def trainable_parameters(injected_dicts: List[Dict[str, STLoRALinear]]):
    params = []
    for d in injected_dicts:
        for layer in d.values():
            params += [p for p in layer.parameters() if p.requires_grad]
    return params


def save_stlora(injected: Dict[str, STLoRALinear], config: dict, path: str):
    """Sadece ST-LoRA parametrelerini (lora_A/B + gate_net) + config kaydeder."""
    state = {}
    for name, layer in injected.items():
        sub = {"lora_A": layer.lora_A.detach().cpu(), "lora_B": layer.lora_B.detach().cpu()}
        if layer.timestep_gate:
            sub["gate_net"] = {k: v.detach().cpu() for k, v in layer.gate_net.state_dict().items()}
        state[name] = sub
    torch.save({"state": state, "config": config}, path)


def load_stlora_into(injected: Dict[str, STLoRALinear], ckpt_path: str, device):
    """Kaydedilmiş ST-LoRA parametrelerini enjekte edilmiş katmanlara yükler."""
    blob = torch.load(ckpt_path, map_location="cpu")
    state = blob["state"]
    for name, layer in injected.items():
        if name not in state:
            continue
        sub = state[name]
        layer.lora_A.data.copy_(sub["lora_A"].to(layer.lora_A.dtype))
        layer.lora_B.data.copy_(sub["lora_B"].to(layer.lora_B.dtype))
        if layer.timestep_gate and "gate_net" in sub:
            layer.gate_net.load_state_dict({k: v for k, v in sub["gate_net"].items()})
    return blob.get("config", {})
