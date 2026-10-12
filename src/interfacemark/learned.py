"""Compact keyed verifier; explicit multi-loss training on cached design features."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .core import unit_carrier


@torch.no_grad()
def features(states: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """[N,C,H,W], [K,C,H,W] -> [N,K,17*C], band-major spectra."""
    x, k = states.float(), keys.to(states.device).float()
    if x.ndim != 4 or k.ndim != 4 or x.shape[1:] != k.shape[1:]:
        raise ValueError("states and keys must have matching C,H,W")
    _, c, h, w = x.shape
    if min(h, w) < 8:
        raise ValueError("spatial dimensions must be at least 8")
    aligned = torch.einsum('nchw,kchw->nkc', x, k) / (h*w)
    independent = torch.cat([x.mean((-2,-1)), x.std((-2,-1), unbiased=False),
        x.diff(dim=-2).abs().mean((-2,-1)), x.diff(dim=-1).abs().mean((-2,-1))], -1)
    xf, kf = torch.fft.rfft2(x, norm='ortho'), torch.fft.rfft2(k, norm='ortho')
    fy = torch.fft.fftfreq(h, device=x.device).abs()[:,None]
    fx = torch.fft.rfftfreq(w, device=x.device).abs()[None,:]
    radius = (fy.square()+fx.square()).sqrt()
    edges = torch.linspace(0, float(radius.max())+1e-7, 7, device=x.device)
    bands = [(radius>=edges[i]) & (radius<edges[i+1]) for i in range(6)]
    if not all(bool(b.any()) for b in bands):
        raise ValueError("empty radial band")
    cross = (xf[:,None]*kf[None].conj()).real
    cf = torch.cat([cross[...,b].mean(-1) for b in bands], -1)
    pf = torch.cat([torch.log1p(xf.abs().square()[...,b].mean(-1)) for b in bands], -1)
    result = torch.cat([aligned, independent[:,None].expand(-1,len(k),-1),
                        cf, pf[:,None].expand(-1,len(k),-1)], -1)
    if result.shape[-1] != 17*c or not torch.isfinite(result).all():
        raise ValueError("invalid compact features")
    return result


class Decoder(nn.Module):
    def __init__(self, dimension: int = 68):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension,192), nn.LayerNorm(192),
            nn.GELU(), nn.Dropout(.12), nn.Linear(192,64), nn.GELU(), nn.Linear(64,1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def multi_loss(sc, sw, swk, tc, tw, weights):
    cls = .5*(F.binary_cross_entropy_with_logits(sc,torch.zeros_like(sc),reduction='none')
              +F.binary_cross_entropy_with_logits(sw,torch.ones_like(sw),reduction='none'))
    rank = F.softplus(.5-sw+sc)
    teacher = .5*(F.smooth_l1_loss(sc,tc.detach(),reduction='none',beta=1.)
                  +F.smooth_l1_loss(sw,tw.detach(),reduction='none',beta=1.))
    wrong = F.binary_cross_entropy_with_logits(swk,torch.zeros_like(swk),reduction='none')
    return ((cls+.25*rank+.20*teacher+.15*wrong)*weights).sum()/weights.sum()


def validate_cache(data):
    clean = data['clean']
    if clean.ndim != 3 or clean.shape[-1] not in (68,272):
        raise ValueError("features must be [images,views,68 or 272]")
    n,v,_ = clean.shape
    if n < 2 or len(data['ids']) != n or len(set(data['ids'])) != n:
        raise ValueError("at least two images with unique IDs are required")
    if len(data['views']) != v or data['views'][0] != 'identity':
        raise ValueError("views must list identity first")
    for name in ('clean','marked','wrong_marked'):
        if data[name].shape != clean.shape or not torch.isfinite(data[name]).all():
            raise ValueError(f"invalid {name}")
    for name in ('projection_clean','projection_marked'):
        if data[name].shape != (n,) or not torch.isfinite(data[name]).all():
            raise ValueError(f"invalid identity {name}")


def train(train_data, val_data, output, epochs=160, patience=30, seed=2026093002, device='cpu'):
    for data in (train_data,val_data):
        validate_cache(data)
    if set(train_data['ids']) & set(val_data['ids']):
        raise ValueError("training/validation image IDs overlap")
    if train_data['views'] != val_data['views'] or train_data['clean'].shape[1:] != val_data['clean'].shape[1:]:
        raise ValueError("training/validation layouts differ")
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    torch.manual_seed(seed)
    d = train_data['clean'].shape[-1]
    population = torch.cat([train_data[k].reshape(-1,d) for k in ('clean','marked')]).double()
    mean, std = population.mean(0).float(), population.std(0,unbiased=False).float().clamp_min(1e-5)
    teacher = train_data['projection_clean'].float()
    tm, ts = teacher.mean(), teacher.std(unbiased=False).clamp_min(1e-5)
    def prepare(data):
        n,v,_ = data['clean'].shape
        tensors = [((data[k].float()-mean)/std).reshape(-1,d).to(device)
                   for k in ('clean','marked','wrong_marked')]
        tensors += [((data[k].float()-tm)/ts)[:,None].expand(-1,v).reshape(-1).to(device)
                    for k in ('projection_clean','projection_marked')]
        weights = torch.ones(n,v); weights[:,0] = 3.
        return tensors+[weights.reshape(-1).to(device)]
    tr, va = prepare(train_data), prepare(val_data)
    model = Decoder(d).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    best, best_epoch, history, best_state = float('inf'), -1, [], None
    def objective(batch):
        c,w,wk,tc,tw,weight = batch
        return multi_loss(model(c),model(w),model(wk),tc,tw,weight)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tr[0]),device=device)
        for ids in order.split(128):
            opt.zero_grad(set_to_none=True)
            loss = objective([x[ids] for x in tr])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
        model.eval()
        with torch.no_grad():
            value = float(objective(va))
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("nonfinite validation loss")
        history.append(value)
        if value < best-1e-5:
            best, best_epoch = value, epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch >= patience:
            break
    payload = dict(model=best_state,mean=mean,std=std,dimension=d,
        teacher_mean=tm,teacher_std=ts,best_epoch=best_epoch,validation_loss=history,
        protocol='multi_loss_identity_teacher_v1',views=train_data['views'],seed=seed,
        train_ids=train_data['ids'],validation_ids=val_data['ids'],
        loss_weights=dict(cls=1.,rank=.25,teacher=.20,wrong=.15))
    output = Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,output)
    return payload


@torch.no_grad()
def score(feature_values, checkpoint):
    d = checkpoint['dimension']
    if feature_values.shape[-1] != d:
        raise ValueError("checkpoint feature dimension mismatch")
    model = Decoder(d).to(feature_values.device).eval()
    model.load_state_dict(checkpoint['model'])
    mean, std = [checkpoint[k].to(feature_values.device) for k in ('mean','std')]
    return model((feature_values.float()-mean)/std)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    t = sub.add_parser('train',help='train from separate design feature caches')
    t.add_argument('--train',required=True); t.add_argument('--validation',required=True)
    t.add_argument('--output',required=True); t.add_argument('--epochs',type=int,default=160)
    t.add_argument('--device',default='cpu')
    s = sub.add_parser('score',help='score saved VAE latents (not initial noise)')
    s.add_argument('--latents',required=True); s.add_argument('--checkpoint',required=True)
    s.add_argument('--key',type=int,required=True); s.add_argument('--output',required=True)
    args = parser.parse_args()
    def load(path):
        return torch.load(path,map_location='cpu',weights_only=True)
    if args.command == 'train':
        result = train(load(args.train),load(args.validation),args.output,epochs=args.epochs,device=args.device)
        print(f"Saved {args.output}; best epoch {result['best_epoch']}")
    else:
        x = load(args.latents)
        k = unit_carrier((1,*x.shape[1:]),args.key)
        values = score(features(x,k)[:,0],load(args.checkpoint))
        torch.save(values,args.output)
        print(f"Saved {len(values)} continuous scores; calibrate on independent clean images.")


if __name__ == '__main__':
    main()
