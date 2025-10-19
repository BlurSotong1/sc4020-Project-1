# rgat_lora_memsafe.py
# LoRA + R-GAT with memory-safe per-relation streaming (Fix 1).
# Trains on WN18RR, FB15K-237, Cora-KG (TSV triples: head \t rel \t tail)
# Saves checkpoints + text reports with AUC and Hits@K.

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


# -------------------------
# Utilities
# -------------------------
def load_triples(path: Path) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.rstrip("\n").split("\t")
            triples.append((h, r, t))
    return triples


def build_id_maps(*triple_lists: List[Tuple[str, str, str]]):
    ents, rels = set(), set()
    for triples in triple_lists:
        for h, r, t in triples:
            ents.add(h); ents.add(t); rels.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(ents))}
    rel2id = {r: i for i, r in enumerate(sorted(rels))}
    return ent2id, rel2id


def triples_to_tensor(triples, ent2id, rel2id, device):
    arr = np.array([(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triples], dtype=np.int64)
    return torch.from_numpy(arr).to(device)


def batches(tensor: torch.Tensor, batch_size: int, shuffle: bool = True):
    N = tensor.size(0)
    idx = torch.randperm(N, device=tensor.device) if shuffle else torch.arange(N, device=tensor.device)
    for i in range(0, N, batch_size):
        yield tensor[idx[i:i + batch_size]]


@torch.no_grad()
def sample_negatives_both(pos_triples: torch.Tensor, num_entities: int, k_neg: int = 10):
    """Return k_neg head- and tail-corrupted negatives per positive (flattened)."""
    B = pos_triples.size(0); dev = pos_triples.device
    # tail corrupt
    tails = torch.randint(0, num_entities, (B, k_neg), device=dev)
    neg_t = pos_triples.unsqueeze(1).expand(B, k_neg, 3).clone()
    neg_t[:, :, 2] = tails
    # head corrupt
    heads = torch.randint(0, num_entities, (B, k_neg), device=dev)
    neg_h = pos_triples.unsqueeze(1).expand(B, k_neg, 3).clone()
    neg_h[:, :, 0] = heads
    return neg_h.reshape(-1, 3), neg_t.reshape(-1, 3)


def _fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"


# -------------------------
# Fix 1: Memory-safe LoRA-RGAT convolution (per-relation streaming)
# -------------------------
class RelationalLoRAGATConv(MessagePassing):
    """
    Memory-safe LoRA + relation-aware GAT.
    Processes edges **per relation** so we never materialize [E, d*k].
    LoRA is applied in INPUT space with A_r, B_r shared by all edges of relation r:
        x_j' = x_j + scale * A_r @ (B_r^T x_j)
    """
    def __init__(self, in_dim, out_dim, num_relations,
                 heads=4, rank=8, adapter_scale=1.0, dropout=0.2,
                 concat=True, negative_slope=0.2, bias=True):
        super().__init__(node_dim=0, aggr="add")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.heads, self.rank = heads, rank
        self.adapter_scale = adapter_scale
        self.concat = concat
        self.dropout = dropout

        # Per-relation LoRA (d x k) each
        self.A = nn.Embedding(num_relations, in_dim * rank)
        self.B = nn.Embedding(num_relations, in_dim * rank)
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)

        # Shared projection
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

        # Attention params
        self.att_l = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_r = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_uniform_(self.att_l); nn.init.xavier_uniform_(self.att_r)

        # Relation bias per head
        self.rel_head_bias = nn.Embedding(num_relations, heads)
        nn.init.zeros_(self.rel_head_bias.weight)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.feat_drop = nn.Dropout(dropout)
        self.att_drop = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, edge_index, edge_type):
        """
        x: [N, d_in], edge_index: [2, E], edge_type: [E]
        """
        N, H, C = x.size(0), self.heads, self.out_dim

        # Output accumulator in head space
        out = x.new_zeros(N, H, C)

        # Process each relation in turn to cap peak memory
        rel_ids = edge_type.unique()

        for r in rel_ids.tolist():
            mask = (edge_type == r)
            if not mask.any():
                continue

            eidx = edge_index[:, mask]          # [2, E_r]
            j_idx, i_idx = eidx[0], eidx[1]     # source, target

            # LoRA for this relation (A_r, B_r)
            A_r = self.A.weight[r].view(self.in_dim, self.rank)    # [d, k]
            B_r = self.B.weight[r].view(self.in_dim, self.rank)    # [d, k]

            # x_j' = x_j + scale * A_r @ (B_r^T x_j)
            x_j = x[j_idx]                                        # [E_r, d]
            BTx = x_j @ B_r                                       # [E_r, k]
            lora_delta = BTx @ A_r.T                               # [E_r, d]
            x_j_adapt = x_j + self.adapter_scale * lora_delta      # [E_r, d]

            # Project to head space
            z_j = self.lin(x_j_adapt).view(-1, H, C)               # [E_r, H, C]
            z_i = self.lin(x[i_idx]).view(-1, H, C)                # [E_r, H, C]

            # Attention logits + head bias for this relation
            alpha = (z_i * self.att_l).sum(-1) + (z_j * self.att_r).sum(-1)  # [E_r, H]
            alpha = alpha + self.rel_head_bias.weight[r].view(1, H)
            alpha = self.leaky_relu(alpha)

            # Normalize over incoming edges per node, per head
            alpha = softmax(alpha, i_idx)                          # [E_r, H]
            alpha = self.att_drop(alpha)

            # Weighted messages
            m = z_j * alpha.unsqueeze(-1)                          # [E_r, H, C]

            # Aggregate to out
            out.index_add_(0, i_idx, m)

        if self.concat:
            out = out.view(N, H * C)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out


# -------------------------
# Encoder (2-layer stack)
# -------------------------
class LoRARelationalGATEncoder(nn.Module):
    def __init__(self, num_entities: int, num_relations: int,
                 emb_dim: int = 128, hidden_dim: int = 128, out_dim: int = 256,
                 heads: int = 4, rank: int = 8, adapter_scale: float = 1.0, dropout: float = 0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        self.conv1 = RelationalLoRAGATConv(
            in_dim=emb_dim, out_dim=hidden_dim, num_relations=num_relations,
            heads=heads, rank=rank, adapter_scale=adapter_scale, dropout=dropout, concat=True)
        self.conv2 = RelationalLoRAGATConv(
            in_dim=hidden_dim * heads, out_dim=out_dim, num_relations=num_relations,
            heads=1, rank=rank, adapter_scale=adapter_scale, dropout=dropout, concat=False)

        self.res_proj = nn.Linear(emb_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor):
        x0 = self.entity_emb.weight
        x = self.conv1(x0, edge_index, edge_type)
        x = F.elu(x)
        x = self.drop(x)
        x = self.conv2(x, edge_index, edge_type)
        x = self.ln(x + self.res_proj(x0))
        return x  # [N, out_dim]


# -------------------------
# Decoders
# -------------------------
class DistMultDecoder(nn.Module):
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, e_h: torch.Tensor, r: torch.Tensor, e_t: torch.Tensor):
        w = self.rel(r)
        return (e_h * w * e_t).sum(dim=1)


class DotProductDecoder(nn.Module):
    def forward(self, e_h: torch.Tensor, r: torch.Tensor, e_t: torch.Tensor):
        return (e_h * e_t).sum(dim=1)


# -------------------------
# Link predictor (encoder+decoder)
# -------------------------
class LinkPredictor(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 edge_index: torch.Tensor, edge_type: torch.Tensor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.edge_index = edge_index
        self.edge_type = edge_type

    def forward(self, triples: torch.Tensor):
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        ent = self.encoder(self.edge_index, self.edge_type)
        return self.decoder(ent[h], r, ent[t])


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate_auc_hits(model: LinkPredictor, triples: torch.Tensor, num_entities: int,
                      batch_size: int = 4096, ks=(1, 5, 10)) -> Dict[str, float]:
    model.eval()

    # --- AUC (1:1 pos/neg, tail corrupt) ---
    scores_all, labels_all = [], []
    for pos in batches(triples, batch_size, shuffle=False):
        B = pos.size(0)
        neg = pos.clone()
        neg[:, 2] = torch.randint(0, num_entities, (B,), device=pos.device)

        s_pos = model(pos)
        s_neg = model(neg)

        scores_all.append(torch.cat([s_pos, s_neg], dim=0).detach().cpu().numpy())
        labels_all.append(np.concatenate([np.ones(B), np.zeros(B)], axis=0))

    auc = float(roc_auc_score(np.concatenate(labels_all), np.concatenate(scores_all)))

    # --- Hits@K (tail ranking vs 99 random) ---
    ent = model.encoder(model.edge_index, model.edge_type)
    hits = {k: 0 for k in ks}; trials = 0
    for pos in batches(triples, batch_size, shuffle=False):
        B = pos.size(0)
        h = pos[:, 0]; r = pos[:, 1]; t_true = pos[:, 2]
        rand_t = torch.randint(0, num_entities, (B, 99), device=pos.device)
        cand_t = torch.cat([t_true.view(-1, 1), rand_t], dim=1)

        e_h = ent[h]; e_c = ent[cand_t]  # [B,d], [B,100,d]
        if isinstance(model.decoder, DistMultDecoder):
            w = model.decoder.rel(r)
            s = ((e_h * w).unsqueeze(1) * e_c).sum(dim=2)
        else:
            s = (e_h.unsqueeze(1) * e_c).sum(dim=2)

        ranks = (s.argsort(dim=1, descending=True) == 0).nonzero()[:, 1] + 1
        for k in ks:
            hits[k] += (ranks <= k).sum().item()
        trials += B

    out = {"AUC": auc}
    for k in ks:
        out[f"Hits@{k}"] = hits[k] / max(trials, 1)
    return out


# -------------------------
# Training
# -------------------------
def train_epoch(model: LinkPredictor, triples: torch.Tensor,
                num_entities: int, optimizer: torch.optim.Optimizer,
                batch_size: int = 2048, k_neg: int = 10) -> float:
    model.train()
    pos_weight = torch.tensor([2.0 * k_neg], device=triples.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total, count = 0.0, 0
    for pos in batches(triples, batch_size, shuffle=True):
        neg_h, neg_t = sample_negatives_both(pos, num_entities, k_neg=k_neg)
        all_trip = torch.cat([pos, neg_h, neg_t], dim=0)
        labels = torch.cat([
            torch.ones(len(pos), device=triples.device),
            torch.zeros(len(neg_h) + len(neg_t), device=triples.device)
        ], dim=0)

        optimizer.zero_grad(set_to_none=True)
        scores = model(all_trip)
        loss = criterion(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += float(loss.item()) * labels.numel()
        count += int(labels.numel())
    return total / max(count, 1)


def train_run(
        dataset_name: str,
        train_p: Path, valid_p: Path, test_p: Path,
        *,
        outdir: Path = Path("results/rgat-lora"),
        epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 10,
        emb_dim: int = 128, hidden_dim: int = 128, out_dim: int = 256,
        heads: int = 4, rank: int = 8, adapter_scale: float = 1.0, dropout: float = 0.2,
        batch_size: int = 2048, k_neg: int = 10,
        use_distmult: bool = True,
) -> Dict[str, Any]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(outdir / dataset_name, exist_ok=True)
    print(f"\n=== {dataset_name} | device: {device} ===")

    # --- Load data ---
    train_list = load_triples(train_p)
    valid_list = load_triples(valid_p)
    test_list = load_triples(test_p)
    ent2id, rel2id = build_id_maps(train_list, valid_list, test_list)
    num_entities, num_relations = len(ent2id), len(rel2id)

    train = triples_to_tensor(train_list, ent2id, rel2id, device)
    valid = triples_to_tensor(valid_list, ent2id, rel2id, device)
    test = triples_to_tensor(test_list, ent2id, rel2id, device)

    # Unified edge_index + types with reverse edges
    src, dst, typ = [], [], []
    for h, r, t in train.tolist():
        src.extend([h, t]); dst.extend([t, h]); typ.extend([r, r])
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type = torch.tensor(typ, dtype=torch.long, device=device)

    # --- Model ---
    enc = LoRARelationalGATEncoder(
        num_entities=num_entities, num_relations=num_relations,
        emb_dim=emb_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        heads=heads, rank=rank, adapter_scale=adapter_scale, dropout=dropout
    ).to(device)
    dec = DistMultDecoder(num_relations, out_dim).to(device) if use_distmult else DotProductDecoder().to(device)
    model = LinkPredictor(enc, dec, edge_index, edge_type).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best = {"epoch": 0, "AUC": -1.0, "Hits@1": 0.0, "Hits@5": 0.0, "Hits@10": 0.0}
    best_state = None
    patience_ctr = 0

    for epoch in tqdm(range(1, epochs + 1), desc=f"{dataset_name}-LoRA-RGAT"):
        tr_loss = train_epoch(model, train, num_entities, opt, batch_size=batch_size, k_neg=k_neg)
        val = evaluate_auc_hits(model, valid, num_entities, batch_size=4096)

        history.append({
            "epoch": epoch,
            "train_loss": float(tr_loss),
            "val_auc": float(val["AUC"]),
            "val_hits1": float(val["Hits@1"]),
            "val_hits5": float(val["Hits@5"]),
            "val_hits10": float(val["Hits@10"]),
        })
        print(f"Epoch {epoch:03d} | loss={tr_loss:.4f} | "
              f"AUC={val['AUC']:.4f} | H@1={val['Hits@1']:.4f} | H@5={val['Hits@5']:.4f} | H@10={val['Hits@10']:.4f}")

        # Early stop on AUC
        if val["AUC"] > best["AUC"]:
            best.update({"epoch": epoch, **val})
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
            print(f"  → new best AUC: {best['AUC']:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    # Restore best & test
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_auc_hits(model, test, num_entities, batch_size=4096)

    # Save checkpoint
    ckpt_path = outdir / dataset_name / f"lora_rgat_{'distmult' if use_distmult else 'dot'}_d{out_dim}_H{heads}_r{rank}.pt"
    torch.save({"state_dict": best_state, "best": best, "test": test_metrics,
                "hparams": {"emb_dim": emb_dim, "hidden_dim": hidden_dim, "out_dim": out_dim,
                            "heads": heads, "rank": rank, "adapter_scale": adapter_scale,
                            "dropout": dropout, "decoder": "DistMult" if use_distmult else "Dot"}},
               ckpt_path)

    # Save report
    rep_path = outdir / dataset_name / f"report_{'distmult' if use_distmult else 'dot'}.txt"
    lines = []
    lines.append(f"LoRA R-GAT — {dataset_name}")
    lines.append("=" * 70)
    lines.append(f"Best (valid) @ epoch {best['epoch']}: "
                 f"AUC={_fmt(best['AUC'])} | H@1={_fmt(best['Hits@1'])} | H@5={_fmt(best['Hits@5'])} | H@10={_fmt(best['Hits@10'])}")
    lines.append(f"Test: AUC={_fmt(test_metrics['AUC'])} | H@1={_fmt(test_metrics['Hits@1'])} | "
                 f"H@5={_fmt(test_metrics['Hits@5'])} | H@10={_fmt(test_metrics['Hits@10'])}")
    lines.append("")
    lines.append("History")
    lines.append("-" * 70)
    lines.append(f"{'epoch':<6} {'loss':<10} {'AUC':<10} {'H@1':<10} {'H@5':<10} {'H@10':<10}")
    for rec in history:
        lines.append(f"{rec['epoch']:<6} {_fmt(rec['train_loss']):<10} {_fmt(rec['val_auc']):<10} "
                     f"{_fmt(rec['val_hits1']):<10} {_fmt(rec['val_hits5']):<10} {_fmt(rec['val_hits10']):<10}")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved checkpoint → {ckpt_path}")
    print(f"Saved report     → {rep_path}\n")

    return {"best": best, "test": test_metrics, "history": history, "ckpt": str(ckpt_path), "report": str(rep_path)}


# -------------------------
# Entry for 3 datasets
# -------------------------

runs = [
    ("WN18RR",
     Path("../WN18RR/train.txt"), Path("../WN18RR/valid.txt"), Path("../WN18RR/test.txt")),
    ("FB15K-237",
     Path("../FB15K-237/train.txt"), Path("../FB15K-237/valid.txt"), Path("../FB15K-237/test.txt")),
    ("Cora",
     Path("data/CORA_KG/train.txt"), Path("data/CORA_KG/valid.txt"), Path("data/CORA_KG/test.txt")),
]

# Global defaults
HP = dict(
    epochs=100, lr=1e-3, weight_decay=1e-4, patience=10,
    emb_dim=128, hidden_dim=128, out_dim=256,
    heads=4, rank=8, adapter_scale=1.0, dropout=0.2,
    batch_size=2048, k_neg=10,
    outdir=Path("results/rgat-lora")
)

PER_DATASET_HP = {
    "FB15K-237": dict(batch_size=2048, k_neg=10, heads=4, emb_dim=128, hidden_dim=128, out_dim=256, rank=8),
}

for name, tr, va, te in runs:
    args = {**HP, **PER_DATASET_HP.get(name, {})}
    # DistMult (recommended)
    train_run(name, tr, va, te, use_distmult=True, **args)
    # Dot product (ablation)
    if name == "WN18RR":
        train_run(name, tr, va, te, use_distmult=False, **args)