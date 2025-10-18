import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
from datetime import datetime

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class RelationalLoRAGINConv(MessagePassing):
    """
    Relational LoRA adapter for GIN-style message passing.

    Message:
        base:    W x_j
        adapter: (A_r B_r^T) x_j               with rank k << d
        total:   W x_j + scale * (A_r B_r^T) x_j

    Update (GIN):
        x_i' = MLP( (1 + eps) * x_i + sum_j message )

    Args:
        emb_dim (int): node feature dimensionality d.
        num_relations (int): number of relation types |R|.
        rank (int): LoRA rank k (e.g., 4–16).
        hidden_layers (int): depth of the node MLP inside the conv.
        train_eps (bool): whether epsilon in GIN update is learnable.
        adapter_scale (float): multiplicative scale on adapter path.
        bias (bool): bias for the shared linear map W.
    """
    def __init__(
            self,
            emb_dim: int,
            num_relations: int,
            rank: int = 8,
            hidden_layers: int = 2,
            train_eps: bool = True,
            adapter_scale: float = 1.0,
            bias: bool = False,
    ):
        super().__init__(aggr="add")
        self.emb_dim = emb_dim
        self.rank = rank
        self.adapter_scale = adapter_scale

        # GIN epsilon
        self.eps = (
            nn.Parameter(torch.zeros(1))
            if train_eps else nn.Parameter(torch.tensor(0.0), requires_grad=False)
        )

        # Shared linear message transform
        self.W = nn.Linear(emb_dim, emb_dim, bias=bias)
        nn.init.xavier_uniform_(self.W.weight)

        # Relation-specific low-rank adapters (stored as embeddings)
        # A_r, B_r ∈ R^{d×k} are stored flattened per-relation.
        self.A = nn.Embedding(num_relations, emb_dim * rank)  # [|R|, d*k]
        self.B = nn.Embedding(num_relations, emb_dim * rank)  # [|R|, d*k]
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)

        # Node MLP (same output dim as input, like GIN)
        layers = [nn.Linear(emb_dim, emb_dim * 2), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(emb_dim * 2, emb_dim * 2), nn.ReLU()]
        layers += [nn.Linear(emb_dim * 2, emb_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(
            self,
            x: torch.Tensor,               # [N, d]
            edge_index: torch.Tensor,      # [2, E]
            edge_type: torch.Tensor,       # [E]
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_type=edge_type)

    def message(
            self,
            x_j: torch.Tensor,             # [E, d] neighbor features
            edge_type: torch.Tensor,       # [E] relation id per edge
    ) -> torch.Tensor:
        # Shared message
        base = self.W(x_j)  # [E, d]

        # Low-rank adapter per relation
        A_flat = self.A(edge_type)                       # [E, d*k]
        B_flat = self.B(edge_type)                       # [E, d*k]
        A = A_flat.view(-1, self.emb_dim, self.rank)     # [E, d, k]
        B = B_flat.view(-1, self.emb_dim, self.rank)     # [E, d, k]

        # Compute (A_r B_r^T) x_j efficiently:
        #   Bj = B_r^T x_j  -> [E, k]
        Bj = torch.einsum('edk,ed->ek', B, x_j)
        #   adapter = A_r Bj -> [E, d]
        adapter = torch.einsum('edk,ek->ed', A, Bj)

        return base + self.adapter_scale * adapter

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.mlp((1.0 + self.eps) * x + aggr_out)


class RelationalLoRAGINEncoder(nn.Module):
    """
    Stacks multiple RelationalLoRAGINConv layers over a learned node embedding table.
    """
    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            emb_dim: int = 128,
            num_layers: int = 3,
            hidden_layers: int = 2,
            dropout: float = 0.1,
            rank: int = 8,
            adapter_scale: float = 1.0,
            train_eps: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.embed.weight)

        self.convs = nn.ModuleList([
            RelationalLoRAGINConv(
                emb_dim=emb_dim,
                num_relations=num_relations,
                rank=rank,
                hidden_layers=hidden_layers,
                train_eps=train_eps,
                adapter_scale=adapter_scale,
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.embed.weight                        # [N, d]
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)       # [N, d]
            x = self.dropout(x)
        return x


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"

def print_comparison_report(
        title: str,
        left_name: str, left_result: Dict[str, Any],
        right_name: str, right_result: Dict[str, Any],
        save_path: Optional[str | Path] = None,
):
    ts = _fmt_ts(datetime.now())

    def block(name, res):
        best = res["best"]; hist = res["history"]
        best_auc = max((h.get("val_auc", float("nan")) for h in hist), default=float("nan"))
        total_epochs = res.get("epochs_trained", len(hist))

        lines = []
        lines.append(f"{name} Training History")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Best Validation AUC: {_fmt(best_auc)}")
        lines.append(f"Total Epochs Trained: {total_epochs}")
        lines.append(f"Early Stopping Best Score: {_fmt(best.get('Hits@10'))} (Hits@10 at epoch {best.get('epoch')})")
        lines.append("")
        lines.append("-" * 90)
        lines.append(f"{'Epoch':<8} {'Train Loss':<14} {'Val AUC':<12} {'Val H@1':<12} {'Val H@5':<12} {'Val H@10':<12}")
        lines.append("-" * 90)
        for rec in hist:
            e = rec.get("epoch")
            lines.append(
                f"{e:<8} "
                f"{_fmt(rec.get('train_loss')):<14} "
                f"{_fmt(rec.get('val_auc')):<12} "
                f"{_fmt(rec.get('val_hits1')):<12} "
                f"{_fmt(rec.get('val_hits5')):<12} "
                f"{_fmt(rec.get('val_hits10')):<12}"
            )
        lines.append("")
        return "\n".join(lines)

    out = []
    out.append(f"{title} - {ts}")
    out.append("=" * 80)
    out.append("")
    out.append(block(left_name, left_result))
    out.append(block(right_name, right_result))

    # Best-at-a-glance
    out.append("Best Validation Metrics Summary")
    out.append("=" * 60)
    for name, res in [(left_name, left_result), (right_name, right_result)]:
        b = res["best"]
        out.append(
            f"{name}: "
            f"AUC={_fmt(b.get('AUC'))} | "
            f"H@1={_fmt(b.get('Hits@1'))} | "
            f"H@5={_fmt(b.get('Hits@5'))} | "
            f"H@10={_fmt(b.get('Hits@10'))} "
            f"(epoch {b.get('epoch')})"
        )
    out.append("")

    report = "\n".join(out)
    print(report)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ Comparison report saved to: {save_path.resolve()}")


def print_training_report(
        model_name: str,
        result: Dict[str, Any],
        header_title: str = "Model Training Results",
        save_path: Optional[str | Path] = None,
):
    ts = _fmt_ts(result.get("end_time", datetime.now()))
    best = result["best"]
    history = result["history"]
    total_epochs = result.get("epochs_trained", len(history))
    best_auc = max((h.get("val_auc", float("nan")) for h in history), default=float("nan"))

    lines = []
    lines.append(f"{header_title} - {ts}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("")
    lines.append(f"{model_name} Training History")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Best Validation AUC: {_fmt(best_auc)}")
    lines.append(f"Total Epochs Trained: {total_epochs}")
    lines.append(f"Early Stopping Best Score: {_fmt(best.get('Hits@10'))} (Hits@10 at epoch {best.get('epoch')})")
    lines.append("")
    lines.append("-" * 90)
    lines.append(f"{'Epoch':<8} {'Train Loss':<14} {'Val AUC':<12} {'Val H@1':<12} {'Val H@5':<12} {'Val H@10':<12}")
    lines.append("-" * 90)

    for rec in history:
        e = rec.get("epoch")
        lines.append(
            f"{e:<8} "
            f"{_fmt(rec.get('train_loss')):<14} "
            f"{_fmt(rec.get('val_auc')):<12} "
            f"{_fmt(rec.get('val_hits1')):<12} "
            f"{_fmt(rec.get('val_hits5')):<12} "
            f"{_fmt(rec.get('val_hits10')):<12}"
        )

    lines.append("")
    report_text = "\n".join(lines)

    print(report_text)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n✅ Report saved to: {save_path.resolve()}")


# ---------- Build edge_index and edge_type from the *train* split ----------
def build_edge_index_and_type_from_typed_dm(dm) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Returns:
        edge_index: [2, E] directed edges
        edge_type:  [E]    relation id per edge (aligned with edge_index columns)
    Uses dm._train_triples directly (already contains reverse triples if add_reverse=True).
    """
    assert hasattr(dm, "_train_triples"), "KGDataModuleTyped expected."
    triples = dm._train_triples  # [N, 3] (h, r, t), torch.long
    if triples.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    h = triples[:, 0]
    r = triples[:, 1]
    t = triples[:, 2]

    edge_index = torch.stack([h, t], dim=0).contiguous()  # directed edges h->t
    edge_type = r.contiguous()                             # relation per edge
    return edge_index, edge_type


# ---------- Typed negative sampling (corrupt head/tail, keep relation) ----------
@torch.no_grad()
def sample_negatives_typed(triples: torch.Tensor, num_entities: int) -> torch.Tensor:
    """
    1:1 negatives per positive (half head-corrupt, half tail-corrupt).
    Input triples: [B,3] (h, r, t)
    Output triples: [B,3] negatives (h', r, t) or (h, r, t')
    """
    B = triples.size(0)
    device = triples.device
    neg = triples.clone()
    flip = torch.rand(B, device=device) < 0.5
    rand_ents = torch.randint(0, num_entities, (B,), device=device)

    # corrupt head
    neg[flip, 0] = rand_ents[flip]
    # corrupt tail
    neg[~flip, 2] = rand_ents[~flip]
    return neg


# ---------- DistMult decoder for typed link prediction ----------
class DistMultDecoder(torch.nn.Module):
    """
    score(h, r, t) = <e_h, w_r, e_t> = sum_d e_h[d] * w_r[d] * e_t[d]
    """
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.rel = torch.nn.Embedding(num_relations, dim)
        torch.nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, z: torch.Tensor, triples: torch.LongTensor) -> torch.Tensor:
        # triples: [B,3] (h, r, t)
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        e_h, e_t, w_r = z[h], z[t], self.rel(r)
        return (e_h * w_r * e_t).sum(dim=1)  # [B]

class DotProductDecoder(torch.nn.Module):
    """
    score(h, r, t) = <e_h, e_t> (relation is ignored)
    """
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, triples: torch.LongTensor) -> torch.Tensor:
        h, t = triples[:, 0], triples[:, 2]
        return (z[h] * z[t]).sum(dim=1)  # [B]

def scores_for_candidates(
        decoder: torch.nn.Module,
        z: torch.Tensor,
        h: torch.Tensor,                # [B]
        r: torch.Tensor,                # [B]
        cand_t: torch.Tensor,           # [B, K]
) -> torch.Tensor:
    """
    Returns [B, K] scores for candidates.
    Fast path for DistMult; generic path for DotProduct.
    """
    if isinstance(decoder, DistMultDecoder):
        e_h = z[h]                       # [B, d]
        w_r = decoder.rel(r)            # [B, d]
        e_c = z[cand_t]                 # [B, K, d]
        s = ((e_h * w_r).unsqueeze(1) * e_c).sum(dim=2)  # [B, K]
        return s
    else:
        # generic: build triples and call decoder
        B, K = cand_t.shape
        h_rep = h.view(B, 1).expand(B, K)
        r_rep = r.view(B, 1).expand(B, K)  # unused by dot, but fine for API
        triples = torch.stack([h_rep, r_rep, cand_t], dim=2).reshape(-1, 3)  # [B*K,3]
        s = decoder(z, triples).view(B, K)
        return s



@torch.no_grad()
def evaluate_metrics_typed(
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        loader: Optional[DataLoader],
        num_entities: int,
        device: torch.device,
        show_tqdm: bool = False,
) -> Dict[str, float]:
    if loader is None:
        return {"AUC": float("nan"), "Hits@1": float("nan"), "Hits@5": float("nan"), "Hits@10": float("nan")}

    encoder.eval(); decoder.eval()
    z = encoder(edge_index.to(device), edge_type.to(device))  # [N, d]

    # --- AUC ---
    all_scores, all_labels = [], []
    it_auc = loader if not show_tqdm else tqdm(loader, leave=False, desc="Eval AUC (typed)")
    for pos, _ in it_auc:
        pos = pos.to(device)
        neg = sample_negatives_typed(pos, num_entities)

        s_pos = decoder(z, pos)
        s_neg = decoder(z, neg)

        all_scores.append(torch.cat([s_pos, s_neg]).detach().cpu().numpy())
        all_labels.append(np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))]))

    scores = np.concatenate(all_scores) if len(all_scores) > 0 else np.array([])
    labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
    if _HAS_SK and len(scores) > 0:
        auc = float(roc_auc_score(labels, scores))
    else:
        auc = float("nan") if len(scores) == 0 else float((scores[labels == 1].mean() > scores[labels == 0].mean()))

    # --- Hits@K (tail ranking) ---
    hits1 = hits5 = hits10 = 0
    trials = 0
    it_hits = loader if not show_tqdm else tqdm(loader, leave=False, desc="Eval Hits (tail)")
    for pos, _ in it_hits:
        pos = pos.to(device)
        B = pos.size(0)
        h, r, t_true = pos[:, 0], pos[:, 1], pos[:, 2]

        rand_t = torch.randint(0, num_entities, (B, 99), device=device)
        cand_t = torch.cat([t_true.view(-1, 1), rand_t], dim=1)  # [B,100]

        s = scores_for_candidates(decoder, z, h, r, cand_t)  # [B,100]
        ranks = s.argsort(dim=1, descending=True)
        true_positions = torch.nonzero(ranks == 0, as_tuple=False)[:, 1] + 1  # 1-based
        hits1  += (true_positions <= 1).sum().item()
        hits5  += (true_positions <= 5).sum().item()
        hits10 += (true_positions <= 10).sum().item()
        trials += B

    return {
        "AUC": auc,
        "Hits@1": hits1 / max(trials, 1),
        "Hits@5": hits5 / max(trials, 1),
        "Hits@10": hits10 / max(trials, 1),
    }


import re

def _safe_filename(s: str) -> str:
    # replace characters that can be annoying in shells/IDEs
    return re.sub(r'[^A-Za-z0-9._\-=/]', '_', s)

def train_linkpred_typed(
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,                 # DistMultDecoder (or another typed decoder)
        dm,                                       # KGDataModuleTyped
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: Optional[torch.device] = None,
        show_tqdm: bool = True,
        save_best_path: Optional[str | Path] = None,
        save_on_improve: bool = True,
        hparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Optimizer only over encoder + decoder
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=lr, weight_decay=weight_decay)

    # Build typed graph from *train* split
    edge_index, edge_type = build_edge_index_and_type_from_typed_dm(dm)
    edge_index = edge_index.to(device)
    edge_type  = edge_type.to(device)

    train_loader = dm.train_loader()
    val_loader   = dm.val_loader()
    num_entities = len(dm.ent2id)
    num_relations = len(dm.rel2id)

    # hparams
    auto_hparams: Dict[str, Any] = {
        "model_name": f"{encoder.__class__.__name__}+{decoder.__class__.__name__}",
        "optimizer": "Adam",
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "typed_graph": True,
        "batch_size": getattr(dm, "batch_size", None),
        "add_reverse": getattr(dm, "add_reverse", None),
        "reverse_relation_strategy": getattr(dm, "reverse_relation_strategy", None),
        "num_nodes": len(dm.ent2id),
        "num_relations": num_relations,
        "enc_emb_dim": getattr(getattr(encoder, "embed", None), "embedding_dim", None),
        "enc_num_layers": len(getattr(encoder, "convs", [])),
        "decoder": decoder.__class__.__name__,
    }
    run_hparams = {**auto_hparams, **(hparams or {})}

    history = []
    best = {"epoch": 0, "AUC": -1.0, "Hits@1": 0.0, "Hits@5": 0.0, "Hits@10": 0.0}
    patience_ctr = 0
    best_state = None

    save_best_path = Path(save_best_path) if save_best_path else None
    if save_best_path:
        save_best_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_iter = range(1, epochs + 1)
    if show_tqdm:
        epoch_iter = tqdm(epoch_iter, desc="Epochs (typed)")

    start_time = datetime.now()

    for epoch in epoch_iter:
        encoder.train(); decoder.train()
        running_loss = 0.0
        running_n = 0

        batch_iter = train_loader
        if show_tqdm:
            batch_iter = tqdm(train_loader, leave=False, desc=f"Train {epoch}")

        # Precompute node embeddings once per epoch for efficiency
        z = encoder(edge_index, edge_type)  # [N, d]

        for pos, _ in batch_iter:
            pos = pos.to(device)
            neg = sample_negatives_typed(pos, num_entities).to(device)

            opt.zero_grad()

            # recompute embeddings for THIS batch so we have a fresh graph
            z = encoder(edge_index, edge_type)              # <— moved inside

            s_pos = decoder(z, pos)
            s_neg = decoder(z, neg)

            scores = torch.cat([s_pos, s_neg], dim=0)
            labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)], dim=0)
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            running_loss += loss.item()
            running_n += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0
            )
            opt.step()

            if show_tqdm:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_n, 1)

        # Validation (fresh z to reflect updated encoder)
        val_metrics = evaluate_metrics_typed(
            encoder, decoder, edge_index, edge_type, val_loader, num_entities, device, show_tqdm=show_tqdm
        )
        if show_tqdm:
            tqdm.write(f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                       f"AUC={val_metrics['AUC']:.4f} | "
                       f"H@1={val_metrics['Hits@1']:.4f} | "
                       f"H@5={val_metrics['Hits@5']:.4f} | "
                       f"H@10={val_metrics['Hits@10']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc": float(val_metrics["AUC"]),
            "val_hits1": float(val_metrics["Hits@1"]),
            "val_hits5": float(val_metrics["Hits@5"]),
            "val_hits10": float(val_metrics["Hits@10"]),
        })
        # Early stopping on Hits@10
        if val_metrics["Hits@10"] > best["Hits@10"]:
            best.update({"epoch": epoch, **val_metrics})
            best_state = {
                "encoder": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
            }
            patience_ctr = 0

            if save_best_path and save_on_improve:

                if save_best_path:
                    save_best_path = Path(_safe_filename(str(save_best_path)))
                    save_best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "encoder_state_dict": best_state["encoder"],
                        "decoder_state_dict": best_state["decoder"],
                        "epoch": epoch,
                        "best_metrics": best,
                        "history": history,
                        "hparams": run_hparams,
                        "timestamp": datetime.now().isoformat(),
                    },
                    save_best_path,
                )
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if show_tqdm:
                    tqdm.write(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    # Restore best
    end_time = datetime.now()
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        if show_tqdm:
            tqdm.write(f"Restored best model from epoch {best['epoch']} | "
                       f"AUC={best['AUC']:.4f} | Hits@10={best['Hits@10']:.4f}")

        # If you prefer single save at end:
        if save_best_path and not save_on_improve:
            torch.save(
                {
                    "encoder_state_dict": best_state["encoder"],
                    "decoder_state_dict": best_state["decoder"],
                    "epoch": best["epoch"],
                    "best_metrics": best,
                    "history": history,
                    "hparams": run_hparams,
                    "timestamp": datetime.now().isoformat(),
                },
                save_best_path,
            )
            if show_tqdm:
                tqdm.write(f"Saved final best checkpoint to {save_best_path}")

    return {
        "best": best,
        "history": history,
        "epochs_trained": history[-1]["epoch"] if history else 0,
        "start_time": start_time,
        "end_time": end_time,
        "checkpoint_path": str(save_best_path) if save_best_path else None,
        "hparams": run_hparams,
    }



# --- 1) LoRA R-GIN + DistMult ---
lora_rank = 8           # try 4–16; 8 is a good start for d=128
adapter_scale = 1.0     # you can anneal or tune this (e.g., 0.5–2.0)

dataset = "WN18RR"
num_layers, hidden_layers, emb_dim = 3,3,128

# --- Dataset setup: defines dm_typed, num_nodes, num_relations ---
from pathlib import Path
from dataset_loader import KGDataModuleTyped

dataset = "WN18RR"  # or "FB15k-237", etc.
train_p = Path("../WN18RR/train.txt")
valid_p = Path("../WN18RR/valid.txt")
test_p  = Path("../WN18RR/test.txt")

# Create the typed KG datamodule
dm_typed = KGDataModuleTyped(
    train_p,
    valid_p,
    test_p,
    batch_size=2048,              # or whatever you use
    add_reverse=True,             # keep as in your code
    reverse_relation_strategy="duplicate_rel",  # creates separate reverse relation ids
    num_workers=2                 # optional
)

# Entity and relation counts for your embedding tables
num_nodes = len(dm_typed.ent2id)     # number of entities (nodes)
num_relations = len(dm_typed.rel2id) # number of relation types (including reverse if added)

print(f"Entities: {num_nodes} | Relations: {num_relations}")

enc_lora = RelationalLoRAGINEncoder(
    num_nodes=num_nodes,
    num_relations=num_relations,
    emb_dim=emb_dim,
    num_layers=num_layers,
    hidden_layers=hidden_layers,
    dropout=0.1,
    rank=8,
    adapter_scale=1.0,
    train_eps=True,
)
dec_lora = DistMultDecoder(num_relations=num_relations, dim=emb_dim)

save_path_lora = f"checkpoints/lora_rel/{dataset}/best_rginlora_distmult_d={emb_dim}_L={num_layers}_mlp={hidden_layers}_rank={lora_rank}.pt"

res_lora = train_linkpred_typed(
    enc_lora, dec_lora, dm_typed,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    show_tqdm=True,
    save_best_path=save_path_lora,   # your train loop already _safe_filename()s it
    save_on_improve=True,
    hparams={
        "dataset": dataset,
        "decoder": "DistMult",
        "emb_dim": emb_dim,
        "num_layers": num_layers,
        "hidden_layers": hidden_layers,
        "rank": lora_rank,
        "adapter_scale": adapter_scale,
        "method": "LoRA-GIN"
    }
)

# Optional: print a standalone training report
print_training_report(
    model_name="LoRA R-GIN + DistMult",
    result=res_lora,
    header_title="Model Training Results (LoRA)",
    save_path=f"results/lora_rel/{dataset}/lora_rgin_distmult_d={emb_dim}_L={num_layers}_mlp={hidden_layers}_rank={lora_rank}.txt"
)





# --- Dataset setup: defines dm_typed, num_nodes, num_relations ---
from pathlib import Path
from dataset_loader import KGDataModuleTyped

dataset = "FB15K-237"  # or "FB15k-237", etc.
train_p = Path("../FB15K-237/train.txt")
valid_p = Path("../FB15K-237/valid.txt")
test_p  = Path("../FB15K-237/test.txt")


num_layers, hidden_layers, emb_dim = 3,3,128


# Create the typed KG datamodule
dm_typed = KGDataModuleTyped(
    train_p,
    valid_p,
    test_p,
    batch_size=2048,              # or whatever you use
    add_reverse=True,             # keep as in your code
    reverse_relation_strategy="duplicate_rel",  # creates separate reverse relation ids
    num_workers=2                 # optional
)

# Entity and relation counts for your embedding tables
num_nodes = len(dm_typed.ent2id)     # number of entities (nodes)
num_relations = len(dm_typed.rel2id) # number of relation types (including reverse if added)

print(f"Entities: {num_nodes} | Relations: {num_relations}")

enc_lora = RelationalLoRAGINEncoder(
    num_nodes=num_nodes,
    num_relations=num_relations,
    emb_dim=emb_dim,
    num_layers=num_layers,
    hidden_layers=hidden_layers,
    dropout=0.1,
    rank=8,
    adapter_scale=1.0,
    train_eps=True,
)
dec_lora = DistMultDecoder(num_relations=num_relations, dim=emb_dim)

save_path_lora = f"checkpoints/lora_rel/{dataset}/best_rginlora_distmult_d={emb_dim}_L={num_layers}_mlp={hidden_layers}_rank={lora_rank}.pt"

res_lora = train_linkpred_typed(
    enc_lora, dec_lora, dm_typed,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    show_tqdm=True,
    save_best_path=save_path_lora,   # your train loop already _safe_filename()s it
    save_on_improve=True,
    hparams={
        "dataset": dataset,
        "decoder": "DistMult",
        "emb_dim": emb_dim,
        "num_layers": num_layers,
        "hidden_layers": hidden_layers,
        "rank": lora_rank,
        "adapter_scale": adapter_scale,
        "method": "LoRA-GIN"
    }
)

# Optional: print a standalone training report
print_training_report(
    model_name="LoRA R-GIN + DistMult",
    result=res_lora,
    header_title="Model Training Results (LoRA)",
    save_path=f"results/lora_rel/{dataset}/lora_rgin_distmult_d={emb_dim}_L={num_layers}_mlp={hidden_layers}_rank={lora_rank}.txt"
)




# --- Dataset setup: defines dm_typed, num_nodes, num_relations ---
from pathlib import Path
from dataset_loader import KGDataModuleTyped

dataset = "Cora"  # or "FB15k-237", etc.
train_p = Path("data/CORA_KG/train.txt")
valid_p = Path("data/CORA_KG/valid.txt")
test_p  = Path("data/CORA_KG/test.txt")


num_layers, hidden_layers, emb_dim = 3,3,128


# Create the typed KG datamodule
dm_typed = KGDataModuleTyped(
    train_p,
    valid_p,
    test_p,
    batch_size=1024,              # or whatever you use
    add_reverse=False,             # keep as in your code
    reverse_relation_strategy="duplicate_rel",  # creates separate reverse relation ids
    num_workers=2                 # optional
)

# Entity and relation counts for your embedding tables
num_nodes = len(dm_typed.ent2id)     # number of entities (nodes)
num_relations = len(dm_typed.rel2id) # number of relation types (including reverse if added)

print(f"Entities: {num_nodes} | Relations: {num_relations}")

enc_lora = RelationalLoRAGINEncoder(
    num_nodes=num_nodes,
    num_relations=num_relations,
    emb_dim=emb_dim,
    num_layers=num_layers,
    hidden_layers=hidden_layers,
    dropout=0.1,
    rank=8,
    adapter_scale=1.0,
    train_eps=True,
)
dec_lora = DistMultDecoder(num_relations=num_relations, dim=emb_dim)

save_path_lora = f"checkpoints/lora_rel/{dataset}/best_rginlora_distmult_d={emb_dim}_L={num_layers}_mlp={hidden_layers}_rank={lora_rank}.pt"

res_lora = train_linkpred_typed(
    enc_lora, dec_lora, dm_typed,
    epochs=500,
    lr=1e-3,
    weight_decay=1e-4,
    patience=100,
    show_tqdm=True,
    save_best_path=save_path_lora,   # your train loop already _safe_filename()s it
    save_on_improve=True,
    hparams={
        "dataset": dataset,
        "decoder": "DistMult",
        "emb_dim": emb_dim,
        "num_layers": num_layers,
        "hidden_layers": hidden_layers,
        "rank": lora_rank,
        "adapter_scale": adapter_scale,
        "method": "LoRA-GIN"
    }
)

# Optional: print a standalone training report
print_training_report(
    model_name="LoRA R-GIN + DistMult",
    result=res_lora,
    header_title="Model Training Results (LoRA)",
    save_path=f"results/lora_rel/{dataset}/lora_rgin_distmult_d={emb_dim}_L={num_layers}_mlp={hidden_layers}_rank={lora_rank}.txt"
)