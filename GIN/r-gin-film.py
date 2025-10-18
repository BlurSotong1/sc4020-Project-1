import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

# ---- trainer_film_gin.py ----------------------------------------------------
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False
    def tqdm(x, **k): return x

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False

# ---------------------------
# FiLM-style Relational GIN
# ---------------------------
class RelationalFiLMGINConv(MessagePassing):
    """
    Relation-aware GIN with FiLM modulation:
      gamma_r, beta_r = MLP_rel(r_emb[r])
      m_{j->i} = gamma_r ⊙ (W x_j) + beta_r
      x_i' = MLP_node( (1 + eps) * x_i + sum_j m_{j->i} )
    """
    def __init__(
            self,
            emb_dim: int,
            num_relations: int,
            hidden_layers: int = 1,   # depth of relation MLP (>=1)
            train_eps: bool = True
    ):
        super().__init__(aggr="add")  # GIN uses sum aggregation

        # relation embeddings (one per edge type)
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # relation MLP that outputs [gamma | beta] of size 2 * emb_dim
        rel_layers = [nn.Linear(emb_dim, emb_dim * 2), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            rel_layers += [nn.Linear(emb_dim * 2, emb_dim * 2), nn.ReLU()]
        rel_layers += [nn.Linear(emb_dim * 2, emb_dim * 2)]
        self.rel_mlp = nn.Sequential(*rel_layers)

        # shared linear transform on node features
        self.W = nn.Linear(emb_dim, emb_dim)

        # node-side GIN MLP (kept simple; you can deepen if you like)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        # learnable epsilon like classic GIN
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        x: [N, d]
        edge_index: [2, E]
        edge_type: [E]  (relation id for each edge)
        """
        # propagate will call message(...) then aggregate, then update(...)
        return self.propagate(edge_index, x=x, edge_type=edge_type)

    def message(self, x_j: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        x_j: [E, d] features of source nodes (neighbors)
        edge_type: [E] relation ids aligned with edges
        """
        r = self.rel_emb(edge_type)                 # [E, d]
        gamma_beta = self.rel_mlp(r)                # [E, 2d]
        gamma, beta = gamma_beta.chunk(2, dim=-1)   # each [E, d]
        return gamma * self.W(x_j) + beta           # FiLM message

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        aggr_out: [N, d] summed messages
        x:         [N, d] original node features, passed from propagate via kwargs
        """
        out = (1.0 + self.eps) * x + aggr_out
        return self.mlp(out)


class RelationalGINEncoder(nn.Module):
    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            emb_dim: int = 128,
            num_layers: int = 3,
            hidden_layers: int = 1,   # depth for the relation FiLM MLP
            dropout: float = 0.1,
            train_eps: bool = True,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.embed.weight)

        self.convs = nn.ModuleList([
            RelationalFiLMGINConv(
                emb_dim=emb_dim,
                num_relations=num_relations,
                hidden_layers=hidden_layers,
                train_eps=train_eps
            )
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        x = self.embed.weight
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = self.dropout(x)
        return x  # [N, emb_dim]


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

import re
def _safe_filename(s: str) -> str:
    # replace characters that can be annoying in shells/IDEs
    return re.sub(r'[^A-Za-z0-9._\-=/]', '_', s)


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





# ------------------------------------------------------------------
# assumes THESE are already defined/imported in your codebase:
#   - build_edge_index_and_type_from_typed_dm(dm) -> (edge_index, edge_type)
#   - sample_negatives_typed(triples, num_entities)
#   - DistMultDecoder / scores_for_candidates
#   - print_training_report / print_comparison_report
# ------------------------------------------------------------------

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
        sampled_k: int = 100,   # number of tails per query (1 gold + k-1 random)
) -> Dict[str, float]:
    """AUC via 1:1 pos/neg; Hits@k via sampled ranking against (sampled_k-1) random tails."""
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

    # --- Hits@K (tail ranking, sampled) ---
    hits1 = hits5 = hits10 = 0
    trials = 0
    it_hits = loader if not show_tqdm else tqdm(loader, leave=False, desc="Eval Hits@K (typed)")
    K = sampled_k
    for pos, _ in it_hits:
        pos = pos.to(device)
        B = pos.size(0)
        h, r, t_true = pos[:, 0], pos[:, 1], pos[:, 2]

        # candidates = {true tail} ∪ {K-1 random tails}
        rand_t = torch.randint(0, num_entities, (B, K - 1), device=device)
        cand_t = torch.cat([t_true.view(-1, 1), rand_t], dim=1)  # [B, K]

        s = scores_for_candidates(decoder, z, h, r, cand_t)  # [B, K]
        ranks = s.argsort(dim=1, descending=True)
        # position of column 0 (the true tail) in the argsorted index
        true_positions = (ranks == 0).nonzero(as_tuple=False)[:, 1] + 1  # 1-based

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


def train_linkpred_film_gin(
        encoder: torch.nn.Module,         # RelationalGINEncoder (FiLM GIN)
        decoder: torch.nn.Module,         # e.g., DistMultDecoder
        dm,                               # your KGDataModuleTyped instance
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: Optional[torch.device] = None,
        show_tqdm: bool = True,
        save_best_path: Optional[str | Path] = None,
        save_on_improve: bool = True,
        hparams: Optional[Dict[str, Any]] = None,
        eval_sampled_k: int = 100,        # eval ranking pool size (1 gold + K-1 random)
) -> Dict[str, Any]:
    """Full-batch FiLM-GIN training loop for typed link prediction (1:1 negatives)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    # Graph from *train* triples only (avoid leakage)
    edge_index, edge_type = build_edge_index_and_type_from_typed_dm(dm)
    edge_index, edge_type = edge_index.to(device), edge_type.to(device)

    train_loader = dm.train_loader()
    val_loader   = dm.val_loader()
    num_entities = len(dm.ent2id)
    num_relations = len(dm.rel2id)

    # hparams block for report
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
        "num_nodes": num_entities,
        "num_relations": num_relations,
        "enc_emb_dim": getattr(getattr(encoder, "embed", None), "embedding_dim", None),
        "enc_num_layers": len(getattr(encoder, "convs", [])),
        "decoder": decoder.__class__.__name__,
    }
    run_hparams = {**auto_hparams, **(hparams or {})}

    # bookkeeping
    history: list[Dict[str, Any]] = []
    best = {"epoch": 0, "AUC": -1.0, "Hits@1": 0.0, "Hits@5": 0.0, "Hits@10": 0.0}
    patience_ctr = 0
    best_state = None

    save_best_path = Path(save_best_path) if save_best_path else None
    if save_best_path:
        save_best_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()
    epoch_iter = range(1, epochs + 1)
    if show_tqdm and _HAS_TQDM:
        epoch_iter = tqdm(epoch_iter, desc="Epochs (FiLM-GIN)")

    for epoch in epoch_iter:
        encoder.train(); decoder.train()
        running_loss = 0.0; running_n = 0

        batch_iter = train_loader
        if show_tqdm and _HAS_TQDM:
            batch_iter = tqdm(train_loader, leave=False, desc=f"Train {epoch}")

        # Full-batch encoder; recompute embeddings each step to reflect parameter updates
        for pos, _ in batch_iter:
            pos = pos.to(device)
            neg = sample_negatives_typed(pos, num_entities).to(device)

            opt.zero_grad()

            z = encoder(edge_index, edge_type)          # [N, d] (full graph)
            s_pos = decoder(z, pos)                     # [B]
            s_neg = decoder(z, neg)                     # [B]

            scores = torch.cat([s_pos, s_neg], dim=0)
            labels = torch.cat(
                [torch.ones_like(s_pos), torch.zeros_like(s_neg)], dim=0
            )
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                max_norm=1.0
            )
            opt.step()

            running_loss += float(loss.item()); running_n += 1
            if show_tqdm and _HAS_TQDM:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_n, 1)

        # ---- validation ----
        val_metrics = evaluate_metrics_typed(
            encoder, decoder, edge_index, edge_type,
            val_loader, num_entities, device,
            show_tqdm=show_tqdm and _HAS_TQDM, sampled_k=eval_sampled_k
        )

        if show_tqdm:
            msg = (f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                   f"AUC={val_metrics['AUC']:.4f} | "
                   f"H@1={val_metrics['Hits@1']:.4f} | "
                   f"H@5={val_metrics['Hits@5']:.4f} | "
                   f"H@10={val_metrics['Hits@10']:.4f}")
            if _HAS_TQDM:
                tqdm.write(msg)
            else:
                print(msg)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc": float(val_metrics["AUC"]),
            "val_hits1": float(val_metrics["Hits@1"]),
            "val_hits5": float(val_metrics["Hits@5"]),
            "val_hits10": float(val_metrics["Hits@10"]),
        })

        # early stopping on Hits@10
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


                torch.save({
                    "encoder_state_dict": best_state["encoder"],
                    "decoder_state_dict": best_state["decoder"],
                    "epoch": epoch,
                    "best_metrics": best,
                    "history": history,
                    "hparams": run_hparams,
                    "timestamp": datetime.now().isoformat(),
                }, save_best_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if show_tqdm:
                    msg = f"Early stopping at epoch {epoch} (patience={patience})."
                    if _HAS_TQDM: tqdm.write(msg)
                    else: print(msg)
                break

    # restore best
    end_time = datetime.now()
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        if show_tqdm:
            msg = (f"Restored best model from epoch {best['epoch']} | "
                   f"AUC={best['AUC']:.4f} | Hits@10={best['Hits@10']:.4f}")
            if _HAS_TQDM: tqdm.write(msg)
            else: print(msg)

        if save_best_path and not save_on_improve:
            torch.save({
                "encoder_state_dict": best_state["encoder"],
                "decoder_state_dict": best_state["decoder"],
                "epoch": best["epoch"],
                "best_metrics": best,
                "history": history,
                "hparams": run_hparams,
                "timestamp": datetime.now().isoformat(),
            }, save_best_path)

    return {
        "best": best,
        "history": history,
        "epochs_trained": history[-1]["epoch"] if history else 0,
        "start_time": start_time,
        "end_time": end_time,
        "checkpoint_path": str(save_best_path) if save_best_path else None,
        "hparams": run_hparams,
    }
# ------------------------------------------------------------------------------



dataset = "WN18RR"
train_p = Path("../WN18RR/train.txt")
valid_p = Path("../WN18RR/valid.txt")
test_p  = Path("../WN18RR/test.txt")

from dataset_loader import KGDataModuleTyped


dm_typed = KGDataModuleTyped(
    train_p, valid_p, test_p,
    add_reverse=True,
    reverse_relation_strategy="duplicate_rel",
)

num_nodes = len(dm_typed.ent2id)
num_relations = len(dm_typed.rel2id)
emb_dim = 128
num_layers = 3
hidden_layers = 3

# --- 1) R-GIN + DistMult ---
enc_dm = RelationalGINEncoder( num_nodes=num_nodes, num_relations=num_relations, emb_dim=emb_dim, num_layers=num_layers, hidden_layers=hidden_layers, dropout=0.1, train_eps=True
                               )
dec_dm = DistMultDecoder(num_relations=num_relations, dim=emb_dim)

res_distmult = train_linkpred_film_gin(
    enc_dm, dec_dm, dm_typed,
    epochs=100, lr=1e-3, weight_decay=1e-4, patience=10,
    show_tqdm=True,
    save_best_path=f"checkpoints/r-gin-film/{dataset}/best_rgin_distmult_em={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.pt",
    save_on_improve=True,
    hparams={"dataset": dataset, "decoder": "DistMult", "emb_dim": emb_dim, "num_layers": num_layers, "hidden_layers": hidden_layers}
)


print_training_report(
    model_name = "R-GIN_embed_rel + Distmult",
    result = res_distmult,
    header_title = "Model Training Results",
    save_path=f"results/r-gin-film/{dataset}/film_rgin_report_em={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.txt"
)



dataset = "FB15K-237"
train_p = Path("../FB15K-237/train.txt")
valid_p = Path("../FB15K-237/valid.txt")
test_p  = Path("../FB15K-237/test.txt")

from dataset_loader import KGDataModuleTyped


dm_typed = KGDataModuleTyped(
    train_p, valid_p, test_p,
    add_reverse=True,
    reverse_relation_strategy="duplicate_rel",
)

num_nodes = len(dm_typed.ent2id)
num_relations = len(dm_typed.rel2id)
emb_dim = 128
num_layers = 3
hidden_layers = 3

# --- 1) R-GIN + DistMult ---
enc_dm = RelationalGINEncoder( num_nodes=num_nodes, num_relations=num_relations, emb_dim=emb_dim, num_layers=num_layers, hidden_layers=hidden_layers, dropout=0.1, train_eps=True
                               )
dec_dm = DistMultDecoder(num_relations=num_relations, dim=emb_dim)

res_distmult = train_linkpred_film_gin(
    enc_dm, dec_dm, dm_typed,
    epochs=100, lr=1e-3, weight_decay=1e-4, patience=10,
    show_tqdm=True,
    save_best_path=f"checkpoints/r-gin-film/{dataset}/best_rgin_distmult_em={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.pt",
    save_on_improve=True,
    hparams={"dataset": dataset, "decoder": "DistMult", "emb_dim": emb_dim, "num_layers": num_layers, "hidden_layers": hidden_layers}
)


print_training_report(
    model_name = "R-GIN_embed_rel + Distmult",
    result = res_distmult,
    header_title = "Model Training Results",
    save_path=f"results/r-gin-film/{dataset}/film_rgin_report_em={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.txt"
)



# cora_to_kg.py
# Convert Cora (PyG) into WN18RR-style TSV triples your loader can read.

import csv
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import coalesce
from torch_geometric.data import Data

# ------------------------- config -------------------------
root = Path("./data")
out_dir = root / "CORA_KG"
out_dir.mkdir(parents=True, exist_ok=True)

relation_name = "cites"            # single relation
is_undirected_split = True         # safer split for citation graphs
add_reverse_edges = True           # also write reverse edges
reverse_relation_strategy = "duplicate_rel"  # or "same_rel"

# --------------------- load + coalesce --------------------
print("📥 Loading Cora via PyG (auto-download if needed)...")
dataset = Planetoid(root=str(root), name="Cora")
data = dataset[0]

# coalesce deduplicates edges; keep them as-is (directed list)
edge_index, _ = coalesce(data.edge_index, None, data.num_nodes, data.num_nodes)

# Build a Data object with the coalesced edges (older PyG has no .replace)
new_data = Data(
    x=data.x,
    y=data.y,
    edge_index=edge_index,
    num_nodes=data.num_nodes,
)

print(f"✅ Cora: num_nodes={data.num_nodes}, edges={edge_index.size(1)}")

# ------------------- train/val/test split -----------------
splitter = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=is_undirected_split,
    add_negative_train_samples=False,  # you sample negatives yourself
)
train_g, val_g, test_g = splitter(new_data)

def pos_edges(g: Data) -> torch.Tensor:
    # RandomLinkSplit attaches edge_label and edge_label_index
    mask = (g.edge_label == 1)
    return g.edge_label_index[:, mask]  # [2, E_pos]

train_edges = pos_edges(train_g)
val_edges   = pos_edges(val_g)
test_edges  = pos_edges(test_g)

print(f"📊 Splits: train={train_edges.size(1)}, val={val_edges.size(1)}, test={test_edges.size(1)}")

# -------------------- triples + saving --------------------
def make_triples(edge_idx: torch.Tensor,
                 rel: str,
                 add_rev: bool,
                 rev_strategy: str) -> list[tuple[str, str, str]]:
    triples = []
    h_list = edge_idx[0].tolist()
    t_list = edge_idx[1].tolist()
    for h, t in zip(h_list, t_list):
        triples.append((f"n{h}", rel, f"n{t}"))
        if add_rev:
            if rev_strategy == "duplicate_rel":
                triples.append((f"n{t}", rel + "_rev", f"n{h}"))
            else:  # same_rel
                triples.append((f"n{t}", rel, f"n{h}"))
    return triples

def save_triples(triples: list[tuple[str, str, str]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(triples)
    print(f"💾 Saved {len(triples):,} triples -> {path}")

train_triples = make_triples(train_edges, relation_name, add_reverse_edges, reverse_relation_strategy)
val_triples   = make_triples(val_edges,   relation_name, add_reverse_edges, reverse_relation_strategy)
test_triples  = make_triples(test_edges,  relation_name, add_reverse_edges, reverse_relation_strategy)

save_triples(train_triples, out_dir / "train.txt")
save_triples(val_triples,   out_dir / "valid.txt")
save_triples(test_triples,  out_dir / "test.txt")

print("✅ Done. Files are WN18RR-style and compatible with your KGDataModuleTyped.")
print(f"Use paths:\n  train: {out_dir/'train.txt'}\n  valid: {out_dir/'valid.txt'}\n  test : {out_dir/'test.txt'}")




from dataset_loader import KGDataModuleTyped
dataset = "Cora"
train_p =  Path("data/CORA_KG/train.txt")
valid_p = Path("data/CORA_KG/valid.txt")
test_p = Path("data/CORA_KG/test.txt")

dm_typed = KGDataModuleTyped(
    train_p, valid_p, test_p,
    add_reverse=False,
    reverse_relation_strategy="duplicate_rel",
    batch_size=512
)

num_nodes = len(dm_typed.ent2id)
num_relations = len(dm_typed.rel2id)
emb_dim = 128
num_layers = 2
hidden_layers = 3

# --- 1) R-GIN + DistMult ---
enc_dm = RelationalGINEncoder( num_nodes=num_nodes, num_relations=num_relations, emb_dim=emb_dim, num_layers=num_layers, hidden_layers=hidden_layers, dropout=0.1, train_eps=True
                               )
dec_dm = DistMultDecoder(num_relations=num_relations, dim=emb_dim)



res_distmult = train_linkpred_film_gin(
    enc_dm, dec_dm, dm_typed,
    epochs=100, lr=1e-3, weight_decay=1e-4, patience=10,
    show_tqdm=True,
    save_best_path=f"checkpoints/r-gin-film/{dataset}/best_rgin_distmult_em={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.pt",
    save_on_improve=True,
    hparams={"dataset": dataset, "decoder": "DistMult", "emb_dim": emb_dim, "num_layers": num_layers, "hidden_layers": hidden_layers}
)

print_training_report(
    model_name = "R-GIN-film + Distmult",
    result = res_distmult,
    header_title = "Model Training Results",
    save_path=f"results/r-gin-film/{dataset}/film_rgin_report_em={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.txt"
)

