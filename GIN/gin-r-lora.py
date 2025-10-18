# rgin_lora.py
# ---------------------------------------------------------------------
# Relation-LoRA GIN for typed link prediction + full trainer & runners
# ---------------------------------------------------------------------
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import MessagePassing
from datetime import datetime

# --- tqdm (fallback to no-op if missing, works in notebook/terminal) ---
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False
    def tqdm(x, **k): return x

# --- sklearn AUC (fallback to simple heuristic if missing) -------------
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


# =========================
#   Utilities / Reporting
# =========================
def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"

# good: single backslash before '-' is enough
def _safe_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._\-=/]', '_', s)
# or: put '-' at the end so it can't become a range
# r'[^A-Za-z0-9._=/-]'

def print_training_report(
        model_name: str,
        result: Dict[str, Any],
        header_title: str = "Model Training Results",
        save_path: Optional[str | Path] = None,
):
    ts = _fmt_ts(result.get("end_time", datetime.now()))
    best = result["best"]; history = result["history"]
    total_epochs = result.get("epochs_trained", len(history))
    best_auc = max((h.get("val_auc", float("nan")) for h in history), default=float("nan"))

    lines = []
    lines.append(f"{header_title} - {ts}")
    lines.append("=" * 80); lines.append("")
    lines.append(f"{model_name} Training History")
    lines.append("=" * 60); lines.append("")
    lines.append(f"Best Validation AUC: {_fmt(best_auc)}")
    lines.append(f"Total Epochs Trained: {total_epochs}")
    lines.append(f"Early Stopping Best Score: {_fmt(best.get('Hits@10'))} (epoch {best.get('epoch')})")
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
    text = "\n".join(lines)
    print(text)
    if save_path:
        save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(text, encoding="utf-8")
        print(f"\n✅ Report saved to: {save_path.resolve()}")

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
        lines.append("=" * 60); lines.append("")
        lines.append(f"Best Validation AUC: {_fmt(best_auc)}")
        lines.append(f"Total Epochs Trained: {total_epochs}")
        lines.append(f"Early Stopping Best Score: {_fmt(best.get('Hits@10'))} (epoch {best.get('epoch')})")
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
    out.append("=" * 80); out.append("")
    out.append(block(left_name, left_result))
    out.append(block(right_name, right_result))
    out.append("Best Validation Metrics Summary")
    out.append("=" * 60)
    for name, res in [(left_name, left_result), (right_name, right_result)]:
        b = res["best"]
        out.append(
            f"{name}: AUC={_fmt(b.get('AUC'))} | "
            f"H@1={_fmt(b.get('Hits@1'))} | "
            f"H@5={_fmt(b.get('Hits@5'))} | "
            f"H@10={_fmt(b.get('Hits@10'))} (epoch {b.get('epoch')})"
        )
    out.append("")
    text = "\n".join(out)
    print(text)
    if save_path:
        save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(text, encoding="utf-8")
        print(f"\n✅ Comparison report saved to: {save_path.resolve()}")


# =========================
#     Relation-LoRA GIN
# =========================
import torch
import torch.nn as nn
from torch_scatter import scatter_add  # make sure torch-scatter is installed

class RelationalLoRAGINConv(nn.Module):
    """
    LoRA message: m_{j->i}^{(r)} = W x_j + scale * A_r (B_r^T x_j)
    Update (GIN): x_i' = MLP( (1 + eps) * x_i + sum_j m_{j->i} )
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
        super().__init__()
        self.emb_dim = emb_dim
        self.rank = rank
        self.adapter_scale = adapter_scale

        # GIN epsilon
        self.eps = nn.Parameter(torch.zeros(1)) if train_eps else nn.Parameter(
            torch.tensor(0.0), requires_grad=False
        )

        # Shared linear
        self.W = nn.Linear(emb_dim, emb_dim, bias=bias)
        nn.init.xavier_uniform_(self.W.weight)

        # Per-relation low-rank adapters (flattened)
        self.A = nn.Embedding(num_relations, emb_dim * rank)  # [|R|, d*k]
        self.B = nn.Embedding(num_relations, emb_dim * rank)  # [|R|, d*k]
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)

        # Node-side GIN MLP
        layers = [nn.Linear(emb_dim, emb_dim * 2), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(emb_dim * 2, emb_dim * 2), nn.ReLU()]
        layers += [nn.Linear(emb_dim * 2, emb_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        x:          [N, d]    node embeddings
        edge_index: [2, E]    (src -> dst)
        edge_type:  [E]       relation id per edge
        """
        src, dst = edge_index[0], edge_index[1]       # [E]
        x_j = x[src]                                  # [E, d]

        # base message
        base = self.W(x_j)                            # [E, d]

        # relation adapters
        A_flat = self.A(edge_type).view(-1, self.emb_dim, self.rank)  # [E, d, k]
        B_flat = self.B(edge_type).view(-1, self.emb_dim, self.rank)  # [E, d, k]

        # (B_r^T x_j): [E, k]
        Bj = torch.einsum('edk,ed->ek', B_flat, x_j)
        # A_r @ (...): [E, d]
        adapter = torch.einsum('edk,ek->ed', A_flat, Bj)

        msg = base + self.adapter_scale * adapter     # [E, d]

        # aggregate to destination nodes
        aggr = scatter_add(msg, dst, dim=0, dim_size=x.size(0))  # [N, d]

        # GIN update
        out = (1.0 + self.eps) * x + aggr
        return self.mlp(out)


class RelationalLoRAGINEncoder(nn.Module):
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
        x = self.embed.weight
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = self.dropout(x)
        return x


# =========================
#          Decoders
# =========================
class DistMultDecoder(nn.Module):
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.rel.weight)

    def forward(self, z: torch.Tensor, triples: torch.LongTensor) -> torch.Tensor:
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        e_h, e_t, w_r = z[h], z[t], self.rel(r)
        return (e_h * w_r * e_t).sum(dim=1)

class DotProductDecoder(nn.Module):
    def forward(self, z: torch.Tensor, triples: torch.LongTensor) -> torch.Tensor:
        h, t = triples[:, 0], triples[:, 2]
        return (z[h] * z[t]).sum(dim=1)

def scores_for_candidates(decoder: nn.Module, z: torch.Tensor,
                          h: torch.Tensor, r: torch.Tensor, cand_t: torch.Tensor) -> torch.Tensor:
    if isinstance(decoder, DistMultDecoder):
        e_h = z[h]                     # [B, d]
        w_r = decoder.rel(r)           # [B, d]
        e_c = z[cand_t]                # [B, K, d]
        return ((e_h * w_r).unsqueeze(1) * e_c).sum(dim=2)
    else:
        B, K = cand_t.shape
        h_rep = h.view(B, 1).expand(B, K)
        r_rep = r.view(B, 1).expand(B, K)
        triples = torch.stack([h_rep, r_rep, cand_t], dim=2).reshape(-1, 3)
        return decoder(z, triples).view(B, K)


# =========================
#      Data helpers
# =========================
def build_edge_index_and_type_from_typed_dm(dm) -> Tuple[torch.LongTensor, torch.LongTensor]:
    assert hasattr(dm, "_train_triples"), "Expected KGDataModuleTyped with _train_triples tensor."
    triples = dm._train_triples  # [M,3] (h,r,t)
    if triples.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    edge_index = torch.stack([h, t], dim=0).contiguous()
    edge_type  = r.contiguous()
    return edge_index, edge_type

@torch.no_grad()
def sample_negatives_typed(triples: torch.Tensor, num_entities: int) -> torch.Tensor:
    B = triples.size(0); device = triples.device
    neg = triples.clone()
    flip = torch.rand(B, device=device) < 0.5
    rand_ents = torch.randint(0, num_entities, (B,), device=device)
    neg[flip, 0] = rand_ents[flip]     # corrupt head
    neg[~flip, 2] = rand_ents[~flip]   # corrupt tail
    return neg


# =========================
#       Evaluation
# =========================
@torch.no_grad()
def evaluate_metrics_typed(
        encoder: nn.Module,
        decoder: nn.Module,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        loader: Optional[DataLoader],
        num_entities: int,
        device: torch.device,
        show_tqdm: bool = False,
        sampled_k: int = 100,
) -> Dict[str, float]:
    if loader is None:
        return {"AUC": float("nan"), "Hits@1": float("nan"), "Hits@5": float("nan"), "Hits@10": float("nan")}

    encoder.eval(); decoder.eval()
    z = encoder(edge_index.to(device), edge_type.to(device))

    # AUC via 1:1 pos/neg
    all_scores, all_labels = [], []
    it_auc = loader if not (show_tqdm and _HAS_TQDM) else tqdm(loader, leave=False, desc="Eval AUC")
    for pos, _ in it_auc:
        pos = pos.to(device)
        neg = sample_negatives_typed(pos, num_entities)
        s_pos = decoder(z, pos); s_neg = decoder(z, neg)
        all_scores.append(torch.cat([s_pos, s_neg]).cpu().numpy())
        all_labels.append(np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))]))
    scores = np.concatenate(all_scores) if all_scores else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    if _HAS_SK and len(scores) > 0:
        auc = float(roc_auc_score(labels, scores))
    else:
        auc = float("nan") if len(scores) == 0 else float((scores[labels == 1].mean() > scores[labels == 0].mean()))

    # Hits@K (sampled)
    hits1 = hits5 = hits10 = 0; trials = 0
    it_hits = loader if not (show_tqdm and _HAS_TQDM) else tqdm(loader, leave=False, desc="Eval Hits@K")
    K = sampled_k
    for pos, _ in it_hits:
        pos = pos.to(device)
        B = pos.size(0)
        h, r, t_true = pos[:, 0], pos[:, 1], pos[:, 2]
        rand_t = torch.randint(0, num_entities, (B, K - 1), device=device)
        cand_t = torch.cat([t_true.view(-1,1), rand_t], dim=1)  # [B,K]
        s = scores_for_candidates(decoder, z, h, r, cand_t)
        ranks = s.argsort(dim=1, descending=True)
        true_pos = (ranks == 0).nonzero(as_tuple=False)[:, 1] + 1
        hits1  += (true_pos <= 1).sum().item()
        hits5  += (true_pos <= 5).sum().item()
        hits10 += (true_pos <= 10).sum().item()
        trials += B

    return {
        "AUC": auc,
        "Hits@1": hits1 / max(trials, 1),
        "Hits@5": hits5 / max(trials, 1),
        "Hits@10": hits10 / max(trials, 1),
    }


# =========================
#        Training
# =========================
def train_linkpred_typed(
        encoder: nn.Module,
        decoder: nn.Module,
        dm,                          # KGDataModuleTyped
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: Optional[torch.device] = None,
        show_tqdm: bool = True,
        save_best_path: Optional[str | Path] = None,
        save_on_improve: bool = True,
        hparams: Optional[Dict[str, Any]] = None,
        eval_sampled_k: int = 100,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder = encoder.to(device), decoder.to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=lr, weight_decay=weight_decay)

    edge_index, edge_type = build_edge_index_and_type_from_typed_dm(dm)
    edge_index, edge_type = edge_index.to(device), edge_type.to(device)

    train_loader = dm.train_loader()
    val_loader   = dm.val_loader()
    num_entities = len(dm.ent2id)
    num_relations = len(dm.rel2id)

    auto_hparams: Dict[str, Any] = {
        "model_name": f"{encoder.__class__.__name__}+{decoder.__class__.__name__}",
        "optimizer": "Adam", "lr": lr, "weight_decay": weight_decay,
        "epochs": epochs, "patience": patience,
        "typed_graph": True,
        "batch_size": getattr(dm, "batch_size", None),
        "add_reverse": getattr(dm, "add_reverse", None),
        "reverse_relation_strategy": getattr(dm, "reverse_relation_strategy", None),
        "num_nodes": num_entities, "num_relations": num_relations,
        "enc_emb_dim": getattr(getattr(encoder, "embed", None), "embedding_dim", None),
        "enc_num_layers": len(getattr(encoder, "convs", [])),
        "decoder": decoder.__class__.__name__,
    }
    run_hparams = {**auto_hparams, **(hparams or {})}

    history = []
    best = {"epoch": 0, "AUC": -1.0, "Hits@1": 0.0, "Hits@5": 0.0, "Hits@10": 0.0}
    patience_ctr = 0; best_state = None

    save_best_path = Path(save_best_path) if save_best_path else None
    if save_best_path: save_best_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_iter = range(1, epochs + 1)
    if show_tqdm and _HAS_TQDM:
        epoch_iter = tqdm(epoch_iter, desc="Epochs (Rel-LoRA GIN)")

    start_time = datetime.now()
    for epoch in epoch_iter:
        encoder.train(); decoder.train()
        running_loss = 0.0; running_n = 0

        batch_iter = train_loader
        if show_tqdm and _HAS_TQDM:
            batch_iter = tqdm(train_loader, leave=False, desc=f"Train {epoch}")

        for pos, _ in batch_iter:
            pos = pos.to(device)
            neg = sample_negatives_typed(pos, num_entities)

            opt.zero_grad()
            z = encoder(edge_index, edge_type)
            s_pos = decoder(z, pos); s_neg = decoder(z, neg)

            scores = torch.cat([s_pos, s_neg], 0)
            labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)], 0)
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()

            running_loss += float(loss.item()); running_n += 1
            if show_tqdm and _HAS_TQDM:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_n, 1)

        # Validation
        val_metrics = evaluate_metrics_typed(
            encoder, decoder, edge_index, edge_type,
            val_loader, num_entities, device,
            show_tqdm=show_tqdm and _HAS_TQDM, sampled_k=eval_sampled_k
        )
        msg = (f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
               f"AUC={val_metrics['AUC']:.4f} | "
               f"H@1={val_metrics['Hits@1']:.4f} | "
               f"H@5={val_metrics['Hits@5']:.4f} | "
               f"H@10={val_metrics['Hits@10']:.4f}")
        if show_tqdm and _HAS_TQDM: tqdm.write(msg)
        else: print(msg)

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
                sp = Path(_safe_filename(str(save_best_path)))
                sp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "encoder_state_dict": best_state["encoder"],
                    "decoder_state_dict": best_state["decoder"],
                    "epoch": epoch,
                    "best_metrics": best,
                    "history": history,
                    "hparams": run_hparams,
                    "timestamp": datetime.now().isoformat(),
                }, sp)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                if show_tqdm and _HAS_TQDM: tqdm.write(f"Early stopping at epoch {epoch} (patience={patience}).")
                else: print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    end_time = datetime.now()
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        decoder.load_state_dict(best_state["decoder"])
        if show_tqdm and _HAS_TQDM:
            tqdm.write(f"Restored best from epoch {best['epoch']} | AUC={best['AUC']:.4f} | H@10={best['Hits@10']:.4f}")
        else:
            print(f"Restored best from epoch {best['epoch']} | AUC={best['AUC']:.4f} | H@10={best['Hits@10']:.4f}")

        if save_best_path and not save_on_improve:
            sp = Path(_safe_filename(str(save_best_path)))
            torch.save({
                "encoder_state_dict": best_state["encoder"],
                "decoder_state_dict": best_state["decoder"],
                "epoch": best["epoch"],
                "best_metrics": best,
                "history": history,
                "hparams": run_hparams,
                "timestamp": datetime.now().isoformat(),
            }, sp)

    return {
        "best": best,
        "history": history,
        "epochs_trained": history[-1]["epoch"] if history else 0,
        "start_time": start_time,
        "end_time": end_time,
        "checkpoint_path": str(save_best_path) if save_best_path else None,
        "hparams": run_hparams,
    }


# =========================
#         Runners
# =========================
def run_lora_rgin(dataset_name: str,
                  train_p: Path, valid_p: Path, test_p: Path,
                  *,
                  emb_dim=128, num_layers=3, hidden_layers=3,
                  rank=8, adapter_scale=1.0,
                  batch_size: int = 2048,
                  add_reverse=True,
                  reverse_relation_strategy="duplicate_rel"):
    from dataset_loader import KGDataModuleTyped

    dm = KGDataModuleTyped(
        train_p, valid_p, test_p,
        batch_size=batch_size,
        add_reverse=add_reverse,
        reverse_relation_strategy=reverse_relation_strategy,
    )
    num_nodes = len(dm.ent2id)
    num_relations = len(dm.rel2id)
    print(f"[{dataset_name}] Entities={num_nodes:,} Relations={num_relations:,}")

    enc = RelationalLoRAGINEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        emb_dim=emb_dim,
        num_layers=num_layers,
        hidden_layers=hidden_layers,
        dropout=0.1,
        rank=rank,
        adapter_scale=adapter_scale,
        train_eps=True,
    )
    dec = DistMultDecoder(num_relations=num_relations, dim=emb_dim)

    save_path = f"checkpoints/r-lora/{dataset_name}/best_rginlora_distmult_d={emb_dim}_L={num_layers}_mlp={hidden_layers}_rank={rank}.pt"

    if dataset_name == 'Cora':
        patience, epochs= 100,300
    else:
        patience=10
        epochs = 100
    res = train_linkpred_typed(
        enc, dec, dm,
        epochs=epochs, lr=1e-3, weight_decay=1e-4, patience=patience,
        show_tqdm=True,
        save_best_path=save_path,
        save_on_improve=True,
        hparams={
            "dataset": dataset_name,
            "decoder": "DistMult",
            "emb_dim": emb_dim,
            "num_layers": num_layers,
            "hidden_layers": hidden_layers,
            "rank": rank,
            "adapter_scale": adapter_scale,
            "method": "Relation-LoRA GIN",
        },
        eval_sampled_k=100,
    )

    print_training_report(
        model_name="Relation-LoRA GIN + DistMult",
        result=res,
        header_title=f"{dataset_name} Results",
        save_path=f"results/r-lora/{dataset_name}/report_rginlora_d={emb_dim}_L={num_layers}_mlp={hidden_layers}_rank={rank}.txt",
    )
    return res



res_wn = run_lora_rgin(
    "WN18RR",
    Path("../WN18RR/train.txt"),
    Path("../WN18RR/valid.txt"),
    Path("../WN18RR/test.txt"),
    emb_dim=128, num_layers=3, hidden_layers=3,
    rank=8, adapter_scale=1.0,
    batch_size=2048,
    add_reverse=True,
    reverse_relation_strategy="duplicate_rel",
)




res_fb = run_lora_rgin(
    "FB15K-237",
    Path("../FB15K-237/train.txt"),
    Path("../FB15K-237/valid.txt"),
    Path("../FB15K-237/test.txt"),
    emb_dim=128, num_layers=3, hidden_layers=3,
    rank=8, adapter_scale=1.0,
    batch_size=2048,
    add_reverse=True,
    reverse_relation_strategy="duplicate_rel",
)



res_cora = run_lora_rgin(
    "Cora",
    Path("data/CORA_KG/train.txt"),
    Path("data/CORA_KG/valid.txt"),
    Path("data/CORA_KG/test.txt"),
    emb_dim=128, num_layers=2, hidden_layers=3,
    rank=8, adapter_scale=1.0,
    batch_size=1024,
    add_reverse=False,   # you used single forward relation in export
    reverse_relation_strategy="duplicate_rel",
)
