import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GINConv
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False
    print("sklearn not found: AUC will be computed via a simple approx (PRNG tie-breaks).")



class GINEncoder(torch.nn.Module):
    """
    Flexible GIN encoder with configurable MLP depth.

    Args:
        num_nodes (int): Number of nodes in the graph.
        hidden_layers (int): Number of hidden layers inside each MLP.
        emb_dim (int): Embedding dimension.
        num_layers (int): Number of GIN layers.
        train_eps (bool): Whether to learn epsilon.
        dropout (float): Dropout rate.
    """
    def __init__(self, num_nodes, hidden_layers=2, emb_dim=128, num_layers=3, train_eps=True, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.embed.weight)

        def make_mlp(in_dim, hidden_dim, out_dim, num_hidden):
            """Builds an MLP with variable hidden depth."""
            layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_hidden - 1):  # add intermediate hidden layers
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        self.convs = nn.ModuleList([
            GINConv(make_mlp(emb_dim, emb_dim * 2, emb_dim, hidden_layers), train_eps=train_eps)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index):
        x = self.embed.weight  # [N, emb_dim]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.dropout(x)
        return x  # node embeddings


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

def build_edge_index_from_dm(dm) -> torch.LongTensor:
    """
    Build an undirected edge_index [2, E] from the *train* split only.
    - For Typed DM: use (h, t) from train triples (relation ignored for GIN).
    - For Collapsed DM: use train pairs.
    Ensures symmetry (adds reverse if missing).
    """
    if hasattr(dm, "_train_triples"):
        ht = dm._train_triples[:, [0, 2]]  # (h, t)
    else:
        ht = dm._train_pairs                # (h, t)
    if ht.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long)

    # ensure both directions
    rev = torch.stack([ht[:,1], ht[:,0]], dim=1)
    edges = torch.cat([ht, rev], dim=0).T.contiguous()  # [2, E]
    return edges


@torch.no_grad()
def sample_negatives(pairs_or_triples: torch.Tensor, num_entities: int, typed: bool) -> torch.Tensor:
    """
    1:1 negatives per positive (half head-corrupt, half tail-corrupt).
    Returns a tensor with same shape as input.
    """
    B = pairs_or_triples.size(0)
    device = pairs_or_triples.device
    rand_ents = torch.randint(0, num_entities, (B,), device=device)

    neg = pairs_or_triples.clone()
    flip = torch.rand(B, device=device) < 0.5
    if typed:
        # triples: (h, r, t)
        neg[flip, 0] = rand_ents[flip]   # corrupt head
        neg[~flip, 2] = rand_ents[~flip] # corrupt tail
    else:
        # pairs: (h, t)
        neg[flip, 0] = rand_ents[flip]
        neg[~flip, 1] = rand_ents[~flip]
    return neg


def dot_scores(z: torch.Tensor, X: torch.Tensor, typed: bool) -> torch.Tensor:
    """
    Dot-product decoder.
    - If typed: X is [B,3] (h,r,t) but r is ignored for dot-product.
    - If untyped: X is [B,2] (h,t).
    """
    if typed:
        h, t = X[:,0], X[:,2]
    else:
        h, t = X[:,0], X[:,1]
    return (z[h] * z[t]).sum(dim=1)  # logits


@torch.no_grad()
def evaluate_metrics(
        encoder: torch.nn.Module,
        edge_index: torch.Tensor,
        loader: Optional[DataLoader],
        num_entities: int,
        typed: bool,
        device: torch.device,
        show_tqdm: bool = False,
) -> Dict[str, float]:
    """
    Evaluates AUC and Hits@1/5/10 using *unfiltered* ranking.
    Embeddings are computed from the train graph (edge_index).
    """
    if loader is None:
        return {"AUC": float("nan"), "Hits@1": float("nan"), "Hits@5": float("nan"), "Hits@10": float("nan")}

    encoder.eval()
    z = encoder(edge_index.to(device))  # [N, d]

    # --- AUC: 1 negative per positive ---
    all_scores = []
    all_labels = []
    it_auc = loader if not show_tqdm else tqdm(loader, leave=False, desc="Eval AUC")
    for X_pos, _ in it_auc:
        X_pos = X_pos.to(device)
        X_neg = sample_negatives(X_pos, num_entities, typed=typed).to(device)

        s_pos = dot_scores(z, X_pos, typed)
        s_neg = dot_scores(z, X_neg, typed)

        all_scores.append(torch.cat([s_pos, s_neg]).detach().cpu().numpy())
        all_labels.append(np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))]))

    scores = np.concatenate(all_scores) if len(all_scores) > 0 else np.array([])
    labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
    if _HAS_SK and len(scores) > 0:
        auc = float(roc_auc_score(labels, scores))
    else:
        if len(scores) == 0:
            auc = float("nan")
        else:
            order = np.argsort(scores)
            ranks = np.empty_like(order); ranks[order] = np.arange(len(scores))
            pos_ranks = ranks[labels == 1]
            neg_ranks = ranks[labels == 0]
            auc = float(np.mean(pos_ranks[:, None] > neg_ranks[None, :]))

    # --- Hits@k ---
    hits1 = hits5 = hits10 = 0
    trials = 0
    it_hits = loader if not show_tqdm else tqdm(loader, leave=False, desc="Eval Hits")
    for X_pos, _ in it_hits:
        X_pos = X_pos.to(device)
        B = X_pos.size(0)
        if typed:
            h, t_true = X_pos[:, 0], X_pos[:, 2]
        else:
            h, t_true = X_pos[:, 0], X_pos[:, 1]

        rand_t = torch.randint(0, num_entities, (B, 99), device=device)
        cand_t = torch.cat([t_true.view(-1, 1), rand_t], dim=1)  # [B,100]

        e_h = z[h]                      # [B,d]
        e_c = z[cand_t]                 # [B,100,d]
        s = (e_h.unsqueeze(1) * e_c).sum(dim=2)  # [B,100]

        ranks = s.argsort(dim=1, descending=True)
        true_positions = torch.nonzero(ranks == 0, as_tuple=False)[:, 1] + 1  # 1-based rank
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


def train_linkpred(
        encoder: torch.nn.Module,
        dm,                                      # KGDataModuleCollapsed or KGDataModuleTyped
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: Optional[torch.device] = None,
        show_tqdm: bool = True,
        save_best_path: Optional[str | Path] = None,
        save_on_improve: bool = True,
        hparams: Optional[Dict[str, Any]] = None,         # <<< NEW
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    edge_index = build_edge_index_from_dm(dm).to(device)
    typed = hasattr(dm, "_train_triples")

    train_loader = dm.train_loader()
    val_loader   = dm.val_loader()
    num_entities = len(dm.ent2id)

    # --------- Build/augment hparams ---------
    auto_hparams: Dict[str, Any] = {
        "model_name": encoder.__class__.__name__,
        "optimizer": "Adam",
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "typed_graph": typed,
        "batch_size": getattr(dm, "batch_size", None),
        "add_reverse": getattr(dm, "add_reverse", None),
        "reverse_relation_strategy": getattr(dm, "reverse_relation_strategy", None),
        "num_nodes": getattr(getattr(encoder, "embed", None), "num_embeddings", None),
        "emb_dim": getattr(getattr(encoder, "embed", None), "embedding_dim", None),
        "num_layers": len(getattr(encoder, "convs", [])),
        "dropout": getattr(encoder, "dropout", None).__dict__.get("p", None) if hasattr(encoder, "dropout") else None,
    }
    # user-supplied hparams override auto
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
        epoch_iter = tqdm(epoch_iter, desc="Epochs")

    start_time = datetime.now()

    for epoch in epoch_iter:
        encoder.train()
        running_loss = 0.0
        running_n    = 0

        batch_iter = train_loader
        if show_tqdm:
            batch_iter = tqdm(train_loader, leave=False, desc=f"Train {epoch}")

        for X_pos, _ in batch_iter:
            X_pos = X_pos.to(device)
            X_neg = sample_negatives(X_pos, num_entities, typed=typed).to(device)

            z = encoder(edge_index)
            s_pos = dot_scores(z, X_pos, typed=typed)
            s_neg = dot_scores(z, X_neg, typed=typed)

            scores = torch.cat([s_pos, s_neg], dim=0)
            labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)], dim=0)

            loss = F.binary_cross_entropy_with_logits(scores, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            opt.step()

            running_loss += loss.item() * labels.numel()
            running_n    += labels.numel()

            if show_tqdm:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_n, 1)

        # ---- Validation
        val_metrics = evaluate_metrics(encoder, edge_index, val_loader, num_entities, typed, device, show_tqdm=show_tqdm)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc": float(val_metrics["AUC"]),
            "val_hits1": float(val_metrics["Hits@1"]),
            "val_hits5": float(val_metrics["Hits@5"]),
            "val_hits10": float(val_metrics["Hits@10"]),
        })

        # ---- Early stopping on Hits@10
        if val_metrics["Hits@10"] > best["Hits@10"]:
            best.update({"epoch": epoch, **val_metrics})
            best_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
            patience_ctr = 0


            # SAVE on improvement (with hparams)
            if save_best_path and save_on_improve:
                if save_best_path:
                    save_best_path = Path(_safe_filename(str(save_best_path)))
                    save_best_path.parent.mkdir(parents=True, exist_ok=True)


                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": epoch,
                        "best_metrics": best,
                        "history": history,
                        "hparams": run_hparams,      # <<< save hparams
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

    # ---- Restore best into encoder
    end_time = datetime.now()
    if best_state is not None:
        encoder.load_state_dict(best_state)
        if show_tqdm:
            tqdm.write(f"Restored best model from epoch {best['epoch']} | "
                       f"AUC={best['AUC']:.4f} | Hits@10={best['Hits@10']:.4f}")

        # SAVE final best if user wanted single save at end
        if save_best_path and not save_on_improve:
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": best["epoch"],
                    "best_metrics": best,
                    "history": history,
                    "hparams": run_hparams,      # <<< save hparams
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
        "hparams": run_hparams,                 # <<< return hparams
    }




from pathlib import Path
from dataset_loader import KGDataModuleCollapsed

dataset = 'WN18RR'
train_p = Path("../WN18RR/train.txt")
valid_p = Path("../WN18RR/valid.txt")
test_p  = Path("../WN18RR/test.txt")


dm = KGDataModuleCollapsed(train_p, valid_p, test_p, add_reverse=True)
hidden_layers,num_layers, emb_dim = 3, 3, 128
encoder = GINEncoder(num_nodes=len(dm.ent2id), hidden_layers=hidden_layers, emb_dim=emb_dim, num_layers=num_layers, dropout=0.1)

result = train_linkpred(
    encoder, dm,
    epochs=100, lr=1e-3, weight_decay=1e-4, patience=10,
    show_tqdm=True,
    save_best_path=f"checkpoints/checkpoints_gin/{dataset}/gin_bestemb={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.pt",
    save_on_improve=True                        # save every improvement
)
print_training_report("GIN", result, save_path=f"results/results_gin/{dataset}/gin_reportemb={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.txt")


from dataset_loader import KGDataModuleCollapsed
dataset = "FB15K-237"
train_p = Path("../FB15K-237/train.txt")
valid_p = Path("../FB15K-237/valid.txt")
test_p  = Path("../FB15K-237/test.txt")

dm = KGDataModuleCollapsed(train_p, valid_p, test_p, add_reverse=True)
hidden_layers,num_layers, emb_dim = 3, 3, 128
encoder = GINEncoder(num_nodes=len(dm.ent2id), hidden_layers=hidden_layers, emb_dim=emb_dim, num_layers=num_layers, dropout=0.1)

result = train_linkpred(
    encoder, dm,
    epochs=100, lr=1e-3, weight_decay=1e-4, patience=10,
    show_tqdm=True,
    save_best_path=f"checkpoints/checkpoints_gin/{dataset}/gin_best_emb={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.pt",
    save_on_improve=True                        # save every improvement
)
print_training_report("GIN", result, save_path=f"results/results_gin/{dataset}/gin_report_emb={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.txt")


from dataset_loader import KGDataModuleCollapsed

train_p = Path("data/CORA_KG/train.txt")
valid_p = Path("data/CORA_KG/valid.txt")
test_p  = Path("data/CORA_KG/test.txt")
dataset = "Cora"
dm = KGDataModuleCollapsed(train_p, valid_p, test_p, add_reverse=True)
hidden_layers,num_layers, emb_dim = 3, 3, 128
encoder = GINEncoder(num_nodes=len(dm.ent2id), hidden_layers=hidden_layers, emb_dim=emb_dim, num_layers=num_layers, dropout=0.1)

result = train_linkpred(
    encoder, dm,
    epochs=100, lr=1e-3, weight_decay=1e-4, patience=10,
    show_tqdm=True,
    save_best_path=f"checkpoints/checkpoints_gin/{dataset}/gin_best_emb={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.pt",
    save_on_improve=True                        # save every improvement
)
print_training_report("GIN", result, save_path=f"results/results_gin/{dataset}/gin_report_emb={emb_dim}_mlp={hidden_layers}_ag={num_layers}_ds={dataset}.txt")