import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, Optional

_train_path = Path("../WN18RR/train.txt")
_test_path = Path("../WN18RR/test.txt")
_valid_path = Path("../WN18RR/valid.txt")

def load_dataset(path: Path) -> list[tuple]:
    """
    parses dataset path into list of tuples.
    """
    datalist = []
    with open(path, "r") as f:
        for line in f:
            head, relation, tail = line.strip().split("\t")
            datalist.append((head, relation, tail))
    return datalist

train_list = load_dataset(_train_path)
valid_list = load_dataset(_valid_path)
test_list = load_dataset(_test_path)

entities, relations = set(), set()
for h, r, t in (train_list + valid_list + test_list):
    entities.add(h); entities.add(t); relations.add(r)

ent2id = {e: i for i, e in enumerate(sorted(entities))}
rel2id = {r: i for i, r in enumerate(sorted(relations))}
num_entities, num_relations = len(ent2id), len(rel2id)
print(f"#entities={num_entities}, #relations={num_relations}")

def triples_to_tensor(triples):
    arr = np.array([(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triples], dtype=np.int64)
    return torch.from_numpy(arr)

train_triples = triples_to_tensor(train_list)
valid_triples = triples_to_tensor(valid_list)
test_triples = triples_to_tensor(test_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

train_triples = train_triples.to(device)
valid_triples = valid_triples.to(device)
test_triples = test_triples.to(device)
print(f"Train: {len(train_triples)}, Valid: {len(valid_triples)}, Test: {len(test_triples)} triples")

rel_edge_index = defaultdict(list)
for h, r, t in train_triples.tolist():
    rel_edge_index[r].append((h, t))
    rel_edge_index[r].append((t, h))

for r in range(num_relations):
    if len(rel_edge_index[r]) == 0:
        rel_edge_index[r] = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        eidx = torch.tensor(rel_edge_index[r], dtype=torch.long).t().contiguous()
        eidx, _ = add_self_loops(eidx, num_nodes=num_entities)
        rel_edge_index[r] = eidx.to(device)

edges_src, edges_dst, edges_type = [], [], []
for h, r, t in train_triples.tolist():
    edges_src.extend([h, t])
    edges_dst.extend([t, h])
    edges_type.extend([r, r])

edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
edge_type = torch.tensor(edges_type, dtype=torch.long, device=device)
print(f"Unified edge_index shape: {edge_index.shape}, edge_type shape: {edge_type.shape}")

class RelationalGATEncoder(nn.Module):
    """
    One GATConv per relation. Messages are summed over relations each layer.
    """
    def __init__(self, num_entities, num_relations,
                 emb_dim=128, hidden_dim=128, out_dim=256,
                 heads=4, dropout=0.2):
        super().__init__()
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        self.gat1 = nn.ModuleDict({
            str(r): GATConv(emb_dim, hidden_dim, heads=heads, dropout=dropout)
            for r in range(num_relations)
        })
        self.gat2 = nn.ModuleDict({
            str(r): GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
            for r in range(num_relations)
        })

        self.res_proj = nn.Linear(emb_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, rel_edge_index: dict[int, torch.Tensor]):
        x0 = self.entity_emb.weight

        outs = []
        for r in range(self.num_relations):
            eidx = rel_edge_index[r]
            if eidx.numel() == 0:
                continue
            outs.append(F.elu(self.gat1[str(r)](x0, eidx)))
        x = torch.stack(outs).sum(0) if outs else torch.zeros_like(self.res_proj.weight[:x0.size(1)])
        x = self.drop(x)

        outs = []
        for r in range(self.num_relations):
            eidx = rel_edge_index[r]
            if eidx.numel() == 0:
                continue
            outs.append(self.gat2[str(r)](x, eidx))
        x = torch.stack(outs).sum(0) if outs else torch.zeros_like(self.res_proj(x0))

        x = self.ln(x + self.res_proj(x0))
        return x

class EmbedRelationGATEncoder(nn.Module):
    """
    GAT with relation embeddings: node features are modulated by relation embeddings
    during message passing (similar to RelationalGIN approach)
    """
    def __init__(self, num_entities, num_relations,
                 emb_dim=128, hidden_dim=128, out_dim=256,
                 heads=4, dropout=0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        
        self.gat1 = GATConv(emb_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        
        self.res_proj = nn.Linear(emb_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, edge_index, edge_type):
        """
        edge_index: [2, E] unified edge index
        edge_type: [E] relation type per edge
        """
        x0 = self.entity_emb.weight
        
        rel_weights = self.rel_emb(edge_type)
        
        x = self.gat1(x0, edge_index)
        x = F.elu(x)
        x = self.drop(x)
        
        x = self.gat2(x, edge_index)
        
        x = self.ln(x + self.res_proj(x0))
        return x

class DistMultDecoder(nn.Module):
    """
    score(h,r,t) = <e_h, w_r, e_t>
    """
    def __init__(self, num_relations, dim):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, e_h, r, e_t):
        w_r = self.rel_emb(r)
        return (e_h * w_r * e_t).sum(dim=1)

class DotProductDecoder(nn.Module):
    """
    score(h,r,t) = <e_h, e_t> (relation is ignored)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, e_h, r, e_t):
        return (e_h * e_t).sum(dim=1)

class LinkPredictor(nn.Module):
    """Flexible link predictor that works with different encoder/decoder combinations"""
    def __init__(self, encoder, decoder, graph_data):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.graph_data = graph_data
        
    def forward(self, triples):
        h = triples[:, 0]; r = triples[:, 1]; t = triples[:, 2]
        
        if isinstance(self.encoder, RelationalGATEncoder):
            ent = self.encoder(self.graph_data)
        elif isinstance(self.encoder, EmbedRelationGATEncoder):
            edge_index, edge_type = self.graph_data
            ent = self.encoder(edge_index, edge_type)
        else:
            raise ValueError(f"Unknown encoder type: {type(self.encoder)}")
        
        return self.decoder(ent[h], r, ent[t])

class RelationalGATLinkPredictor(nn.Module):
    """Legacy wrapper for backward compatibility"""
    def __init__(self, num_entities, num_relations, rel_edge_index,
                 out_dim=256, **gat_kwargs):
        super().__init__()
        self.encoder = RelationalGATEncoder(num_entities, num_relations,
                                            out_dim=out_dim, **gat_kwargs)
        self.decoder = DistMultDecoder(num_relations, out_dim)
        self.rel_edge_index = rel_edge_index

    def forward(self, triples):
        h = triples[:, 0]; r = triples[:, 1]; t = triples[:, 2]
        ent = self.encoder(self.rel_edge_index)
        return self.decoder(ent[h], r, ent[t])

def batches(tensor, batch_size, shuffle=True):
    N = tensor.size(0)
    idx = torch.randperm(N, device=tensor.device) if shuffle else torch.arange(N, device=tensor.device)
    for i in range(0, N, batch_size):
        yield tensor[idx[i:i+batch_size]]

@torch.no_grad()
def sample_negatives_both(pos_triples, num_entities, k_neg=10):
    """
    Returns flattened head- and tail-corrupted negatives:
      neg_h: [B*k,3], neg_t: [B*k,3]
    """
    B = pos_triples.size(0)
    device = pos_triples.device

    tails = torch.randint(0, num_entities, (B, k_neg), device=device)
    neg_t = pos_triples.unsqueeze(1).repeat(1, k_neg, 1)
    neg_t[:, :, 2] = tails

    heads = torch.randint(0, num_entities, (B, k_neg), device=device)
    neg_h = pos_triples.unsqueeze(1).repeat(1, k_neg, 1)
    neg_h[:, :, 0] = heads

    return neg_h.view(-1, 3), neg_t.view(-1, 3)

def train_one_epoch(model, triples, optimizer, batch_size=2048, k_neg=10):
    model.train()

    pos_weight = torch.tensor([2.0 * k_neg], device=triples.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss, total_items = 0.0, 0
    for pos in batches(triples, batch_size, shuffle=True):
        neg_h, neg_t = sample_negatives_both(pos, num_entities, k_neg=k_neg)
        all_trip = torch.cat([pos, neg_h, neg_t], dim=0)
        labels   = torch.cat([
            torch.ones(len(pos), device=triples.device),
            torch.zeros(len(neg_h) + len(neg_t), device=triples.device)
        ], dim=0)

        scores = model(all_trip)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_items += labels.numel()

    return total_loss / total_items

@torch.no_grad()
def evaluate_auc_hits(model, triples, batch_size=4096, hits_k_list=[1, 5, 10]):
    """
    Evaluate AUC and Hits@k for multiple k values.
    
    Args:
        hits_k_list: List of k values for Hits@k metric (default: [1, 5, 10])
    """
    model.eval()
    scores_all, labels_all = [], []
    for pos in batches(triples, batch_size, shuffle=False):
        B = pos.size(0)
        neg = pos.clone()
        neg[:, 2] = torch.randint(0, num_entities, (B,), device=pos.device)

        s_pos = model(pos)
        s_neg = model(neg)

        scores_all.append(torch.cat([s_pos, s_neg], dim=0).cpu().numpy())
        labels_all.append(np.concatenate([np.ones(B), np.zeros(B)], axis=0))

    auc = roc_auc_score(np.concatenate(labels_all), np.concatenate(scores_all))

    hits_dict = {k: 0 for k in hits_k_list}
    trials = 0
    if isinstance(model, LinkPredictor):
        if isinstance(model.encoder, RelationalGATEncoder):
            ent = model.encoder(model.graph_data)
        elif isinstance(model.encoder, EmbedRelationGATEncoder):
            edge_index, edge_type = model.graph_data
            ent = model.encoder(edge_index, edge_type)
        else:
            raise ValueError(f"Unknown encoder type: {type(model.encoder)}")
    elif hasattr(model, 'encoder'):
        if hasattr(model, 'rel_edge_index'):
            ent = model.encoder(model.rel_edge_index)
        else:
            raise ValueError("Cannot determine how to call encoder")
    else:
        raise ValueError("Model has no encoder attribute")
    
    for pos in batches(triples, batch_size, shuffle=False):
        B = pos.size(0)
        h = pos[:, 0]; r = pos[:, 1]; t_true = pos[:, 2]

        rand_t = torch.randint(0, num_entities, (B, 99), device=pos.device)
        cand_t = torch.cat([t_true.unsqueeze(1), rand_t], dim=1)

        e_h = ent[h]
        e_c = ent[cand_t]
        
        if isinstance(model.decoder, DistMultDecoder):
            w_r = model.decoder.rel_emb(r)
            s = ((e_h * w_r).unsqueeze(1) * e_c).sum(dim=2)
        elif isinstance(model.decoder, DotProductDecoder):
            s = (e_h.unsqueeze(1) * e_c).sum(dim=2)
        else:
            raise ValueError(f"Unknown decoder type: {type(model.decoder)}")
        
        ranks = (s.argsort(dim=1, descending=True) == 0).nonzero()[:, 1] + 1
        
        for k in hits_k_list:
            hits_dict[k] += (ranks <= k).sum().item()
        trials += B

    result = {"AUC": float(auc)}
    for k in hits_k_list:
        result[f"Hits@{k}"] = hits_dict[k] / trials
    
    return result

# ============================================================================
# ABLATION STUDY TRAINING FUNCTION
# ============================================================================

def train_model(model, train_triples, valid_triples, test_triples,
                epochs=100, lr=1e-3, weight_decay=1e-4, patience=10,
                batch_size=2048, k_neg=10, model_name="model"):
    """
    Train a model with early stopping and return results dict.
    Evaluates on test set after training completes.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = []
    best = {"epoch": 0, "AUC": -1.0, "Hits@1": 0.0, "Hits@5": 0.0, "Hits@10": 0.0}
    patience_counter = 0
    best_state = None
    
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    for epoch in tqdm(range(1, epochs + 1), desc=f"{model_name}"):
        loss = train_one_epoch(model, train_triples, optimizer, batch_size=batch_size, k_neg=k_neg)
        metrics = evaluate_auc_hits(model, valid_triples, batch_size=4096, hits_k_list=[1, 5, 10])
        
        history.append({
            "epoch": epoch,
            "train_loss": float(loss),
            "val_auc": float(metrics["AUC"]),
            "val_hits1": float(metrics["Hits@1"]),
            "val_hits5": float(metrics["Hits@5"]),
            "val_hits10": float(metrics["Hits@10"]),
        })
        
        print(f"Epoch {epoch:03d} | loss={loss:.4f} | "
              f"AUC={metrics['AUC']:.4f} | "
              f"H@1={metrics['Hits@1']:.4f} | "
              f"H@5={metrics['Hits@5']:.4f} | "
              f"H@10={metrics['Hits@10']:.4f}")
        
        # Early stopping on AUC
        if metrics["AUC"] > best["AUC"]:
            best.update({"epoch": epoch, **metrics})
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  → New best AUC: {best['AUC']:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break
    
    end_time = datetime.now()
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model from epoch {best['epoch']} | "
              f"AUC={best['AUC']:.4f} | Hits@10={best['Hits@10']:.4f}")
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print(f"FINAL TEST SET EVALUATION")
    print(f"{'='*70}")
    test_metrics = evaluate_auc_hits(model, test_triples, batch_size=4096, hits_k_list=[1, 5, 10])
    print(f"Test AUC:    {test_metrics['AUC']:.4f}")
    print(f"Test Hits@1: {test_metrics['Hits@1']:.4f}")
    print(f"Test Hits@5: {test_metrics['Hits@5']:.4f}")
    print(f"Test Hits@10: {test_metrics['Hits@10']:.4f}")
    print(f"{'='*70}\n")
    
    return {
        "best": best,
        "test": test_metrics,
        "history": history,
        "epochs_trained": len(history),
        "start_time": start_time,
        "end_time": end_time,
        "model_state": best_state
    }

# ============================================================================
# REPORTING FUNCTIONS
# ============================================================================

def _fmt(x):
    try:
        return f"{float(x):.4f}"
    except:
        return "nan"

def print_comparison_report(title, left_name, left_result, right_name, right_result, save_path=None):
    """Print and save comparison report for two models."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"{title}")
    lines.append(f"{'='*80}\n")
    
    # Best validation metrics summary
    lines.append("BEST VALIDATION METRICS (used for model selection)")
    lines.append("-"*80)
    for name, res in [(left_name, left_result), (right_name, right_result)]:
        b = res["best"]
        lines.append(
            f"{name:30s} | "
            f"AUC={_fmt(b.get('AUC'))} | "
            f"H@1={_fmt(b.get('Hits@1'))} | "
            f"H@5={_fmt(b.get('Hits@5'))} | "
            f"H@10={_fmt(b.get('Hits@10'))} "
            f"(epoch {b.get('epoch')})"
        )
    lines.append("")
    
    # Test metrics summary
    lines.append("FINAL TEST SET PERFORMANCE (unbiased estimate)")
    lines.append("-"*80)
    for name, res in [(left_name, left_result), (right_name, right_result)]:
        t = res["test"]
        lines.append(
            f"{name:30s} | "
            f"AUC={_fmt(t.get('AUC'))} | "
            f"H@1={_fmt(t.get('Hits@1'))} | "
            f"H@5={_fmt(t.get('Hits@5'))} | "
            f"H@10={_fmt(t.get('Hits@10'))}"
        )
    lines.append("")
    
    # Detailed histories
    for name, res in [(left_name, left_result), (right_name, right_result)]:
        lines.append(f"\n{name} - Training History")
        lines.append("-"*90)
        lines.append(f"{'Epoch':<8} {'Train Loss':<14} {'Val AUC':<12} {'H@1':<12} {'H@5':<12} {'H@10':<12}")
        lines.append("-"*90)
        for rec in res["history"]:
            lines.append(
                f"{rec['epoch']:<8} "
                f"{_fmt(rec['train_loss']):<14} "
                f"{_fmt(rec['val_auc']):<12} "
                f"{_fmt(rec['val_hits1']):<12} "
                f"{_fmt(rec['val_hits5']):<12} "
                f"{_fmt(rec['val_hits10']):<12}"
            )
        lines.append("")
    
    report = "\n".join(lines)
    print(report)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {save_path}")

# ============================================================================
# ABLATION STUDY CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("RGAT ABLATION STUDY CONFIGURATION")
print("="*80)
print("""
Available Ablation Studies:
1. Decoder Comparison: DistMult vs DotProduct (Attention-per-Relation encoder)
2. Encoder Comparison: Attention-per-Relation vs Embed-Relation (DistMult decoder)
3. Combined Variant: Embed-Relation + DotProduct vs Baseline

Configure below by setting flags to True/False
""")

# ============================================================================
# CONFIGURATION - MODIFY THESE TO RUN DIFFERENT ABLATION STUDIES
# ============================================================================

RUN_DECODER_ABLATION = True
RUN_ENCODER_ABLATION = True
RUN_COMBINED_ABLATION = True

CONFIG = {
    "emb_dim": 128,
    "hidden_dim": 128,
    "out_dim": 256,
    "heads": 4,
    "dropout": 0.2,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 10,
    "batch_size": 2048,
    "k_neg": 10,
}

print("\nModel Hyperparameters:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print("")

# ============================================================================
# ABLATION 1: DECODER COMPARISON (DistMult vs DotProduct)
# ============================================================================

if RUN_DECODER_ABLATION:
    print("\n" + "🔬"*40)
    print("ABLATION STUDY 1: DECODER COMPARISON")
    print("🔬"*40)
    print("Comparing: DistMult Decoder vs DotProduct Decoder")
    print("Encoder: Attention-per-Relation (baseline)")
    print("")
    
    # Model 1: Attention-per-Relation + DistMult
    enc_distmult = RelationalGATEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        out_dim=CONFIG["out_dim"],
        heads=CONFIG["heads"],
        dropout=CONFIG["dropout"]
    )
    dec_distmult = DistMultDecoder(num_relations, CONFIG["out_dim"])
    model_distmult = LinkPredictor(enc_distmult, dec_distmult, rel_edge_index).to(device)
    
    res_distmult = train_model(
        model_distmult, train_triples, valid_triples, test_triples,
        epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"], batch_size=CONFIG["batch_size"], k_neg=CONFIG["k_neg"],
        model_name="Attention-per-Relation + DistMult"
    )
    
    # Model 2: Attention-per-Relation + DotProduct
    enc_dot = RelationalGATEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        out_dim=CONFIG["out_dim"],
        heads=CONFIG["heads"],
        dropout=CONFIG["dropout"]
    )
    dec_dot = DotProductDecoder()
    model_dot = LinkPredictor(enc_dot, dec_dot, rel_edge_index).to(device)
    
    res_dot = train_model(
        model_dot, train_triples, valid_triples, test_triples,
        epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"], batch_size=CONFIG["batch_size"], k_neg=CONFIG["k_neg"],
        model_name="Attention-per-Relation + DotProduct"
    )
    
    # Print comparison
    print_comparison_report(
        title="DECODER ABLATION: DistMult vs DotProduct",
        left_name="Attention-per-Relation + DistMult",
        left_result=res_distmult,
        right_name="Attention-per-Relation + DotProduct",
        right_result=res_dot,
        save_path="results/ablation_decoder_comparison.txt"
    )
    
    # Save checkpoints
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': res_distmult["model_state"],
        'config': CONFIG,
        'results': res_distmult
    }, "checkpoints/attention_per_rel_distmult.pt")
    torch.save({
        'model_state_dict': res_dot["model_state"],
        'config': CONFIG,
        'results': res_dot
    }, "checkpoints/attention_per_rel_dotproduct.pt")
    print("Decoder ablation checkpoints saved\n")

# ============================================================================
# ABLATION 2: ENCODER COMPARISON (Attention-per-Relation vs Embed-Relation)
# ============================================================================

if RUN_ENCODER_ABLATION:
    print("\n" + "🔬"*40)
    print("ABLATION STUDY 2: ENCODER COMPARISON")
    print("🔬"*40)
    print("Comparing: Attention-per-Relation vs Embed-Relation")
    print("Decoder: DistMult (baseline)")
    print("")
    
    # Model 1: Attention-per-Relation + DistMult (baseline)
    enc_attn = RelationalGATEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        out_dim=CONFIG["out_dim"],
        heads=CONFIG["heads"],
        dropout=CONFIG["dropout"]
    )
    dec_attn = DistMultDecoder(num_relations, CONFIG["out_dim"])
    model_attn = LinkPredictor(enc_attn, dec_attn, rel_edge_index).to(device)
    
    res_attn = train_model(
        model_attn, train_triples, valid_triples, test_triples,
        epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"], batch_size=CONFIG["batch_size"], k_neg=CONFIG["k_neg"],
        model_name="Attention-per-Relation + DistMult"
    )
    
    # Model 2: Embed-Relation + DistMult
    enc_embed = EmbedRelationGATEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        out_dim=CONFIG["out_dim"],
        heads=CONFIG["heads"],
        dropout=CONFIG["dropout"]
    )
    dec_embed = DistMultDecoder(num_relations, CONFIG["out_dim"])
    model_embed = LinkPredictor(enc_embed, dec_embed, (edge_index, edge_type)).to(device)
    
    res_embed = train_model(
        model_embed, train_triples, valid_triples, test_triples,
        epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"], batch_size=CONFIG["batch_size"], k_neg=CONFIG["k_neg"],
        model_name="Embed-Relation + DistMult"
    )
    
    # Print comparison
    print_comparison_report(
        title="ENCODER ABLATION: Attention-per-Relation vs Embed-Relation",
        left_name="Attention-per-Relation + DistMult",
        left_result=res_attn,
        right_name="Embed-Relation + DistMult",
        right_result=res_embed,
        save_path="results/ablation_encoder_comparison.txt"
    )
    
    # Save checkpoints
    torch.save({
        'model_state_dict': res_attn["model_state"],
        'config': CONFIG,
        'results': res_attn
    }, "checkpoints/attention_per_rel_distmult.pt")
    torch.save({
        'model_state_dict': res_embed["model_state"],
        'config': CONFIG,
        'results': res_embed
    }, "checkpoints/embed_rel_distmult.pt")
    print("Encoder ablation checkpoints saved\n")

# ============================================================================
# ABLATION 3: COMBINED VARIANT (Embed-Relation + DotProduct)
# ============================================================================

if RUN_COMBINED_ABLATION:
    print("\n" + "🔬"*40)
    print("ABLATION STUDY 3: COMBINED VARIANT")
    print("🔬"*40)
    print("Comparing: Embed-Relation + DotProduct vs Attention-per-Relation + DistMult (baseline)")
    print("Testing: Does combining both simplifications (encoder + decoder) still work?")
    print("")
    
    # Model 1: Baseline (Attention-per-Relation + DistMult)
    enc_baseline = RelationalGATEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        out_dim=CONFIG["out_dim"],
        heads=CONFIG["heads"],
        dropout=CONFIG["dropout"]
    )
    dec_baseline = DistMultDecoder(num_relations, CONFIG["out_dim"])
    model_baseline = LinkPredictor(enc_baseline, dec_baseline, rel_edge_index).to(device)
    
    res_baseline = train_model(
        model_baseline, train_triples, valid_triples, test_triples,
        epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"], batch_size=CONFIG["batch_size"], k_neg=CONFIG["k_neg"],
        model_name="Attention-per-Relation + DistMult (Baseline)"
    )
    
    # Model 2: Combined Variant (Embed-Relation + DotProduct)
    enc_combined = EmbedRelationGATEncoder(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        out_dim=CONFIG["out_dim"],
        heads=CONFIG["heads"],
        dropout=CONFIG["dropout"]
    )
    dec_combined = DotProductDecoder()
    model_combined = LinkPredictor(enc_combined, dec_combined, (edge_index, edge_type)).to(device)
    
    res_combined = train_model(
        model_combined, train_triples, valid_triples, test_triples,
        epochs=CONFIG["epochs"], lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"], batch_size=CONFIG["batch_size"], k_neg=CONFIG["k_neg"],
        model_name="Embed-Relation + DotProduct (Variant)"
    )
    
    # Print comparison
    print_comparison_report(
        title="COMBINED ABLATION: Embed-Relation + DotProduct vs Baseline",
        left_name="Attention-per-Relation + DistMult (Baseline)",
        left_result=res_baseline,
        right_name="Embed-Relation + DotProduct (Variant)",
        right_result=res_combined,
        save_path="results/ablation_combined_variant.txt"
    )
    
    # Save checkpoints
    torch.save({
        'model_state_dict': res_baseline["model_state"],
        'config': CONFIG,
        'results': res_baseline
    }, "checkpoints/baseline_attn_distmult.pt")
    torch.save({
        'model_state_dict': res_combined["model_state"],
        'config': CONFIG,
        'results': res_combined
    }, "checkpoints/combined_embed_dotproduct.pt")
    print("Combined ablation checkpoints saved\n")

print("\n" + "="*80)
print("ABLATION STUDY COMPLETE!")
print("="*80)
print("\nResults saved to:")
if RUN_DECODER_ABLATION:
    print("  - results/ablation_decoder_comparison.txt")
if RUN_ENCODER_ABLATION:
    print("  - results/ablation_encoder_comparison.txt")
if RUN_COMBINED_ABLATION:
    print("  - results/ablation_combined_variant.txt")
print("\nCheckpoints saved to checkpoints/ directory")
print("="*80 + "\n")