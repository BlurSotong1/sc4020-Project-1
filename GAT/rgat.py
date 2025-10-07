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

# Data loading
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

# Build ID maps
entities, relations = set(), set()
for h, r, t in (train_list + valid_list):
    entities.add(h); entities.add(t); relations.add(r)

ent2id = {e: i for i, e in enumerate(sorted(entities))}
rel2id = {r: i for i, r in enumerate(sorted(relations))}
num_entities, num_relations = len(ent2id), len(rel2id)
print(f"#entities={num_entities}, #relations={num_relations}")

def triples_to_tensor(triples):
    arr = np.array([(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triples], dtype=np.int64)
    return torch.from_numpy(arr)

train_triples = triples_to_tensor(train_list)  # [N,3]
valid_triples = triples_to_tensor(valid_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

train_triples = train_triples.to(device)
valid_triples = valid_triples.to(device)

# Build relation-aware edge_index dict (add reverse edges + self loops)
rel_edge_index = defaultdict(list)
for h, r, t in train_triples.tolist():
    rel_edge_index[r].append((h, t))
    rel_edge_index[r].append((t, h))  # reverse

for r in range(num_relations):
    if len(rel_edge_index[r]) == 0:
        # ensure key exists even if relation absent in train
        rel_edge_index[r] = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        eidx = torch.tensor(rel_edge_index[r], dtype=torch.long).t().contiguous()
        eidx, _ = add_self_loops(eidx, num_nodes=num_entities)
        rel_edge_index[r] = eidx.to(device)

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
        x0 = self.entity_emb.weight  # [N, emb_dim]

        # layer 1: per-relation attention then sum
        outs = []
        for r in range(self.num_relations):
            eidx = rel_edge_index[r]
            if eidx.numel() == 0:
                continue
            outs.append(F.elu(self.gat1[str(r)](x0, eidx)))
        x = torch.stack(outs).sum(0) if outs else torch.zeros_like(self.res_proj.weight[:x0.size(1)])
        x = self.drop(x)

        # layer 2: per-relation attention then sum
        outs = []
        for r in range(self.num_relations):
            eidx = rel_edge_index[r]
            if eidx.numel() == 0:
                continue
            outs.append(self.gat2[str(r)](x, eidx))
        x = torch.stack(outs).sum(0) if outs else torch.zeros_like(self.res_proj(x0))

        # residual + norm
        x = self.ln(x + self.res_proj(x0))
        return x  # [N, out_dim]

class DistMultDecoder(nn.Module):
    """
    score(h,r,t) = <e_h, w_r, e_t>
    """
    def __init__(self, num_relations, dim):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, e_h, r, e_t):
        w_r = self.rel_emb(r)  # [B, d]
        return (e_h * w_r * e_t).sum(dim=1)  # [B]

class RelationalGATLinkPredictor(nn.Module):
    def __init__(self, num_entities, num_relations, rel_edge_index,
                 out_dim=256, **gat_kwargs):
        super().__init__()
        self.encoder = RelationalGATEncoder(num_entities, num_relations,
                                            out_dim=out_dim, **gat_kwargs)
        self.decoder = DistMultDecoder(num_relations, out_dim)
        self.rel_edge_index = rel_edge_index

    def forward(self, triples):  # triples: [B,3] (h,r,t)
        h = triples[:, 0]; r = triples[:, 1]; t = triples[:, 2]
        ent = self.encoder(self.rel_edge_index)  # [N, d]
        return self.decoder(ent[h], r, ent[t])   # logits [B]

# -------------------------
# Utilities
# -------------------------
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

    # Tail corruption
    tails = torch.randint(0, num_entities, (B, k_neg), device=device)
    neg_t = pos_triples.unsqueeze(1).repeat(1, k_neg, 1)
    neg_t[:, :, 2] = tails

    # Head corruption
    heads = torch.randint(0, num_entities, (B, k_neg), device=device)
    neg_h = pos_triples.unsqueeze(1).repeat(1, k_neg, 1)
    neg_h[:, :, 0] = heads

    return neg_h.view(-1, 3), neg_t.view(-1, 3)

def train_one_epoch(model, triples, optimizer, batch_size=2048, k_neg=10):
    model.train()
    # Balance: 1 pos : (2*k_neg) neg
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
def evaluate_auc_hits(model, triples, batch_size=4096, hits_k=10):
    model.eval()
    # --- AUC: 1 negative per positive (tail corruption) ---
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

    # --- Hits@k (unfiltered): rank true tail among 99 random tails + 1 true ---
    hits, trials = 0, 0
    # Precompute entity encs once for speed
    ent = model.encoder(model.rel_edge_index)  # [N, d]
    for pos in batches(triples, batch_size, shuffle=False):
        B = pos.size(0)
        h = pos[:, 0]; r = pos[:, 1]; t_true = pos[:, 2]

        # 99 random negatives
        rand_t = torch.randint(0, num_entities, (B, 99), device=pos.device)
        cand_t = torch.cat([t_true.unsqueeze(1), rand_t], dim=1)  # [B,100]

        e_h = ent[h]                          # [B,d]
        w_r = model.decoder.rel_emb(r)        # [B,d]
        e_c = ent[cand_t]                     # [B,100,d]

        # DistMult score(h,r,?) for all candidates
        s = ((e_h * w_r).unsqueeze(1) * e_c).sum(dim=2)  # [B,100]
        ranks = (s.argsort(dim=1, descending=True) == 0).nonzero()[:, 1] + 1
        hits += (ranks <= hits_k).sum().item()
        trials += B

    return {"auc": float(auc), f"hits@{hits_k}": hits / trials}

num_entities = len(ent2id)
model = RelationalGATLinkPredictor(
    num_entities=num_entities,
    num_relations=num_relations,
    rel_edge_index=rel_edge_index,
    emb_dim=128, hidden_dim=128, out_dim=256,
    heads=4, dropout=0.2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

history = {
    'train_loss': [],
    'val_auc': [],
    'val_hits@10': []
}

best_val_auc = 0.0
patience = 10
patience_counter = 0
best_model_state = None

EPOCHS = 100
for epoch in tqdm(range(1, EPOCHS + 1), desc="epoch"):
    loss = train_one_epoch(model, train_triples, optimizer, batch_size=2048, k_neg=10)
    metrics = evaluate_auc_hits(model, valid_triples, batch_size=4096, hits_k=10)
    
    history['train_loss'].append(loss)
    history['val_auc'].append(metrics['auc'])
    history['val_hits@10'].append(metrics['hits@10'])
    
    print(f"Epoch {epoch:02d} | loss={loss:.4f} | AUC={metrics['auc']:.4f} | Hits@10={metrics['hits@10']:.4f}")
    
    if metrics['auc'] > best_val_auc:
        best_val_auc = metrics['auc']
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        print(f"  → New best AUC: {best_val_auc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience})")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nRestored best model with validation AUC: {best_val_auc:.4f}")


model_save_path = "best_rgat_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_auc': best_val_auc,
    'history': history
}, model_save_path)
print(f"Best model saved to {model_save_path}")


history_file = "rgat_training_history.txt"
with open(history_file, 'w') as f:
    f.write("Relational GAT Training History\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Best Validation AUC: {best_val_auc:.4f}\n")
    f.write(f"Total Epochs Trained: {len(history['train_loss'])}\n\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Val AUC':<15} {'Val Hits@10':<15}\n")
    f.write("-" * 60 + "\n")
    for i in range(len(history['train_loss'])):
        f.write(f"{i+1:<8} {history['train_loss'][i]:<15.4f} "
                f"{history['val_auc'][i]:<15.4f} {history['val_hits@10'][i]:<15.4f}\n")
print(f"Training history saved to {history_file}")