from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

_train_path = Path("../WN18RR/train.txt")
_test_path = Path("../WN18RR/test.txt")
_valid_path = Path("../WN18RR/valid.txt")

def load_dataset(path:Path) -> list[tuple]:
    """
    parses dataset path into list of tuples.
    """
    datalist = []
    with open(path, "r") as f:
        for line in f:
            head, relation,tail = line.strip().split("\t")
            datalist.append((head,relation,tail))

    return datalist

train_dataset = load_dataset(_train_path)
valid_dataset = load_dataset(_valid_path)
test_dataset = load_dataset(_test_path)

entities = set()
relations = set()
for h, r, t in (train_dataset + valid_dataset + test_dataset):
    entities.add(h); entities.add(t); relations.add(r)

ent2id = {e:i for i,e in enumerate(sorted(entities))}
rel2id = {r:i for i,r in enumerate(sorted(relations))}

num_entities  = len(ent2id)
num_relations = len(rel2id)
print(f"#entities={num_entities}, #relations={num_relations}")

def triples_to_tensor(triples_list):
    arr = np.array([(ent2id[h], rel2id[r], ent2id[t]) for h,r,t in triples_list], dtype=np.int64)
    return torch.from_numpy(arr)

train_triples = triples_to_tensor(train_dataset)
valid_triples = triples_to_tensor(valid_dataset)
test_triples = triples_to_tensor(test_dataset)

edges = []
for h, r, t in train_triples.tolist():
    edges.append((h, t))
    edges.append((t, h))
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print("edge_index:", edge_index.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

edge_index, _ = add_self_loops(edge_index, num_nodes=num_entities)
edge_index = edge_index.to(device)
train_triples = train_triples.to(device)
valid_triples = valid_triples.to(device)
test_triples = test_triples.to(device)
print(f"Train: {len(train_triples)}, Valid: {len(valid_triples)}, Test: {len(test_triples)} triples")

class GATLinkEncoder(nn.Module):
    def __init__(self, num_entities, emb_dim=128, hidden_dim=128, out_dim=256,
                 heads=4, dropout=0.2):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)

        self.gat1 = GATConv(emb_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)

        self.res_proj = nn.Linear(emb_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, edge_index):
        x0 = self.entity_emb.weight
        x = F.elu(self.gat1(x0, edge_index))
        x = self.drop(x)
        x = self.gat2(x, edge_index)
        x = x + self.res_proj(x0)
        x = self.ln(x)
        return x

class DotProductDecoder(nn.Module):
    def forward(self, e_h, e_t):
        return (e_h * e_t).sum(dim=1)

class LinkPredictor(nn.Module):
    def __init__(self, num_entities, edge_index, out_dim=200, **gat_kwargs):
        super().__init__()
        self.encoder = GATLinkEncoder(num_entities, out_dim=out_dim, **gat_kwargs)
        self.decoder = DotProductDecoder()
        self.edge_index = edge_index

    def forward(self, triples):
        h = triples[:, 0]
        t = triples[:, 2]
        ent = self.encoder(self.edge_index)
        e_h, e_t = ent[h], ent[t]
        return self.decoder(e_h, e_t)

@torch.no_grad()
def sample_negatives(pos_triples, num_entities, corrupt_tail=True):
    """Return negatives by corrupting tail (or head). shape matches pos_triples."""
    neg = pos_triples.clone()
    if corrupt_tail:
        neg[:, 2] = torch.randint(0, num_entities, (pos_triples.size(0),), device=pos_triples.device)
    else:
        neg[:, 0] = torch.randint(0, num_entities, (pos_triples.size(0),), device=pos_triples.device)
    return neg

def batches(tensor, batch_size, shuffle=True):
    N = tensor.size(0)
    idx = torch.randperm(N, device=tensor.device) if shuffle else torch.arange(N, device=tensor.device)
    for i in range(0, N, batch_size):
        part = tensor[idx[i:i+batch_size]]
        yield part


def sample_negatives_both(pos_triples, num_entities, k_neg=20):
    B = pos_triples.size(0)
    device = pos_triples.device

    tails = torch.randint(0, num_entities, (B, k_neg), device=device)
    neg_tail = pos_triples.unsqueeze(1).repeat(1, k_neg, 1)
    neg_tail[:, :, 2] = tails

    heads = torch.randint(0, num_entities, (B, k_neg), device=device)
    neg_head = pos_triples.unsqueeze(1).repeat(1, k_neg, 1)
    neg_head[:, :, 0] = heads

    return neg_head.view(-1, 3), neg_tail.view(-1, 3)

def train_one_epoch(model, triples, optimizer, batch_size=2048, k_neg=20):
    model.train()
    pos_weight = torch.tensor([(2.0 * k_neg)], device=triples.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_loss, seen = 0.0, 0
    for pos in batches(triples, batch_size, shuffle=True):
        neg_h, neg_t = sample_negatives_both(pos, num_entities, k_neg=k_neg)

        all_triples = torch.cat([pos, neg_h, neg_t], dim=0)
        labels = torch.cat([
            torch.ones(len(pos), device=triples.device),
            torch.zeros(len(neg_h) + len(neg_t), device=triples.device)
        ])

        scores = model(all_triples)
        loss = bce(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        seen += labels.numel()

    return total_loss / seen

@torch.no_grad()
def evaluate_auc_hits(model, triples, batch_size=4096, hits_k_list=[1, 5, 10]):
    """Quick sanity metrics (UNFILTERED):
       - ROC-AUC on positive vs randomly corrupted negatives
       - Hits@k on head/tail corruption with 100 negatives per positive
       
    Args:
        hits_k_list: List of k values for Hits@k metric (default: [1, 5, 10])
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    scores_all, labels_all = [], []

    for pos in batches(triples, batch_size, shuffle=False):
        neg = sample_negatives(pos, num_entities, corrupt_tail=True)
        s_pos = model(pos)
        s_neg = model(neg)
        scores_all.append(torch.cat([s_pos, s_neg], 0).cpu().numpy())
        labels_all.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))], 0))
    scores_all = np.concatenate(scores_all, 0)
    labels_all = np.concatenate(labels_all, 0)
    auc = roc_auc_score(labels_all, scores_all)

    hits_dict = {k: 0 for k in hits_k_list}
    n_trials = 0
    
    for pos in batches(triples, batch_size, shuffle=False):
        B = pos.size(0)
        true_t = pos[:, 2]
        rand_t = torch.randint(0, num_entities, (B, 99), device=pos.device)
        tails = torch.cat([true_t.unsqueeze(1), rand_t], dim=1)

        ent = model.encoder(model.edge_index)
        e_h = ent[pos[:, 0]]
        e_candidates = ent[tails]
        s = (e_h.unsqueeze(1) * e_candidates).sum(dim=2)
        ranks = (s.argsort(dim=1, descending=True) == 0).nonzero()[:,1] + 1
        
        for k in hits_k_list:
            hits_dict[k] += (ranks <= k).sum().item()
        n_trials += B
    
    result = {"auc": float(auc)}
    for k in hits_k_list:
        result[f"hits@{k}"] = hits_dict[k] / n_trials
    
    return result

model = LinkPredictor(num_entities, edge_index, out_dim=200,
                      emb_dim=128, hidden_dim=128, heads=4, dropout=0.3).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Metrics tracking
history = {
    'train_loss': [],
    'val_auc': [],
    'val_hits@1': [],
    'val_hits@5': [],
    'val_hits@10': []
}

# Early stopping parameters
best_val_auc = 0.0
patience = 10
patience_counter = 0
best_model_state = None

EPOCHS = 100
for epoch in tqdm(range(1, EPOCHS+1), desc="epoch:"):
    loss = train_one_epoch(model, train_triples, opt, batch_size=2048)
    metrics = evaluate_auc_hits(model, valid_triples, batch_size=4096, hits_k_list=[1, 5, 10])
    
    history['train_loss'].append(loss)
    history['val_auc'].append(metrics['auc'])
    history['val_hits@1'].append(metrics['hits@1'])
    history['val_hits@5'].append(metrics['hits@5'])
    history['val_hits@10'].append(metrics['hits@10'])
    
    print(f"Epoch {epoch:02d} | loss={loss:.4f} | AUC={metrics['auc']:.4f} | "
          f"H@1={metrics['hits@1']:.4f} | H@5={metrics['hits@5']:.4f} | H@10={metrics['hits@10']:.4f}")
    
    # Early stopping check
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

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nRestored best model with validation AUC: {best_val_auc:.4f}")

# Evaluate on test set
print("\n" + "="*70)
print("FINAL TEST SET EVALUATION")
print("="*70)
test_metrics = evaluate_auc_hits(model, test_triples, batch_size=4096, hits_k_list=[1, 5, 10])
print(f"Test AUC:    {test_metrics['auc']:.4f}")
print(f"Test Hits@1: {test_metrics['hits@1']:.4f}")
print(f"Test Hits@5: {test_metrics['hits@5']:.4f}")
print(f"Test Hits@10: {test_metrics['hits@10']:.4f}")
print("="*70 + "\n")

# Save the best model
model_save_path = "best_gat_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'best_val_auc': best_val_auc,
    'test_metrics': test_metrics,
    'history': history
}, model_save_path)
print(f"Best model saved to {model_save_path}")

# Save training history to text file
history_file = "training_history.txt"
with open(history_file, 'w') as f:
    f.write("GAT Training History\n")
    f.write("=" * 80 + "\n\n")
    
    # Validation metrics
    f.write("VALIDATION METRICS (used for model selection)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Best Validation AUC: {best_val_auc:.4f}\n")
    f.write(f"Total Epochs Trained: {len(history['train_loss'])}\n\n")
    
    # Test metrics
    f.write("TEST METRICS (final unbiased estimate)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Test AUC:    {test_metrics['auc']:.4f}\n")
    f.write(f"Test Hits@1: {test_metrics['hits@1']:.4f}\n")
    f.write(f"Test Hits@5: {test_metrics['hits@5']:.4f}\n")
    f.write(f"Test Hits@10: {test_metrics['hits@10']:.4f}\n\n")
    
    # Training history
    f.write("TRAINING HISTORY\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Val AUC':<12} {'H@1':<12} {'H@5':<12} {'H@10':<12}\n")
    f.write("-" * 80 + "\n")
    for i in range(len(history['train_loss'])):
        f.write(f"{i+1:<8} {history['train_loss'][i]:<12.4f} "
                f"{history['val_auc'][i]:<12.4f} {history['val_hits@1'][i]:<12.4f} "
                f"{history['val_hits@5'][i]:<12.4f} {history['val_hits@10'][i]:<12.4f}\n")
print(f"Training history saved to {history_file}")