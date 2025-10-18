import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader



# Helpers
def _read_triples(path: Path) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            h, r, t = row
            triples.append((h, r, t))
    return triples


def _build_id_maps(
        train_p: Path,
        valid_p: Optional[Path] = None,
        test_p: Optional[Path] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build ent2id and rel2id from all splits available."""
    ents, rels = set(), set()
    for p in [train_p, valid_p, test_p]:
        if p is None:
            continue
        for h, r, t in _read_triples(p):
            ents.add(h); ents.add(t); rels.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(ents))}
    rel2id = {r: i for i, r in enumerate(sorted(rels))}
    return ent2id, rel2id


# -----------------------
# Datasets
# -----------------------
class _PairsDataset(Dataset):
    """Untyped pairs (h, t), label=1 for each positive edge."""
    def __init__(self, pairs: torch.LongTensor):
        # pairs: [N,2]
        self.pairs = pairs
        self.labels = torch.ones(pairs.size(0), dtype=torch.float32)

    def __len__(self):
        return self.pairs.size(0)

    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


class _TriplesDataset(Dataset):
    """Typed triples (h, r, t), label=1 for each positive triple."""
    def __init__(self, triples: torch.LongTensor):
        # triples: [N,3]
        self.triples = triples
        self.labels = torch.ones(triples.size(0), dtype=torch.float32)

    def __len__(self):
        return self.triples.size(0)

    def __getitem__(self, idx):
        return self.triples[idx], self.labels[idx]


# ============================================================
# Collapsed edge types (untyped) datamodule
# ============================================================
class KGDataModuleCollapsed:
    """
    Produces untyped (h, t) pairs from KG triples.

    Public methods:
      - train_loader()
      - val_loader()
      - test_loader()

    Args:
        train_path, valid_path, test_path: Path to split files (WN18RR format).
        batch_size: int
        shuffle: bool (applied to train loader only)
        num_workers: int (DataLoader)
        add_reverse: if True, also add (t, h) for every (h, t)
    """
    def __init__(
            self,
            train_path: Path,
            valid_path: Optional[Path] = None,
            test_path: Optional[Path] = None,
            batch_size: int = 2048,
            shuffle: bool = True,
            num_workers: int = 0,
            add_reverse: bool = True,
    ):
        self.train_path = Path(train_path)
        self.valid_path = Path(valid_path) if valid_path else None
        self.test_path  = Path(test_path)  if test_path  else None

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.add_reverse = add_reverse

        # Build ids
        self.ent2id, self.rel2id = _build_id_maps(self.train_path, self.valid_path, self.test_path)

        # Build tensors per split
        self._train_pairs = self._make_pairs(self.train_path)
        self._valid_pairs = self._make_pairs(self.valid_path) if self.valid_path else None
        self._test_pairs  = self._make_pairs(self.test_path)  if self.test_path  else None

        # Datasets
        self._train_ds = _PairsDataset(self._train_pairs)
        self._valid_ds = _PairsDataset(self._valid_pairs) if self._valid_pairs is not None else None
        self._test_ds  = _PairsDataset(self._test_pairs)  if self._test_pairs  is not None else None

    def _make_pairs(self, path: Path) -> torch.LongTensor:
        pairs = []
        for h, _, t in _read_triples(path):
            h_id, t_id = self.ent2id[h], self.ent2id[t]
            pairs.append((h_id, t_id))
            if self.add_reverse:
                pairs.append((t_id, h_id))
        if not pairs:
            return torch.empty(0, 2, dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long)

    # -------- public loaders --------
    def train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=False
        )

    def val_loader(self) -> Optional[DataLoader]:
        if self._valid_ds is None:
            return None
        return DataLoader(
            self._valid_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=False
        )

    def test_loader(self) -> Optional[DataLoader]:
        if self._test_ds is None:
            return None
        return DataLoader(
            self._test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=False
        )


# ============================================================
# Typed (with edge types) datamodule
# ============================================================
class KGDataModuleTyped:
    """
    Produces typed (h, r, t) triples from KG files.

    Public methods:
      - train_loader()
      - val_loader()
      - test_loader()

    Args:
        train_path, valid_path, test_path: Path to split files.
        batch_size, shuffle, num_workers: DataLoader args.
        add_reverse: If True, add reverse links.
        reverse_relation_strategy:
            'duplicate_rel' -> create an inverse relation id per r (e.g., r#inv)
            'same_rel'      -> reuse the same r id for reverse triple
    """
    def __init__(
            self,
            train_path: Path,
            valid_path: Optional[Path] = None,
            test_path: Optional[Path] = None,
            batch_size: int = 2048,
            shuffle: bool = True,
            num_workers: int = 0,
            add_reverse: bool = True,
            reverse_relation_strategy: str = "duplicate_rel",
    ):
        assert reverse_relation_strategy in ("duplicate_rel", "same_rel")
        self.train_path = Path(train_path)
        self.valid_path = Path(valid_path) if valid_path else None
        self.test_path  = Path(test_path)  if test_path  else None

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.add_reverse = add_reverse
        self.reverse_relation_strategy = reverse_relation_strategy

        # Base id maps from original relations
        self.ent2id, base_rel2id = _build_id_maps(self.train_path, self.valid_path, self.test_path)

        # Possibly extend rel2id with inverse relations
        if add_reverse and reverse_relation_strategy == "duplicate_rel":
            # Make space for inverse ids
            self.rel2id = dict(base_rel2id)
            self.inv_of = {}  # map original rel -> inverse rel id
            next_id = max(self.rel2id.values(), default=-1) + 1
            # Ensure deterministic order
            for r in sorted(base_rel2id.keys()):
                inv_name = r + "_rev"
                if inv_name not in self.rel2id:
                    self.rel2id[inv_name] = next_id
                    self.inv_of[r] = next_id
                    next_id += 1
        else:
            self.rel2id = base_rel2id
            self.inv_of = None  # not used

        # Build tensors per split
        self._train_triples = self._make_triples(self.train_path)
        self._valid_triples = self._make_triples(self.valid_path) if self.valid_path else None
        self._test_triples  = self._make_triples(self.test_path)  if self.test_path  else None

        # Datasets
        self._train_ds = _TriplesDataset(self._train_triples)
        self._valid_ds = _TriplesDataset(self._valid_triples) if self._valid_triples is not None else None
        self._test_ds  = _TriplesDataset(self._test_triples)  if self._test_triples  is not None else None

    def _make_triples(self, path: Path) -> torch.LongTensor:
        if path is None:
            return torch.empty(0, 3, dtype=torch.long)

        rows = []
        for h, r, t in _read_triples(path):
            h_id, t_id = self.ent2id[h], self.ent2id[t]
            r_id = self.rel2id[r]  # original direction
            rows.append((h_id, r_id, t_id))

            if self.add_reverse:
                if self.reverse_relation_strategy == "duplicate_rel":
                    r_inv_id = self.inv_of[r]  # created in __init__
                    rows.append((t_id, r_inv_id, h_id))
                else:  # same_rel
                    rows.append((t_id, r_id, h_id))

        if not rows:
            return torch.empty(0, 3, dtype=torch.long)
        return torch.tensor(rows, dtype=torch.long)

    # -------- public loaders --------
    def train_loader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, shuffle=self.shuffle,
            num_workers=self.num_workers, pin_memory=False
        )

    def val_loader(self) -> Optional[DataLoader]:
        if self._valid_ds is None:
            return None
        return DataLoader(
            self._valid_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=False
        )

    def test_loader(self) -> Optional[DataLoader]:
        if self._test_ds is None:
            return None
        return DataLoader(
            self._test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=False
        )


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    train_p = Path("../WN18RR/train.txt")
    valid_p = Path("../WN18RR/valid.txt")
    test_p  = Path("../WN18RR/test.txt")

    # Collapsed (untyped) pairs
    dm_pairs = KGDataModuleCollapsed(train_p, valid_p, test_p, batch_size=1024, add_reverse=True)
    train_loader = dm_pairs.train_loader()
    X, y = next(iter(train_loader))  # X: [B,2], y: [B]
    print("Pairs batch:", X.shape, y.shape)

    # Typed triples with inverse relations
    dm_typed = KGDataModuleTyped(
        train_p, valid_p, test_p,
        batch_size=1024,
        add_reverse=True,
        reverse_relation_strategy="duplicate_rel",  # or "same_rel"
    )
    train_loader2 = dm_typed.train_loader()
    X2, y2 = next(iter(train_loader2))  # X2: [B,3], y2: [B]
    print("Triples batch:", X2.shape, y2.shape)