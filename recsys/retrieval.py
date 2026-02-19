from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.history_embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)
        self.user_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.item_proj = nn.Linear(embedding_dim, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.history_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        for layer in self.user_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.item_proj.weight)
        nn.init.zeros_(self.item_proj.bias)

    def encode_user(self, user_ids: torch.Tensor, history_items: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_embedding(user_ids)
        hist_vec = self.history_embedding(history_items)
        mask = history_mask.unsqueeze(-1)
        masked = hist_vec * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        hist_mean = masked.sum(dim=1) / denom

        combined = torch.cat([user_vec, hist_mean], dim=-1)
        return F.normalize(self.user_proj(combined), dim=-1)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_vec = self.item_embedding(item_ids)
        return F.normalize(self.item_proj(item_vec), dim=-1)


@dataclass
class RetrievalTrainConfig:
    embedding_dim: int = 64
    epochs: int = 5
    batch_size: int = 512
    num_negatives: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    max_history_len: int = 50
    steps_per_epoch: int = 500
    seed: int = 42


@dataclass
class RetrievalTrainEvent:
    user_idx: int
    history_items: List[int]
    target_item: int


def build_user_history(
    interactions,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
    ts_col: str = "timestamp",
) -> Dict[int, List[int]]:
    if interactions.empty:
        return {}
    ordered = interactions.sort_values([user_col, ts_col])
    grouped = ordered.groupby(user_col)[item_col].agg(list)
    return {int(user): [int(item) for item in items] for user, items in grouped.items()}


def build_retrieval_train_events(
    interactions,
    min_history: int = 1,
    max_events_per_user: int = 200,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
    ts_col: str = "timestamp",
) -> List[RetrievalTrainEvent]:
    if interactions.empty:
        return []

    events: List[RetrievalTrainEvent] = []
    ordered = interactions.sort_values([user_col, ts_col])
    for user_idx, group in ordered.groupby(user_col):
        user = int(user_idx)
        items = [int(v) for v in group[item_col].tolist()]
        if len(items) <= min_history:
            continue

        positions = list(range(min_history, len(items)))
        if max_events_per_user > 0 and len(positions) > max_events_per_user:
            # Bias toward recent behavior by keeping the latest training events.
            positions = positions[-max_events_per_user:]

        for pos in positions:
            history = items[:pos]
            target = items[pos]
            events.append(
                RetrievalTrainEvent(
                    user_idx=user,
                    history_items=history,
                    target_item=target,
                )
            )
    return events


def _pad_histories(histories: Sequence[Sequence[int]], max_history_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(histories)
    arr = torch.zeros((batch_size, max_history_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_history_len), dtype=torch.float32)
    for idx, history in enumerate(histories):
        tail = list(history)[-max_history_len:]
        if not tail:
            continue
        length = len(tail)
        arr[idx, :length] = torch.tensor(tail, dtype=torch.long)
        mask[idx, :length] = 1.0
    return arr, mask


def _sample_event_batch(
    rng: np.random.Generator,
    events: Sequence[RetrievalTrainEvent],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    event_indices = rng.integers(0, len(events), size=batch_size, endpoint=False)
    sampled_users = np.zeros(batch_size, dtype=np.int64)
    pos_items = np.zeros(batch_size, dtype=np.int64)
    histories: List[List[int]] = []

    for i, event_idx in enumerate(event_indices.tolist()):
        event = events[int(event_idx)]
        sampled_users[i] = int(event.user_idx)
        pos_items[i] = int(event.target_item)
        histories.append(event.history_items)

    return sampled_users, pos_items, histories


def _sample_negatives(
    rng: np.random.Generator,
    batch_users: np.ndarray,
    user_seen_sets: Dict[int, set[int]],
    num_items: int,
    num_negatives: int,
    item_sampling_probs: np.ndarray | None = None,
) -> np.ndarray:
    if item_sampling_probs is None:
        negatives = rng.integers(0, num_items, size=(batch_users.shape[0], num_negatives), endpoint=False, dtype=np.int64)
    else:
        sampled = rng.choice(num_items, size=(batch_users.shape[0], num_negatives), replace=True, p=item_sampling_probs)
        negatives = sampled.astype(np.int64)
    for idx, user in enumerate(batch_users.tolist()):
        seen = user_seen_sets.get(int(user), set())
        for j in range(num_negatives):
            if negatives[idx, j] in seen:
                if item_sampling_probs is None:
                    candidate = int(rng.integers(0, num_items, endpoint=False))
                else:
                    candidate = int(rng.choice(num_items, p=item_sampling_probs))
                retries = 0
                while candidate in seen and retries < 50:
                    if item_sampling_probs is None:
                        candidate = int(rng.integers(0, num_items, endpoint=False))
                    else:
                        candidate = int(rng.choice(num_items, p=item_sampling_probs))
                    retries += 1
                negatives[idx, j] = candidate
    return negatives


def _build_negative_item_probs(user_seen_history: Dict[int, List[int]], num_items: int) -> np.ndarray:
    counts = np.ones(num_items, dtype=np.float64)
    for items in user_seen_history.values():
        for item in items:
            idx = int(item)
            if 0 <= idx < num_items:
                counts[idx] += 1.0
    scaled = np.power(counts, 0.75)
    scaled_sum = float(scaled.sum())
    if scaled_sum <= 0:
        return np.full(num_items, 1.0 / num_items, dtype=np.float64)
    return scaled / scaled_sum


def train_two_tower(
    train_events: Sequence[RetrievalTrainEvent],
    user_seen_history: Dict[int, List[int]],
    num_users: int,
    num_items: int,
    config: RetrievalTrainConfig,
    device: str = "cpu",
) -> Tuple[TwoTowerModel, Dict[str, float]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    model = TwoTowerModel(num_users=num_users, num_items=num_items, embedding_dim=config.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    if len(train_events) == 0:
        raise ValueError("No retrieval training events found. Ensure train split has enough history.")
    user_seen_sets = {int(u): set(int(i) for i in items) for u, items in user_seen_history.items()}
    item_sampling_probs = _build_negative_item_probs(user_seen_history, num_items=num_items)

    losses: List[float] = []
    steps_per_epoch = config.steps_per_epoch
    if steps_per_epoch <= 0:
        steps_per_epoch = max(1, int(math.ceil(len(train_events) / config.batch_size)))

    for epoch in range(config.epochs):
        epoch_losses: List[float] = []
        model.train()

        for _ in range(steps_per_epoch):
            batch_users, pos_items, histories = _sample_event_batch(
                rng=rng,
                events=train_events,
                batch_size=config.batch_size,
            )
            neg_items = _sample_negatives(
                rng=rng,
                batch_users=batch_users,
                user_seen_sets=user_seen_sets,
                num_items=num_items,
                num_negatives=config.num_negatives,
                item_sampling_probs=item_sampling_probs,
            )

            user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            pos_tensor = torch.tensor(pos_items, dtype=torch.long, device=device)
            neg_tensor = torch.tensor(neg_items, dtype=torch.long, device=device)
            history_tensor, history_mask = _pad_histories(histories, max_history_len=config.max_history_len)
            history_tensor = history_tensor.to(device)
            history_mask = history_mask.to(device)

            user_vec = model.encode_user(user_tensor, history_tensor, history_mask)
            pos_vec = model.encode_item(pos_tensor)
            neg_vec = model.encode_item(neg_tensor.view(-1)).view(config.batch_size, config.num_negatives, -1)

            pos_logits = (user_vec * pos_vec).sum(dim=-1)
            neg_logits = torch.einsum("bd,bnd->bn", user_vec, neg_vec)
            # Pairwise ranking objective to push positives above sampled negatives.
            diff = pos_logits.unsqueeze(1) - neg_logits
            loss = -F.logsigmoid(diff).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"[retrieval] epoch={epoch + 1}/{config.epochs} loss={mean_loss:.4f}")

    return model, {
        "train_loss": float(np.mean(losses)),
        "last_epoch_loss": float(losses[-1]),
        "num_train_events": int(len(train_events)),
        "steps_per_epoch": int(steps_per_epoch),
    }


def compute_item_embeddings(model: TwoTowerModel, num_items: int, batch_size: int = 4096, device: str = "cpu") -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            item_ids = torch.arange(start, end, dtype=torch.long, device=device)
            vec = model.encode_item(item_ids)
            embeddings.append(vec.cpu().numpy().astype(np.float32))
    return np.vstack(embeddings)


def compute_user_embedding(
    model: TwoTowerModel,
    user_idx: int,
    history: Sequence[int],
    max_history_len: int,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
        history_tensor, history_mask = _pad_histories([list(history)], max_history_len=max_history_len)
        history_tensor = history_tensor.to(device)
        history_mask = history_mask.to(device)
        vec = model.encode_user(user_tensor, history_tensor, history_mask)
    return vec.squeeze(0).cpu().numpy().astype(np.float32)


def compute_user_embeddings_batch(
    model: TwoTowerModel,
    user_histories: Mapping[int, Sequence[int]],
    max_history_len: int,
    batch_size: int = 1024,
    device: str = "cpu",
) -> Dict[int, np.ndarray]:
    model.eval()
    users = [int(u) for u in user_histories.keys()]
    out: Dict[int, np.ndarray] = {}
    if not users:
        return out

    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            batch_users = users[start : start + batch_size]
            histories = [list(user_histories[u]) for u in batch_users]
            history_tensor, history_mask = _pad_histories(histories, max_history_len=max_history_len)
            user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            history_tensor = history_tensor.to(device)
            history_mask = history_mask.to(device)
            vecs = model.encode_user(user_tensor, history_tensor, history_mask).cpu().numpy().astype(np.float32)
            for user, vec in zip(batch_users, vecs):
                out[int(user)] = vec
    return out


class NumpyIndex:
    def __init__(self, item_vectors: np.ndarray) -> None:
        self.item_vectors = item_vectors.astype(np.float32)

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.item_vectors @ query.astype(np.float32)
        if top_k >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, top_k)[:top_k]
            idx = idx[np.argsort(-scores[idx])]
        return scores[idx], idx


class CandidateIndex:
    def __init__(self, item_vectors: np.ndarray, faiss_index=None) -> None:
        self.item_vectors = item_vectors.astype(np.float32)
        self.faiss_index = faiss_index
        self.numpy_index = NumpyIndex(item_vectors)

    @classmethod
    def from_item_vectors(cls, item_vectors: np.ndarray) -> "CandidateIndex":
        normalized = item_vectors / np.clip(np.linalg.norm(item_vectors, axis=1, keepdims=True), a_min=1e-12, a_max=None)
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(normalized.shape[1])
            index.add(normalized.astype(np.float32))
            return cls(item_vectors=normalized, faiss_index=index)
        return cls(item_vectors=normalized, faiss_index=None)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.item_vectors)
        metadata = {
            "use_faiss": bool(self.faiss_index is not None and FAISS_AVAILABLE),
            "num_items": int(self.item_vectors.shape[0]),
            "dim": int(self.item_vectors.shape[1]),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        if metadata["use_faiss"]:
            faiss.write_index(self.faiss_index, str(path.with_suffix(".faiss")))

    @classmethod
    def load(cls, path: Path) -> "CandidateIndex":
        vectors = np.load(path.with_suffix(".npy"))
        meta = json.loads(path.with_suffix(".json").read_text())
        use_faiss = bool(meta.get("use_faiss", False) and FAISS_AVAILABLE)
        if use_faiss:
            index = faiss.read_index(str(path.with_suffix(".faiss")))
            return cls(item_vectors=vectors, faiss_index=index)
        return cls(item_vectors=vectors, faiss_index=None)

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        query = query_vector.astype(np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        if self.faiss_index is not None:
            scores, ids = self.faiss_index.search(query.reshape(1, -1), top_k)
            return scores[0], ids[0]
        return self.numpy_index.search(query, top_k)


def recommend_from_index(
    query_vector: np.ndarray,
    index: CandidateIndex,
    top_k: int,
    seen_items: Iterable[int] | None = None,
) -> Tuple[List[int], List[float]]:
    seen = set(int(i) for i in seen_items) if seen_items is not None else set()
    filtered_items: List[int] = []
    filtered_scores: List[float] = []
    num_items_total = int(index.item_vectors.shape[0])

    raw_top_k = min(num_items_total, max(top_k * 2, top_k + len(seen)))
    while True:
        scores, item_ids = index.search(query_vector, raw_top_k)
        filtered_items.clear()
        filtered_scores.clear()

        for score, item in zip(scores.tolist(), item_ids.tolist()):
            if int(item) < 0:
                continue
            if int(item) in seen:
                continue
            filtered_items.append(int(item))
            filtered_scores.append(float(score))
            if len(filtered_items) >= top_k:
                break

        if len(filtered_items) >= top_k or raw_top_k >= num_items_total:
            break
        raw_top_k = min(num_items_total, raw_top_k * 2)

    return filtered_items, filtered_scores


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_two_tower_model(
    model: TwoTowerModel,
    path_prefix: Path,
    num_users: int,
    num_items: int,
    embedding_dim: int,
    max_history_len: int,
) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path_prefix.with_suffix(".pt"))
    metadata = {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "embedding_dim": int(embedding_dim),
        "max_history_len": int(max_history_len),
    }
    path_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_two_tower_model(path_prefix: Path, device: str = "cpu") -> Tuple[TwoTowerModel, dict]:
    meta = json.loads(path_prefix.with_suffix(".json").read_text())
    model = TwoTowerModel(
        num_users=int(meta["num_users"]),
        num_items=int(meta["num_items"]),
        embedding_dim=int(meta["embedding_dim"]),
    ).to(device)
    state_dict = torch.load(path_prefix.with_suffix(".pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta
