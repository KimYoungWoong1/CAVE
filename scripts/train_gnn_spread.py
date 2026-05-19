"""GNN 전파 위험도 모델 학습.

기본 학습 모드는 CAVE 피해 산정 입력에 맞춘 synthetic spread-risk graph다.
UPFD (GossipCop, Dou et al. ACL 2021)는 실제 소셜미디어 전파 구조
벤치마크로 별도 실행할 수 있다.

학습된 모델: models/gnn_spread_model.pt

실행:
  python scripts/train_gnn_spread.py
  python scripts/train_gnn_spread.py --epochs 80
  python scripts/train_gnn_spread.py --upfd  # UPFD 벤치마크 학습
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layers.gnn_spread_model import (  # noqa: E402
    MODEL_META_PATH,
    MODEL_PATH,
    NODE_FEATURE_DIM,
    SpreadRiskGCN,
    extract_structural_features,
)

DATA_DIR = ROOT / "data" / "upfd"


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN 전파 위험도 모델 학습")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--synthetic", action="store_true",
                        help="legacy alias: synthetic spread-risk graph 사용")
    parser.add_argument("--upfd", action="store_true",
                        help="UPFD GossipCop 실제 전파 그래프 벤치마크로 학습")
    parser.add_argument("--samples", type=int, default=1500,
                        help="합성 모드: 클래스당 그래프 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    device = _select_device(args.device)
    print(f"device: {device}")

    data_source = "upfd" if args.upfd else "synthetic_risk"

    if data_source == "upfd":
        if not _upfd_available():
            raise SystemExit("UPFD raw 데이터가 부족합니다. 기본 synthetic 모드로 실행하거나 데이터를 준비하세요.")
        train_loader, val_loader, test_loader = _load_upfd(args.batch_size)
        print("데이터: UPFD GossipCop (실제 소셜미디어 전파 그래프, Dou et al. ACL 2021)")
    else:
        train_loader, val_loader, test_loader = _load_synthetic(
            args.samples, args.seed, args.batch_size
        )
        print(f"데이터: 합성 spread-risk 그래프 (CAVE 피해 산정용, 클래스당 {args.samples}개)")

    model = SpreadRiskGCN(hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        val_acc = _evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss={loss:.4f}  val_acc={val_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    train_acc = _evaluate(model, train_loader, device)
    val_acc = _evaluate(model, val_loader, device)
    test_acc = _evaluate(model, test_loader, device)
    print(
        f"\nbest_val_acc={best_val_acc:.3f}  "
        f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  test_acc={test_acc:.3f}"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")
    _write_metadata(
        data_source=data_source,
        args=args,
        train_acc=train_acc,
        val_acc=val_acc,
        best_val_acc=best_val_acc,
        test_acc=test_acc,
    )

    _print_inference_demo()


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────

def _upfd_available() -> bool:
    raw_dir = DATA_DIR / "gossipcop" / "raw"
    required = ["node_graph_id.npy", "graph_labels.npy", "A.txt",
                "train_idx.npy", "val_idx.npy", "test_idx.npy",
                "new_profile_feature.npz"]
    return all((raw_dir / f).exists() for f in required)


def _load_upfd(batch_size: int):
    from torch_geometric.datasets import UPFD

    splits = {}
    for split in ("train", "val", "test"):
        raw = UPFD(str(DATA_DIR), "gossipcop", "profile", split)
        processed = []
        for data in raw:
            data.x = extract_structural_features(data)
            processed.append(data)
        splits[split] = processed

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    n_test  = len(splits["test"])
    n_fake  = sum(int(d.y.item()) for d in splits["train"] + splits["val"] + splits["test"])
    n_total = n_train + n_val + n_test
    print(f"샘플 — 학습: {n_train}, 검증: {n_val}, 테스트: {n_test}  "
          f"(전체 {n_total}개, 가짜: {n_fake}, 진짜: {n_total - n_fake})")

    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(splits["val"],   batch_size=batch_size)
    test_loader  = DataLoader(splits["test"],  batch_size=batch_size)
    return train_loader, val_loader, test_loader


def _load_synthetic(samples_per_class: int, seed: int, batch_size: int):
    dataset = _build_synthetic_dataset(samples_per_class, seed)
    random.Random(seed).shuffle(dataset)

    n = len(dataset)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_list = dataset[:n_train]
    val_list   = dataset[n_train:n_train + n_val]
    test_list  = dataset[n_train + n_val:]

    n_fake = sum(int(d.y.item()) for d in dataset)
    print(f"샘플 — 학습: {len(train_list)}, 검증: {len(val_list)}, 테스트: {len(test_list)}  "
          f"(고위험: {n_fake}, 저위험: {n - n_fake})")

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_list,   batch_size=batch_size)
    test_loader  = DataLoader(test_list,  batch_size=batch_size)
    return train_loader, val_loader, test_loader


# ── 합성 데이터 생성 ──────────────────────────────────────────────────────────

def _build_synthetic_dataset(samples_per_class: int, seed: int = 42) -> list[Data]:
    """Bi-GCN 논문 Table 1 실측 통계 기반 합성 전파 그래프."""
    rng = random.Random(seed)
    dataset: list[Data] = []

    for _ in range(samples_per_class):
        num_nodes   = rng.randint(60, 250)
        extra_ratio = rng.uniform(0.08, 0.25)
        dataset.append(_make_ba_graph(num_nodes, extra_ratio, label=1, rng=rng))

    for _ in range(samples_per_class):
        num_nodes   = rng.randint(4, 45)
        extra_ratio = rng.uniform(0.0, 0.05)
        star_prob   = rng.uniform(0.3, 0.7)
        dataset.append(_make_low_risk_graph(num_nodes, extra_ratio, star_prob, label=0, rng=rng))

    return dataset


def _make_ba_graph(num_nodes: int, extra_ratio: float, label: int, rng: random.Random) -> Data:
    edges_src: list[int] = []
    edges_dst: list[int] = []
    deg = [1] + [0] * (num_nodes - 1)

    for new_node in range(1, num_nodes):
        total = sum(deg[:new_node])
        r = rng.random() * total
        cumsum, parent = 0.0, 0
        for j in range(new_node):
            cumsum += deg[j]
            if r <= cumsum:
                parent = j
                break
        edges_src += [parent, new_node]
        edges_dst += [new_node, parent]
        deg[parent] += 1
        deg[new_node] += 1

    for _ in range(int(num_nodes * extra_ratio)):
        u, v = rng.randint(0, num_nodes - 1), rng.randint(0, num_nodes - 1)
        if u != v:
            edges_src += [u, v]
            edges_dst += [v, u]

    num_nodes = _append_redistribution_motifs(
        edges_src,
        edges_dst,
        num_nodes,
        has_variants=rng.random() < 0.75,
        on_closed_platforms=rng.random() < 0.65,
        reappeared_after_deletion=rng.random() < 0.45,
        rng=rng,
    )

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=num_nodes,
                y=torch.tensor(label, dtype=torch.long))
    data.x = extract_structural_features(data)
    return data


def _make_low_risk_graph(
    num_nodes: int, extra_ratio: float, star_prob: float, label: int, rng: random.Random
) -> Data:
    edges_src: list[int] = []
    edges_dst: list[int] = []

    for node in range(1, num_nodes):
        parent = 0 if rng.random() < star_prob else rng.randint(0, node - 1)
        edges_src += [parent, node]
        edges_dst += [node, parent]

    for _ in range(int(num_nodes * extra_ratio)):
        u, v = rng.randint(0, num_nodes - 1), rng.randint(0, num_nodes - 1)
        if u != v:
            edges_src += [u, v]
            edges_dst += [v, u]

    num_nodes = _append_redistribution_motifs(
        edges_src,
        edges_dst,
        num_nodes,
        has_variants=rng.random() < 0.08,
        on_closed_platforms=rng.random() < 0.05,
        reappeared_after_deletion=rng.random() < 0.03,
        rng=rng,
    )

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=num_nodes,
                y=torch.tensor(label, dtype=torch.long))
    data.x = extract_structural_features(data)
    return data


def _append_redistribution_motifs(
    edges_src: list[int],
    edges_dst: list[int],
    num_nodes: int,
    *,
    has_variants: bool,
    on_closed_platforms: bool,
    reappeared_after_deletion: bool,
    rng: random.Random,
) -> int:
    next_node = num_nodes

    if has_variants:
        variant_hub = next_node
        next_node += 1
        _connect(edges_src, edges_dst, 0, variant_hub)
        for _ in range(2):
            leaf = next_node
            next_node += 1
            _connect(edges_src, edges_dst, variant_hub, leaf)

    if on_closed_platforms:
        closed_hub = next_node
        next_node += 1
        _connect(edges_src, edges_dst, 0, closed_hub)
        fanout = min(4, max(1, num_nodes // 20))
        for _ in range(fanout):
            target = rng.randint(1, max(num_nodes - 1, 1))
            _connect(edges_src, edges_dst, closed_hub, target)

    if reappeared_after_deletion:
        parent = rng.randint(1, max(num_nodes - 1, 1))
        first = next_node
        second = next_node + 1
        next_node += 2
        _connect(edges_src, edges_dst, parent, first)
        _connect(edges_src, edges_dst, first, second)

    return next_node


def _connect(edges_src: list[int], edges_dst: list[int], source: int, target: int) -> None:
    edges_src += [source, target]
    edges_dst += [target, source]


# ── 학습 유틸 ─────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.binary_cross_entropy(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = (out >= 0.5).long()
            correct += int((pred == batch.y).sum())
            total += batch.y.size(0)
    return correct / total if total > 0 else 0.0


def _select_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda")
    if preference == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _write_metadata(
    data_source: str,
    args: argparse.Namespace,
    train_acc: float,
    val_acc: float,
    best_val_acc: float,
    test_acc: float,
) -> None:
    meta = {
        "task": "spread_risk_prediction",
        "data_source": data_source,
        "model_class": "SpreadRiskGCN",
        "feature_order": [
            "depth_norm",
            "degree_log_norm",
            "is_leaf",
            "centrality_norm",
        ],
        "node_feature_dim": NODE_FEATURE_DIM,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "samples_per_class": args.samples if data_source == "synthetic_risk" else None,
        "seed": args.seed,
        "train_accuracy": round(float(train_acc), 6),
        "val_accuracy": round(float(val_acc), 6),
        "best_val_accuracy": round(float(best_val_acc), 6),
        "test_accuracy": round(float(test_acc), 6),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Synthetic spread-risk GCN for CAVE DamageInputs approximate propagation graphs with redistribution motifs."
            if data_source == "synthetic_risk"
            else "UPFD GossipCop structural benchmark; not the default CAVE damage input distribution."
        ),
    }
    MODEL_META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"메타데이터 저장: {MODEL_META_PATH}")


def _print_inference_demo() -> None:
    from layers.damage_score import DamageInputs
    from layers.gnn_spread_model import predict_spread_risk

    examples = [
        ("딥페이크 성범죄 (고위험)", DamageInputs.example_deepfake_sexual()),
        ("금융사기 (중위험)",        DamageInputs.example_financial_fraud()),
        ("단순 배포 (저위험)",       DamageInputs(num_posts=3, num_shares=20, num_views=500)),
    ]
    print("\n── 추론 데모 ────────────────────────────────────")
    for label, inp in examples:
        score = predict_spread_risk(inp)
        bar = "█" * int((score or 0) * 20)
        print(f"{label:22s}  spread_risk={score:.3f}  {bar}")


if __name__ == "__main__":
    main()
