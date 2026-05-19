"""GNN 기반 딥페이크 콘텐츠 전파 위험도 모델.

프로덕션/데모 추론: DamageInputs 메타데이터
  → 합성 전파 위험 그래프(approximate propagation graph)
  → 2-layer GCN
  → 확산 위험도 스코어 (0~1).

UPFD는 실제 소셜미디어 전파 구조 벤치마크로 사용할 수 있지만,
피해 규모 산정 입력은 실제 repost edge가 아니라 메타데이터이므로
기본 체크포인트는 synthetic spread-risk task에 맞춰 학습한다.

설계 원칙:
  - 노드 피처를 텍스트/프로필이 아닌 '구조적 피처'로만 구성
    → 학습 데이터(UPFD)와 추론 시 근사 그래프 간 피처 공간 일치
  - 모델 파일 없으면 None 반환 → damage_score.py가 기존 heuristic으로 fallback
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import degree

if TYPE_CHECKING:
    from layers.damage_score import DamageInputs


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "gnn_spread_model.pt"
MODEL_META_PATH = Path(__file__).resolve().parents[1] / "models" / "gnn_spread_model_meta.json"
NODE_FEATURE_DIM = 4  # [depth_norm, degree_log_norm, is_leaf, centrality_norm]

_model_cache: Optional[nn.Module] = None


# ── 모델 정의 ─────────────────────────────────────────────────────────────────

class SpreadRiskGCN(nn.Module):
    """2-layer GCN for content spread risk prediction.

    graph-level output: sigmoid probability that the propagation graph
    represents high-risk viral spread.
    """

    def __init__(self, in_channels: int = NODE_FEATURE_DIM, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.head(x)).squeeze(-1)


# ── 구조적 피처 추출 ─────────────────────────────────────────────────────────

def extract_structural_features(data: Data) -> torch.Tensor:
    """전파 그래프에서 구조적 노드 피처 추출 (NODE_FEATURE_DIM 차원).

    [depth_norm, degree_log_norm, is_leaf, centrality_norm]
    - UPFD 실제 그래프와 근사 그래프 모두에서 동일하게 계산 가능.
    - 텍스트·프로필 피처 불필요 → 추론 시 피처 공간 일치 보장.
    """
    n = data.num_nodes
    edge_index = data.edge_index

    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros(n, NODE_FEATURE_DIM)

    deg = degree(edge_index[0], num_nodes=n, dtype=torch.float)
    max_deg = float(deg.max().item()) if n > 0 else 1.0

    degree_log_norm = torch.log1p(deg) / math.log1p(max_deg + 1e-8)
    is_leaf = (deg <= 1.0).float()

    depth = _bfs_depth(edge_index, n, root=0)
    max_depth = float(depth.max().item()) or 1.0
    depth_norm = depth / (max_depth + 1e-8)

    total_edges = float(edge_index.size(1)) + 1e-8
    centrality = deg / total_edges
    centrality_norm = centrality / (centrality.max() + 1e-8)

    return torch.stack([depth_norm, degree_log_norm, is_leaf, centrality_norm], dim=1)


def _bfs_depth(edge_index: torch.Tensor, num_nodes: int, root: int = 0) -> torch.Tensor:
    """루트에서 BFS 깊이 계산."""
    depth = torch.zeros(num_nodes)
    if num_nodes <= 1 or edge_index.numel() == 0:
        return depth

    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            adj[s].append(d)
            adj[d].append(s)

    visited = {root}
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                depth[nb] = depth[node] + 1.0
                queue.append(nb)
    return depth


# ── 근사 전파 그래프 생성 ─────────────────────────────────────────────────────

def _compute_virality(inputs: "DamageInputs") -> float:
    """메타데이터에서 전파 바이럴리티 지표 (0~1) 계산.

    - 많은 공유·조회·빠른 확산 → 높은 바이럴리티 → BA 구조
    - 적은 공유·느린 확산 → 낮은 바이럴리티 → 별형(star) 구조
    """
    has_activity = any((
        inputs.num_posts > 0,
        inputs.num_platforms > 0,
        inputs.num_shares > 0,
        inputs.num_views > 0,
    ))
    if not has_activity:
        return 0.0

    post_f  = math.log1p(inputs.num_posts)  / math.log1p(2000)
    share_f = math.log1p(inputs.num_shares) / math.log1p(500_000)
    view_f  = math.log1p(inputs.num_views)  / math.log1p(5_000_000)
    speed_f = (
        max(0.0, 1.0 - inputs.spread_speed_hours / 168.0)
        if inputs.spread_speed_hours > 0 else 0.5
    )
    bool_boost = (
        0.10 * float(inputs.has_variants)
        + 0.10 * float(inputs.on_closed_platforms)
        + 0.05 * float(inputs.reappeared_after_deletion)
    )
    raw = 0.30 * share_f + 0.25 * view_f + 0.20 * post_f + 0.15 * speed_f + 0.10
    return min(float(raw) + bool_boost, 1.0)


def build_approximate_graph(inputs: "DamageInputs") -> Data:
    """소셜미디어 메타데이터로 근사 전파 그래프 생성.

    합성 학습 데이터 분포에 정렬 (train_gnn_spread.py 기준):
    - 저위험: n=4-45 소형 별형/랜덤 (extra_ratio≈0-0.05)
    - 고위험: n=60-250 대형 BA 선호 연결 (extra_ratio≈0.08-0.25)

    바이럴리티 1.5제곱 비선형 스케일링으로 중위험이 전이 영역에 위치:
    - v≈0.38 → n≈36 (저위험 학습 분포 내)
    - v≈0.60 → n≈60 (전이 영역, 경계)
    - v≈0.90 → n≈103 (고위험 학습 분포 내)
    """
    virality = _compute_virality(inputs)

    # 비선형 스케일 (x=1.5): 저위험→n≈35, 중위험→n≈60, 고위험→n≈100
    num_nodes = max(15, int(10 + virality ** 1.5 * 109))

    # 별형 확률: 고위험=0.10 (거의 BA), 저위험=0.90 (거의 별형)
    star_prob = max(0.05, 1.0 - virality)

    rng = random.Random(42)
    edges_src: list[int] = []
    edges_dst: list[int] = []
    deg = [1] + [0] * (num_nodes - 1)

    for new_node in range(1, num_nodes):
        if rng.random() < star_prob:
            parent = 0  # 루트 직결 (별형)
        else:
            # 선호적 연결 (Barabási-Albert)
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

    # 재공유 엣지: 합성 고위험 extra_ratio(0.08-0.25)에 정렬
    extra = int(num_nodes * virality * 0.15)
    for _ in range(extra):
        u = rng.randint(0, num_nodes - 1)
        v_node = rng.randint(0, num_nodes - 1)
        if u != v_node:
            edges_src += [u, v_node]
            edges_dst += [v_node, u]

    num_nodes = _append_redistribution_motifs(
        edges_src,
        edges_dst,
        num_nodes,
        has_variants=bool(inputs.has_variants),
        on_closed_platforms=bool(inputs.on_closed_platforms),
        reappeared_after_deletion=bool(inputs.reappeared_after_deletion),
        rng=rng,
    )

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=num_nodes)


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
    """Add small graph motifs for variants, closed platforms, and reappearance.

    The base graph approximates propagation scale. These motifs encode
    redistribution mechanisms as topology: variants become sibling branches,
    closed platforms become a secondary hub, and reappearance becomes a delayed
    chain from a non-root node.
    """
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


# ── 추론 인터페이스 ───────────────────────────────────────────────────────────

def predict_spread_risk(inputs: "DamageInputs") -> Optional[float]:
    """DamageInputs에서 GNN 기반 전파 위험도 예측 (0~1).

    모델 파일(models/gnn_spread_model.pt)이 없으면 None 반환.
    train_gnn_spread.py 실행 후 사용 가능.
    """
    global _model_cache
    if not MODEL_PATH.exists():
        return None

    if _model_cache is None:
        meta = load_model_metadata() or {}
        hidden = int(meta.get("hidden", 64)) if isinstance(meta.get("hidden", 64), (int, float)) else 64
        model = SpreadRiskGCN(hidden=hidden)
        state = _load_checkpoint_state(MODEL_PATH)
        model.load_state_dict(state)
        model.eval()
        _model_cache = model

    graph = build_approximate_graph(inputs)
    graph.x = extract_structural_features(graph)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)

    with torch.no_grad():
        score = _model_cache(graph.x, graph.edge_index, graph.batch)
    return float(score.item())


def load_model_metadata() -> Optional[dict]:
    """학습 출처/성능 메타데이터를 읽는다."""
    if not MODEL_META_PATH.exists():
        return None
    try:
        return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_checkpoint_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint
