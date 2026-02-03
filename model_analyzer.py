"""
ONNX Model Cut Boundary Analyzer

Usage:
    python model_analyzer.py --onnx original_model/vit_w8a8/model.onnx
    python model_analyzer.py --onnx original_modelvit_w8a8/model.onnx --output my_cuts.txt
"""

import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import deque

import onnx
import onnx_graphsurgeon as gs


# =============================================================================
# Graph Analysis
# =============================================================================

def build_adjacency(graph: gs.Graph) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    succs: Dict[str, Set[str]] = {n.name: set() for n in graph.nodes}
    preds: Dict[str, Set[str]] = {n.name: set() for n in graph.nodes}

    for node in graph.nodes:
        for out_t in node.outputs:
            if out_t is None:
                continue
            for consumer in out_t.outputs:
                if consumer is None or not consumer.name:
                    continue
                succs[node.name].add(consumer.name)
                preds[consumer.name].add(node.name)

    return succs, preds


def get_start_end_nodes(
    graph: gs.Graph,
    preds: Dict[str, Set[str]],
    succs: Dict[str, Set[str]],
) -> Tuple[Set[str], Set[str]]:
    input_tensor_names = {t.name for t in graph.inputs}

    start_nodes = set()
    end_nodes = set()

    for n in graph.nodes:
        if n.op == "Constant":
            continue

        for inp in n.inputs:
            if inp is not None and inp.name in input_tensor_names:
                start_nodes.add(n.name)
                break

        if not succs[n.name]:
            end_nodes.add(n.name)

    return start_nodes, end_nodes


def can_reach(
    starts: Set[str],
    ends: Set[str],
    succs: Dict[str, Set[str]],
    exclude: Set[str] = None,
) -> bool:
    if exclude is None:
        exclude = set()

    visited = set()
    queue = deque([s for s in starts if s not in exclude])

    while queue:
        curr = queue.popleft()
        if curr in visited:
            continue
        visited.add(curr)

        if curr in ends:
            return True

        for nxt in succs.get(curr, []):
            if nxt not in exclude and nxt not in visited:
                queue.append(nxt)

    return False


def topological_sort_names(
    graph: gs.Graph,
    succs: Dict[str, Set[str]],
    preds: Dict[str, Set[str]],
) -> List[str]:
    in_degree = {n.name: len(preds[n.name]) for n in graph.nodes}
    queue = deque([name for name, deg in in_degree.items() if deg == 0])

    result = []
    while queue:
        curr = queue.popleft()
        result.append(curr)

        for nxt in succs.get(curr, []):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return result


# =============================================================================
# Bottleneck Detection & QDQ Grouping
# =============================================================================

def find_bottleneck_nodes(
    graph: gs.Graph,
    succs: Dict[str, Set[str]],
    preds: Dict[str, Set[str]],
) -> List[str]:
    start_nodes, end_nodes = get_start_end_nodes(graph, preds, succs)

    bottlenecks = []
    for node in graph.nodes:
        if node.op == "Constant":
            continue
        if not can_reach(start_nodes, end_nodes, succs, exclude={node.name}):
            bottlenecks.append(node.name)

    topo_order = topological_sort_names(graph, succs, preds)
    topo_index = {name: i for i, name in enumerate(topo_order)}
    bottlenecks.sort(key=lambda x: topo_index.get(x, 0))

    return bottlenecks


def _group_qdq_bottlenecks(
    bottlenecks: List[str],
    succs: Dict[str, Set[str]],
) -> List[Dict]:
    """Group bottleneck q/dq triplets: BaseOp -> _q -> _dq"""
    bottleneck_set = set(bottlenecks)
    consumed = set()
    groups = []

    for bn in bottlenecks:
        if bn in consumed:
            continue

        bn_succs = succs.get(bn, set())
        q_node = None
        dq_node = None

        for s in bn_succs:
            if s in bottleneck_set and s not in consumed and s.endswith("_q"):
                q_candidate = s
                q_succs = succs.get(q_candidate, set())
                for ss in q_succs:
                    if ss in bottleneck_set and ss not in consumed and ss.endswith("_dq"):
                        q_node = q_candidate
                        dq_node = ss
                        break
                if q_node:
                    break

        if q_node and dq_node:
            groups.append({"display_name": bn, "cut_node": q_node})
            consumed.update([bn, q_node, dq_node])
        else:
            groups.append({"display_name": bn, "cut_node": bn})
            consumed.add(bn)

    return groups


# =============================================================================
# Indexing
# =============================================================================

def index_bottleneck_cuts(graph: gs.Graph) -> Tuple[List[Dict], int]:
    """
    Bottleneck cut boundary indexing (1-based).
    Returns (cuts_list, total_node_count).
    """
    succs, preds = build_adjacency(graph)
    bottlenecks = find_bottleneck_nodes(graph, succs, preds)
    start_nodes, _ = get_start_end_nodes(graph, preds, succs)
    groups = _group_qdq_bottlenecks(bottlenecks, succs)

    all_cuts = []
    index = 1

    for i, group in enumerate(groups):
        if i == 0 and (group["cut_node"] in start_nodes or group["display_name"] in start_nodes):
            continue

        all_cuts.append({
            "index": index,
            "display_name": group["display_name"],
            "cut_node": group["cut_node"],
        })
        index += 1

    total_nodes = len(graph.nodes)
    return all_cuts, total_nodes


# =============================================================================
# Output
# =============================================================================

def format_cut_lines(all_cuts: List[Dict], total_nodes: int) -> List[str]:
    lines = []
    lines.append("=" * 100)
    lines.append("Cut Boundary Index")
    lines.append("=" * 100)

    for cut in all_cuts:
        display = cut["display_name"]
        cut_node = cut["cut_node"]
        if display != cut_node:
            lines.append(f"[{cut['index']:4d}] {display}  (cut at: {cut_node})")
        else:
            lines.append(f"[{cut['index']:4d}] {display}")

    lines.append("=" * 100)
    lines.append(f"Total: {len(all_cuts)} cut boundaries  (graph nodes: {total_nodes})")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="ONNX model cut boundary analyzer (bottleneck-based)",
    )
    parser.add_argument("--onnx", type=str, required=True, help="ONNX model path (relative to this script)")
    parser.add_argument("--output", type=str, default=None, help="Save result to .txt file")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / args.onnx

    if not model_path.exists():
        print(f"Error: {model_path} not found")
        return

    print(f"Loading model: {model_path}")
    model = onnx.load(str(model_path))

    print("Running shape inference...")
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)

    print(f"Graph: {len(graph.nodes)} nodes\n")

    all_cuts, total_nodes = index_bottleneck_cuts(graph)

    lines = format_cut_lines(all_cuts, total_nodes)
    for line in lines:
        print(line)

    if args.output:
        output_path = script_dir / args.output
    else:
        output_path = model_path.with_name(model_path.stem + "_cuts.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()