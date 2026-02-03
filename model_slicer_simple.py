"""
ONNX Model Slicer (Quantization-aware) - Backward Reachability

Backward reachability 기반 슬라이서.
cut node에서 역방향으로 도달 가능한 노드를 해당 slice 이전으로 배정.

핵심 원리:
  Forward:  A[i] = cut_i에서 forward reachable  → S0 = All - A[0], S1 = A[0] - A[1], ...
  Backward: B[i] = cut_i에서 backward reachable → S0 = B[0], S1 = B[1] - B[0], S_last = All - B[-1]

  Forward에서 weight q/dq가 S0에 몰리는 이유:
    cut에서 forward로 도달 불가 → All - A[0]에 전부 포함

  Backward에서 자연 해결:
    cut_i에서 backward → activation 경로 역추적 → 해당 layer의 weight q/dq에 도달
    → 차집합으로 자연스럽게 소비자 slice에 배정

사용법:
    python model_slicer_q.py --onnx vit_w8a8/model.onnx --list
    python model_slicer_q.py --onnx vit_w8a8/model.onnx --slice 13,21
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import deque

import onnx
import onnx_graphsurgeon as gs


# =============================================================================
# Cut Boundary Indexer
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


def get_start_end_nodes(graph: gs.Graph,
                        preds: Dict[str, Set[str]],
                        succs: Dict[str, Set[str]]) -> Tuple[Set[str], Set[str]]:
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


def can_reach(starts: Set[str], ends: Set[str],
              succs: Dict[str, Set[str]],
              exclude: Set[str] = None) -> bool:
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


def topological_sort_names(graph: gs.Graph,
                           succs: Dict[str, Set[str]],
                           preds: Dict[str, Set[str]]) -> List[str]:
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


def find_bottleneck_nodes(graph: gs.Graph,
                          succs: Dict[str, Set[str]],
                          preds: Dict[str, Set[str]]) -> List[str]:
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


def _group_qdq_bottlenecks(bottlenecks: List[str], succs: Dict[str, Set[str]]) -> List[Dict]:
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
            groups.append({
                'display_name': bn,
                'cut_node': q_node,
            })
            consumed.update([bn, q_node, dq_node])
        else:
            groups.append({
                'display_name': bn,
                'cut_node': bn,
            })
            consumed.add(bn)

    return groups


def index_bottleneck_cuts(graph: gs.Graph) -> List[Dict]:
    """병목 노드 cut boundary 인덱싱 (1부터 시작)"""
    succs, preds = build_adjacency(graph)
    bottlenecks = find_bottleneck_nodes(graph, succs, preds)
    start_nodes, _ = get_start_end_nodes(graph, preds, succs)
    groups = _group_qdq_bottlenecks(bottlenecks, succs)

    all_cuts = []
    index = 1

    for i, group in enumerate(groups):
        if i == 0 and group['cut_node'] in start_nodes:
            continue
        if i == 0 and group['display_name'] in start_nodes:
            continue

        all_cuts.append({
            'index': index,
            'display_name': group['display_name'],
            'nodes': [group['cut_node']],
        })
        index += 1

    return all_cuts


def print_cut_boundaries(all_cuts: List[Dict], total_nodes: int):
    print("=" * 100)
    print("Cut Boundary Index")
    print("=" * 100)

    for cut in all_cuts:
        display = cut.get('display_name', cut['nodes'][0])
        cut_node = cut['nodes'][0]
        if display != cut_node:
            print(f"[{cut['index']:4d}] {display}  (cut at: {cut_node})")
        else:
            print(f"[{cut['index']:4d}] {display}")

    print("=" * 100)
    print(f"Total: {len(all_cuts)} cut boundaries  (graph nodes: {total_nodes})")


# =============================================================================
# Model Slicer - Backward Reachability
# =============================================================================

def _backward_reachable_nodes(graph: gs.Graph, start_nodes: List[gs.Node]) -> Dict[str, gs.Node]:
    """
    Backward traversal from start_nodes.
    Follows: node -> input tensors -> producer nodes (recursively)
    Returns all nodes reachable by going backward (including start_nodes themselves).
    """
    visited: Set[str] = set()
    collected: Dict[str, gs.Node] = {}
    queue: List[gs.Node] = list(start_nodes)

    while queue:
        node = queue.pop()
        if node is None or not node.name or node.name in visited:
            continue

        visited.add(node.name)
        collected[node.name] = node

        # Traverse backward via input tensor producers
        for inp_t in node.inputs:
            if inp_t is None:
                continue
            for producer in inp_t.inputs:  # gs.Tensor.inputs = list of producer nodes
                queue.append(producer)

    return collected


def _dedup_tensors(tensors: List[gs.Tensor]) -> List[gs.Tensor]:
    dedup: Dict[str, gs.Tensor] = {}
    for t in tensors:
        if t is None or not getattr(t, "name", None):
            continue
        dedup[t.name] = t
    return list(dedup.values())


def _produced_tensors(nodes: List[gs.Node], all_tensors_by_name: Dict[str, gs.Tensor]) -> Dict[str, gs.Tensor]:
    produced: Dict[str, gs.Tensor] = {}
    for n in nodes:
        if n is None:
            continue
        for t in n.outputs:
            if t is None or not t.name:
                continue
            produced[t.name] = all_tensors_by_name.get(t.name, t)
    return produced


def _check_interface_tensors(tensors: List[gs.Tensor], tag: str):
    for t in tensors:
        if t is None or not t.name:
            continue
        if getattr(t, "dtype", None) is None or getattr(t, "shape", None) is None:
            raise RuntimeError(
                f"[{tag}] Interface tensor '{t.name}' missing dtype/shape after infer_shapes(). "
                "Cannot safely slice here."
            )


def slice_onnx_model(
    graph: gs.Graph,
    original_model: onnx.ModelProto,
    cut_node_names_list: List[List[str]],
    save_prefix: str = "submodel"
):
    """
    Backward reachability 기반 ONNX 모델 슬라이싱.

    B[i] = cut_i에서 backward reachable (cut_i 자신 포함)
    S0       = B[0]
    S[k]     = B[k] - B[k-1]   (1 <= k <= n_cuts-1)
    S[last]  = All - B[-1]

    cut node 자체는 B[i]에 포함되므로 이전 slice에 배정된다.
    """

    # Lookup tables
    all_tensors_by_name: Dict[str, gs.Tensor] = {
        t.name: t for t in graph.tensors().values()
        if t is not None and getattr(t, "name", None)
    }
    orig_inputs_by_name: Dict[str, gs.Tensor] = {
        t.name: t for t in graph.inputs
        if t is not None and getattr(t, "name", None)
    }
    orig_outputs: List[gs.Tensor] = [
        t for t in graph.outputs
        if t is not None and getattr(t, "name", None)
    ]

    name_to_node = {n.name: n for n in graph.nodes if n is not None and n.name}

    # Locate cut nodes
    cut_nodes_list: List[List[gs.Node]] = []
    for i, cut_names in enumerate(cut_node_names_list):
        cut_nodes = [name_to_node[name] for name in cut_names if name in name_to_node]
        if not cut_nodes:
            raise ValueError(f"No valid cut nodes found at cut[{i}] with names={cut_names}")
        cut_nodes_list.append(cut_nodes)

    n_cuts = len(cut_nodes_list)
    n_slices = n_cuts + 1

    # =========================================================================
    # Phase 1: Backward reachability partition
    #
    #   B[i] = backward reachable from cut_i (including cut_i itself)
    #   Expected nesting: B[0] ⊆ B[1] ⊆ ... ⊆ B[n-1]
    #
    #   S0       = B[0]
    #   S[k]     = B[k] - B[k-1]   for k = 1..n_cuts-1
    #   S[last]  = All - B[-1]
    # =========================================================================
    B_dicts: List[Dict[str, gs.Node]] = []
    B_sets: List[Set[str]] = []

    for cut_nodes in cut_nodes_list:
        reachable = _backward_reachable_nodes(graph, cut_nodes)
        B_dicts.append(reachable)
        B_sets.append(set(reachable.keys()))

    # Sanity: check nesting
    for i in range(n_cuts - 1):
        if not B_sets[i + 1].issuperset(B_sets[i]):
            print(
                f"  [WARN] B[{i+1}] does NOT contain B[{i}]. "
                "Cuts may be out of topological order."
            )

    all_nodes: List[gs.Node] = [
        n for n in graph.nodes
        if n is not None and getattr(n, "name", None)
    ]
    all_names: Set[str] = set(n.name for n in all_nodes)

    slices_nodes: List[List[gs.Node]] = []

    # S0 = B[0]
    if n_cuts > 0:
        s0_names = B_sets[0]
        slices_nodes.append([B_dicts[0][name] for name in s0_names if name in B_dicts[0]])
    else:
        slices_nodes.append(list(all_nodes))

    # Middle slices: S[k] = B[k] - B[k-1]
    for k in range(1, n_cuts):
        sk_names = B_sets[k] - B_sets[k - 1]
        slices_nodes.append([B_dicts[k][name] for name in sk_names if name in B_dicts[k]])

    # Last slice: S[last] = All - B[-1]
    if n_cuts > 0:
        last_names = all_names - B_sets[-1]
        slices_nodes.append([n for n in all_nodes if n.name in last_names])
    else:
        slices_nodes.append([])

    # =========================================================================
    # Partition validation
    # =========================================================================
    slice_name_sets = [set(n.name for n in nodes) for nodes in slices_nodes]
    union = set().union(*slice_name_sets) if slice_name_sets else set()

    if union != all_names:
        missing = all_names - union
        extra = union - all_names
        print(f"  [WARN] Partition mismatch: missing={len(missing)}, extra={len(extra)}")
        if missing:
            print(f"  [WARN] Missing nodes (first 10): {list(missing)[:10]}")
        if extra:
            print(f"  [WARN] Extra nodes (first 10): {list(extra)[:10]}")
        raise RuntimeError(f"Slice partition mismatch. missing={len(missing)}, extra={len(extra)}")

    for i in range(n_slices):
        for j in range(i + 1, n_slices):
            overlap = slice_name_sets[i] & slice_name_sets[j]
            if overlap:
                raise RuntimeError(
                    f"Slice overlap: S{i} & S{j} share {len(overlap)} nodes. "
                    f"First 5: {list(overlap)[:5]}"
                )

    # =========================================================================
    # Phase 2: Interface tensor computation
    # =========================================================================
    produced_by_slice: List[Dict[str, gs.Tensor]] = []
    producer_slice: Dict[str, int] = {}

    for k in range(n_slices):
        produced = _produced_tensors(slices_nodes[k], all_tensors_by_name)
        produced_by_slice.append(produced)
        for tname in produced.keys():
            producer_slice.setdefault(tname, k)

    # Determine slice inputs
    slice_inputs: List[List[gs.Tensor]] = [[] for _ in range(n_slices)]
    interface_inputs: List[List[gs.Tensor]] = [[] for _ in range(n_slices)]

    for k in range(n_slices):
        inputs_k: List[gs.Tensor] = []
        iface_k: List[gs.Tensor] = []

        for n in slices_nodes[k]:
            for t in n.inputs:
                if t is None or not getattr(t, "name", None):
                    continue

                # Original model input
                if t.name in orig_inputs_by_name:
                    inputs_k.append(orig_inputs_by_name[t.name])
                    continue

                # Produced locally
                if t.name in produced_by_slice[k]:
                    continue

                # Produced by earlier slice -> interface
                p = producer_slice.get(t.name, None)
                if p is not None and p < k:
                    iface_k.append(all_tensors_by_name.get(t.name, t))
                    continue

                # Else: initializer/constant -> internal

        inputs_k = _dedup_tensors(inputs_k)
        iface_k = _dedup_tensors(iface_k)
        _check_interface_tensors(iface_k, f"interface_inputs_S{k}")

        slice_inputs[k] = _dedup_tensors(inputs_k + iface_k)
        interface_inputs[k] = iface_k

    # Last slice: output passthrough
    last = n_slices - 1
    last_extra_inputs: List[gs.Tensor] = []

    for out in orig_outputs:
        p = producer_slice.get(out.name, None)
        if p is not None and p < last:
            t = all_tensors_by_name.get(out.name, out)
            last_extra_inputs.append(t)

    last_extra_inputs = _dedup_tensors(last_extra_inputs)
    _check_interface_tensors(last_extra_inputs, "output_passthrough_to_last")
    slice_inputs[last] = _dedup_tensors(slice_inputs[last] + last_extra_inputs)

    # Determine slice outputs
    slice_outputs: List[List[gs.Tensor]] = [[] for _ in range(n_slices)]

    for m in range(1, n_slices):
        for t in interface_inputs[m]:
            p = producer_slice.get(t.name, None)
            if p is not None and p < m:
                slice_outputs[p].append(all_tensors_by_name.get(t.name, t))

    slice_outputs[last] = [all_tensors_by_name.get(t.name, t) for t in orig_outputs]

    # Linear relay for long edges (producer -> consumer skipping intermediate slices)
    for m in range(1, n_slices):
        for t in interface_inputs[m]:
            if t is None or not getattr(t, "name", None):
                continue

            p = producer_slice.get(t.name, None)
            if p is None or p >= m:
                continue

            for k in range(p + 1, m):
                relay_t = all_tensors_by_name.get(t.name, t)
                slice_inputs[k] = _dedup_tensors(slice_inputs[k] + [relay_t])
                slice_outputs[k] = _dedup_tensors(slice_outputs[k] + [relay_t])

    # Final dedup and checks
    for k in range(n_slices):
        slice_outputs[k] = _dedup_tensors(slice_outputs[k])
        slice_inputs[k] = _dedup_tensors(slice_inputs[k])

        if k != last:
            _check_interface_tensors(slice_outputs[k], f"interface_outputs_S{k}")

        non_orig_inputs = [
            tt for tt in slice_inputs[k]
            if tt is not None and getattr(tt, "name", None) and tt.name not in orig_inputs_by_name
        ]
        _check_interface_tensors(non_orig_inputs, f"interface_inputs_S{k}")

    # =========================================================================
    # Phase 3: Export
    # =========================================================================
    def _export_slice(slice_nodes, inputs, outputs, slice_name, save_path):
        g = gs.Graph(nodes=slice_nodes, inputs=inputs, outputs=outputs, name=slice_name)
        g.cleanup().toposort()
        m = gs.export_onnx(g)

        m.ir_version = original_model.ir_version
        m.opset_import.clear()
        m.opset_import.extend(original_model.opset_import)

        data_filename = save_path.replace(".onnx", ".data")
        data_location = data_filename.split("/")[-1].split("\\")[-1]

        onnx.save(
            m,
            save_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_location,
            size_threshold=0,
        )

        _internalize_small_tensors(save_path, threshold=1024)

    def _internalize_small_tensors(model_path: str, threshold: int = 1024):
        """Small external tensors -> inline (ORT shape inference compatibility)"""
        model_dir = os.path.dirname(os.path.abspath(model_path))
        model = onnx.load(model_path, load_external_data=False)

        modified = False
        for init in model.graph.initializer:
            if not init.external_data:
                continue

            ext_info = {}
            for field in init.external_data:
                ext_info[field.key] = field.value

            location = ext_info.get("location", "")
            offset = int(ext_info.get("offset", "0"))
            length = int(ext_info.get("length", "0"))

            if length == 0 or length > threshold:
                continue

            ext_path = os.path.join(model_dir, location)
            if not os.path.exists(ext_path):
                continue

            with open(ext_path, "rb") as f:
                f.seek(offset)
                raw_bytes = f.read(length)

            init.ClearField("external_data")
            init.data_location = onnx.TensorProto.DEFAULT
            init.raw_data = raw_bytes
            modified = True

        if modified:
            onnx.save(model, model_path)

    for k in range(n_slices):
        save_path = f"{save_prefix}_S{k}.onnx"
        _export_slice(
            slice_nodes=slices_nodes[k],
            inputs=slice_inputs[k],
            outputs=slice_outputs[k],
            slice_name=f"Slice{k}",
            save_path=save_path
        )

    # Summary
    print(f"\nSaved {n_slices} submodels with prefix '{save_prefix}'")
    for k in range(n_slices):
        iface_in = len(interface_inputs[k])
        print(
            f"  S{k}: nodes={len(slices_nodes[k])}, "
            f"inputs={len(slice_inputs[k])}, outputs={len(slice_outputs[k])}, "
            f"interface_in={iface_in}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ONNX Model Slicer (Quantization-aware) - Backward Reachability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory convention:
  Original model : original_model/<folder>/<model>.onnx
  Sliced output  : sliced_model/<folder>/<model>_S0.onnx, <model>_S1.onnx, ...

Examples:
  python model_slicer_q.py --onnx vit_w8a8/model.onnx --list
  python model_slicer_q.py --onnx vit_w8a8/model.onnx --slice 13,21
        """
    )
    parser.add_argument("--onnx", type=str, required=True, help="Subpath under original_model/")
    parser.add_argument("--list", action="store_true", help="List cut boundary nodes")
    parser.add_argument("--slice", type=str, help="Cut boundary indices (comma separated)")

    args = parser.parse_args()

    # Resolve paths
    subpath = Path(args.onnx)
    model_name = subpath.stem
    folder_path = subpath.parent
    input_path = Path("original_model") / subpath

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    print(f"Loading model: {input_path}")
    original_model = onnx.load(str(input_path))

    print("Running shape inference...")
    inferred_model = onnx.shape_inference.infer_shapes(original_model)
    graph = gs.import_onnx(inferred_model)

    print(f"Graph: {len(graph.nodes)} nodes\n")

    all_cuts = index_bottleneck_cuts(graph)

    if args.list:
        print_cut_boundaries(all_cuts, len(graph.nodes))
        return

    if args.slice:
        try:
            cut_indices = [int(x.strip()) for x in args.slice.split(",") if x.strip()]
        except ValueError:
            print(f"Error: Invalid index format: {args.slice}")
            return

        index_to_nodes = {cut['index']: cut['nodes'] for cut in all_cuts}

        for idx in cut_indices:
            if idx not in index_to_nodes:
                print(f"Error: Invalid index {idx}. Valid: 1 ~ {len(all_cuts)}")
                return

        sorted_cuts = sorted(cut_indices)
        cut_node_names_list = [index_to_nodes[idx] for idx in sorted_cuts]

        print(f"Cut boundaries: {sorted_cuts}")
        for i, (idx, nodes) in enumerate(zip(sorted_cuts, cut_node_names_list)):
            display = next((c['display_name'] for c in all_cuts if c['index'] == idx), nodes[0])
            print(f"  Cut {i+1} (index={idx}): {display}")

        output_dir = Path("sliced_model") / folder_path
        output_dir.mkdir(parents=True, exist_ok=True)
        save_prefix = str(output_dir / model_name)

        slice_onnx_model(
            graph=graph,
            original_model=original_model,
            cut_node_names_list=cut_node_names_list,
            save_prefix=save_prefix
        )
    else:
        print_cut_boundaries(all_cuts, len(graph.nodes))
        print("\nTo slice: python model_slicer_q.py --onnx <path> --slice 13,21")


if __name__ == "__main__":
    main()