import argparse
from typing import List, Dict, Set, Tuple

import onnx
import onnx_graphsurgeon as gs


def _forward_reachable_nodes(graph: gs.Graph, start_nodes: List[gs.Node]) -> Dict[str, gs.Node]:
    # Forward traversal from start_nodes to collect all reachable nodes (unique by node.name)
    visited: Set[str] = set()
    collected: Dict[str, gs.Node] = {}
    queue: List[gs.Node] = list(start_nodes)

    while queue:
        node = queue.pop()
        if node is None or not node.name or node.name in visited:
            continue

        visited.add(node.name)
        collected[node.name] = node

        # Traverse forward via output tensor consumers
        for out_t in node.outputs:
            if out_t is None:
                continue
            for consumer in out_t.outputs:
                queue.append(consumer)

    return collected


def _dedup_tensors(tensors: List[gs.Tensor]) -> List[gs.Tensor]:
    # De-duplicate by tensor name (keep last seen)
    dedup: Dict[str, gs.Tensor] = {}
    for t in tensors:
        if t is None or not getattr(t, "name", None):
            continue
        dedup[t.name] = t
    return list(dedup.values())


def _produced_tensors(nodes: List[gs.Node], all_tensors_by_name: Dict[str, gs.Tensor]) -> Dict[str, gs.Tensor]:
    # Map tensor name -> tensor for tensors produced by given nodes
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
    # Interface tensors must have dtype/shape (after shape inference) to be safely stitched
    for t in tensors:
        if t is None or not t.name:
            continue
        if getattr(t, "dtype", None) is None or getattr(t, "shape", None) is None:
            raise RuntimeError(
                f"[{tag}] Interface tensor '{t.name}' missing dtype/shape even after infer_shapes(). "
                "Cannot safely slice here."
            )


def slice_onnx_model_nstage(
    onnx_path: str,
    cut_node_names_list: List[List[str]],  # n cutpoints => n+1 stages
    save_prefix: str = "submodelN"
):
    # ---------------------------
    # 0) Load + shape inference + import
    # ---------------------------
    original_model = onnx.load(onnx_path)
    inferred_model = onnx.shape_inference.infer_shapes(original_model)
    graph = gs.import_onnx(inferred_model)

    # Lookup tables
    all_tensors_by_name: Dict[str, gs.Tensor] = {
        t.name: t for t in graph.tensors().values() if t is not None and getattr(t, "name", None)
    }
    orig_inputs_by_name: Dict[str, gs.Tensor] = {
        t.name: t for t in graph.inputs if t is not None and getattr(t, "name", None)
    }
    orig_outputs: List[gs.Tensor] = [t for t in graph.outputs if t is not None and getattr(t, "name", None)]

    # ---------------------------
    # 1) Locate cut nodes (for each cutpoint)
    # ---------------------------
    cut_nodes_list: List[List[gs.Node]] = []
    for i, cut_names in enumerate(cut_node_names_list):
        cut_set = set(cut_names)
        cut_nodes = [n for n in graph.nodes if n is not None and n.name in cut_set]
        if not cut_nodes:
            raise ValueError(f"No valid cut nodes found at cut[{i}] with names={cut_names}")
        cut_nodes_list.append(cut_nodes)

    n_cuts = len(cut_nodes_list)
    n_stages = n_cuts + 1

    # ---------------------------
    # 2) Build reachable sets A[i] = nodes after cut i
    #    Expect (often) A[0] ⊇ A[1] ⊇ ... ⊇ A[n-1] for clean partitioning
    # ---------------------------
    A_dicts: List[Dict[str, gs.Node]] = []
    A_sets: List[Set[str]] = []

    for i, cut_nodes in enumerate(cut_nodes_list):
        reachable = _forward_reachable_nodes(graph, cut_nodes)
        A_dicts.append(reachable)
        A_sets.append(set(reachable.keys()))

    # Optional sanity: warn if reachability nesting is violated
    for i in range(n_cuts - 1):
        if not A_sets[i].issuperset(A_sets[i + 1]):
            print(
                f"[WARN] Reachability nesting violated: A[{i}] does NOT contain A[{i+1}]. "
                "Cutpoints may be out-of-order topologically; partition may still work but can be surprising."
            )

    # ---------------------------
    # Construct stage node lists using set differences
    #    S[0] = All - A[0]
    #    S[k] = A[k-1] - A[k]  (1..n-1)
    #    S[n] = A[n-1]
    # ---------------------------
    all_nodes: List[gs.Node] = [n for n in graph.nodes if n is not None and getattr(n, "name", None)]
    all_names: Set[str] = set(n.name for n in all_nodes)

    stages_nodes: List[List[gs.Node]] = []

    # Stage 0
    s0_names = all_names - A_sets[0] if n_cuts > 0 else all_names
    stages_nodes.append([n for n in all_nodes if n.name in s0_names])

    # Middle stages
    for k in range(1, n_stages - 1):
        sk_names = A_sets[k - 1] - A_sets[k]
        # Use A_dicts[k-1] to recover node objects
        stages_nodes.append([A_dicts[k - 1][name] for name in sk_names if name in A_dicts[k - 1]])

    # Last stage
    if n_cuts > 0:
        last_names = A_sets[-1]
        stages_nodes.append([A_dicts[-1][name] for name in last_names if name in A_dicts[-1]])
    else:
        stages_nodes.append([])  # Should never happen because n_stages = n_cuts+1 => at least 1

    # Sanity: ensure partition coverage/disjointness
    stage_name_sets = [set(n.name for n in nodes) for nodes in stages_nodes]
    union = set().union(*stage_name_sets) if stage_name_sets else set()
    if union != all_names:
        missing = all_names - union
        extra = union - all_names
        raise RuntimeError(f"Stage partition mismatch. missing={len(missing)}, extra={len(extra)}")

    for i in range(n_stages):
        for j in range(i + 1, n_stages):
            if stage_name_sets[i] & stage_name_sets[j]:
                raise RuntimeError(f"Stage overlap detected between S{i} and S{j}.")

    # ---------------------------
    # Produced tensors per stage + producer_stage map
    # ---------------------------
    produced_by_stage: List[Dict[str, gs.Tensor]] = []
    producer_stage: Dict[str, int] = {}

    for k in range(n_stages):
        produced = _produced_tensors(stages_nodes[k], all_tensors_by_name)
        produced_by_stage.append(produced)
        for tname in produced.keys():
            producer_stage.setdefault(tname, k)

    # ---------------------------
    # Determine stage inputs
    #    stage_inputs[k] = (orig inputs used by stage) + (any tensor produced by earlier stage and consumed here)
    # ---------------------------
    stage_inputs: List[List[gs.Tensor]] = [[] for _ in range(n_stages)]
    interface_inputs: List[List[gs.Tensor]] = [[] for _ in range(n_stages)]  # only non-orig, produced earlier

    for k in range(n_stages):
        inputs_k: List[gs.Tensor] = []
        iface_k: List[gs.Tensor] = []

        for n in stages_nodes[k]:
            for t in n.inputs:
                if t is None or not getattr(t, "name", None):
                    continue

                # Original model input used by this stage
                if t.name in orig_inputs_by_name:
                    inputs_k.append(orig_inputs_by_name[t.name])
                    continue

                # Tensor produced by an earlier stage => interface input
                p = producer_stage.get(t.name, None)
                if p is not None and p < k:
                    iface_k.append(all_tensors_by_name.get(t.name, t))
                    continue

                # Else: likely initializer/constant/internal tensor; keep it internal (do not add as graph input)

        # De-dup and record
        inputs_k = _dedup_tensors(inputs_k)
        iface_k = _dedup_tensors(iface_k)

        # Interface tensors must have dtype/shape for safe stitching
        _check_interface_tensors(iface_k, f"interface_inputs_S{k}")

        stage_inputs[k] = _dedup_tensors(inputs_k + iface_k)
        interface_inputs[k] = iface_k

    # ---------------------------
    # 6) Ensure last stage exposes original outputs (signature preservation)
    #    If an original output is produced in an earlier stage, make it a pass-through input of last stage.
    # ---------------------------
    last = n_stages - 1
    last_extra_inputs: List[gs.Tensor] = []

    for out in orig_outputs:
        out_name = out.name
        p = producer_stage.get(out_name, None)
        if p is not None and p < last:
            t = all_tensors_by_name.get(out_name, out)
            last_extra_inputs.append(t)

    last_extra_inputs = _dedup_tensors(last_extra_inputs)
    _check_interface_tensors(last_extra_inputs, "output_passthrough_to_last")

    stage_inputs[last] = _dedup_tensors(stage_inputs[last] + last_extra_inputs)

    # ---------------------------
    # Determine stage outputs
    #    Rule: if a tensor is consumed as an interface input in a future stage,
    #          it must be produced as an output of its producer stage.
    #    Last stage outputs = original outputs (with pass-through handled above).
    # ---------------------------
    stage_outputs: List[List[gs.Tensor]] = [[] for _ in range(n_stages)]

    # For each stage m, look at its interface inputs; ensure producer stage outputs include them
    for m in range(1, n_stages):
        for t in interface_inputs[m]:
            p = producer_stage.get(t.name, None)
            if p is None:
                continue
            if p < m:
                stage_outputs[p].append(all_tensors_by_name.get(t.name, t))

    # Last stage must expose original outputs
    stage_outputs[last] = [all_tensors_by_name.get(t.name, t) for t in orig_outputs]

    # ---------------------------
    # 7) FORCE LINEAR RELAY (stability-critical)
    #    Ensure long edges (producer -> consumer where consumer > producer+1) are relayed through
    #    all intermediate stages so that each stage can be executed in a strict linear pipeline:
    #      S0 -> S1 -> S2 -> ... -> S(last)
    #
    #    Implementation:
    #      If tensor t is needed as an interface input in stage m, produced at stage p:
    #        For every intermediate stage k in (p+1 .. m-1):
    #          - add t to stage_inputs[k]
    #          - add t to stage_outputs[k]
    #
    #    Why this prevents gs.cleanup() from dropping relay tensors:
    #      A relay tensor that appears in graph.outputs is always preserved by cleanup(),
    #      even if no node consumes it inside that subgraph.
    # ---------------------------
    for m in range(1, n_stages):
        for t in interface_inputs[m]:
            if t is None or not getattr(t, "name", None):
                continue

            p = producer_stage.get(t.name, None)
            if p is None or p >= m:
                continue

            # Relay through intermediate stages: p+1 ... m-1
            for k in range(p + 1, m):
                relay_t = all_tensors_by_name.get(t.name, t)

                # Stage k must accept relay tensor as input (even if unused by nodes)
                stage_inputs[k] = _dedup_tensors(stage_inputs[k] + [relay_t])

                # Stage k must also expose relay tensor as output (so cleanup() keeps it)
                stage_outputs[k] = _dedup_tensors(stage_outputs[k] + [relay_t])

    # ---------------------------
    # 8) De-dup + dtype/shape checks (include relays)
    # ---------------------------
    for k in range(n_stages):
        stage_outputs[k] = _dedup_tensors(stage_outputs[k])
        stage_inputs[k] = _dedup_tensors(stage_inputs[k])

        # All non-last stage outputs are inter-stage interface outputs (including relays)
        if k != last:
            _check_interface_tensors(stage_outputs[k], f"interface_or_relay_outputs_S{k}")

        # All stage inputs that are not original inputs can include interface/relay tensors
        # (We only validate non-orig inputs here.)
        non_orig_inputs = [tt for tt in stage_inputs[k] if tt is not None and getattr(tt, "name", None) and tt.name not in orig_inputs_by_name]
        _check_interface_tensors(non_orig_inputs, f"interface_or_relay_inputs_S{k}")

    # If some non-last stage has empty outputs, that's fine (it means nothing downstream needs from it)

    # ---------------------------
    # Construct + export each stage graph
    # ---------------------------
    def _export_stage(
        stage_nodes: List[gs.Node],
        inputs: List[gs.Tensor],
        outputs: List[gs.Tensor],
        stage_name: str,
        save_path: str
    ):
        # Build stage graph
        g = gs.Graph(nodes=stage_nodes, inputs=inputs, outputs=outputs, name=stage_name)

        # Cleanup & toposort (relay tensors survive because they are graph.outputs)
        g.cleanup().toposort()

        # Export ONNX
        m = gs.export_onnx(g)

        # Keep original IR and opsets
        m.ir_version = original_model.ir_version
        m.opset_import.clear()
        m.opset_import.extend(original_model.opset_import)

        onnx.save(m, save_path)

    # Export all stages
    for k in range(n_stages):
        save_path = f"{save_prefix}_S{k}.onnx"
        _export_stage(
            stage_nodes=stages_nodes[k],
            inputs=stage_inputs[k],
            outputs=stage_outputs[k],
            stage_name=f"Stage{k}",
            save_path=save_path
        )

    # ---------------------------
    # 9) Print summary
    # ---------------------------
    print(f"[INFO] Saved {n_stages} submodels with prefix='{save_prefix}'")
    for k in range(n_stages):
        print(
            f"[INFO] S{k}: nodes={len(stages_nodes[k])}, "
            f"inputs={len(stage_inputs[k])}, outputs={len(stage_outputs[k])}, "
            f"iface_in={len(interface_inputs[k])}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-stage ONNX slicer (n cutpoints => n+1 stages)")
    parser.add_argument("--onnx", type=str, required=True, help="Path to input ONNX model")
    parser.add_argument(
        "--cut",
        action="append",
        required=True,
        help=(
            "One cutpoint as a comma-separated node-name list. "
            "Repeat --cut multiple times to add multiple cutpoints.\n"
            "Example: --cut nodeA,nodeB --cut nodeC"
        ),
    )
    parser.add_argument("--prefix", type=str, default="submodelN", help="Output ONNX prefix")

    args = parser.parse_args()

    # Parse cuts: each --cut "a,b,c" -> ["a","b","c"]
    cut_node_names_list: List[List[str]] = []
    for cut_str in args.cut:
        names = [s.strip() for s in cut_str.split(",") if s.strip()]
        cut_node_names_list.append(names)

    slice_onnx_model_nstage(
        onnx_path=args.onnx,
        cut_node_names_list=cut_node_names_list,
        save_prefix=args.prefix,
    )
