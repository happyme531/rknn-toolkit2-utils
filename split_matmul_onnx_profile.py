#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx import mapping as onnx_mapping


def _build_initializer_map(model: onnx.ModelProto) -> Dict[str, onnx.TensorProto]:
    return {init.name: init for init in model.graph.initializer}


def _tensorproto_to_numpy(tensor: onnx.TensorProto) -> np.ndarray:
    return onnx.numpy_helper.to_array(tensor)


def _numpy_to_tensorproto(name: str, array: np.ndarray) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(array, name=name)


def _count_value_uses(nodes: List[onnx.NodeProto]) -> Dict[str, int]:
    uses: Dict[str, int] = {}
    for n in nodes:
        for i in n.input:
            if i:
                uses[i] = uses.get(i, 0) + 1
    return uses


def _collect_producers(nodes: List[onnx.NodeProto]) -> Dict[str, onnx.NodeProto]:
    producers: Dict[str, onnx.NodeProto] = {}
    for n in nodes:
        for o in n.output:
            if o:
                producers[o] = n
    return producers


def _get_const_node_value(node: onnx.NodeProto) -> Optional[np.ndarray]:
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
            return _tensorproto_to_numpy(attr.t)
    return None


def _unique_name(base: str, existing: Set[str]) -> str:
    if base not in existing:
        existing.add(base)
        return base
    idx = 1
    while f"{base}_{idx}" in existing:
        idx += 1
    name = f"{base}_{idx}"
    existing.add(name)
    return name


def _find_matmul_nodes(model: onnx.ModelProto, name_pattern: str) -> List[onnx.NodeProto]:
    """按名称正则匹配线性类算子，包含 MatMul 与 Gemm。"""
    pattern = re.compile(name_pattern)
    return [
        n
        for n in model.graph.node
        if n.op_type in ("MatMul", "Gemm") and n.name and pattern.search(n.name or "")
    ]


def _get_b_weight_array_for_node(
    node: onnx.NodeProto,
    model: onnx.ModelProto,
    value_uses: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    inits = _build_initializer_map(model)
    producers = _collect_producers(list(model.graph.node))
    b_name = node.input[1]
    if b_name in inits:
        return _tensorproto_to_numpy(inits[b_name])
    prod = producers.get(b_name)
    if prod is not None and prod.op_type == "Constant":
        val = _get_const_node_value(prod)
        if val is None:
            raise ValueError(f"Constant 节点 {prod.name} 没有值")
        return val
    raise ValueError(f"无法获取节点 {node.name} 的权重（既不是 initializer 也不是 Constant）")


def _build_single_matmul_model(
    b_array: np.ndarray,
    seqlen: int,
    input_dtype: Optional[np.dtype] = None,
    node_name: str = "TargetMatMul",
) -> onnx.ModelProto:
    if b_array.ndim != 2:
        raise ValueError(f"权重必须为 2D，实际为 {b_array.shape}")
    k_dim, n_dim = b_array.shape
    if input_dtype is None:
        input_dtype = b_array.dtype

    # ONNX dtype
    onnx_dtype = onnx_mapping.NP_TYPE_TO_TENSOR_TYPE[input_dtype]

    graph = helper.make_graph(
        name="MatMulSubgraph",
        inputs=[
            helper.make_tensor_value_info("a", onnx_dtype, [1, seqlen, k_dim])
        ],
        outputs=[
            helper.make_tensor_value_info("y", onnx_dtype, [1, seqlen, n_dim])
        ],
        nodes=[],
    )

    # Add weight initializer
    graph.initializer.extend([_numpy_to_tensorproto("B", b_array)])

    # Add MatMul
    mm = helper.make_node("MatMul", inputs=["a", "B"], outputs=["y"], name=node_name)
    graph.node.extend([mm])

    opset = helper.make_operatorsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset])
    return model


def _save_model_to_temp(model: onnx.ModelProto, suffix: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="matmul_prof_")
    path = os.path.join(tmpdir, suffix)
    onnx.save(model, path)
    return path


def _ensure_ir_version_set(model: onnx.ModelProto) -> None:
    # 如果原模型缺失或为0，设置为当前ONNX IR版本
    if not getattr(model, "ir_version", None) or model.ir_version == 0:
        model.ir_version = onnx.IR_VERSION


def _save_large_model_external(model: onnx.ModelProto, output_path: str) -> None:
    """将模型以 external data 方式保存，避免 >2GB 的protobuf限制。"""
    _ensure_ir_version_set(model)
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    # 统一将所有tensor写到一个外部文件，阈值设为1KB以强制外部化
    ext_filename = os.path.basename(output_path) + "_data"
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=ext_filename,
        size_threshold=1024,
    )


def _run_rknn_runner_and_get_time_us(runner_script: str, onnx_path: str) -> Optional[int]:
    try:
        out = subprocess.check_output([
            "python3",
            runner_script,
            "--onnx",
            onnx_path,
        ], stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] 运行 runner 失败: {e}\n输出:\n{e.output}")
        return None

    # 匹配 "Total Time(us): 431198"
    m = None
    for match in re.finditer(r"Total Time\(us\):\s*(\d+)", out):
        m = match
    if m is None:
        print(f"[WARN] 未在输出中找到 Total Time(us)，输出如下:\n{out}")
        return None
    return int(m.group(1))


def _profile_best_factor(
    model: onnx.ModelProto,
    name_pattern: str,
    seqlen: int,
    max_factor: int,
    runner_script: str,
) -> int:
    # 1) 找到所有匹配 MatMul/Gemm，确保有效权重形状一致（Gemm 需考虑 transB）
    nodes = _find_matmul_nodes(model, name_pattern)
    if not nodes:
        raise ValueError("未找到任何匹配的 MatMul 节点")

    def _get_attr_int(attrs: List[onnx.AttributeProto], name: str, default: int) -> int:
        for a in attrs:
            if a.name == name and a.type == onnx.AttributeProto.INT:
                return int(a.i)
        return default

    shapes: List[Tuple[int, int]] = []
    b_arrays_eff: List[np.ndarray] = []  # 有效的 B'(K,N)
    for n in nodes:
        b_arr = _get_b_weight_array_for_node(n, model)
        if b_arr.ndim != 2:
            raise ValueError(f"节点 {n.name} 权重不是 2D: {b_arr.shape}")
        if n.op_type == "Gemm":
            transB = _get_attr_int(list(n.attribute), "transB", 0)
            b_eff = b_arr if transB == 0 else b_arr.T
        else:
            b_eff = b_arr
        b_arrays_eff.append(b_eff)
        shapes.append(tuple(b_eff.shape))

    if len(set(shapes)) != 1:
        raise RuntimeError("[TODO] 匹配的 MatMul 权重形状不一致，请按需拆分或分组后再试。")

    b_array = b_arrays_eff[0]
    k_dim, _ = b_array.shape

    # 2) 构建不拆分的单算子子模型
    base_model = _build_single_matmul_model(b_array=b_array, seqlen=seqlen, input_dtype=b_array.dtype)
    base_onnx = _save_model_to_temp(base_model, "matmul_base.onnx")

    # 3) 跑一次 baseline
    base_time = _run_rknn_runner_and_get_time_us(runner_script, base_onnx)
    if base_time is None:
        raise RuntimeError("无法获取 baseline 模型的运行时间")
    best_factor = 1
    best_time = base_time
    print(f"[PROFILE] baseline: {best_time} us @ factor=1")

    # 4) 对每个可行的 factor 进行拆分并测试
    candidates = []
    for f in range(2, max_factor + 1):
        if k_dim % f == 0:
            candidates.append(f)

    if not candidates:
        print("[INFO] 无可行拆分因子（k 不能被 2..max_factor 整除），返回 baseline")
        return best_factor

    # 把 base 子模型用已有 split_matmul 转换（按 name 匹配单个节点）
    for f in candidates:
        try:
            tmp_model = onnx.load(base_onnx)
            # 我们给单个节点命名为 TargetMatMul
            new_m, cnt = split_matmul(tmp_model, name_pattern="^TargetMatMul$", factor=f)
            assert cnt == 1
            sp_onnx = _save_model_to_temp(new_m, f"matmul_split_f{f}.onnx")
            t = _run_rknn_runner_and_get_time_us(runner_script, sp_onnx)
            if t is None:
                continue
            print(f"[PROFILE] factor={f}: {t} us")
            if t < best_time:
                best_time = t
                best_factor = f
        except Exception as e:
            print(f"[WARN] 测试 factor={f} 出错: {e}")

    return best_factor


def split_matmul(
    model: onnx.ModelProto,
    name_pattern: str,
    factor: int,
) -> Tuple[onnx.ModelProto, int]:
    if factor <= 1:
        raise ValueError("factor 必须大于 1")

    pattern = re.compile(name_pattern)

    graph = model.graph
    inits = _build_initializer_map(model)
    nodes = list(graph.node)
    value_uses = _count_value_uses(nodes)
    producers = _collect_producers(nodes)

    # 用于生成唯一名称
    existing_value_names: Set[str] = set()
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        existing_value_names.add(vi.name)
    for init in graph.initializer:
        existing_value_names.add(init.name)
    for n in nodes:
        for nm in list(n.input) + list(n.output):
            if nm:
                existing_value_names.add(nm)

    # 遍历并替换
    new_nodes: List[onnx.NodeProto] = []
    replace_output: Dict[str, str] = {}
    replaced_count = 0
    # 记录被移除的常量输出名（来自 Constant 节点）
    constant_outputs_to_remove: Set[str] = set()

    for node in nodes:
        # 先替换其输入名（如果前面有输出被替换）
        updated_inputs = [replace_output.get(inp, inp) for inp in node.input]

        # 仅处理 MatMul
        if node.op_type == "MatMul" and node.name and pattern.search(node.name or ""):
            if len(node.input) != 2:
                print(f"[WARN] 跳过节点 {node.name}: 输入数量不是 2")
                # 保留原节点（但维持已经替换过的输入）
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                # 复制属性
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue
            print(f"[INFO] 处理节点 {node.name}")
            a_name = updated_inputs[0]
            b_name = updated_inputs[1]
            y_name = node.output[0]

            # 解析 B（权重）
            b_array: Optional[np.ndarray] = None
            b_const_output_name_to_remove: Optional[str] = None

            if b_name in inits:
                b_array = _tensorproto_to_numpy(inits[b_name])
            else:
                # 尝试 Constant 节点
                prod = producers.get(b_name)
                if prod is not None and prod.op_type == "Constant":
                    b_array = _get_const_node_value(prod)
                    # 如果只被这个 MatMul 使用，可在之后移除
                    if value_uses.get(b_name, 0) == 1:
                        b_const_output_name_to_remove = b_name

            if b_array is None:
                print(f"[WARN] 跳过节点 {node.name}: 无法获取权重（既不是 initializer 也不是 Constant）")
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue

            if b_array.ndim != 2:
                print(f"[WARN] 跳过节点 {node.name}: 权重维度不是 2D，实际 {b_array.shape}")
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue

            k_dim, n_dim = b_array.shape[0], b_array.shape[1]
            if k_dim % factor != 0:
                print(
                    f"[WARN] 跳过节点 {node.name}: 权重首维 {k_dim} 无法被 factor {factor} 整除"
                )
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue

            chunk = k_dim // factor
            split_sizes = [chunk] * factor

            # 为 A 创建 Split（沿最后一维）。Split-13 使用第二个输入作为 split 张量
            a_split_outputs: List[str] = [
                _unique_name(f"{a_name}_split_{node.name}_{i}", existing_value_names) for i in range(factor)
            ]
            split_sizes_name = _unique_name(
                f"{node.name}_Split_sizes", existing_value_names
            )
            split_sizes_tensor = _numpy_to_tensorproto(
                split_sizes_name, np.asarray(split_sizes, dtype=np.int64)
            )
            graph.initializer.append(split_sizes_tensor)
            split_a = helper.make_node(
                "Split",
                inputs=[a_name, split_sizes_name],
                outputs=a_split_outputs,
                name=_unique_name(f"{node.name}_SplitA", existing_value_names),
            )
            # 属性：axis=-1
            split_a.attribute.extend([helper.make_attribute("axis", -1)])
            new_nodes.append(split_a)

            # 拆分 B 为多个 initializer
            new_b_names: List[str] = []
            for i in range(factor):
                start = i * chunk
                end = (i + 1) * chunk
                part = b_array[start:end, :]
                new_b_name = _unique_name(f"{b_name}_part_{i}", existing_value_names)
                graph.initializer.append(_numpy_to_tensorproto(new_b_name, part))
                new_b_names.append(new_b_name)

            # 为每个分块创建 MatMul
            part_outputs: List[str] = []
            for i in range(factor):
                part_y = _unique_name(f"{node.name}_partY_{i}", existing_value_names)
                mm = helper.make_node(
                    "MatMul",
                    inputs=[a_split_outputs[i], new_b_names[i]],
                    outputs=[part_y],
                    name=_unique_name(f"{node.name}_MatMul_{i}", existing_value_names),
                )
                part_outputs.append(part_y)
                new_nodes.append(mm)

            # Sum 聚合
            sum_y = _unique_name(f"{node.name}_SumY", existing_value_names)
            sum_node = helper.make_node(
                "Sum",
                inputs=part_outputs,
                outputs=[sum_y],
                name=_unique_name(f"{node.name}_Sum", existing_value_names),
            )
            new_nodes.append(sum_node)

            # 将原输出名替换为 sum 输出
            replace_output[y_name] = sum_y
            replaced_count += 1

            # 如果 B 来自 Constant 且仅被该 MatMul 使用，则标记删除
            if b_const_output_name_to_remove is not None:
                constant_outputs_to_remove.add(b_const_output_name_to_remove)

            continue  # 不保留原 MatMul

        # 处理 Gemm（在 alpha=1.0, beta=1.0, transA=0 的前提下，将其展开为 Split/MatMul/Sum/(Add)）
        if node.op_type == "Gemm" and node.name and pattern.search(node.name or ""):
            # 解析属性
            def _get_attr(attrs, name, default):
                for a in attrs:
                    if a.name == name:
                        if a.type == onnx.AttributeProto.FLOAT:
                            return float(a.f)
                        if a.type == onnx.AttributeProto.INT:
                            return int(a.i)
                return default

            alpha = _get_attr(node.attribute, "alpha", 1.0)
            beta = _get_attr(node.attribute, "beta", 1.0)
            transA = _get_attr(node.attribute, "transA", 0)
            transB = _get_attr(node.attribute, "transB", 0)

            # 严格断言（按用户要求）
            assert alpha == 1.0, f"Gemm {node.name}: 仅支持 alpha==1.0，实际 {alpha}"
            assert beta == 1.0, f"Gemm {node.name}: 仅支持 beta==1.0，实际 {beta}"
            assert transA == 0, f"Gemm {node.name}: 仅支持 transA==0，实际 {transA}"
            if transB not in (0, 1):
                print(f"[WARN] 跳过节点 {node.name}: 不支持的 transB={transB}")
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue

            # A, B, (C) 名称
            if len(updated_inputs) < 2:
                print(f"[WARN] 跳过节点 {node.name}: 输入数量不足")
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue

            a_name = updated_inputs[0]
            b_name = updated_inputs[1]
            c_name = None
            if len(updated_inputs) >= 3 and updated_inputs[2]:
                c_name = updated_inputs[2]
            y_name = node.output[0]

            # 解析 B（权重）
            b_array: Optional[np.ndarray] = None
            b_const_output_name_to_remove: Optional[str] = None

            if b_name in inits:
                b_array = _tensorproto_to_numpy(inits[b_name])
            else:
                prod = producers.get(b_name)
                if prod is not None and prod.op_type == "Constant":
                    b_array = _get_const_node_value(prod)
                    if value_uses.get(b_name, 0) == 1:
                        b_const_output_name_to_remove = b_name

            if b_array is None:
                print(f"[WARN] 跳过节点 {node.name}: 无法获取权重（既不是 initializer 也不是 Constant）")
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue

            if b_array.ndim != 2:
                print(f"[WARN] 跳过节点 {node.name}: 权重维度不是 2D，实际 {b_array.shape}")
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue

            # 若 transB==1，离线转置为 (K, N)
            if transB == 0:
                b_k_n = b_array
            else:
                b_k_n = b_array.T

            k_dim, n_dim = b_k_n.shape[0], b_k_n.shape[1]
            if k_dim % factor != 0:
                print(
                    f"[WARN] 跳过节点 {node.name}: 权重K维 {k_dim} 无法被 factor {factor} 整除"
                )
                kept = onnx.helper.make_node(
                    node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
                )
                kept.attribute.extend(node.attribute)
                new_nodes.append(kept)
                continue

            chunk = k_dim // factor
            split_sizes = [chunk] * factor

            # 对 A 沿最后一维（K）做 Split
            a_split_outputs: List[str] = [
                _unique_name(f"{a_name}_split_{node.name}_{i}", existing_value_names) for i in range(factor)
            ]
            split_sizes_name = _unique_name(
                f"{node.name}_Split_sizes", existing_value_names
            )
            split_sizes_tensor = _numpy_to_tensorproto(
                split_sizes_name, np.asarray(split_sizes, dtype=np.int64)
            )
            graph.initializer.append(split_sizes_tensor)
            split_a = helper.make_node(
                "Split",
                inputs=[a_name, split_sizes_name],
                outputs=a_split_outputs,
                name=_unique_name(f"{node.name}_SplitA", existing_value_names),
            )
            split_a.attribute.extend([helper.make_attribute("axis", -1)])
            new_nodes.append(split_a)

            # 拆分 B'(K,N) 为多个 initializer
            new_b_names: List[str] = []
            for i in range(factor):
                start = i * chunk
                end = (i + 1) * chunk
                part = b_k_n[start:end, :]
                new_b_name = _unique_name(f"{b_name}_part_{i}", existing_value_names)
                graph.initializer.append(_numpy_to_tensorproto(new_b_name, part))
                new_b_names.append(new_b_name)

            # 为每个分块创建 MatMul
            part_outputs: List[str] = []
            for i in range(factor):
                part_y = _unique_name(f"{node.name}_partY_{i}", existing_value_names)
                mm = helper.make_node(
                    "MatMul",
                    inputs=[a_split_outputs[i], new_b_names[i]],
                    outputs=[part_y],
                    name=_unique_name(f"{node.name}_MatMul_{i}", existing_value_names),
                )
                part_outputs.append(part_y)
                new_nodes.append(mm)

            # Sum 聚合
            sum_y = _unique_name(f"{node.name}_SumY", existing_value_names)
            sum_node = helper.make_node(
                "Sum",
                inputs=part_outputs,
                outputs=[sum_y],
                name=_unique_name(f"{node.name}_Sum", existing_value_names),
            )
            new_nodes.append(sum_node)

            final_y = sum_y
            # 若存在 C，进行 Add
            if c_name is not None:
                add_y = _unique_name(f"{node.name}_AddY", existing_value_names)
                add_node = helper.make_node(
                    "Add",
                    inputs=[sum_y, c_name],
                    outputs=[add_y],
                    name=_unique_name(f"{node.name}_AddC", existing_value_names),
                )
                new_nodes.append(add_node)
                final_y = add_y

            # 将原输出名替换为最终输出
            replace_output[y_name] = final_y
            replaced_count += 1

            # 如果 B 来自 Constant 且仅被该 Gemm 使用，则标记删除
            if b_const_output_name_to_remove is not None:
                constant_outputs_to_remove.add(b_const_output_name_to_remove)

            continue  # 不保留原 Gemm

        # 非目标节点或未匹配正则：按更新后的输入复制
        kept = onnx.helper.make_node(
            node.op_type, inputs=updated_inputs, outputs=node.output, name=node.name
        )
        kept.attribute.extend(node.attribute)
        # 如果是 Constant 且输出在待删除集合中，跳过
        if kept.op_type == "Constant" and any(
            (out in constant_outputs_to_remove) for out in kept.output
        ):
            continue
        new_nodes.append(kept)

    # 替换图输出名（若恰好直接从该 MatMul 输出）
    for go in graph.output:
        if go.name in replace_output:
            go.name = replace_output[go.name]

    # 应用新的节点列表
    del graph.node[:]
    graph.node.extend(new_nodes)

    # 清理未使用的 initializer（按新图重新统计使用）
    new_uses = _count_value_uses(new_nodes)
    kept_inits: List[onnx.TensorProto] = []
    for init in graph.initializer:
        if init.name in new_uses or any(inp.name == init.name for inp in graph.input):
            kept_inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)

    return model, replaced_count


def main():
    parser = argparse.ArgumentParser(
        description="按名称正则匹配 MatMul，并沿共享维度拆分为多个更小的 MatMul + Sum；支持对子模型进行性能 profile 以选择最佳拆分倍数"
    )
    parser.add_argument("--input", required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--output", required=False, help="输出 ONNX 模型路径")
    parser.add_argument(
        "--pattern",
        required=True,
        help="用于匹配 MatMul 节点名称的正则表达式（按 node.name 匹配）",
    )
    parser.add_argument("--factor", type=int, help="拆分倍数，>1。若设置 --profile 则忽略该值")
    parser.add_argument(
        "--infer-shape",
        action="store_true",
        help="保存前执行一次 shape inference",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="启用对子模型（原始与拆分）进行性能测试，自动挑选最快的拆分倍数",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        help="当 --profile 启用时，指定该算子输入的序列长度（输入形状为 1 x seqlen x dim）",
    )
    parser.add_argument(
        "--max-factor",
        type=int,
        default=16,
        help="当 --profile 启用时，候选拆分因子的最大值（会筛选能整除 dim 的因子）",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="rknn_perf_runner.py",
        help="用于在设备上转换+运行 ONNX 的 runner 脚本路径（子进程调用）",
    )
    args = parser.parse_args()

    model = onnx.load(args.input)

    if args.profile:
        if not args.seqlen or args.seqlen <= 0:
            raise ValueError("--profile 需要提供正整数 --seqlen")

        best_factor = _profile_best_factor(
            model=model,
            name_pattern=args.pattern,
            seqlen=args.seqlen,
            max_factor=args.max_factor,
            runner_script=args.runner,
        )
        print(f"[INFO] profile 选择的最佳拆分倍数：{best_factor}")
        if args.output:
            new_model, cnt = split_matmul(model, args.pattern, best_factor)
            if args.infer_shape:
                try:
                    new_model = onnx.shape_inference.infer_shapes(new_model)
                except Exception as e:
                    print(f"[WARN] 形状推理失败: {e}")
            _ensure_ir_version_set(new_model)
            _save_large_model_external(new_model, args.output)
            try:
                onnx.checker.check_model(args.output)
            except Exception as e:
                print(f"[WARN] 模型校验失败（已保存 external data）：{e}")
            print(f"完成：拆分 {cnt} 个 MatMul，已保存至 {args.output}")
        return

    if args.factor is None:
        raise ValueError("未启用 --profile 时必须指定 --factor")

    new_model, cnt = split_matmul(model, args.pattern, args.factor)

    if args.infer_shape:
        try:
            new_model = onnx.shape_inference.infer_shapes(new_model)
        except Exception as e:
            print(f"[WARN] 形状推理失败: {e}")

    if not args.output:
        raise ValueError("必须指定 --output")

    _ensure_ir_version_set(new_model)
    _save_large_model_external(new_model, args.output)
    try:
        onnx.checker.check_model(args.output)
    except Exception as e:
        print(f"[WARN] 模型校验失败（已保存 external data）：{e}")
    print(f"完成：拆分 {cnt} 个 MatMul，已保存至 {args.output}")


if __name__ == "__main__":
    main()


