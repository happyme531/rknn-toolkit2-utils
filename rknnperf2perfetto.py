#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from datetime import datetime
import os

def parse_rknn_log(log_file):
    """解析RKNN性能日志文件"""
    operators = []
    operators_by_id = {}  # 使用字典确保操作ID唯一
    
    # 读取原始日志
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 检查是否带前缀
    has_prefix = "D RKNN: [" in content[:1000]
    
    # 如果带前缀，去掉所有前缀部分
    if has_prefix:
        print("检测到带前缀的日志格式，进行预处理...")
        # 使用正则表达式匹配并移除前缀
        content = re.sub(r'D RKNN: \[\d+:\d+:\d+\.\d+\] ', '', content)
    
    # 查找表格部分
    table_start = content.find("Network Layer Information Table")
    if table_start == -1:
        print("未找到'Network Layer Information Table'")
        return []
    
    # 从表格之后开始查找操作
    start_idx = content.find("\n", table_start)
    table_header_idx = content.find("ID   OpType", start_idx)
    if table_header_idx == -1:
        print("未找到表格头部")
        return []
    
    # 跳过表格头部和分隔线
    lines = content[table_header_idx:].split('\n')
    parsing_started = False
    
    # 查找操作行
    op_lines = []
    for line in lines:
        if "ID   OpType" in line:
            # 找到表头，但还不开始解析
            continue
        
        if line.strip().startswith("----"):
            # 这是分隔线
            if not parsing_started:
                # 第一次遇到分隔线后，开始解析
                parsing_started = True
            continue
        
        if parsing_started:
            if line.strip().startswith("Total"):
                # 结束表格
                break
            
            # 此时应该是操作行
            line = line.strip()
            if re.match(r'^\d+\s+\S+', line):
                op_lines.append(line)
    
    print(f"找到 {len(op_lines)} 行可能是操作的行")
    
    # 解析操作行
    for line in op_lines:
        # 直接从行开始解析，提取ID和其他字段
        parts = line.split()
        
        if not parts:
            continue
            
        try:
            op_id = int(parts[0])
        except ValueError:
            # 如果第一个字段不是数字，这可能不是一个操作行
            continue
        
        # 如果行足够长，提取常见字段
        if len(parts) >= 5:
            try:
                op_type = parts[1]
                data_type = parts[2]
                target = parts[3]
                
                # 使用更可靠的方法查找时间字段和RW字段
                time_us = 0
                rw_kb = 0
                cycles = "0/0/0"
                mac_usage = "0/0/0"
                workload = "0%/0%/0%"
                input_shape = ""
                output_shape = ""
                full_name = ""
                
                # 寻找数字字段（时间和RW）
                number_fields = []
                for i, p in enumerate(parts):
                    if p.isdigit() and i > 3:  # 跳过ID字段
                        number_fields.append((i, int(p)))
                
                # 如果找到至少2个数字字段
                if len(number_fields) >= 2:
                    # 倒数第二个数字通常是时间
                    time_us = number_fields[-2][1]
                    # 倒数第一个数字通常是RW
                    rw_kb = number_fields[-1][1]
                
                # 寻找Cycles字段 (格式如 "数字/数字/数字")
                for i, p in enumerate(parts):
                    if re.match(r'\d+/\d+/\d+', p):
                        cycles = p
                        break
                
                # 寻找workload字段（带有%符号）
                for i, p in enumerate(parts):
                    if '%' in p:
                        workload = p
                        break
                
                # 查找mac_usage字段（格式如 "数字.数字/数字.数字/数字.数字"）
                for i, p in enumerate(parts):
                    if re.match(r'[\d\.]+/[\d\.]+/[\d\.]+', p) and '%' not in p:
                        mac_usage = p
                        break
                
                # 全名通常是最后一个字段
                if len(parts) > 10:
                    full_name = parts[-1]
                
                # 输入和输出形状在前面部分，但可能包含逗号和括号，难以准确提取
                # 我们使用一种简单的启发式方法，寻找括号
                shape_parts = []
                for p in parts:
                    if '(' in p and ')' in p:
                        shape_parts.append(p)
                
                if len(shape_parts) >= 1:
                    output_shape = shape_parts[-1]
                if len(shape_parts) >= 2:
                    input_shape = shape_parts[0]
                
                # 创建操作字典，并添加到operators_by_id中
                operators_by_id[op_id] = {
                    'id': op_id,
                    'op_type': op_type,
                    'data_type': data_type,
                    'target': target,
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'cycles': cycles,
                    'time_us': time_us,
                    'mac_usage': mac_usage,
                    'workload': workload,
                    'rw_kb': rw_kb,
                    'full_name': full_name
                }
            except Exception as e:
                # 如果发生错误，跳过这一行，继续处理下一行
                print(f"解析行时出错 (ID: {op_id}): {str(e)}")
                
                # 如果找不到更多信息，至少保留ID、类型和目标设备
                if op_id not in operators_by_id:
                    operators_by_id[op_id] = {
                        'id': op_id,
                        'op_type': parts[1] if len(parts) > 1 else "Unknown",
                        'data_type': parts[2] if len(parts) > 2 else "Unknown",
                        'target': parts[3] if len(parts) > 3 else "Unknown",
                        'input_shape': "",
                        'output_shape': "",
                        'cycles': "0/0/0",
                        'time_us': 0,
                        'mac_usage': "0/0/0",
                        'workload': "0%/0%/0%",
                        'rw_kb': 0,
                        'full_name': parts[-1] if len(parts) > 4 else ""
                    }
    
    # 将字典转换为列表
    operators = list(operators_by_id.values())
    
    print(f"解析完成，找到 {len(operators)} 个操作")
    if operators:
        print(f"最大操作ID: {max([op['id'] for op in operators])}")
    
    return operators

def generate_chrome_trace(operators, output_file):
    """生成Chrome Trace格式的JSON文件"""
    trace_events = []
    timestamp = 0  # 全局时间戳，确保所有事件在时间上对齐
    
    # 定义处理器与线程ID的映射关系
    target_tid_map = {
        'CPU': 1,
        'NPU': 2
    }
    
    # 为不同目标处理器创建名称
    metadata_events = [
        {
            "name": "thread_name",
            "ph": "M",  # Metadata
            "pid": 1,
            "tid": 1,
            "args": {"name": "CPU 线程"}
        },
        {
            "name": "thread_name",
            "ph": "M",  # Metadata
            "pid": 1,
            "tid": 2,
            "args": {"name": "NPU 线程"}
        }
    ]
    trace_events.extend(metadata_events)
    
    # 根据操作ID排序，确保按执行顺序处理
    sorted_operators = sorted(operators, key=lambda op: op['id'])
    
    # 收集所有操作类型，为每种类型分配一个颜色
    op_types = set()
    for op in operators:
        op_types.add(op['op_type'])
    
    # 为每个操作创建开始和结束事件
    for op in sorted_operators:
        target = op['target']
        tid = target_tid_map.get(target, 3)  # 默认给其他设备一个不同的tid
        
        # 格式化名称，避免重复
        full_name = op['full_name']
        op_type = op['op_type']
        
        # 如果full_name已包含op_type，则直接使用full_name
        if full_name.startswith(op_type):
            display_name = full_name
        else:
            # 否则，组合op_type和full_name
            if ":" in full_name and full_name.split(":")[0] == op_type:
                # 避免类型重复
                display_name = full_name
            else:
                display_name = f"{op_type}:{full_name}"
        
        # 添加操作开始事件
        trace_events.append({
            "name": display_name,
            "cat": op_type,  # 使用操作类型作为分类
            "ph": "B",  # 表示开始事件
            "ts": timestamp,  # 微秒为单位的时间戳
            "pid": 1,  # 进程ID，这里简单使用1
            "tid": tid,  # 线程ID，根据目标设备区分
            "args": {
                "id": op['id'],
                "data_type": op['data_type'],
                "target": op['target'],
                "input_shape": op['input_shape'],
                "output_shape": op['output_shape'],
                "cycles": op['cycles'],
                "mac_usage": op['mac_usage'],
                "workload": op['workload'],
                "rw_kb": op['rw_kb'],
                "time_us": op['time_us']
            }
        })
        
        # 更新时间戳为当前操作结束的时间
        timestamp += op['time_us']
        
        # 添加操作结束事件
        trace_events.append({
            "name": display_name,
            "cat": op_type,  # 使用操作类型作为分类
            "ph": "E",  # 表示结束事件
            "ts": timestamp,
            "pid": 1,
            "tid": tid
        })
    
    # 创建完整的Trace JSON
    trace_json = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",
        "otherData": {
            "version": "RKNN Performance Log to Perfetto",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # 写入JSON文件
    with open(output_file, 'w') as f:
        json.dump(trace_json, f, indent=2)
    
    print(f"生成的Chrome Trace文件：{output_file}")
    print(f"该文件可以在Perfetto UI (https://ui.perfetto.dev) 中加载查看")
    print(f"NPU和CPU的操作将在不同行显示，且时间轴对齐")
    print(f"操作已按类型分类，共 {len(op_types)} 种不同类型")

def main():
    parser = argparse.ArgumentParser(description='将RKNN性能日志转换为Perfetto可读取的Chrome Trace格式')
    parser.add_argument('input', help='RKNN性能日志文件路径')
    parser.add_argument('-o', '--output', help='输出的Chrome Trace文件路径')
    parser.add_argument('-d', '--debug', action='store_true', help='启用调试模式，打印更多信息')
    
    args = parser.parse_args()
    
    if not args.output:
        # 如果没有指定输出文件，使用输入文件名加上.trace.json后缀
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}.trace.json"
    
    # 解析RKNN日志
    operators = parse_rknn_log(args.input)
    
    if not operators:
        print("未能从日志文件中解析出操作信息，请检查日志格式是否正确")
        return
    
    print(f"成功解析 {len(operators)} 个操作")
    
    # 生成Chrome Trace文件
    generate_chrome_trace(operators, args.output)

if __name__ == "__main__":
    main()
