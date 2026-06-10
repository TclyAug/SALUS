import os
import sys
import networkx as nx
from collections import defaultdict
import pickle
import itertools
import math
from heapq import heappush, heappop
from itertools import combinations

# inputnum = 8
outputnum = 3
similarity_threshold = 0.4

def save_graph(graph, file_name):
    # 设置目标目录和文件路径
    output_directory = "../testdag"
    os.makedirs(output_directory, exist_ok=True) 
    output_file = os.path.join(output_directory, file_name)
    with open(output_file, 'wb') as f:
        pickle.dump(graph, f)

def parse_blif(file_path):
    inputs = []
    outputs = []
    current_node = None
    node_inputs = []
    truth_table = []
    continuation_line = ""
    
    dag = nx.MultiDiGraph()

    with open(file_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.endswith("\\"):
                continuation_line += line[:-1].strip() + " "
                continue
            else:
                line = continuation_line + line
                continuation_line = ""

            if line.startswith(".inputs"):
                inputs = line.split()[1:]
                for inp in inputs:
                    dag.add_node(inp)
            elif line.startswith(".outputs"):
                outputs = line.split()[1:]
            elif line.startswith(".names"):
                if current_node:
                    dag.nodes[current_node]['truth_table'] = truth_table
                    for inp in node_inputs:
                        dag.add_edge(inp, current_node)

                parts = line.split()
                current_node = parts[-1]  
                node_inputs = parts[1:-1] 
                truth_table = [] 
                dag.add_node(current_node) 
            elif line.startswith(".end"):
                if current_node:
                    dag.nodes[current_node]['truth_table'] = truth_table
                    for inp in node_inputs:
                        dag.add_edge(inp, current_node)
                break
            elif current_node:
                truth_table.append(line.strip())

    for node in dag.nodes:
        if 'truth_table' not in dag.nodes[node]:
            dag.nodes[node]['truth_table'] = []

    return dag

def format_truth_table(truth_table, num_inputs):
    total_entries = 2 ** num_inputs  
    formatted = ["X"] * total_entries
    for line in truth_table:
        line = line.strip()
        if not line or " " not in line:
            continue
        inputs, output = line.split()
        possible_inputs = [""]
        for char in inputs:
            if char == "-":
                possible_inputs = [p + "0" for p in possible_inputs] + [p + "1" for p in possible_inputs]
            else:
                possible_inputs = [p + char for p in possible_inputs]
        for binary_input in possible_inputs:
            index = int(binary_input, 2)
            formatted[index] = output  
    default_value = "1" if "0" in formatted else "0" 
    for i in range(total_entries):
        if formatted[i] == "X": 
            formatted[i] = default_value

    return "".join(formatted)


def layerize_dag(dag):
    layers = list(nx.topological_sort(dag))
    node_layer = {}
    for node in layers:
        max_input_layer = -1
        for input_node in dag.predecessors(node):
            if input_node in node_layer:
                max_input_layer = max(max_input_layer, node_layer[input_node])
        
        node_layer[node] = max_input_layer + 1

    grouped_layers = defaultdict(list)
    for node, layer in node_layer.items():
        grouped_layers[layer].append(node)
        
    max_layer = max(grouped_layers.keys())
    return [grouped_layers[i] for i in range(max_layer + 1)] 


def compute_sensitivity(truth_table, input_var_indices, num_inputs):
    """
    计算每个输入变量对函数输出的敏感度。
    
    :param truth_table: 字符串形式的真值表，如 '0101'
    :param input_var_indices: 输入变量的索引列表
    :param num_inputs: 输入变量的总数
    :return: 敏感度字典 {input_var: sensitivity}
    """
    sensitivity = {}
    num_entries = len(truth_table)
    
    for var in input_var_indices:
        mask = 1 << (num_inputs - var - 1)
        changes = 0
        
        for i in range(num_entries):
            flipped_i = i ^ mask
            if flipped_i < num_entries:
                if truth_table[i] != truth_table[flipped_i]:
                    changes += 1
        # 每次变化只计算一次
        sensitivity[var] = changes / (num_entries)
    
    return sensitivity

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)


def group_nodes(layer, dag, inputnum, outputnum):
    node_inputs = {}
    node_sensitivities = {}
    
    all_inputs = set(inp for node in layer for inp in dag.predecessors(node))
    sorted_inputs = sorted(all_inputs)
    input_mapping = {inp: idx for idx, inp in enumerate(sorted_inputs)}

    for node in layer:
        inputs = list(dag.predecessors(node))
        node_inputs[node] = set(inputs)
        truth_table = dag.nodes[node].get('truth_table', '')
        if not truth_table:
            raise ValueError(f"Node {node} requires a truth_table.")
        # node_sensitivities[node] = compute_sensitivity(truth_table, 
        #                                                [input_mapping[inp] for inp in inputs], 
        #                                                len(sorted_inputs))
    def calculate_overlap(node1, node2):
        return len(node_inputs[node1].intersection(node_inputs[node2]))
    sorted_layer = sorted(layer, key=lambda node: sum(
        calculate_overlap(node, other_node) for other_node in layer if other_node != node
    ), reverse=True)
    groups = []
    for node in sorted_layer:
        placed = False
        for group in groups:
            group_inputs = set(inp for n in group for inp in node_inputs[n])
            group_inputs.update(node_inputs[node])            
            if len(group_inputs) <= inputnum and len(group) < outputnum:
                group.append(node)
                placed = True
                break

        if not placed:
            groups.append([node])
    
    return groups

def format_dag_before_merge(dag):
    for node in list(dag.nodes):  
        if dag.in_degree(node) == 0:  
            dag.nodes[node]['type'] = 'input'
            # new_node = f"output_{node}"  
            # dag.add_node(new_node, type='output')  
            # dag.add_edge(node, new_node)
    for u, v, k in dag.edges(keys=True):
        dag.edges[u, v, k]['pre'] = u
    for node in dag.nodes:
        if dag.nodes[node].get('type') == 'input':
            continue
        input_count = len(list(dag.predecessors(node)))
        # if input_count <= outputnum:
        #     dag.nodes[node]['type'] = 'pbs'
        # else:
        dag.nodes[node]['type'] = 'mux1'
        
def format_dag_after_merge(dag):
    for node in dag.nodes:
        if dag.nodes[node].get('type') == 'input':
            continue
        input_num = len(list(dag.in_edges(node)))
        if input_num <= outputnum:
            dag.nodes[node]['type'] = 'pbs'
            
def merge_nodes(layers, dag, inputnum, outputnum):
    dag_copy = dag.copy()
    for i in range(len(layers) - 1, 0, -1):  
        layer = layers[i]
        groups = group_nodes(layer, dag, inputnum, outputnum)
        for node_list in groups:
            if len(node_list) > 1 : 
                new_node = "-".join(node_list) 
                dag.add_node(new_node)  
                predecessors = set()
                successors = set()  
                for node in node_list:
                    for succ in dag.successors(node):
                        dag.add_edge(new_node, succ, pre=node)
                        successors.add(succ)  
                    for pred in dag.predecessors(node):
                        predecessors.add(pred)
                for pred in predecessors:
                    dag.add_edge(pred, new_node)
                input_nodes, new_table = generate_truth_table(dag_copy , node_list)
                new_truth_table = combine_truth_table(new_table)
                dag.nodes[new_node]['truth_table'] = new_truth_table 
                dag.nodes[new_node]['type'] =  'mux3'
                
                for node in node_list:
                    dag.remove_node(node)
    return dag

def generate_truth_table(dag, node_ids_to_merge):
    all_inputs_set = set()
    node_to_local_predecessors = {}

    for node_id in node_ids_to_merge:
        preds = list(dag.predecessors(node_id))  # 直接前序节点
        node_to_local_predecessors[node_id] = preds
        all_inputs_set.update(preds)

    merged_inputs = sorted(all_inputs_set)
    num_merged_inputs = len(merged_inputs)

    total_combos = list(itertools.product([0,1], repeat=num_merged_inputs))

    new_truth_tables = {}

    for node_id in node_ids_to_merge:
        old_tt = dag.nodes[node_id]["truth_table"]  # 旧的0/1串, 长度=2^(fanin)
        local_preds = node_to_local_predecessors[node_id]
        local_preds_sorted = sorted(local_preds)  # 以保证 index 一致

        fanin = len(local_preds_sorted)
        if len(old_tt) != 2**fanin:
            raise ValueError(f"truth_table length({len(old_tt)}) do not match ({fanin})")

        new_tt_bits = []

        for combo in total_combos:
            index_for_old_tt = 0
            for pred_id in local_preds_sorted:
                idx_in_merged = merged_inputs.index(pred_id)
                bit_val = combo[idx_in_merged]
                index_for_old_tt = (index_for_old_tt << 1) | bit_val
            out_bit = old_tt[index_for_old_tt]
            new_tt_bits.append(out_bit)

        new_tt_str = "".join(new_tt_bits)

        new_truth_tables[node_id] = new_tt_str
        
    return merged_inputs, new_truth_tables

def combine_truth_table(truth_tables):
    # 将每个真值表拆解为列表（每个二进制字符为一个元素）
    binary_values = [list(val) for val in truth_tables.values()]
    # 存储每一列合并后的二进制数
    merged_columns = []
    # 遍历每一列，按列合并并转为整数
    for i in range(len(binary_values[0])):  # 遍历每一列
        column = ''.join([binary_values[j][i] for j in range(len(binary_values))])
        merged_columns.append(int(column, 2))  # 转为二进制整数
    return merged_columns

def format_dag_truthtable(dag):
    for node in dag.nodes:
        truth_table = dag.nodes[node].get('truth_table', [])

        if isinstance(truth_table, str):
            if ',' in truth_table:
                truth_table = truth_table
            else:
                chars = list(truth_table)
                numeric_vals = [int(c) for c in chars if c.isdigit()]
                truth_table = "[" + ",".join(str(x) for x in numeric_vals) + "]"
        elif isinstance(truth_table, list):
            truth_table = "[" + ",".join(map(str, truth_table)) + "]"
        else:
            truth_table = str(truth_table)
        dag.nodes[node]['truth_table'] = truth_table
        
def print_dag_info(dag):
    inputs = [node for node in dag.nodes if dag.in_degree(node) == 0]
    outputs = [node for node in dag.nodes if dag.out_degree(node) == 0]
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")
    for node in dag.nodes:
        if dag.nodes[node].get('type') == 'input':
            continue
        
        predecessors = list(dag.predecessors(node))
        successors = list(dag.successors(node))
        nodetype = dag.nodes[node].get('type', [])
        truth_table = dag.nodes[node].get('truth_table', [])
        
        print(f"Node: {node}, Type:{nodetype}")
        print(f"  Pre: {predecessors}, Suc: {successors}")

    # total_nodes = len(dag.nodes) - len(inputs)
    # print(f"Node#: {total_nodes}")
            
def checkdag(dag):
    inputs = [node for node in dag.nodes if dag.in_degree(node) == 0]
    outputs = [node for node in dag.nodes if dag.out_degree(node) == 0]
    for out_node in outputs:
        if not any(nx.has_path(dag, inp, out_node) for inp in inputs):
            raise ValueError(
                f"Output node {out_node} is not connected from any input node."
            )
    for node in dag.nodes:
        if dag.nodes[node].get('type') == 'input':
            continue
        if not(isinstance(dag.nodes[node]['truth_table'], str)):
            raise ValueError(
                f"Node {node} truthtable type error."
            )
    print("checkdag: PASS")
            
def main():
    if len(sys.argv) != 3:
        print("input error")
        sys.exit(1)
        
    circuit_name = sys.argv[1]
    inputnum = int(sys.argv[2])
    
    file_path = os.path.join("../testcircuit", f"{circuit_name}.blif")
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        sys.exit(1)
    
    dag = parse_blif(file_path)
    
    for node in dag.nodes:
        if 'truth_table' not in dag.nodes[node]:
            dag.nodes[node]['truth_table'] = []
        dag.nodes[node]['truth_table'] = format_truth_table(dag.nodes[node]['truth_table'], len(list(dag.predecessors(node))))

    #print_dag_info(dag)
    format_dag_before_merge(dag)
    
    dag_copy = dag.copy()
    format_dag_truthtable(dag_copy)
    filename = f"{circuit_name}_dag.pkl"
    save_graph(dag_copy, filename)  # 存储图到文件
    dag_copy.clear()
    
    layers = layerize_dag(dag)
    merge_nodes(layers, dag, inputnum, outputnum)
    
    format_dag_after_merge(dag)
    format_dag_truthtable(dag)
    print_dag_info(dag)   
    checkdag(dag)
    filename = f"{circuit_name}_dag_m.pkl"
    save_graph(dag, filename)  
    
if __name__ == "__main__":
    main()
