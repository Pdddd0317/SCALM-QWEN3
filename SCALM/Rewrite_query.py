import os
from deep_lake import query_database
import time
from openai import OpenAI
from prompt_txt import prompt_txt
from json_processing import extract_and_combine_json_arrays, add_line
import re
import glob
import multiprocessing
import json




# 132 125
# sol_files = glob.glob(os.path.join(r"C:\Users\33519\Desktop\exp\code\de", "*.sol"))
save_session_path = 'session_History'
save_audit_path = 'Audit_report'


def read_by_chunks(lines, chunk_size=20):
    chunk = []
    chunk_number = 1
    for i, line in enumerate(lines):
        if i % chunk_size == 0 and i != 0:
            yield f'#{chunk_number}\n' + ''.join(chunk)
            chunk = []
            chunk_number += 1
        chunk.append(line)
    if chunk:
        yield f'#{chunk_number}\n' + ''.join(chunk)


def extract_contract_context(code):
    """
    Extract contract context information (state variables, events, modifiers, etc.)
    
    Returns:
        Dictionary containing various context information
    """
    lines = code.split('\n')
    context = {
        'contract_name': None,
        'state_variables': [],
        'events': [],
        'modifiers': [],
        'structs': [],
        'enums': [],
        'imports': [],
        'pragmas': []
    }
    
    i = 0
    in_contract = False
    contract_start = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Extract pragma
        if line.startswith('pragma '):
            context['pragmas'].append(lines[i])
        
        # Extract import
        elif line.startswith('import '):
            context['imports'].append(lines[i])
        
        # Detect contract start
        elif re.match(r'^\s*contract\s+(\w+)', line):
            match = re.match(r'^\s*contract\s+(\w+)', line)
            context['contract_name'] = match.group(1)
            in_contract = True
            contract_start = i
        
        elif in_contract:
            # Extract state variables (not functions, not events, not modifiers)
            if (not line.startswith('function ') and 
                not line.startswith('event ') and
                not line.startswith('modifier ') and
                not line.startswith('struct ') and
                not line.startswith('enum ') and
                not line.startswith('constructor') and
                not line.startswith('//') and
                not line.startswith('/*') and
                not line.startswith('*') and
                line and 
                ';' in line and
                not line.startswith('}')):
                
                # Check if it's a state variable declaration
                if any(keyword in line for keyword in ['mapping', 'uint', 'int', 'address', 'bool', 'string', 'bytes']):
                    # May span multiple lines, collect complete declaration
                    var_lines = [lines[i]]
                    temp_i = i
                    while temp_i < len(lines) and ';' not in lines[temp_i]:
                        temp_i += 1
                        if temp_i < len(lines):
                            var_lines.append(lines[temp_i])
                    
                    var_code = '\n'.join(var_lines)
                    context['state_variables'].append({
                        'code': var_code,
                        'line': i + 1
                    })
            
            # Extract event definitions
            elif line.startswith('event '):
                event_lines = [lines[i]]
                temp_i = i
                while temp_i < len(lines) and ';' not in lines[temp_i]:
                    temp_i += 1
                    if temp_i < len(lines):
                        event_lines.append(lines[temp_i])
                
                event_code = '\n'.join(event_lines)
                # Extract event name
                event_match = re.match(r'^\s*event\s+(\w+)', line)
                event_name = event_match.group(1) if event_match else 'Unknown'
                
                context['events'].append({
                    'name': event_name,
                    'code': event_code,
                    'line': i + 1
                })
            
            # Extract modifiers
            elif line.startswith('modifier '):
                modifier_match = re.match(r'^\s*modifier\s+(\w+)', line)
                modifier_name = modifier_match.group(1) if modifier_match else 'Unknown'
                
                # Find modifier end
                brace_count = 0
                modifier_lines = []
                for j in range(i, len(lines)):
                    modifier_lines.append(lines[j])
                    for char in lines[j]:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                context['modifiers'].append({
                                    'name': modifier_name,
                                    'code': '\n'.join(modifier_lines),
                                    'line': i + 1
                                })
                                i = j
                                break
                    if brace_count == 0:
                        break
            
            # Extract structs
            elif line.startswith('struct '):
                struct_match = re.match(r'^\s*struct\s+(\w+)', line)
                struct_name = struct_match.group(1) if struct_match else 'Unknown'
                
                brace_count = 0
                struct_lines = []
                for j in range(i, len(lines)):
                    struct_lines.append(lines[j])
                    for char in lines[j]:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                context['structs'].append({
                                    'name': struct_name,
                                    'code': '\n'.join(struct_lines),
                                    'line': i + 1
                                })
                                i = j
                                break
                    if brace_count == 0:
                        break
            
            # Extract enums
            elif line.startswith('enum '):
                enum_match = re.match(r'^\s*enum\s+(\w+)', line)
                enum_name = enum_match.group(1) if enum_match else 'Unknown'
                
                enum_lines = [lines[i]]
                temp_i = i
                while temp_i < len(lines) and '}' not in lines[temp_i]:
                    temp_i += 1
                    if temp_i < len(lines):
                        enum_lines.append(lines[temp_i])
                
                context['enums'].append({
                    'name': enum_name,
                    'code': '\n'.join(enum_lines),
                    'line': i + 1
                })
        
        i += 1
    
    return context


def extract_function_variables(func_code):
    """
    Extract variable names used in function code
    
    Returns:
        List of variable names
    """
    variables = set()
    
    # Match variable usage patterns (simple version)
    # Match identifier[...] or identifier.
    patterns = [
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\[',  # Array/mapping access
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.',  # Member access
        r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', # Assignment
        r'=\s*([a-zA-Z_][a-zA-Z0-9_]*)',   # Read
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, func_code)
        variables.update(matches)
    
    # Filter out keywords and common functions
    keywords = {
        'require', 'assert', 'revert', 'return', 'if', 'else', 'for', 'while',
        'msg', 'block', 'tx', 'this', 'super', 'true', 'false',
        'uint', 'uint256', 'int', 'address', 'bool', 'string', 'bytes',
        'public', 'private', 'internal', 'external', 'view', 'pure', 'payable'
    }
    
    return [v for v in variables if v not in keywords]


def extract_functions_from_solidity(code):
    """
    Extract all functions from Solidity code, preserving comments and function information
    Returns: List of functions, each containing name, signature, code block, starting line number, etc.
    """
    lines = code.split('\n')
    functions = []
    
    # Regular expression to match function definitions
    function_pattern = re.compile(
        r'^\s*(function\s+(\w+)\s*\([^)]*\)[^{;]*(?:\{|;))',
        re.MULTILINE
    )
    
    # Match special functions like constructor, fallback, receive
    special_function_pattern = re.compile(
        r'^\s*((constructor|fallback|receive)\s*\([^)]*\)[^{;]*(?:\{|;))',
        re.MULTILINE
    )
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is the start of a function definition
        func_match = re.match(r'^\s*function\s+(\w+)', line)
        special_match = re.match(r'^\s*(constructor|fallback|receive)\s*\(', line)
        
        if func_match or special_match:
            func_name = func_match.group(1) if func_match else special_match.group(1)
            start_line = i
            
            # Collect function signature (may span multiple lines)
            signature_lines = []
            temp_i = i
            brace_found = False
            semicolon_found = False
            
            while temp_i < len(lines):
                signature_lines.append(lines[temp_i])
                if '{' in lines[temp_i]:
                    brace_found = True
                    break
                if ';' in lines[temp_i]:  # Interface function or abstract function
                    semicolon_found = True
                    temp_i += 1
                    signature = '\n'.join(signature_lines)
                    functions.append({
                        'name': func_name,
                        'signature': signature.strip(),
                        'code': signature.strip(),
                        'start_line': start_line + 1,
                        'end_line': temp_i,
                        'is_abstract': True,
                        'called_functions': []
                    })
                    i = temp_i
                    break
                temp_i += 1
            
            # If neither { nor ; found, skip this function (prevent infinite loop)
            if not brace_found and not semicolon_found:
                i += 1
                continue
            
            if brace_found:
                # Find the end of function body (match braces)
                brace_count = 0
                func_start = temp_i
                matched = False
                
                for j in range(temp_i, len(lines)):
                    for char in lines[j]:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_line = j
                                
                                # Extract complete function code (including preceding comments)
                                comment_start = start_line
                                # Search upward for related comments
                                for k in range(start_line - 1, -1, -1):
                                    stripped = lines[k].strip()
                                    if stripped.startswith('//') or stripped.startswith('/*') or \
                                       stripped.startswith('*') or stripped.endswith('*/') or stripped == '':
                                        comment_start = k
                                    else:
                                        break
                                
                                # Extract complete code block
                                full_code = '\n'.join(lines[comment_start:end_line + 1])
                                func_body = '\n'.join(lines[start_line:end_line + 1])
                                signature = '\n'.join(signature_lines)
                                
                                # Extract function calls
                                called_functions = extract_function_calls(func_body)
                                
                                functions.append({
                                    'name': func_name,
                                    'signature': signature.strip(),
                                    'code': full_code,
                                    'body_only': func_body,
                                    'start_line': comment_start + 1,
                                    'end_line': end_line + 1,
                                    'is_abstract': False,
                                    'called_functions': called_functions
                                })
                                
                                i = end_line + 1
                                matched = True
                                break
                    if matched:
                        break
                
                # If brace matching failed, skip this line (prevent infinite loop)
                if not matched:
                    i += 1
        else:
            i += 1
    
    return functions


def extract_function_calls(code):
    """
    Extract all function calls from code
    Returns: List of called function names
    """
    # First remove function signature lines to avoid treating the function itself as a call
    # Match function definitions: function functionName(...) or constructor(...)
    lines = code.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip function definition lines
        if re.match(r'^\s*(function\s+\w+|constructor)\s*\(', line):
            continue
        filtered_lines.append(line)
    
    filtered_code = '\n'.join(filtered_lines)
    
    # Match function calls: functionName(...)
    call_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    calls = call_pattern.findall(filtered_code)
    
    # Filter out common keywords and type conversions
    keywords = {
        'if', 'for', 'while', 'require', 'assert', 'revert', 
        'uint', 'uint256', 'int', 'address', 'bool', 'string',
        'uint8', 'uint16', 'uint32', 'uint64', 'uint128',
        'int8', 'int16', 'int32', 'int64', 'int128', 'int256',
        'bytes', 'bytes1', 'bytes2', 'bytes4', 'bytes8', 'bytes16', 'bytes32'
    }
    
    # Deduplicate and filter
    unique_calls = []
    seen = set()
    for call in calls:
        if call not in keywords and call not in seen:
            unique_calls.append(call)
            seen.add(call)
    
    return unique_calls


def build_function_call_graph(functions):
    """
    Build function call dependency graph
    
    Args:
        functions: List of functions
    
    Returns:
        Dictionary with function names as keys and dependency information as values
    """
    func_map = {f['name']: f for f in functions}
    call_graph = {}
    
    for func in functions:
        func_name = func['name']
        call_graph[func_name] = {
            'function': func,
            'calls': [],  # Directly called functions
            'called_by': [],  # Which functions call this
            'all_dependencies': set()  # All dependencies (recursive)
        }
    
    # Establish call relationships
    for func in functions:
        func_name = func['name']
        for called in func['called_functions']:
            if called in call_graph:
                call_graph[func_name]['calls'].append(called)
                call_graph[called]['called_by'].append(func_name)
    
    # Recursively calculate all dependencies
    def get_all_dependencies(func_name, visited=None):
        if visited is None:
            visited = set()
        if func_name in visited or func_name not in call_graph:
            return set()
        
        visited.add(func_name)
        dependencies = set(call_graph[func_name]['calls'])
        
        for called in call_graph[func_name]['calls']:
            dependencies.update(get_all_dependencies(called, visited))
        
        return dependencies
    
    for func_name in call_graph:
        call_graph[func_name]['all_dependencies'] = get_all_dependencies(func_name)
    
    return call_graph


def get_function_with_dependencies(func_name, func_map, call_graph, max_depth=3):
    """
    Get function and all its dependent functions' code (recursive, with depth limit)
    
    Args:
        func_name: Function name
        func_map: Mapping from function names to function objects
        call_graph: Function call graph
        max_depth: Maximum recursion depth
    
    Returns:
        List containing main function and all dependent functions
    """
    result = []
    visited = set()
    
    def collect_dependencies(name, depth=0):
        if depth > max_depth or name in visited or name not in func_map:
            return
        
        visited.add(name)
        func = func_map[name]
        result.append({
            'name': name,
            'function': func,
            'depth': depth,
            'is_main': depth == 0
        })
        
        # Recursively collect called functions
        if name in call_graph:
            for called in call_graph[name]['calls']:
                collect_dependencies(called, depth + 1)
    
    collect_dependencies(func_name)
    return result


def query_function_with_context(func_context, call_graph, func_map):
    """
    Query vector database for function, including all functions it calls
    
    Args:
        func_context: Main function's context information
        call_graph: Function call graph
        func_map: Function mapping
    
    Returns:
        Vector retrieval results for all related functions
    """
    retrieval_results = {}
    
    # 1. Retrieve main function
    try:
        main_result = query_database(func_context['code'])
        retrieval_results['main_function'] = main_result
    except Exception as e:
        print(f"Query main function {func_context['name']} failed: {e}")
        retrieval_results['main_function'] = None
    
    # 2. Retrieve all called functions
    retrieval_results['called_functions'] = {}
    for called_func in func_context.get('called_functions_detail', []):
        if not called_func['is_main']:  # Skip main function itself
            try:
                called_result = query_database(called_func['function']['code'])
                retrieval_results['called_functions'][called_func['name']] = called_result
            except Exception as e:
                print(f"Query called function {called_func['name']} failed: {e}")
                retrieval_results['called_functions'][called_func['name']] = None
    
    return retrieval_results


def read_by_functions(code, query_db=True, include_call_graph=True, max_depth=2):
    """
    Slice code by function level and attach relevant context information to each function
    
    Args:
        code: Solidity source code
        query_db: Whether to query vector database for similar code
        include_call_graph: Whether to include function call graph information
        max_depth: Maximum depth for recursively retrieving called functions
    
    Returns:
        Generator that yields a dictionary containing function information each time
    """
    functions = extract_functions_from_solidity(code)
    
    # Build mapping from function names to function objects
    func_map = {f['name']: f for f in functions}
    
    # Build function call graph
    call_graph = build_function_call_graph(functions) if include_call_graph else {}
    
    for idx, func in enumerate(functions, 1):
        func_name = func['name']
        
        # Get function and all its dependencies
        dependencies = get_function_with_dependencies(func_name, func_map, call_graph, max_depth)
        
        # Build function context
        context = {
            'index': idx,
            'name': func['name'],
            'signature': func['signature'],
            'code': func['code'],
            'start_line': func['start_line'],
            'end_line': func['end_line'],
            'called_functions': func['called_functions'],
            'called_functions_detail': dependencies,  # Detailed dependency information
            'called_functions_code': []  # Maintain backward compatibility
        }
        
        # Add call graph information
        if include_call_graph and func_name in call_graph:
            context['call_graph_info'] = {
                'directly_calls': call_graph[func_name]['calls'],
                'called_by': call_graph[func_name]['called_by'],
                'all_dependencies': list(call_graph[func_name]['all_dependencies'])
            }
        
        # Build list of called functions' code (backward compatibility)
        for dep in dependencies:
            if not dep['is_main']:  # Exclude main function itself
                context['called_functions_code'].append({
                    'name': dep['name'],
                    'signature': dep['function']['signature'],
                    'code': dep['function']['code'],
                    'depth': dep['depth']
                })
        
        # If needed, retrieve similar code from vector database (including called functions)
        if query_db:
            try:
                # Retrieve main function and all called functions
                retrieval_results = query_function_with_context(context, call_graph, func_map)
                context['retrieval_results'] = retrieval_results
                # Maintain backward compatibility
                context['similar_vulnerable_code'] = retrieval_results.get('main_function')
            except Exception as e:
                print(f"Database query failed: {e}")
                context['similar_vulnerable_code'] = None
                context['retrieval_results'] = None
        
        yield context


def format_function_context(func_context, bad_text_start, bad_text_end, include_full_context=True):
    """
    Format function context for sending to GPT
    
    Args:
        func_context: Function context dictionary
        bad_text_start: Start marker for RAG retrieval results
        bad_text_end: End marker for RAG retrieval results
        include_full_context: Whether to include complete call chain and retrieval results
    
    Returns:
        Formatted string
    """
    content = f"### Function #{func_context['index']}: {func_context['name']}\n"
    content += f"Location: Lines {func_context['start_line']}-{func_context['end_line']}\n\n"
    
    # Add function call graph information (if available)
    if 'call_graph_info' in func_context:
        graph_info = func_context['call_graph_info']
        if graph_info['directly_calls']:
            content += f"**Directly calls:** {', '.join(graph_info['directly_calls'])}\n"
        if graph_info['called_by']:
            content += f"**Called by:** {', '.join(graph_info['called_by'])}\n"
        if graph_info['all_dependencies']:
            content += f"**All dependencies (recursive):** {', '.join(graph_info['all_dependencies'])}\n"
        content += "\n"
    
    # Main function code
    content += f"**Main Function Code:**\n```solidity\n{func_context['code']}\n```\n\n"
    
    # Add detailed information about called functions
    if include_full_context and func_context.get('called_functions_code'):
        content += "=" * 60 + "\n"
        content += "**Functions Called by This Function (with code):**\n"
        content += "=" * 60 + "\n\n"
        
        for called in func_context['called_functions_code']:
            depth_indent = "  " * called.get('depth', 1)
            content += f"{depth_indent}➤ Function: **{called['name']}** (Depth: {called.get('depth', 1)})\n"
            content += f"{depth_indent}  Signature: `{called['signature']}`\n"
            content += f"{depth_indent}  Code:\n"
            content += f"```solidity\n{called['code']}\n```\n\n"
    
    # Add similar vulnerable code retrieved from vector database
    if func_context.get('retrieval_results'):
        retrieval_results = func_context['retrieval_results']
        
        # Main function retrieval results
        if retrieval_results.get('main_function'):
            content += bad_text_start
            content += f"**Similar vulnerable patterns for main function '{func_context['name']}':**\n\n"
            content += retrieval_results['main_function']
            content += "\n"
        
        # Called functions retrieval results
        if include_full_context and retrieval_results.get('called_functions'):
            for func_name, similar_code in retrieval_results['called_functions'].items():
                if similar_code:
                    content += f"\n**Similar vulnerable patterns for called function '{func_name}':**\n\n"
                    content += similar_code
                    content += "\n"
        
        content += bad_text_end
    elif func_context.get('similar_vulnerable_code'):
        # Backward compatibility
        content += bad_text_start
        content += func_context['similar_vulnerable_code']
        content += bad_text_end
    
    return content


def visualize_call_graph(functions, output_file='call_graph.txt'):
    """
    Visualize function call graph, output as text format
    
    Args:
        functions: List of functions
        output_file: Output file path
    """
    call_graph = build_function_call_graph(functions)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Function Call Graph Visualization\n")
        f.write("=" * 80 + "\n\n")
        
        for func_name, info in call_graph.items():
            f.write(f"Function: {func_name}\n")
            f.write("-" * 60 + "\n")
            
            if info['calls']:
                f.write(f"  Calls: {', '.join(info['calls'])}\n")
            else:
                f.write(f"  Calls: (none)\n")
            
            if info['called_by']:
                f.write(f"  Called by: {', '.join(info['called_by'])}\n")
            else:
                f.write(f"  Called by: (none)\n")
            
            if info['all_dependencies']:
                f.write(f"  All dependencies: {', '.join(info['all_dependencies'])}\n")
            else:
                f.write(f"  All dependencies: (none)\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Call Graph Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total functions: {len(call_graph)}\n")
        
        # Find entry point functions (not called by any function)
        entry_points = [name for name, info in call_graph.items() if not info['called_by']]
        if entry_points:
            f.write(f"Entry point functions: {', '.join(entry_points)}\n")
        
        # Find most complex function (most dependencies)
        most_complex = max(call_graph.items(), key=lambda x: len(x[1]['all_dependencies']))
        f.write(f"Most complex function: {most_complex[0]} ")
        f.write(f"(depends on {len(most_complex[1]['all_dependencies'])} functions)\n")
    
    print(f"Call graph saved to {output_file}")


def save_comparison(original_code, sol_file, output_dir='comparison_output'):
    """
    Save original code and function-sliced code blocks for comparison
    
    Args:
        original_code: Original Solidity code
        sol_file: Source file path (used to generate output filename)
        output_dir: Output directory
    
    Returns:
        Dictionary of output file paths
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate base filename
    base_name = os.path.splitext(os.path.basename(sol_file))[0]
    
    # 1. Save original code
    original_file = os.path.join(output_dir, f"{base_name}_original.sol")
    with open(original_file, 'w', encoding='utf-8') as f:
        f.write("// ========================================\n")
        f.write("// Original Complete Code\n")
        f.write("// ========================================\n\n")
        f.write(original_code)
    
    # 2. Extract functions and save sliced code blocks
    functions = extract_functions_from_solidity(original_code)
    call_graph = build_function_call_graph(functions)
    
    # Save all function slices
    sliced_file = os.path.join(output_dir, f"{base_name}_sliced_functions.txt")
    with open(sliced_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Function Slicing Results\n")
        f.write(f"Original file: {sol_file}\n")
        f.write(f"Total functions: {len(functions)}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, func in enumerate(functions, 1):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Function #{idx}: {func['name']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Location: Lines {func['start_line']} - {func['end_line']}\n")
            f.write(f"Signature: {func['signature']}\n")
            
            if func['name'] in call_graph:
                info = call_graph[func['name']]
                f.write(f"Calls: {', '.join(info['calls']) if info['calls'] else '(none)'}\n")
                f.write(f"Called by: {', '.join(info['called_by']) if info['called_by'] else '(none)'}\n")
                f.write(f"All dependencies: {', '.join(info['all_dependencies']) if info['all_dependencies'] else '(none)'}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("Code block:\n")
            f.write("-" * 80 + "\n")
            f.write(func['code'])
            f.write("\n")
    
    # 3. Extract contract context information
    contract_context = extract_contract_context(original_code)
    
    # 4. Save comparison report (one file per function, including relevant context)
    functions_dir = os.path.join(output_dir, f"{base_name}_functions")
    if not os.path.exists(functions_dir):
        os.makedirs(functions_dir)
    
    for idx, func in enumerate(functions, 1):
        func_file = os.path.join(functions_dir, f"{idx:02d}_{func['name']}.sol")
        
        # Analyze variables used by function
        used_vars = extract_function_variables(func['code'])
        
        with open(func_file, 'w', encoding='utf-8') as f:
            f.write(f"// ========================================\n")
            f.write(f"// Function: {func['name']}\n")
            f.write(f"// Lines: {func['start_line']}-{func['end_line']}\n")
            f.write(f"// Signature: {func['signature']}\n")
            if func['name'] in call_graph:
                info = call_graph[func['name']]
                f.write(f"// Calls: {', '.join(info['calls']) if info['calls'] else '(none)'}\n")
            f.write(f"// ========================================\n\n")
            
            # Add pragma and import
            if contract_context['pragmas']:
                f.write("// Pragma declarations\n")
                for pragma in contract_context['pragmas']:
                    f.write(pragma + '\n')
                f.write('\n')
            
            if contract_context['imports']:
                f.write("// Import statements\n")
                for import_stmt in contract_context['imports']:
                    f.write(import_stmt + '\n')
                f.write('\n')
            
            # Add relevant state variables
            relevant_vars = []
            for var in contract_context['state_variables']:
                # Check if function uses this variable
                var_names = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', var['code'])
                if any(vn in used_vars for vn in var_names):
                    relevant_vars.append(var)
            
            if relevant_vars:
                f.write("// ========================================\n")
                f.write("// Relevant State Variables\n")
                f.write("// ========================================\n")
                for var in relevant_vars:
                    f.write(var['code'] + '\n')
                f.write('\n')
            
            # Add relevant event definitions
            # Check if events are triggered in function (emit or direct call)
            relevant_events = []
            for event in contract_context['events']:
                if event['name'] in func['code']:
                    relevant_events.append(event)
            
            if relevant_events:
                f.write("// ========================================\n")
                f.write("// Relevant Events\n")
                f.write("// ========================================\n")
                for event in relevant_events:
                    f.write(event['code'] + '\n')
                f.write('\n')
            
            # Add relevant modifiers
            relevant_modifiers = []
            for modifier in contract_context['modifiers']:
                if modifier['name'] in func['signature']:
                    relevant_modifiers.append(modifier)
            
            if relevant_modifiers:
                f.write("// ========================================\n")
                f.write("// Relevant Modifiers\n")
                f.write("// ========================================\n")
                for modifier in relevant_modifiers:
                    f.write(modifier['code'] + '\n\n')
            
            # Write function code
            f.write("// ========================================\n")
            f.write("// Function Code\n")
            f.write("// ========================================\n")
            f.write(func['code'])
    
    # 5. Save call graph
    call_graph_file = os.path.join(output_dir, f"{base_name}_call_graph.txt")
    visualize_call_graph(functions, call_graph_file)
    
    # 6. Generate comparison summary
    summary_file = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Code Slicing Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source file: {sol_file}\n")
        f.write(f"Contract name: {contract_context['contract_name']}\n")
        f.write(f"Original code lines: {len(original_code.split(chr(10)))}\n")
        f.write(f"Extracted functions: {len(functions)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Contract Context Information\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"State variables: {len(contract_context['state_variables'])}\n")
        f.write(f"Event definitions: {len(contract_context['events'])}\n")
        f.write(f"Modifiers: {len(contract_context['modifiers'])}\n")
        f.write(f"Structs: {len(contract_context['structs'])}\n")
        f.write(f"Enums: {len(contract_context['enums'])}\n\n")
        
        if contract_context['state_variables']:
            f.write("State variables list:\n")
            for idx, var in enumerate(contract_context['state_variables'], 1):
                var_preview = var['code'][:60].replace('\n', ' ')
                f.write(f"  {idx}. {var_preview}...\n")
            f.write('\n')
        
        if contract_context['events']:
            f.write("Events list:\n")
            for idx, event in enumerate(contract_context['events'], 1):
                f.write(f"  {idx}. {event['name']}\n")
            f.write('\n')
        
        if contract_context['modifiers']:
            f.write("Modifiers list:\n")
            for idx, modifier in enumerate(contract_context['modifiers'], 1):
                f.write(f"  {idx}. {modifier['name']}\n")
            f.write('\n')
        
        f.write("=" * 80 + "\n")
        f.write("Functions List\n")
        f.write("=" * 80 + "\n\n")
        for idx, func in enumerate(functions, 1):
            f.write(f"{idx}. {func['name']}\n")
            f.write(f"   Location: Lines {func['start_line']}-{func['end_line']}\n")
            if func['name'] in call_graph:
                info = call_graph[func['name']]
                if info['calls']:
                    f.write(f"   Calls: {', '.join(info['calls'])}\n")
                if info['called_by']:
                    f.write(f"   Called by: {', '.join(info['called_by'])}\n")
            f.write('\n')
        f.write(f"2. Function slicing summary: {sliced_file}\n")
        f.write(f"3. Individual function files directory: {functions_dir}\n")
        f.write(f"4. Call graph: {call_graph_file}\n")
        f.write(f"5. This summary file: {summary_file}\n")
    
    print(f"\n✅ Comparison files generated:")
    print(f"   📄 Original code: {original_file}")
    print(f"   📑 Function slicing summary: {sliced_file}")
    print(f"   📁 Individual function directory: {functions_dir} ({len(functions)} files with context)")
    print(f"   📊 Call graph: {call_graph_file}")
    print(f"   📋 Summary report: {summary_file}")
    print(f"\n💡 Context information:")
    print(f"   - State variables: {len(contract_context['state_variables'])}")
    print(f"   - Event definitions: {len(contract_context['events'])}")
    print(f"   - Modifiers: {len(contract_context['modifiers'])}")
    print(f"   ✅ Each function file automatically includes its used state variables and event definitions")
    
    return {
        'original': original_file,
        'sliced': sliced_file,
        'functions_dir': functions_dir,
        'call_graph': call_graph_file,
        'summary': summary_file
    }


def compare_slicing_methods(code, output_file='slicing_comparison.txt'):
    """
    Compare original code with results from different slicing methods
    
    Args:
        code: Solidity source code
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Code Slicing Methods Comparison\n")
        f.write("=" * 80 + "\n\n")
        
        # Original code statistics
        lines = code.split('\n')
        f.write(f"Original code total lines: {len(lines)}\n")
        f.write(f"Original code characters: {len(code)}\n\n")
        
        # Method 1: Line-based slicing (original method)
        f.write("=" * 80 + "\n")
        f.write("Method 1: Line-based slicing (chunk_size=20)\n")
        f.write("=" * 80 + "\n\n")
        
        chunks_by_line = list(read_by_chunks(lines, chunk_size=20))
        f.write(f"Slice count: {len(chunks_by_line)}\n")
        for idx, chunk in enumerate(chunks_by_line, 1):
            chunk_lines = len(chunk.split('\n'))
            f.write(f"  Chunk #{idx}: {chunk_lines} lines\n")
        
        # Method 2: Function-based slicing (new method)
        f.write("\n" + "=" * 80 + "\n")
        f.write("Method 2: Function-based slicing\n")
        f.write("=" * 80 + "\n\n")
        
        functions = extract_functions_from_solidity(code)
        f.write(f"Function count: {len(functions)}\n\n")
        
        for idx, func in enumerate(functions, 1):
            func_lines = func['end_line'] - func['start_line'] + 1
            f.write(f"  Function #{idx}: {func['name']}\n")
            f.write(f"    Lines: {func_lines}\n")
            f.write(f"    Location: {func['start_line']}-{func['end_line']}\n")
            f.write(f"    Calls functions: {', '.join(func['called_functions']) if func['called_functions'] else '(none)'}\n")
        
        # Comparison summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Line-based slicing:\n")
        f.write(f"  - Slice count: {len(chunks_by_line)}\n")
        f.write(f"  - Advantages: Simple, fixed size\n")
        f.write(f"  - Disadvantages: May break function integrity, lacks semantic information\n\n")
        
        f.write(f"Function-based slicing:\n")
        f.write(f"  - Slice count: {len(functions)}\n")
        f.write(f"  - Advantages: Maintains function integrity, includes semantics and call relationships\n")
        f.write(f"  - Disadvantages: Uneven function sizes\n")
    
    print(f"Slicing methods comparison saved to: {output_file}")



def ask_gpt_with_retries(content, session_history, max_retries=5):
    """
    使用 Xinference 本地模型 qwen3 的大模型调用函数
    完全替换原 GPT 调用函数
    """

    # === 保持 SCALM 所需的会话结构 ===
    session_history.append({"role": "user", "content": content})

    # === 你的本地 Xinference 设置 ===
    client = OpenAI(
        base_url="http://192.168.100.38:9997/v1",   # 完全按照你成功测试的写法
        api_key="xinference"                        # 任意字符串即可
    )

    retries = 0
    while retries < max_retries:
        try:
            # === 按你成功的结构调用 ===
            response = client.chat.completions.create(
                model="qwen3",
                messages=session_history,
                max_tokens=1024,
                temperature=0.4
            )

            answer = response.choices[0].message.content
            # 加入到会话历史
            session_history.append({"role": "assistant", "content": answer})

            return answer

        except Exception as e:
            print(f"[ask_gpt_with_retries] 调用失败：{e}")
            retries += 1
            time.sleep(2)

    return None



def save_session_history(session_history, sol_file):
    new_filename = os.path.splitext(os.path.basename(sol_file))[0] + '_session_history.json'
    session_history_path = os.path.join(save_session_path, new_filename)
    with open(session_history_path, 'w', encoding="utf-8") as f:
        json.dump(session_history, f, ensure_ascii=False, indent=2)
    # print("Session history saved.")
    return session_history_path


def merge_content(prompt, code, bad_text_start, bad_text_end):
    content = code + bad_text_start
    answers = query_database(code)
    # for answer in answers:
    #     content += answer
    content += answers
    content += bad_text_end
    # print(content)
    return content


def process_file_by_function(sol_file):
    """
    Process smart contract file using function-level slicing
    This is a new processing method that slices by function rather than by line count
    """
    session_history = []
    print(f"Processing {sol_file} with function-level slicing")

    try:
        with open(sol_file, 'r', encoding="utf-8") as f:
            code = f.read()

        prompt, bad_text_start, prompt2, prompt3, norminal_text, bad_text_end = prompt_txt()
        
        # Initialize prompt
        init_prompt = """I will provide you with a smart contract code divided by functions. 
Each function will be marked with its name, line numbers, and related context information.
Your task is to carefully record each function.
When I send "Code Entry Complete", reply with "All content recorded".
Please focus on accurate recording."""
        
        answer = ask_gpt_with_retries(init_prompt, session_history)
        
        # Slice by function level and send to GPT
        function_contexts = list(read_by_functions(code, query_db=True))
        
        for func_context in function_contexts:
            content = format_function_context(func_context, bad_text_start, bad_text_end)
            session_history.append({"role": "user", "content": content})
        
        # Notify code input complete
        answer = ask_gpt_with_retries("Code Entry Complete", session_history)
        
        # Perform security audit for each function
        for func_context in function_contexts:
            audit_prompt = f"""Based on the fundamental principles of smart contract security, 
does the function '{func_context['name']}' (lines {func_context['start_line']}-{func_context['end_line']}) 
contain any bad practices or vulnerabilities?

Consider:
1. The function's own code
2. Functions it calls: {', '.join(func_context['called_functions']) if func_context['called_functions'] else 'None'}
3. Similar vulnerable patterns from the database

Be careful not to audit the database examples themselves."""
            
            ask_gpt_with_retries(audit_prompt, session_history)
        
        # Finally generate audit report
        final_prompt = norminal_text + """\n
Please generate a comprehensive JSON list of all bad practices and vulnerabilities found.
The JSON should contain: bad_practice_id, title, type, bad_practice_code_block, risk_level(1-5), 
description, recommendation."""
        
        ask_gpt_with_retries(final_prompt, session_history)
        
        # Save session history
        session_history_file = save_session_history(session_history, sol_file)
        
        # Generate audit report
        modified_json_data = add_line(session_history_file, code)
        report = os.path.splitext(os.path.basename(sol_file))[0] + '_Audit report.txt'
        report_file_path = os.path.join(save_audit_path, report)
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(modified_json_data)
        
        print(f"{sol_file} has been processed successfully with function-level analysis.")
        
    except Exception as e:
        print(f"Error processing file {sol_file}: {e}")
        import traceback
        traceback.print_exc()

# Main function
def process_file(sol_file):
    session_history = []
    print(f"Processing {sol_file}")

    try:
        with open(sol_file, 'r', encoding="utf-8") as f:
            code = f.read()

        prompt, bad_text_start, prompt2, prompt3, norminal_text, bad_text_end = prompt_txt()
        code_lines = code.splitlines()  # Split string into lines
        # print(prompt2)
        answer = ask_gpt_with_retries(prompt2, session_history)
        for chunk in read_by_chunks(code_lines):
            content = merge_content('', chunk, bad_text_start, bad_text_end)
            session_history.append({"role": "user", "content": content})
        answer = ask_gpt_with_retries("Code Entry Complete", session_history)

        for index, content in enumerate(read_by_chunks(code_lines)):
            if index == 0:
                ask_gpt_with_retries(norminal_text + f'\nBased on the fundamental principles of smart contract security, does the  smart contract of #{index+1} contain any bad practices? Does it contain SWC-116	Block Values as a Proxy for Time vulnerabilities?\nBe careful not to audit the database for similar bad practice codes.', session_history)
            else:
                ask_gpt_with_retries(
                    f'Based on the fundamental principles of smart contract security, does the  smart contract of #{index+1} contain any bad practices? Does it contain any vulnerabilities?', session_history)

        session_history_file = save_session_history(session_history, sol_file)

        modified_json_data = add_line(session_history_file, code)
        report = os.path.splitext(os.path.basename(sol_file))[0] + '_Audit report.txt'
        report_file_path = os.path.join(save_audit_path, report)
        with open(os.path.join(report_file_path), "w", encoding="utf-8") as f:
            f.write(modified_json_data)
        print(f"{sol_file} has been processed.")
        
    except Exception as e:
        print(f"Error processing file {sol_file}: {e}")
        import traceback
        traceback.print_exc()


# Without segmentation
def process_file2(sol_file):
    session_history = []
    # print(f"Processing {sol_file}")
    try:
        with open(sol_file, 'r', encoding="utf=8") as f:
            code = f.read()

        prompt, bad_text_start, prompt2, prompt3, norminal_text, bad_text_end = prompt_txt()
        # print(prompt2)
        answer = ask_gpt_with_retries(prompt3, session_history)
        session_history.append({"role": "user", "content": code})
            # answer = ask_gpt_with_retries(content, session_history)
        answer = ask_gpt_with_retries("Code Entry Complete", session_history)
        # print(session_history)

        answer = ask_gpt_with_retries(norminal_text + f'\nBased on the fundamental principles of smart contract security, does the code of smart contract contain any bad practices? \nBe careful not to audit the database for similar bad practice codes.', session_history)

        session_history_file = save_session_history(session_history, sol_file)

        modified_json_data = add_line(session_history_file, code)
        report = os.path.splitext(os.path.basename(sol_file))[0] + '_Audit report.txt'
        report_file_path = os.path.join(save_audit_path, report)
        with open(os.path.join(report_file_path), "w", encoding="utf-8") as f:
            f.write(modified_json_data)
        print(f"{sol_file} has been processed.")
    except Exception as e:
        print(f"Error processing file {sol_file}: {e}")
def main():
    # method 1
    sol_files = [
        r"C:\Users\33519\Desktop\exp\latest\*.sol"]

    with multiprocessing.Pool() as pool:
        pool.map(process_file, sol_files)



    #method 2
    # for file in sol_files:
    #     process_file(file)
        # time.sleep(120)  # Wait 60 seconds

    # method 3
    # input_file_path = r'C:\Users\33519\Desktop\exp\no_RAG\tp\no_keyword_files.txt'
    # sol_files = []
    # # Read file containing converted paths
    # with open(input_file_path, 'r') as file:
    #     for line in file:
    #         # Remove newline at end of line
    #         sol_path = line.strip()
    #         # Add path to list
    #         sol_files.append(sol_path)

    # Process each file path
    with multiprocessing.Pool() as pool:
        pool.map(process_file, sol_files)


    # Open and process files (e.g., read content)
    print("All files processed.")

if __name__ == '__main__':
    main()





