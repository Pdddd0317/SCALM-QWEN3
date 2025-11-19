"""
Build Vector Database Using Function-Level Slicing
Contains complete function call relationships (up to 3 levels of recursion)

Core Features:
1. Extract functions containing SWC annotations from Solidity code
2. Extract called function code for each function (3 levels of recursion)
3. Store function + call context as a whole in vector database
4. Support returning complete function context during retrieval

New Features:
- Only extract functions containing SWC annotation markers (e.g., // SWC-Code With No Effects: L302-307)
- Automatically identify and record SWC vulnerability types in each function
- Save SWC annotation information in metadata for subsequent analysis
"""

import os
import glob
import json
import shutil
from Rewrite_query import (
    extract_functions_from_solidity,
    extract_contract_context,
    build_function_call_graph,
    get_function_with_dependencies,
    extract_function_variables
)
from deep_lake import embedding_function as api_embedding_function
from deeplake.core.vectorstore import VectorStore


# ============================================
# Configuration
# ============================================
VECTOR_STORE_PATH = r'database\DApp_func_level_SWC'  # New function-level vector database
MAX_RECURSION_DEPTH = 3  # Maximum recursion depth of 3 layers

# Embedding model configuration
USE_LOCAL_EMBEDDING = True  # Set to True to use local model
LOCAL_MODEL_NAME = 'all-MiniLM-L6-v2'  # Local model name (consistent with database)
LOCAL_DEVICE = 'cuda'  # Default to use CUDA (GPU), can change to 'cpu' if no GPU

# Global embedding function and thread lock
import threading
_embedding_function = None
_embedding_lock = threading.Lock()


# ============================================
# Embedding Function Selection
# ============================================

def get_embedding_function(use_local=None, model_name=None, device=None):
    """
    Get embedding function (local or API, thread-safe)
    
    Args:
        use_local: Whether to use local model (uses global config when None)
        model_name: Local model name (uses global config when None)
        device: Device selection 'cuda' or 'cpu' (uses global config when None)
    
    Returns:
        Embedding function
    """
    global _embedding_function
    
    # Use global configuration
    if use_local is None:
        use_local = USE_LOCAL_EMBEDDING
    if model_name is None:
        model_name = LOCAL_MODEL_NAME
    if device is None:
        device = LOCAL_DEVICE
    
    # Double-checked locking (thread-safe)
    if _embedding_function is not None:
        return _embedding_function
    
    with _embedding_lock:
        # Check again to prevent multiple threads from entering simultaneously
        if _embedding_function is not None:
            return _embedding_function
        
        # Initialize based on configuration (within lock protection)
        if use_local:
            try:
                from local_embedding import get_local_embedding_model
                print(f"✓ Using local embedding model: {model_name}")
                print(f"✓ Device: {device.upper()}")
                
                # Load model with specified device
                model = get_local_embedding_model(model_name=model_name, device=device)
                _embedding_function = lambda texts: model.embed(texts, show_progress=False)
            except ImportError:
                print("⚠️  local_embedding module not found, falling back to API")
                _embedding_function = api_embedding_function
            except Exception as e:
                print(f"⚠️  Local model loading failed: {e}")
                print("   Falling back to API")
                _embedding_function = api_embedding_function
        else:
            print("✓ Using OpenAI API embedding")
            _embedding_function = api_embedding_function
    
    return _embedding_function


# ============================================
# SWC Annotation Extraction and Filtering
# ============================================

def extract_swc_annotations(code):
    """
    Extract SWC annotations and their line numbers from code
    
    Args:
        code: Complete Solidity code
    
    Returns:
        List of [(line_number, SWC_description), ...]
    
    Example:
        // SWC-Code With No Effects: L302-307
        // SWC-Integer Overflow and Underflow: L2-308
    """
    import re
    # Regular expression to match SWC annotations
    swc_pattern = r'//\s*SWC[^\n]+'
    
    lines = code.split('\n')
    swc_annotations = []
    
    for i, line in enumerate(lines, 1):
        if re.search(swc_pattern, line, re.IGNORECASE):
            swc_annotations.append((i, line.strip()))
    
    return swc_annotations


def find_function_containing_line(functions, line_number):
    """
    Find the function containing the specified line number
    
    Args:
        functions: List of functions
        line_number: Target line number
    
    Returns:
        Function object containing that line number, or None if not found
    """
    for func in functions:
        if func['start_line'] <= line_number <= func['end_line']:
            return func
    return None


def filter_functions_with_swc(code, functions):
    """
    Filter out functions containing SWC annotations (function-level only)
    
    Args:
        code: Complete Solidity code
        functions: List of all extracted functions
    
    Returns:
        List of functions that contain SWC markers inside the function body.
        (File-level SWC will be completely ignored)
    """
    swc_annotations = extract_swc_annotations(code)

    if not swc_annotations:
        return []

    swc_functions = {}  # name -> {function, swc_annotations}

    for line_num, annotation in swc_annotations:
        func = find_function_containing_line(functions, line_num)

        # -----------------------------
        # ① 情况一：SWC 在函数体内 → 保留
        # -----------------------------
        if func:
            func_name = func['name']
            if func_name not in swc_functions:
                swc_functions[func_name] = {
                    'function': func,
                    'swc_annotations': []
                }
            swc_functions[func_name]['swc_annotations'].append({
                'line': line_num,
                'annotation': annotation
            })

        # -----------------------------
        # ② 情况二：SWC 不在任何函数内 → 直接跳过
        # -----------------------------
        else:
            # skip file-level SWC silently
            continue

    # -----------------------------
    # 收集最终结果
    # -----------------------------
    result = []
    for func_name, data in swc_functions.items():
        func = data['function']
        func['swc_info'] = data['swc_annotations']  # attach SWC info
        result.append(func)

    return result



def extract_swc_type_from_annotation(annotation):
    """
    Extract vulnerability type from SWC annotation
    
    Example:
        "// SWC-Code With No Effects: L302-307" -> "SWC-Code_With_No_Effects"
    """
    import re
    match = re.search(r'SWC-([^:]+)', annotation, re.IGNORECASE)
    if match:
        swc_type = match.group(1).strip()
        # Replace spaces with underscores
        swc_type = swc_type.replace(' ', '_')
        return f"SWC-{swc_type}"
    return "Unknown"


# ============================================
# Core Functionality: Build Function Context
# ============================================

def build_function_context_for_storage(func, func_map, call_graph, contract_context, max_depth=3):
    """
    Build complete context information for function (for storage in database)
    
    Args:
        func: Main function information
        func_map: Mapping from function names to function objects
        call_graph: Function call graph
        contract_context: Contract context (state variables, events, etc.)
        max_depth: Maximum recursion depth
    
    Returns:
        Complete function context string
    """
    context_parts = []
    
    # 1. Add function basic information
    context_parts.append(f"// ========================================")
    context_parts.append(f"// Function: {func['name']}")
    context_parts.append(f"// Lines: {func['start_line']}-{func['end_line']}")
    
    # Add SWC annotation information (if any)
    if 'swc_info' in func and func['swc_info']:
        context_parts.append(f"// SWC Vulnerabilities Found:")
        for swc_item in func['swc_info']:
            swc_type = extract_swc_type_from_annotation(swc_item['annotation'])
            context_parts.append(f"//   - {swc_type} at line {swc_item['line']}")
            context_parts.append(f"//     {swc_item['annotation']}")
    
    context_parts.append(f"// ========================================\n")
    
    # 2. Add pragma declarations
    if contract_context.get('pragmas'):
        context_parts.append("// Pragma declarations")
        for pragma in contract_context['pragmas']:
            context_parts.append(pragma)
        context_parts.append("")
    
    # 3. Add relevant state variables
    used_vars = extract_function_variables(func['code'])
    relevant_vars = []
    for var in contract_context.get('state_variables', []):
        var_names = var['code'].split()
        if any(vn in used_vars for vn in var_names):
            relevant_vars.append(var)
    
    if relevant_vars:
        context_parts.append("// ========================================")
        context_parts.append("// State Variables (used by this function)")
        context_parts.append("// ========================================")
        for var in relevant_vars:
            context_parts.append(var['code'])
        context_parts.append("")
    
    # 4. Add relevant event definitions
    relevant_events = []
    for event in contract_context.get('events', []):
        if event['name'] in func['code']:
            relevant_events.append(event)
    
    if relevant_events:
        context_parts.append("// ========================================")
        context_parts.append("// Events (triggered by this function)")
        context_parts.append("// ========================================")
        for event in relevant_events:
            context_parts.append(event['code'])
        context_parts.append("")
    
    # 5. Add main function code
    context_parts.append("// ========================================")
    context_parts.append("// Main Function Code")
    context_parts.append("// ========================================")
    context_parts.append(func['code'])
    context_parts.append("")
    
    # 6. Recursively add called function code (up to 3 layers)
    if func['name'] in call_graph:
        dependencies = get_function_with_dependencies(
            func['name'], func_map, call_graph, max_depth=max_depth
        )
        
        # Exclude main function itself, only include called functions
        called_funcs = [dep for dep in dependencies if not dep['is_main']]
        
        if called_funcs:
            context_parts.append("// ========================================")
            context_parts.append("// Called Functions (with full code)")
            context_parts.append("// ========================================")
            
            for dep in called_funcs:
                depth_indent = "  " * dep['depth']
                context_parts.append(f"\n{depth_indent}// Function: {dep['name']} (Depth: {dep['depth']})")
                context_parts.append(f"{depth_indent}// Called by: {func['name']}")
                context_parts.append(dep['function']['code'])
                context_parts.append("")
    
    return '\n'.join(context_parts)


def create_function_level_database(vulnerability_files, vector_store_path=VECTOR_STORE_PATH, 
                                   max_retries=None, retry_delay=10, quota_retry_delay=60,
                                   use_local_embedding=None, local_model_name=None, device=None):
    """
    Build vector database using function-level slicing
    
    Args:
        vulnerability_files: List of vulnerability code files (.sol file paths)
        vector_store_path: Vector database storage path
        max_retries: Maximum retry count for single function (None means infinite retries)
        retry_delay: Normal error retry delay (seconds)
        quota_retry_delay: Quota error retry delay (seconds, recommended 60-120)
        use_local_embedding: Whether to use local embedding model (uses global config when None)
        local_model_name: Local model name (uses global config when None)
        device: Device selection 'cuda' or 'cpu' (uses global config when None)
    """
    import time
    
    print("=" * 80)
    print("Building Vector Database Using Function-Level Slicing")
    print("=" * 80)
    print(f"Vector database path: {vector_store_path}")
    print(f"Maximum recursion depth: {MAX_RECURSION_DEPTH}")
    print(f"Files to process: {len(vulnerability_files)}")
    print(f"Maximum retries: {'Infinite' if max_retries is None else max_retries}")
    print(f"Normal error retry delay: {retry_delay}s")
    print(f"Quota error retry delay: {quota_retry_delay}s")
    
    # Get embedding function
    embedding_func = get_embedding_function(use_local_embedding, local_model_name, device)
    
    print("=" * 80 + "\n")
    
    # Create or load vector database (with corruption recovery)
    try:
        vector_store = VectorStore(path=vector_store_path)
        print("✓ Successfully loaded existing database\n")
    except Exception as e:
        error_msg = str(e).lower()
        if 'corrupt' in error_msg or 'integrity' in error_msg:
            print("⚠️  Database corruption detected, attempting recovery with reset=True...")
            try:
                import deeplake
                # First try loading with reset=True
                dataset = deeplake.load(vector_store_path, reset=True)
                vector_store = VectorStore(path=vector_store_path)
                print("✓ Successfully recovered database\n")
            except Exception as reset_error:
                print(f"❌ Recovery failed: {reset_error}")
                print("\nRecommended actions:")
                print(f"1. Delete corrupted database: {vector_store_path}")
                print("2. Re-run this script to create new database")
                
                user_choice = input("\nDelete corrupted database and recreate? (y/n): ").strip().lower()
                if user_choice == 'y':
                    if os.path.exists(vector_store_path):
                        shutil.rmtree(vector_store_path)
                        print(f"✓ Deleted: {vector_store_path}")
                    vector_store = VectorStore(path=vector_store_path)
                    print("✓ Created new database\n")
                else:
                    print("Operation cancelled")
                    return None
        else:
            raise
    
    total_functions = 0
    processed_files = 0
    failed_functions = []  # Record failed functions
    
    for file_idx, file_path in enumerate(vulnerability_files, 1):
            
        try:
            print(f"\n[{file_idx}/{len(vulnerability_files)}] Processing file: {file_path}")
            
            # Read code
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Extract functions and context
            functions = extract_functions_from_solidity(code)
            contract_context = extract_contract_context(code)
            
            if not functions:
                print(f"  ⚠️  No functions found, skipping")
                continue
            
            print(f"  ✓ Found {len(functions)} functions")
            
            # ===== New: Filter out functions containing SWC annotations =====
            swc_functions = filter_functions_with_swc(code, functions)
            
            if not swc_functions:
                print(f"  ⚠️  No functions with SWC annotations found, skipping")
                continue
            
            print(f"  ✓ Found {len(swc_functions)} functions with SWC annotations")
            
            # Use filtered function list
            functions = swc_functions
            
            # Build function name mapping and call graph
            func_map = {f['name']: f for f in functions}
            call_graph = build_function_call_graph(functions)
            
            # Build complete context for each function and store
            for func in functions:
                success = False
                last_error = None
                attempt = 0
                
                # Retry mechanism (supports infinite retries)
                while not success:
                    try:
                        # Build complete context for function (including called functions)
                        full_context = build_function_context_for_storage(
                            func, func_map, call_graph, contract_context, 
                            max_depth=MAX_RECURSION_DEPTH
                        )
                        
                        # Extract vulnerability type (from filename or path)
                        vuln_type = extract_vulnerability_type(file_path)
                        
                        # Extract more precise vulnerability types from SWC annotations
                        swc_types = []
                        if 'swc_info' in func:
                            for swc_item in func['swc_info']:
                                swc_type = extract_swc_type_from_annotation(swc_item['annotation'])
                                swc_types.append(swc_type)
                        
                        # Prepare metadata
                        metadata = {
                            "source_file": file_path,
                            "contract_name": contract_context.get('contract_name', 'Unknown'),
                            "function_name": func['name'],
                            "start_line": func['start_line'],
                            "end_line": func['end_line'],
                            "vulnerability_type": vuln_type,
                            "swc_annotations": json.dumps(func.get('swc_info', [])),
                            "swc_types": json.dumps(swc_types),
                            "has_dependencies": len(func['called_functions']) > 0,
                            "called_functions": json.dumps(func['called_functions']),
                            "is_abstract": func.get('is_abstract', False)
                        }
                        
                        # Store to vector database
                        vector_store.add(
                            text=[full_context],
                            embedding_function=embedding_func,
                            embedding_data=[full_context],
                            metadata=[metadata]
                        )
                        
                        # Prepare SWC information for display
                        swc_display = ""
                        if 'swc_info' in func and func['swc_info']:
                            swc_count = len(func['swc_info'])
                            swc_display = f" [SWC: {swc_count} vulnerabilities]"
                        
                        print(f"    ✓ {func['name']}{swc_display} (Calls: {func['called_functions'] if func['called_functions'] else 'none'})")
                        total_functions += 1
                        success = True
                        break  # Success, exit retry loop
                        
                    except Exception as func_error:
                        last_error = func_error
                        error_msg = str(func_error).lower()
                        attempt += 1
                        
                        # Check if maximum retry count exceeded (if set)
                        if max_retries is not None and attempt >= max_retries:
                            print(f"    ❌ {func['name']} - Maximum retry count ({max_retries}) reached, giving up")
                            failed_functions.append({
                                'file': file_path,
                                'function': func['name'],
                                'error': str(last_error)[:100] if last_error else 'Unknown'
                            })
                            break
                        
                        # Check if quota error - wait longer before retrying
                        if "quota" in error_msg or "403" in error_msg or "insufficient" in error_msg:
                            retry_label = f"({attempt}/{max_retries})" if max_retries else f"(attempt {attempt})"
                            print(f"    ⚠️  {func['name']} - API quota insufficient, retrying in {quota_retry_delay}s {retry_label}...")
                            time.sleep(quota_retry_delay)
                            continue
                        
                        # Check if data format error
                        elif "arrays to stack" in error_msg or "sequence" in error_msg:
                            retry_label = f"({attempt}/{max_retries})" if max_retries else f"(attempt {attempt})"
                            print(f"    ⚠️  {func['name']} - Data format error, retrying in {retry_delay}s {retry_label}...")
                            time.sleep(retry_delay)
                            continue
                        
                        # Check if network error
                        elif "timeout" in error_msg or "connection" in error_msg:
                            retry_label = f"({attempt}/{max_retries})" if max_retries else f"(attempt {attempt})"
                            print(f"    ⚠️  {func['name']} - Network error, retrying in {retry_delay}s {retry_label}...")
                            time.sleep(retry_delay)
                            continue
                        
                        # Other errors - incremental wait time
                        else:
                            wait_time = min(retry_delay * (attempt), 60)  # Maximum wait 60 seconds
                            retry_label = f"({attempt}/{max_retries})" if max_retries else f"(attempt {attempt})"
                            print(f"    ⚠️  {func['name']} - Error: {str(func_error)[:50]}")
                            print(f"         Retrying in {wait_time}s {retry_label}...")
                            time.sleep(wait_time)
                            continue
            
            processed_files += 1
            
        except Exception as e:
            print(f"  ❌ File processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "=" * 80)
    print("Database Build Complete!")
    print("=" * 80)
    print(f"✓ Successfully processed files: {processed_files}/{len(vulnerability_files)}")
    print(f"✓ Successfully processed functions: {total_functions}")
    print(f"✓ Failed functions: {len(failed_functions)}")
    print(f"✓ Database path: {vector_store_path}")
    
    # If there are failed functions, save failure list
    if failed_functions:
        failed_log = os.path.join(os.path.dirname(vector_store_path) or '.', 'failed_functions.json')
        with open(failed_log, 'w', encoding='utf-8') as f:
            json.dump(failed_functions, f, indent=2, ensure_ascii=False)
        print(f"⚠️  Failed functions list saved: {failed_log}")
        print(f"   Can use resume_database_creation() to continue processing failed functions")
    
    print("=" * 80)
    
    return vector_store


def extract_vulnerability_type(file_path):
    """
    Extract vulnerability type from file path
    
    Examples:
        "vulnerabilities/SWC-107_reentrancy.sol" -> "SWC-107"
        "extracted_SWCs/overflow_vulnerability.sol" -> "Overflow"
    """
    filename = os.path.basename(file_path).lower()
    
    # Try to match SWC number
    import re
    swc_match = re.search(r'swc[-_]?(\d+)', filename)
    if swc_match:
        return f"SWC-{swc_match.group(1)}"
    
    # Try to match common vulnerability keywords
    vulnerability_keywords = {
        'reentrancy': 'SWC-107',
        'overflow': 'SWC-101',
        'underflow': 'SWC-101',
        'timestamp': 'SWC-116',
        'delegatecall': 'SWC-112',
        'unchecked': 'SWC-104',
        'tx.origin': 'SWC-115',
        'selfdestruct': 'SWC-106',
    }
    
    for keyword, swc in vulnerability_keywords.items():
        if keyword in filename:
            return swc
    
    return 'Unknown'


# ============================================
# Retrieval Functionality: Enhanced Version
# ============================================

def query_function_level_database(query_code, vector_store_path=VECTOR_STORE_PATH, top_k=2, 
                                  use_local_embedding=None, device=None):
    """
    Retrieve similar code from function-level vector database
    
    Args:
        query_code: Code to retrieve
        vector_store_path: Vector database path
        top_k: Return top k most similar results
        use_local_embedding: Whether to use local embedding model
        device: Device selection 'cuda' or 'cpu'
    
    Returns:
        List of retrieval results, each containing code and metadata
    """
    vector_store = VectorStore(path=vector_store_path)
    
    # Get embedding function
    embedding_func = get_embedding_function(use_local_embedding, device=device)
    
    # Execute retrieval
    search_results = vector_store.search(
        embedding_data=query_code,
        embedding_function=embedding_func,
        k=top_k
    )
    
    results = []
    for i in range(min(top_k, len(search_results['text']))):
        result = {
            'text': search_results['text'][i],
            'metadata': search_results['metadata'][i] if 'metadata' in search_results else {},
            'score': search_results.get('score', [None])[i] if 'score' in search_results else None
        }
        results.append(result)
    
    return results


def enhanced_query_with_context(func_context, vector_store_path=VECTOR_STORE_PATH):
    """
    Enhanced retrieval: Retrieve for both function and its called functions
    
    Args:
        func_context: Function context (from read_by_functions)
        vector_store_path: Vector database path
    
    Returns:
        Complete retrieval results, including main function and called functions retrieval results
    
    Note:
    - Main function retrieval results already include functions it calls (from same contract)
    - No longer retrieve called functions separately to avoid confusion with functions from different contracts
    """
    retrieval_results = {
        'main_function': None,
        'called_functions': {},
        'main_metadata': None,
        'called_metadata': {}
    }
    
    # 1. Retrieve main function (including complete context: main function + called functions)
    try:
        main_results = query_function_level_database(
            func_context['code'], 
            vector_store_path, 
            top_k=1
        )
        if main_results:
            retrieval_results['main_function'] = main_results[0]['text']
            retrieval_results['main_metadata'] = main_results[0]['metadata']
            
            # Note: main_function already contains complete call context
            # Database storage format is:
            # - Main Function Code
            # - Called Functions (with full code)
            # Therefore no need to retrieve called functions separately
            
    except Exception as e:
        print(f"Query main function failed: {e}")
    
    # 2. No longer retrieve called functions separately to avoid confusion
    # Reason:
    # - Main function retrieval results already include called functions from same contract
    # - Separate retrieval might return same-name functions from other contracts, causing context confusion
    # 
    # If need to view similar examples for called functions too:
    # - Clearly mark this as "independent reference example"
    # - Or only supplement retrieval when main function retrieval results are insufficient
    
    return retrieval_results


# ============================================
# Main Function and Utility Functions
# ============================================

def collect_vulnerability_files(base_dir, pattern='**/*.sol'):
    """
    Collect vulnerability code files
    
    Args:
        base_dir: Base directory
        pattern: File matching pattern
    
    Returns:
        List of file paths
    """
    files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    return files


def resume_database_creation(base_dir, vector_store_path=VECTOR_STORE_PATH, 
                            failed_log='database/failed_functions.json'):
    """
    Resume and continue building database from failure log
    
    Args:
        base_dir: Vulnerability code base directory
        vector_store_path: Vector database path
        failed_log: Failed functions log path
    """
    print("\n" + "=" * 80)
    print("Resuming Database Build from Failure Log")
    print("=" * 80 + "\n")
    
    # Read failure log
    if not os.path.exists(failed_log):
        print(f"❌ Failure log does not exist: {failed_log}")
        print("   Please run create_function_level_database() first")
        return None
    
    with open(failed_log, 'r', encoding='utf-8') as f:
        failed_functions = json.load(f)
    
    print(f"✓ Read {len(failed_functions)} failed functions")
    
    # Group by file
    files_to_retry = {}
    for item in failed_functions:
        file_path = item['file']
        if file_path not in files_to_retry:
            files_to_retry[file_path] = []
        files_to_retry[file_path].append(item['function'])
    
    print(f"✓ Involves {len(files_to_retry)} files\n")
    
    # Reprocess these files
    file_list = list(files_to_retry.keys())
    
    print("Starting reprocessing...")
    vector_store = create_function_level_database(
        file_list, 
        vector_store_path=vector_store_path,
        max_retries=None,  # Infinite retries
        retry_delay=10,
        quota_retry_delay=60,
        device='cuda'  # Default to use CUDA
    )
    
    return vector_store


def save_progress(processed_files, total_files, vector_store_path):
    """
    Save processing progress
    
    Args:
        processed_files: List of processed files
        total_files: List of total files
        vector_store_path: Database path
    """
    progress_file = os.path.join(os.path.dirname(vector_store_path) or '.', 'progress.json')
    progress = {
        'processed': processed_files,
        'total': total_files,
        'remaining': [f for f in total_files if f not in processed_files],
        'progress_percent': len(processed_files) / len(total_files) * 100 if total_files else 0
    }
    
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Progress saved: {progress_file}")


def test_query(test_code, vector_store_path=VECTOR_STORE_PATH):
    """
    Test retrieval functionality
    
    Args:
        test_code: Test code
        vector_store_path: Vector database path
    """
    print("\n" + "=" * 80)
    print("Testing Retrieval Functionality")
    print("=" * 80)
    print(f"\nCode to retrieve:\n{test_code[:200]}...\n")
    
    results = query_function_level_database(test_code, vector_store_path, top_k=3)
    
    print(f"Found {len(results)} similar results:\n")
    
    for idx, result in enumerate(results, 1):
        print(f"{'='*80}")
        print(f"Result #{idx}")
        print(f"{'='*80}")
        
        metadata = result.get('metadata', {})
        print(f"Function name: {metadata.get('function_name', 'Unknown')}")
        print(f"Source: {metadata.get('source_file', 'Unknown')}")
        print(f"Vulnerability type: {metadata.get('vulnerability_type', 'Unknown')}")
        print(f"Called functions: {metadata.get('called_functions', '[]')}")
        print(f"Similarity score: {result.get('score', 'N/A')}")
        print(f"\nCode:\n{result['text'][:300]}...")
        print()


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("Function-Level Vector Database Build Tool")
    print("=" * 80 + "\n")
    
    # Select operation
    print("Please select an operation:")
    print("1. Build new database")
    print("2. Resume from failure log")
    print("3. Test retrieval functionality")
    print("4. View database statistics")
    
    choice = input("\nPlease enter your choice (1/2/3/4): ").strip()
    
    if choice == '1':
        # Build database
        print("\nPlease provide vulnerability code file path:")
        print("Example: C:\\Users\\33519\\Desktop\\DAppSCAN\\extracted_SWCs")
        
        # base_dir = input("Directory path: ").strip()
        base_dir = r'D:\SCALM\SCALM-ALL\DAppSCAN-main\DAppSCAN-source'
        if not os.path.exists(base_dir):
            print(f"❌ Directory does not exist: {base_dir}")
            return
        
        # Collect files
        print("\nCollecting files...")
        vuln_files = collect_vulnerability_files(base_dir)
        print(f"✓ Found {len(vuln_files)} .sol files")
        
        if len(vuln_files) == 0:
            print("❌ No .sol files found")
            return
        
        # Display first 5 files
        print("\nFile examples:")
        for f in vuln_files[:5]:
            print(f"  - {f}")
        if len(vuln_files) > 5:
            print(f"  ... and {len(vuln_files) - 5} more files")
        
        print("\nEmbedding model selection:")
        print("1. OpenAI API (requires quota)")
        print("2. Local model (free, no quota limit)")
        
        embed_choice = input("\nPlease select embedding model (1/2, default 1): ").strip() or '1'
        
        if embed_choice == '2':
            use_local = True
            print("\nRecommended local models:")
            print("1. all-MiniLM-L6-v2 (Lightweight, 80MB, recommended)")
            print("2. all-mpnet-base-v2 (High quality, 420MB)")
            print("3. BAAI/bge-large-en (Highest quality, 1.3GB)")
            
            model_choice = input("\nPlease select model (1/2/3, default 3): ").strip() or '3'
            model_map = {
                '1': 'all-MiniLM-L6-v2',
                '2': 'all-mpnet-base-v2',
                '3': 'BAAI/bge-large-en'
            }
            local_model = model_map.get(model_choice, 'all-MiniLM-L6-v2')
            
            print(f"\nWill use local model: {local_model}")
            
            # Device selection
            print("\nDevice selection:")
            print("1. CUDA (GPU acceleration, recommended)")
            print("2. CPU")
            
            device_choice = input("\nPlease select device (1/2, default 1-CUDA): ").strip() or '1'
            if device_choice == '2':
                device = 'cpu'
                print("✓ Using CPU")
            else:
                device = 'cuda'
                print("✓ Using CUDA (GPU)")
            
            print("⚠️  First run will automatically download model, please be patient")
        else:
            use_local = False
            local_model = None
            device = None
        
        print("\nRetry strategy:")
        print("1. Infinite retries (recommended) - Will keep retrying until success")
        print("2. Limited retries - Custom maximum retry count")
        
        retry_choice = input("\nPlease select retry strategy (1/2, default 1): ").strip() or '1'
        
        if retry_choice == '2':
            max_retries_input = input("Please enter maximum retry count (e.g. 10): ").strip()
            max_retries = int(max_retries_input) if max_retries_input.isdigit() else 10
        else:
            max_retries = None  # Infinite retries
        
        # If using local model, set quota delay to 0
        if use_local:
            quota_retry_delay = 0
        else:
            quota_delay = input("Quota error retry delay (seconds, default 60): ").strip()
            quota_retry_delay = int(quota_delay) if quota_delay.isdigit() else 60
        
        confirm = input("\nStart building database? (y/n): ").strip().lower()
        if confirm == 'y':
            create_function_level_database(
                vuln_files,
                max_retries=max_retries,
                retry_delay=10,
                quota_retry_delay=quota_retry_delay,
                use_local_embedding=use_local,
                local_model_name=local_model,
                device=device
            )
        else:
            print("Operation cancelled")
    
    elif choice == '2':
        # Resume from failure log
        print("\nResume from failure log")
        
        failed_log = 'database/failed_functions.json'
        if not os.path.exists(failed_log):
            print(f"❌ Failure log does not exist: {failed_log}")
            print("   If no failed functions, no need to resume")
            return
        
        print(f"✓ Found failure log: {failed_log}")
        
        # base_dir = input("Please enter original vulnerability code directory: ").strip()
        base_dir = r'extracted_SWCs'
        
        confirm = input("\nStart recovery processing? (y/n): ").strip().lower()
        if confirm == 'y':
            resume_database_creation(base_dir, VECTOR_STORE_PATH, failed_log)
        else:
            print("Operation cancelled")
    
    elif choice == '3':
        # Test retrieval
        print("\nTest retrieval functionality")
        print("Please enter test code file path:")
        test_file = input("File path: ").strip()
        
        if not os.path.exists(test_file):
            print(f"❌ File does not exist: {test_file}")
            return
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_code = f.read()
        
        test_query(test_code)
    
    elif choice == '4':
        # View statistics
        print("\nDatabase statistics feature under development...")
        print(f"Database path: {VECTOR_STORE_PATH}")
        
        if os.path.exists(VECTOR_STORE_PATH):
            vector_store = VectorStore(path=VECTOR_STORE_PATH)
            print(f"✓ Database exists")
            # More statistics can be added here
        else:
            print("❌ Database does not exist")
    
    else:
        print("❌ Invalid choice")


if __name__ == '__main__':
    # Example usage
    print(__doc__)
    
    # If run directly, start interactive interface
    main()

