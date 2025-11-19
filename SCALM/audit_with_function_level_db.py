"""
Smart Contract Auditing Using Function-Level Vector Database
Integrates function-level slicing of code to be audited with function-level database retrieval

Core Process:
1. Code to audit -> Function-level slicing (including call relationships, up to 3 layers)
2. Each function -> Retrieve similar functions from function-level database
3. Retrieval results contain complete function context (called functions, etc.)
4. Audit using Three-Layer Reasoning Validation Framework:
   - Layer 1: Syntax Validation - Basic code-level vulnerabilities
   - Layer 2: Design Pattern Validation - Structural issues
   - Layer 3: Architecture Validation - System-level risks
5. Send to GPT-4 for auditing, applying Step-Back abstraction method

Three-Layer Reasoning Validation Framework:
- Target_Code: Target contract code to be audited
- Similar_Code: Similar vulnerability patterns retrieved from vector database (as reference)
- Through progressive three-layer validation, comprehensively analyze security risks from syntax to architecture

Optimization Features:
- Each function audited independently: Audit immediately after retrieval, no need to wait for all functions to be retrieved
- API call count: N times (N = number of functions), one complete audit per function
- Real-time feedback: Get results immediately after each function audit completes
- Support dynamic model switching: Called through Rewrite_query module, allows experiment scripts to dynamically replace models
"""

import sys
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

import Rewrite_query  # Correct import: import whole module ONLY

from create_function_level_database import (
    enhanced_query_with_context,
    VECTOR_STORE_PATH
)
from prompt_txt import prompt_txt
from json_processing import add_line



# ============================================
# Configuration
# ============================================
SAVE_SESSION_PATH = 'session_History'
SAVE_AUDIT_PATH = 'Audit_report'
USE_FUNCTION_LEVEL_DB = True  # Whether to use function-level database


# ============================================
# Format Function Context (Enhanced Version)
# ============================================

def format_enhanced_function_context(func_context, retrieval_results, bad_text_start, bad_text_end):
    """
    Format function context with enhanced retrieval results
    
    Args:
        func_context: Function context
        retrieval_results: Enhanced retrieval results (including metadata)
        bad_text_start: RAG marker start
        bad_text_end: RAG marker end
    
    Returns:
        Formatted string
    """
    content = []
    
    # 1. Function basic information
    content.append(f"### Function #{func_context['index']}: {func_context['name']}")
    content.append(f"Location: Lines {func_context['start_line']}-{func_context['end_line']}\n")
    
    # 2. Call relationships
    if func_context.get('call_graph_info'):
        graph_info = func_context['call_graph_info']
        if graph_info['directly_calls']:
            content.append(f"**Directly calls:** {', '.join(graph_info['directly_calls'])}")
        if graph_info['called_by']:
            content.append(f"**Called by:** {', '.join(graph_info['called_by'])}")
        if graph_info['all_dependencies']:
            content.append(f"**All dependencies:** {', '.join(graph_info['all_dependencies'])}")
        content.append("")
    
    # 3. Main function code (Target_Code)
    content.append("**Target_Code (Main Function to Audit):**")
    content.append("```solidity")
    content.append(func_context['code'])
    content.append("```\n")
    
    # 4. Called functions code
    if func_context.get('called_functions_code'):
        content.append("=" * 60)
        content.append("**Functions Called by This Function:**")
        content.append("=" * 60 + "\n")
        
        for called in func_context['called_functions_code']:
            depth_indent = "  " * called.get('depth', 1)
            content.append(f"{depth_indent}➤ Function: **{called['name']}** (Depth: {called.get('depth', 1)})")
            content.append(f"{depth_indent}  Signature: `{called['signature']}`")
            content.append(f"{depth_indent}  Code:")
            content.append("```solidity")
            content.append(called['code'])
            content.append("```\n")
    
    # 5. Similar code retrieved from function-level database (enhanced version)
    if retrieval_results:
        content.append(bad_text_start)
        content.append("**=== Similar_Code (Reference Patterns from Database - DO NOT AUDIT) ===**\n")
        
        # Main function retrieval results
        if retrieval_results.get('main_function'):
            content.append(f"**Similar vulnerable function from database:**")
            
            # Add metadata
            main_meta = retrieval_results.get('main_metadata', {})
            if main_meta:
                content.append(f"- Function: {main_meta.get('function_name', 'Unknown')}")
                content.append(f"- Vulnerability: {main_meta.get('vulnerability_type', 'Unknown')}")
                content.append(f"- Source: {os.path.basename(main_meta.get('source_file', 'Unknown'))}")
                content.append(f"- Has dependencies: {main_meta.get('has_dependencies', False)}\n")
            
            content.append("```solidity")
            content.append(retrieval_results['main_function'])
            content.append("```\n")
        
        # Called functions retrieval results
        if retrieval_results.get('called_functions'):
            for func_name, similar_code in retrieval_results['called_functions'].items():
                if similar_code:
                    content.append(f"\n**Similar pattern for called function '{func_name}':**")
                    
                    # Add metadata
                    called_meta = retrieval_results.get('called_metadata', {}).get(func_name, {})
                    if called_meta:
                        content.append(f"- Function: {called_meta.get('function_name', 'Unknown')}")
                        content.append(f"- Vulnerability: {called_meta.get('vulnerability_type', 'Unknown')}\n")
                    
                    content.append("```solidity")
                    content.append(similar_code)
                    content.append("```\n")
        
        content.append("**=== End of Similar_Code Reference Section ===**")
        content.append(bad_text_end)
    
    return '\n'.join(content)


# ============================================
# Main Audit Process
# ============================================

def audit_with_function_level_db(sol_file, output_dir=None):
    """
    Smart contract auditing using function-level database
    
    Args:
        sol_file: Smart contract file to audit
        output_dir: Output directory (optional)
    
    Returns:
        Audit report path
    """
    print("=" * 80)
    print(f"Auditing file: {sol_file}")
    print(f"Using function-level database: {USE_FUNCTION_LEVEL_DB}")
    print(f"Database path: {VECTOR_STORE_PATH}")
    print(f"Audit mode: Each function audited independently (audit immediately after retrieval)")
    print("=" * 80 + "\n")
    
    session_history = []
    
    try:
        # 1. Read code
        with open(sol_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        print(f"✓ Code lines: {len(code.split(chr(10)))}")
        
        # 3. Get prompts
        prompt, bad_text_start, prompt2, prompt3, norminal_text, bad_text_end = prompt_txt()
        
        # 4. Function-level slicing
        print("\nPerforming function-level slicing...")
        function_contexts = list(Rewrite_query.read_by_functions(
            code, 
            query_db=False,  # Don't use old retrieval method
            include_call_graph=True, 
            max_depth=3  # Maximum recursion depth of 3 layers
        ))
        
        print(f"✓ Extracted {len(function_contexts)} functions")
        
        # 5. Retrieve and audit each function immediately
        print("\nStarting function-by-function retrieval and auditing...")
        all_function_results = []  # Store audit results for each function
        
        for idx, func_context in enumerate(function_contexts, 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(function_contexts)}] Processing function: {func_context['name']}")
            print('='*70)
            
            # 5.1 Retrieve similar code
            print(f"  Step 1/2: Retrieving similar code from function-level database...", end='')
            if USE_FUNCTION_LEVEL_DB:
                try:
                    retrieval_results = enhanced_query_with_context(
                        func_context, 
                        VECTOR_STORE_PATH
                    )
                    func_context['enhanced_retrieval'] = retrieval_results
                    print(" ✓")
                except Exception as e:
                    print(f" ❌ Retrieval failed: {e}")
                    func_context['enhanced_retrieval'] = None
            else:
                func_context['enhanced_retrieval'] = None
                print(" (Skipped)")
            
            # 5.2 Audit this function immediately
            print(f"  Step 2/2: Auditing function '{func_context['name']}'...")
            
            # Build audit prompt for single function
            single_function_prompt = f"""You are a smart contract security auditor using a multi-layer reasoning framework.

I will provide you with a function from a smart contract. The function includes:
- Target_Code: The function's own code (the contract to audit)
- Functions it calls (with full code)
- Similar_Code: Contextual snippets from vector DB similarity search (vulnerable patterns from our function-level database)

**Three-Layer Reasoning Validation Framework:**

**Layer 1 (Syntax Validation)**
Abstraction: What are the fundamental syntax rules and security-critical coding conventions in Solidity that prevent basic vulnerabilities?
- Focus on: Missing checks, unchecked returns, improper visibility, dangerous operations
- Examples: Reentrancy guards, integer overflow/underflow, uninitialized storage pointers

**Layer 2 (Design Pattern Validation)**
Abstraction: What are the canonical secure design patterns and anti-patterns for smart contracts regarding access control, state management, and external interactions?
- Focus on: Access control patterns, state transition logic, event emissions, error handling
- Examples: Checks-Effects-Interactions pattern, proper modifier usage, secure oracle integration

**Layer 3 (Architecture Validation)**
Abstraction: What architectural principles ensure secure smart contract ecosystems, considering composition risks and trust boundaries?
- Focus on: Trust assumptions, external dependencies, upgrade patterns, cross-contract interactions
- Examples: Secure composition, trust boundary violations, centralization risks

**Your Analysis Process:**
1. Compare Target_Code with Similar_Code patterns from the database
2. Apply Step-Back abstraction: Identify high-level security principles
3. Map principles to code implementation
4. Perform progressive three-layer verification (Layer 1 → Layer 2 → Layer 3)
5. Document findings with complete context

---

Here is the function to audit:

"""
            
            # 格式化函数上下文
            function_content = format_enhanced_function_context(
                func_context,
                func_context.get('enhanced_retrieval'),
                bad_text_start,
                bad_text_end
            )
            
            # 添加分析指令
            analysis_instruction = """\n
---

**Now, apply the Three-Layer Reasoning Validation Framework to analyze this function.**

**Progressive Three-Layer Verification:**

**Layer 1 (Syntax Validation):**
- Check for missing checks, unchecked external calls, improper visibility
- Verify return values are handled, arithmetic is safe
- Identify dangerous operations (delegatecall, selfdestruct, inline assembly)

**Layer 2 (Design Pattern Validation):**
- Verify access control mechanisms (modifiers, require statements)
- Check state management (Checks-Effects-Interactions pattern)
- Validate external interactions (reentrancy protection, pull over push)
- Confirm proper event emissions and error handling

**Layer 3 (Architecture Validation):**
- Assess trust assumptions and boundaries
- Evaluate external dependencies and composition risks
- Check for centralization risks and upgrade vulnerabilities
- Validate cross-contract interaction safety

**Important:** Only audit Target_Code. Similar_Code from database is for reference only.

Please provide a comprehensive JSON list of all vulnerabilities found in THIS FUNCTION.

**Format Requirements:**
Each entry must contain:
- bad_practice_id: Unique identifier (e.g., "001", "002")
- title: Concise vulnerability name
- type: Vulnerability category (e.g., "Reentrancy", "Access Control", "Integer Overflow")
- bad_practice_code_block: The vulnerable code snippet
- risk_level: Severity from 1-5 (1=Low, 2=Medium-Low, 3=Medium, 4=High, 5=Critical)
- reason: Detailed explanation 
- improvement_suggestion: Specific, actionable remediation steps
- function_name: Name of the vulnerable function

Output as properly formatted JSON array. If no vulnerabilities found, return an empty array [].
"""
            
            # Send audit request
            full_audit_prompt = single_function_prompt + function_content + analysis_instruction
            
            # Create independent session history for each function
            function_session = []
            answer = Rewrite_query.ask_gpt_with_retries(full_audit_prompt, function_session)
            
            # Save audit results for this function
            all_function_results.append({
                'function_name': func_context['name'],
                'function_index': idx,
                'session_history': function_session,
                'audit_result': answer
            })
            
            print(f"  ✓ Function '{func_context['name']}' audit completed")
        
        print("\n" + "="*70)
        print(f"✓ All {len(function_contexts)} functions audit completed")
        print("="*70)
        
        # 6. Merge all function audit results into main session history
        print("\nMerging all function audit results...")
        for result in all_function_results:
            # Add each function's session history to main session history
            session_history.extend(result['session_history'])
        
        # 9. Save results
        print("\nSaving audit results...")
        session_history_file = Rewrite_query.save_session_history(session_history, sol_file)
        print(f"✓ Session history saved: {session_history_file}")
        
        # Generate final report
        modified_json_data = add_line(session_history_file, code)
        report_name = os.path.splitext(os.path.basename(sol_file))[0] + '_Audit_report_func_level.txt'
        
        # Determine output directory and create it
        target_dir = output_dir if output_dir else SAVE_AUDIT_PATH
        os.makedirs(target_dir, exist_ok=True)
        
        report_path = os.path.join(target_dir, report_name)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(modified_json_data)
        
        print(f"✓ Audit report saved: {report_path}")
        
        print("\n" + "=" * 80)
        print("✅ Audit completed!")
        print(f"✓ Total API calls: {len(function_contexts)} times (once per function)")
        print(f"✓ Number of audited functions: {len(function_contexts)}")
        print("=" * 80)
        
        return report_path
        
    except Exception as e:
        print(f"\n❌ Audit failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def batch_audit(sol_files, output_dir=None):
    """
    Batch audit multiple smart contracts
    
    Args:
        sol_files: List of smart contract files
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print(f"Batch audit: {len(sol_files)} files")
    print("=" * 80 + "\n")
    
    results = []
    for idx, sol_file in enumerate(sol_files, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(sol_files)}] Processing: {sol_file}")
        print('='*80)
        
        report_path = audit_with_function_level_db(sol_file, output_dir)
        results.append({
            'file': sol_file,
            'report': report_path,
            'success': report_path is not None
        })
    
    # Print summary
    print("\n" + "=" * 80)
    print("Batch Audit Summary")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"Success: {success_count}/{len(sol_files)}")
    
    print("\nSuccessful audits:")
    for r in results:
        if r['success']:
            print(f"  ✓ {os.path.basename(r['file'])} -> {r['report']}")
    
    print("\nFailed audits:")
    for r in results:
        if not r['success']:
            print(f"  ❌ {os.path.basename(r['file'])}")
    
    print("=" * 80)


# ============================================
# Main Function
# ============================================

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("Function-Level Database Smart Contract Audit Tool")
    print("=" * 80 + "\n")
    
    print("Please select an operation:")
    print("1. Audit single file")
    print("2. Batch audit directory")
    print("3. Test mode (using example file)")
    
    choice = input("\nPlease enter your choice (1/2/3): ").strip()
    
    if choice == '1':
        # Single file
        # sol_file = input("\nPlease enter .sol file path: ").strip()
        sol_file = r'D:\SCALM\SCALM-ALL\SCALM\1\AloeBlend.sol'
        if not os.path.exists(sol_file):
            print(f"❌ File does not exist: {sol_file}")
            return
        
        audit_with_function_level_db(sol_file)
    
    elif choice == '2':
        # Batch audit
        import glob
        
        dir_path = input("\nPlease enter directory path: ").strip()
        if not os.path.exists(dir_path):
            print(f"❌ Directory does not exist: {dir_path}")
            return
        
        sol_files = glob.glob(os.path.join(dir_path, "*.sol"))
        if not sol_files:
            print(f"❌ No .sol files found")
            return
        
        print(f"\nFound {len(sol_files)} files")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            batch_audit(sol_files)
        else:
            print("Operation cancelled")
    
    elif choice == '3':
        # Test mode
        print("\nTest mode: Using example file")
        test_file = r"D:\SCALM\SCALM-ALL\SCALM\1\AloeBlend.sol"
        
        if os.path.exists(test_file):
            audit_with_function_level_db(test_file)
        else:
            print(f"❌ Test file does not exist: {test_file}")
            print("Please provide a test file path:")
            test_file = input("File path: ").strip()
            if os.path.exists(test_file):
                audit_with_function_level_db(test_file)
            else:
                print(f"❌ File does not exist: {test_file}")
    
    else:
        print("❌ Invalid choice")


if __name__ == '__main__':
    main()

