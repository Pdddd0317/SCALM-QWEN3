import re
import json
import difflib
import os

def extract_and_combine_json_arrays(session_history_file):
    # Read JSON format session_history file
    with open(session_history_file, 'r', encoding="utf-8") as file:
        session_history = json.load(file)
    
    # Extract JSON arrays from all message content (mainly from GPT responses)
    combined_text = []
    for message in session_history:
        if isinstance(message, dict) and 'content' in message:
            combined_text.append(message['content'])
    
    # Merge all text content
    text = '\n'.join(combined_text)
    
    combined_json_array = []
    
    # Method 1: Try to directly extract complete JSON arrays (more robust approach)
    # Find all [ ... ] matches
    bracket_depth = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '[':
            if bracket_depth == 0:
                start_idx = i
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
            if bracket_depth == 0 and start_idx != -1:
                # Found complete array
                array_string = text[start_idx:i+1]
                
                # Filter out obviously invalid small fragments (length < 20 or no braces)
                if len(array_string) < 20 or '{' not in array_string:
                    start_idx = -1
                    continue
                
                try:
                    json_array = json.loads(array_string)
                    if isinstance(json_array, list) and len(json_array) > 0:
                        # Ensure array contains dictionary objects
                        if isinstance(json_array[0], dict):
                            combined_json_array.extend(json_array)
                except json.JSONDecodeError as e:
                    # If entire array parsing fails, try to fix common issues
                    try:
                        # Remove possible markdown code block markers
                        cleaned = array_string.strip()
                        if cleaned.startswith('```'):
                            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                            cleaned = re.sub(r'```\s*$', '', cleaned)
                        
                        json_array = json.loads(cleaned)
                        if isinstance(json_array, list) and len(json_array) > 0:
                            if isinstance(json_array[0], dict):
                                combined_json_array.extend(json_array)
                    except json.JSONDecodeError as e2:
                        # Only output error when parsing longer valid fragments
                        if len(array_string) > 100 and '{' in array_string:
                            print(f"⚠️  JSON parsing warning: {str(e2)[:100]}")
                            print(f"   Failed fragment (first 200 chars): {array_string[:200]}...")
                
                start_idx = -1

    # Reassign 'bad_practice_id' starting from 1
    for i, item in enumerate(combined_json_array, start=1):
        if isinstance(item, dict):
            item['bad_practice_id'] = i
    
    return combined_json_array

# json_code = extract_and_combine_json_arrays()
# print(json_code)



def find_most_similar_line(target_str, text):
    lines = text.split('\n')  # Split text by lines
    highest_similarity = 0.0  # Store highest similarity
    most_similar_line = None  # Store most similar line
    line_number = 0           # Store line number of most similar line
    current_line_number = 0   # Current line number

    # Iterate through each line to calculate similarity
    for line in lines:
        current_line_number += 1
        similarity = difflib.SequenceMatcher(None, target_str, line).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_line = line
            line_number = current_line_number

    return line_number, most_similar_line, highest_similarity


def add_line(session_history_file, code):
    json_code = extract_and_combine_json_arrays(session_history_file)
    
    # Only output warning when no valid data extracted
    if not json_code:
        print("⚠️  Warning: Failed to extract valid audit results from session history")
        return json.dumps([], indent=2, ensure_ascii=False)
    
    # Display number of extracted vulnerabilities
    print(f"✓ Successfully extracted {len(json_code)} audit findings")
    
    for item in json_code:
        try:
            # Find closest matching code block
            line_num, similar_line, similarity = find_most_similar_line(item['bad_practice_code_block'], code)
            item['line'] = line_num
        except Exception as e:
            # Only output detailed errors in debug mode
            # print(f"Error processing item: {e}")
            continue

    # Convert modified data back to JSON string
    modified_json_data = json.dumps(json_code, indent=2, ensure_ascii=False)
    return modified_json_data









