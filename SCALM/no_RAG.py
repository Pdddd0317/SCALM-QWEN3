import os
import time
import openai
from prompt_txt import prompt_txt
from json_processing import extract_and_combine_json_arrays, add_line
from openai import OpenAI
import glob
import multiprocessing
import json


openai.max_retries = 10

sol_files = glob.glob(os.path.join(r"C:\Users\33519\Desktop\SWC样本", "SWC-*.txt"))
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


def ask_gpt_with_retries(content, session_history, max_retries=5):
    retries = 0
    session_history.append({"role": "user", "content": content})
    # print(content)
    while retries < max_retries:
        try:
            client = OpenAI(api_key="sk-xxxxxxx593",
                            base_url="https://api.xxxxx/v1/", max_retries=5)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=session_history,
                temperature=0.4,
                # stream=True,
            )
            session_history.append({"role": "assistant", "content": completion.choices[0].message.content})
            print(completion.choices[0].message.content)
            return completion.choices[0].message.content
        except Exception as e:
            print(f"error：{e}")
            print("session_history:", session_history)
            retries += 1
            time.sleep(5)  #
    return None

def save_session_history(session_history, sol_file):
    new_filename = os.path.splitext(os.path.basename(sol_file))[0] + '_session_history.txt'
    session_history_path = os.path.join(save_session_path, new_filename)
    with open(session_history_path, 'w', encoding="utf-8") as f:
        for message in session_history:
            f.write(f"{message['role']}: {message['content']}\n")
    # print("Session history saved.")
    return session_history_path


# main函数
def process_file(sol_file):
    session_history = []
    print(f"Processing {sol_file}")

    # sol_files = glob.glob(os.path.join(r"C:\Users\33519\Desktop\SWC样本", "*.txt"))
    # for sol_file in sol_files:
    # print(sol_file)
    with open(sol_file, 'r', encoding="utf=8") as f:
        code = f.read()

    prompt, bad_text_start, prompt2, norminal_text, bad_text_end = prompt_txt()
    code_lines = code.splitlines()
    # print(prompt2)
    answer = ask_gpt_with_retries(prompt2, session_history)
    for chunk in read_by_chunks(code_lines):
        answer = ask_gpt_with_retries(chunk, session_history)
    answer = ask_gpt_with_retries("Code Entry Complete", session_history)
    # print(session_history)
    # 计算chunk数量，调用ask_gpt函数,从1开始
    for index, content in enumerate(read_by_chunks(code_lines)):
        if index == 0:
            ask_gpt_with_retries(norminal_text + f'Based on the fundamental principles of smart contract security, does the  smart contract of #{index+1} contain any bad practices? Be careful not to audit the database for similar bad practice codes.', session_history)
        else:
            ask_gpt_with_retries(
                f'Based on the fundamental principles of smart contract security, does the  smart contract of #{index+1} contain any bad practices? ', session_history)


    session_history_file = save_session_history(session_history, sol_file)

    modified_json_data = add_line(session_history_file,code)
    report = os.path.splitext(os.path.basename(sol_file))[0] + '_Audit report.txt'
    report_file_path = os.path.join(save_audit_path, report)
    with open(os.path.join(report_file_path), "w", encoding="utf-8") as f:
        f.write(modified_json_data)

def main():

    with multiprocessing.Pool() as pool:
        pool.map(process_file, sol_files)
    print("所有文件处理完毕。")

if __name__ == '__main__':

    main()





