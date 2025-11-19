def prompt_txt():
    # prompt2 = """In the upcoming process of code, I will provide you with sections of the code in batches.      Each batch of code sections will start with a specific number, which consists of a '#' symbol followed by an Arabic numeral, such as '#1', '#8', etc.      Your task is to carefully record each batch of paper sections and their corresponding numbers.      Please note that while recording these sections, you do not need to perform any form of analysis or provide summaries;      simply reply with 'Recorded' after recording each batch.      When I send the instruction '<Code Entry Complete>', it means that all sections of the code have been fully provided, at which time you should reply with 'All content recorded'.      Please focus on accurate recording and avoid performing any actions beyond the scope of these instructions.      Now, please prepare to receive and record the code sections that will be provided."""
    prompt2 = """In the upcoming process of code, I will provide you with sections of the code in batches.        Each batch of code sections will start with a specific number, which consists of a '#' symbol followed by an Arabic numeral, such as '#1', '#8', etc.        Your task is to carefully record each batch of paper sections and their corresponding numbers.        Please note that while recording these sections, you do not need to perform any form of analysis or provide summaries;             When I send the instruction '<Code Entry Complete>', it means that all sections of the code have been fully provided, at which time you should reply with 'All content recorded'.        Please focus on accurate recording and avoid performing any actions beyond the scope of these instructions.        Now, please prepare to receive and record the code sections that will be provided."""
    prompt3 = """In the upcoming process of code, I will provide you with sections of the code.             Your task is to carefully record the code.            Please note that while recording these sections, you do not need to perform any form of analysis or provide summaries;                 When I send the instruction '<Code Entry Complete>', it means that all sections of the code have been fully provided, at which time you should reply with 'All content recorded'.            Please focus on accurate recording and avoid performing any actions beyond the scope of these instructions.            Now, please prepare to receive and record the code sections that will be provided."""
    prompt = """
            Based on the fundamental principles of smart contract security, does the following smart contract contain any bad practices?
            ########################################################
            """


    bad_text_start = """
    ######################Here is similar bad practice code in the database(The following are not subject to audit)###########################

    """
    bad_text_end = """
        ######################The above is similar bad practice code in the database(No audit is required for the above)###########################

        """
    norminal_text = """Notice!  From now on you will be executing new commands.  If there are bad practices, please generate a JSON list of smart contract code bad practices.  The JSON list of bad practices  should be provided in JSON format, containing the following keys: bad_practice_id(i.e.,1,2), title, type, bad_practice_code_block, risk_level(1-5), reason, improvement_suggestion.
    Do not include information other than the audit report in json format."""
    return prompt, bad_text_start, prompt2, prompt3, norminal_text, bad_text_end