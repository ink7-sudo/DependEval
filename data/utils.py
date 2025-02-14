import re

task1_prompt_template_length2="""
{code_content}
Based on the above code snippets, complete the following instructions and output according to the format specified in Step 3:
	1.	Identify the segment in one file (#file 1) that is invoked by another file (#file 2) (excluding the import parts) and specify the relevant code segment in the called file.
	2.	Modify the given code to implement {function}. This requires modifying the part being called in #file 1 and the way #file 2 calls #file 1. If new code snippets are needed, add them to the end of each respective file.
	3.	Output format:
    """

task1_json = """
{
   "called_code_segment": "#file 1 segment being invoked (excluding `import`)",
   "invoking_code_segment": "#file 2 segment invoking #file 1 (excluding `import`)",
   "feature_description": "Description of the new feature",
   "detailed_feature_description": "General explanation of the modification approach",
   "modified_complete_code": "Provide the complete code with the required modifications. Output the modified code snippets. Use comments like #Modify for modified parts and #New for newly added parts to indicate whether the change is an addition or modification."
}"""

task2_prompt_template="""There are files {filename}. 
Analyze their content and determine the dependency relationship between files. 
Output with the following request. 
1. Don't give analysis process and using the same file title. 
2. Simply and only output the dependency relationship list using its file names with the format ['a.py','b.py','c.py'] if b depends on a , c depends on b. 
3. You must strictly output the response with the format:['a.py','b.py','c.py']

Example output:
["file1.py", "file2.py", "file3.py"]

Here's the code snippet:
{code_content}
"""

task4_prompt_template ="""
You are an AI assistant tasked with generating a project structure based on the given repository information. Your task:

1. Analyze the project description, function, and file information provided.
2. Generate a project structure that shows the dependencies between files.
3. Return the structure in the format [[file1, file2, file3], [file4, file5], ...], where each sublist represents a chain of dependencies (file2 calls file1, file3 calls file2, etc.).
4. Provide only the list structure, without any additional explanation.
5. You must strictly follow the output format.

Here's the project information:

Project Description: {description}
Project Function: {function}

Files in the project:
{files}

Based on this information, please generate the project structure showing the dependencies between files. Remember to return only the list structure without any additional explanation.

Example output format:
[["file1.py", "file2.py", "file3.py"], ["file4.py", "file5.py"]]
"""
def construct_task1_prompt(datapoint):
    function = datapoint["feature_description"]
    code_content = datapoint["content"]
    return task1_prompt_template_length2.format(function=function, code_content=code_content) + task1_json

def construct_task2_prompt(datapoint):
    filename = datapoint["files"]
    code_content = datapoint["content"]
    filename = ', '.join(filename)
    return task2_prompt_template.format(filename=filename, code_content=code_content)

def construct_task4_prompt(datapoint):
    description = datapoint["description"]
    function = datapoint["function"]
    files = "\n".join([f"- {file['file']}: {file['function']}" for file in datapoint['files']])
    return task4_prompt_template.format(description=description, function=function, files=files)
    

def construct_prompt(
    data: dict, 
    language: str = "python",
    tokenizer= None,
    max_token_nums: int = 50000,
    task: str = "task2"
    ) -> str:
    """
    Construct the prompt for next line prediction.

    :param data: data point from the dataset
    :param language: the language of the code
    :param tokenizer: the tokenizer of the evaluation model
    :param max_token_nums: the maximum number of tokens constraint for the prompt

    :return: the constructed prompt
    """

    if task == "task2":
        cross_file_prompt = construct_task2_prompt(data)
    elif task == "task1":
        cross_file_prompt = construct_task1_prompt(data)
    elif task == "task4":
        cross_file_prompt = construct_task4_prompt(data)    

    # if we assign the tokenizer and the max_token_nums, we will truncate the cross-file prompt to meet the constraint
    if tokenizer is not None and max_token_nums is not None:
    
        cross_file_prompt_token_nums = len(tokenizer.encode(cross_file_prompt))
      
        exceed_token_nums = cross_file_prompt_token_nums - max_token_nums

        if exceed_token_nums > 0:
            # split the cross-file prompt into lines
            cross_file_prompt_lines = cross_file_prompt.split("\n")
            # drop lines from end until the extra token number is less than 0
            for i in range(len(cross_file_prompt_lines)-1, -1, -1):
                exceed_token_nums -= len(tokenizer.encode(cross_file_prompt_lines[i]))
                if exceed_token_nums < 0:
                    break
            
            # join the lines back
            cross_file_prompt = "\n".join(cross_file_prompt_lines[:i]) + "\n\n"

    # combine the cross-file prompt and in-file prompt
    prompt = cross_file_prompt

    # normalize some empty lines
    prompt = re.sub(r'\n{4,}', '\n\n', prompt)

    return prompt

