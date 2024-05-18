#!/usr/bin/env python3.8
# Daniel Stribling  |  ORCID: 0000-0002-0649-9506
# University of Florida
# ModelStudentV2_GPT4o Project
# Changelog:
#   Version 1.0.0 - 2025-05-18 - Initial version


"""
Script to query OpenAI GPT4 with examination questions and store the results.

This script is designed to query OpenAI GPT-4v and GPT-4o with assessment questions from higher
education courses and store the results.
This is being performed as part of a research project to evaluate the capabilities of OpenAI GPT4
to provide logically consistent and factually correct answers for questions in the biomedical
sciences.
"""

import os
import copy
import openai
import sys
import base64
import json
import datetime
from Settings_GPT4o_Exam import PROMPT_TEMPLATE_MAIN, \
                                BLANK_INSTRUCTIONS, \
                                USER_INIT_STATEMENT, \
                                INIT_STATEMENT_MAIN, \
                                PACKAGE_VERSION, SCRIPT_VERSION, \
                                DEF_GPT_MODEL, CHAT_URL, CONFIRM_CONTINUE, \
                                MODEL_PARAM_SETS, DEF_IO_DIR

# Constants for the script.
START_AT_PROMPT = 0
IO_DIR = DEF_IO_DIR
USE_MODELS = ['GPT4V']#, 'GPT4O']

# Attempt to find the API key in a file, else ask for API_Key.
def load_api_key(api_key_file='API_KEY.txt'):
    """Attempt to find the API key in a file, else ask for API_Key."""
    os.path.abspath(api_key_file)
    if os.path.isfile(api_key_file):
        with open(api_key_file, 'r') as api_key_file_obj:
            api_key = api_key_file_obj.read().strip()
    else:
        api_key = input('\nEnter API Key:\n')
    return api_key

#Base64 encode image
def base64_encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Print script version, time and date, and openai module version.
def print_script_info():
    """Print script version, time and date, and openai module version."""
    print('Package Version:', PACKAGE_VERSION)
    print('Script Version:', SCRIPT_VERSION)
    print('Date and Time:', datetime.datetime.now())
    print('OpenAI Module Version:', openai.__version__)
    print('Model Param Sets:')
    for key in MODEL_PARAM_SETS:
        print('    ', key, ':', MODEL_PARAM_SETS[key])
    print()

# Add to the list of prompt components with a specified role and content.
def add_to_prompt(initial_prompt, role, content):
    """Add to the list of prompt components with a specified role and content."""
    ret_prompt = []
    for item in initial_prompt:
        ret_prompt.append(copy.deepcopy(item))
    ret_prompt.append({'role': role, 'content': content})
    return ret_prompt

# Create initial prompt by adding initial components.
def prep_prompt(prompt_template,
                assistant_init_statement,
                user_init_statement=USER_INIT_STATEMENT,
                ):
    """Create initial prompt by adding initial components."""
    question_prefixes = []
    question_prefixes = add_to_prompt(question_prefixes, 'system', prompt_template.lstrip())
    question_prefixes = add_to_prompt(
        question_prefixes,
        'user',
        user_init_statement
    )
    question_prefixes = add_to_prompt(
        question_prefixes,
        'assistant',
        assistant_init_statement
    )
    return question_prefixes

# Process the response from GPT4. Returns:
#   response, finish_reason, tokens_str, details, usage
def process_gpt_response(response_obj,
                         print_completion=True,
                         print_response=True,
                         print_tokens=True,
                         print_details=True,
                         ):
    """
    Process the response from GPT4.

    Returns: response, finish_reason, tokens_str, and details.
    """
    # Check for successful response.
    # response_dict = {k: v for k, v in response_obj}
    # if 'choices' not in response_dict:
    #     print("No Response")
    #     print('Obj:')
    #     print(response_obj)
    #     return 'No response from GPT!', 'No_Response', None, None, None

    # Extract response and finish reason.
    choice_obj = response_obj.choices[0]
    response = choice_obj.message.content.strip()
    finish_reason = choice_obj.finish_reason

    # Store details
    detail_keys = ['created', 'id', 'model', 'object']
    details = {'finish_reason': finish_reason}
    for key in detail_keys:
        details[key] = getattr(response_obj, key)

    # Store usage
    usage = {}
    for key in ['completion_tokens', 'prompt_tokens', 'total_tokens']:
        usage[key] = getattr(response_obj.usage, key)

    # Create tokens report string
    tokens_str = ''
    tokens_str += 'Prompt: ' + str(usage['prompt_tokens']) + '  '
    tokens_str += 'Response: ' + str(usage['completion_tokens']) + '  '
    tokens_str += 'Total: ' + str(usage['total_tokens']) + ' (of ' + "nocap" + ')  '

    # If enabled, print the response, tokens, and/or details.
    if print_completion:
        print('API Query Complete...')
    if print_response:
        print('Response:')
        print(response)
    if print_tokens:
        print('Tokens: ' + tokens_str + '\n')
    if print_details:
        print('Details:')
        for key in detail_keys:
            print('   ', key.title() + ':',  details[key])
        print()

    return response, finish_reason, tokens_str, details, usage


# Query GPT4 with prepared prompt, process the response, and return details.
def query_gpt(client, prompt, model=DEF_GPT_MODEL):
    """Query model with prepared prompt, process the response, and return details."""
    response_obj = client.chat.completions.create(
        model=model,
        messages=prompt,
        # temperature=float(temperature),
    )

    response, finish_reason, tokens, details, usage = process_gpt_response(
        response_obj,
        print_completion=True,
        print_response=False,
        print_tokens=True,
        print_details=True,
    )
    return response, finish_reason, tokens, details, usage

# If the content contains a line starting with "![", replace with image-containing
# query.
def process_content(content, IO_DIR=IO_DIR, verbose=False):
    """If the content contains a line starting with "![", replace with image-containing query."""
    if '![' not in content:
        if verbose:
            print('    No Images in Question.')
        return content
    else:
        content_text = ""
        content_text_item = {}
        content_images = []
        for line in content.splitlines(True):
            if not line.startswith('!['):
                content_text += line
            else:
                image_location = line.split('![')[1].rstrip().rstrip(']')
                if image_location.startswith('http'):
                    image_info = {
                        "type": "image_url",
                        "image_url": {
                            "url": image_location
                        }
                    }
                    print("    Using Image: ", image_location)
                    content_images.append(image_info)
                else:
                    full_image_file_path = os.path.join(IO_DIR, image_location)
                    if not os.path.isfile(full_image_file_path):
                        message = 'Image file not found: ' + full_image_file_path
                        raise FileNotFoundError(message)
                    if not (image_location.endswith('.jpg') or image_location.endswith('.jpeg')):
                        message = 'Image file must be in JPEG format: ' + full_image_file_path
                        raise ValueError(message)
                    else:
                        print("    Using Image: ", full_image_file_path)
                    b64_image = base64_encode_image(full_image_file_path)
                    image_info = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }
                    }
                    content_images.append(image_info)
        if content_text.strip():
            content_text_item = {"type": "text", "text": content_text.strip()}
        return [content_text_item] + content_images

# Class to print and write output of the GPT4 query / response conversation to a file.
class Query_Reporter():
    """Class to print and write output of the GPT4 query / response conversation to a file."""

    def __init__(self, file_name, initial_dialog, model):
        """Hold place for module docstring."""
        self.file_name = file_name
        self.file_obj = open(self.file_name, 'w')
        self.model = model
        self.initialize()
        for entry in initial_dialog:
            self.report(entry)

    def __call__(self, dialog, do_print=True):
        """Call the report method if class object is called."""
        self.report(dialog, do_print=do_print)

    def initialize(self):
        """Initialize the file with conversation details."""
        init_str = 'Conversation Details:\n'
        init_str += '    Script Version: ' + SCRIPT_VERSION + '\n'
        init_str += '    Performed: ' + str(datetime.datetime.now()) + '\n'
        init_str += '    Chat URL: ' + CHAT_URL + '\n'
        init_str += '    Model: ' + self.model + '\n'
        init_str += '    Max Context Window: ' + "NO-LIMIT" + '\n'
        init_str += '\n'
        self.file_obj.write(init_str)

    def report(self, dialog, do_print=True):
        """Report a dialog element by writing to the file handle and printing to the screen."""
        header_bar = ' ' + ('-' * 5) + ' '
        write_str = header_bar + dialog['role'] + header_bar + '\n'
        # if content contains images... skip these.
        if isinstance(dialog['content'], list):
            for item in dialog['content']:
                if 'text' in item:
                    write_str += item['text'].rstrip() + '\n\n'
                else:
                    write_str += '    [Image]\n\n'
        else:
            write_str += dialog['content'].rstrip() + '\n\n'
        self.file_obj.write(write_str)
        self.file_obj.flush()
        if do_print:
            print(write_str)

    def add_details(self, details, usage):
        """Add details of the query to the report."""
        self.file_obj.write('Details:\n')
        for key in details:
            self.file_obj.write('    ' + key.title() + ': ' + str(details[key]) + '\n')
        self.file_obj.write('\n')
        self.file_obj.write('Usage:\n')
        for key in usage:
            self.file_obj.write('    ' + key.title() + ': ' + str(usage[key]) + '\n')
        self.file_obj.write('\n')
        self.file_obj.flush()

    def close(self):
        """Close the file handle."""
        self.file_obj.close()


# Execute Functionality
if __name__ == '__main__':
    # Select json file(s) containing data about the test to be examined.
    #   mapping keys in this file must include:
    #   course_id, course_name, course_level, extra details
    #   The questions file must be named: course_id + '_questions.txt'
    #   The output file will be named: course_id + '_out_[].txt'

    target_exam_file_names = [
        'Adv_Vir_settings.json',
    ]

    api_key = load_api_key()

    for target_exam_file_name in target_exam_file_names:
        # Create paths to relevant files and directories for the script.
        target_exam_file = os.path.abspath(os.path.join(IO_DIR, target_exam_file_name))

        # Report beginning of test:
        print('\nBeginning GPT test with file:', target_exam_file, '\n')

        # Load the exam parameters
        with open(target_exam_file, 'r') as target_exam_file_obj:
            exam_parameters = json.load(target_exam_file_obj)
            exam_id = exam_parameters['course_id']
            questions_file_name = exam_id + '_questions.txt'

        # Load the questions file
        full_questions_file_name = os.path.join(IO_DIR, questions_file_name)
        if not os.path.isfile(full_questions_file_name):
            message = 'Questions file not found: ' + full_questions_file_name
            raise FileNotFoundError(message)

        with open(full_questions_file_name, 'r') as questions_file_obj:
            exam_prompts = [(q.strip() + '\n') for q in questions_file_obj.read().split('-&-')]

        # Test all exam prompts for image content.
        print('Testing Question Image Locations...')
        for prompt_i, exam_prompt in enumerate(exam_prompts, start=1):
            print('Prompt:', prompt_i)
            process_content(exam_prompt, verbose=True)
        print('\nImage Testing Complete.\n')

        # Test with each model.
        for model_title, model_params in MODEL_PARAM_SETS.items():
            # Load new instance of client
            client = openai.OpenAI(api_key=api_key)

            use_model = (model_params)

            # Setup prompt
            initial_prompt = prep_prompt(PROMPT_TEMPLATE_MAIN, INIT_STATEMENT_MAIN)
            initial_prompt[0]['content'] = initial_prompt[0]['content'].format(
                exam_parameters['course_level'],
                exam_parameters['course_name'],
                exam_parameters['extra_details'],
            )

            # Set up the output file
            out_file_name = exam_id + '_out_' + model_title + '.txt'
            full_out_file_name = os.path.join(IO_DIR, out_file_name)
            use_out_file_name = full_out_file_name

            # Query GPT4 with all questions
            last_prompt_set = copy.deepcopy(initial_prompt)
            query_reporter = Query_Reporter(use_out_file_name, last_prompt_set, use_model)

            for prompt_i, exam_prompt in enumerate(exam_prompts, start=1):
                # Report skipping or beginning prompt
                if prompt_i < START_AT_PROMPT:
                    print('Skipping prompt', prompt_i, '...')
                    continue
                else:
                    print('------ Model:', use_model, 'Prompt:', prompt_i, 'Date/Time:', datetime.datetime.now(), '------')

                # Prepare next prompt
                ready_content = process_content(exam_prompt)
                query_reporter({'role': 'user', 'content': ready_content})
                next_prompt_set = add_to_prompt(last_prompt_set, 'user', ready_content)

                # Query GPT4 and process response
                response, finish_reason, tokens, details, usage = query_gpt(
                    client, next_prompt_set, use_model)
                query_reporter({'role': 'assistant', 'content': response})
                if finish_reason == 'No_Response':
                    print('Exiting...')
                    sys.exit()
                query_reporter.add_details(details, usage)
                last_prompt_set = add_to_prompt(next_prompt_set, 'assistant', response)

                # manually require continue if enabled
                if CONFIRM_CONTINUE and exam_prompt != exam_prompts[-1]:
                    input('\nContinue?\n')

            # At completion of this prompt-style, close the output file
            query_reporter.close()

            # Report completion of this section:
            print('\nCompleted assessment with model: ' + use_model
                  + ' for exam: ' + exam_id + '.\n')

    print('\nDone.\n')
