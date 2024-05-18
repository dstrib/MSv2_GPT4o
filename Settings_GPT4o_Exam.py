#!/usr/bin/env python3.8
# Daniel Stribling  |  ORCID: 0000-0002-0649-9506
# University of Florida
# ModelStudentV2_GPT4o Project
# Changelog:
#   Version 1.0.0 - 2025-05-18 - Initial version

"""Settings for testing the GPT-4o model on science exams."""

import textwrap
import os

# Constants
PACKAGE_VERSION = 'v1.0.0'
SCRIPT_VERSION = 'Query_GPT4_Exam.py ' + PACKAGE_VERSION
GPT4V_MODEL = 'gpt-4-turbo-2024-04-09'
GPT4O_MODEL = 'gpt-4o-2024-05-13'
DEF_GPT_MODEL = GPT4V_MODEL
CHAT_URL = 'https://api.openai.com/v1/chat/completions'
CONFIRM_CONTINUE = False

MODEL_PARAM_SETS = {
    'GPT4V': (GPT4V_MODEL),
    'GPT4O': (GPT4O_MODEL),
}

DEF_IO_DIR = os.path.abspath('.')

# Prompt template
# The first {} is the training level of the exam,
# the second {} is the name of the field,
# and the third {} is any details on exam format.
PROMPT_TEMPLATE_MAIN = textwrap.dedent("""\
    Please provide correct and very concise answers to the following questions from
    a {}-level exam from {}.
    Use technical or advanced language as appropriate to the level of the course
    to answer the question correctly.

    {}

    Some questions may have multiple parts, denoted by letters after the question number.
    For example: 1A and 1B. When answering multiple part questions, refer to the answer of
    previous parts of the question as necessary to answer each question correctly.

    Do not include numeric lists in the answer unless the question specifically asks for one.

    If the question asks for a sketch, provide detailed instructions to draw the requested sketch.

    Answers will be provided back to the examiner by copying and pasting answers into a
    document.

    For formatting of any responses, provide each answer as
    plain text without markup symbols or header markers such as "**" or "--".

    """)

# Instructions placeholder for no specific exam format.
BLANK_INSTRUCTIONS = """"""

# Initialization Statements
USER_INIT_STATEMENT = "I am ready to provide you with questions to answer."
INIT_STATEMENT_MAIN = textwrap.dedent("""\
    I am ready to answer the questions correctly and concisely at the appropriate
    level, without using numerical lists unless the question specifically asks for
    one. I will not include markup symbols or header markers
    such as "**" or "--".
    """)
