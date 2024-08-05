# %%
# Assume openai>=1.0.0
from tqdm import tqdm
from openai import OpenAI
import datetime
import os
import json
import random
from categories import *

"""
nohup python wizardlm8x22b_api.py &

"""


def load_env_file(file_path):
    with open(file_path) as f:
        for line in f:
            # 空行やコメント行を無視
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value


# .envファイルの内容を読み込みます
load_env_file('.env')

# 環境変数の取得
API_KEY = os.getenv('API_KEY')

# %%
now = datetime.datetime.now()
formatted_date = now.strftime("%Y%m%d_%H%M%S")
save_dir = "out_api"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/{formatted_date}.jsonl"

# %%

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepinfra.com/v1/openai",
)


def ask(role, q, temperature=0.7):
    chat_completion = openai.chat.completions.create(
        model="microsoft/WizardLM-2-8x22B",
        messages=[{"role": "system", "content": role},
                  {"role": "user", "content": q}],
        temperature=temperature,
        max_tokens=2048,
    )

    return chat_completion.choices[0].message.content


def ask_and_log(role, q, save_path):
    a = ask(role, q)
    d = {
        "system": role,
        "instruction": q,
        "output": a,
        "text": f"user: {q}\nassistant: {a}"
    }
    with open(save_path, "a") as f:
        f.write(json.dumps(d, ensure_ascii=False)+"\n")

    return a

# %%


def prepare_initial_command():

    job = random.choice(job_list)
    character = random.choice(character_list)
    role = f"{job}. You are {character}"
    genre = random.choice(genre_list)+","+random.choice(genre_list)
    level = random.choice(levels)
    quiz_type = random.choice(
        ["mathematical problem", "reasoning quiz",
         "logical quiz", "coding problem",
         "logical puzzle", "reasoning puzzle",

         ])
    command = f"""Prepare a {quiz_type}.
        - Output only one question, which is not too long.
        - NEVER the answer, hints, or any other things.
        - Topic: {genre}.
        - Level: {level}.
        """
    return role, command


# %%


# %%


def get_longest_phrase_length(text):
    # 区切り文字として、スペース、カンマ、句読点、改行を指定
    delimiters = r'[ ,。！？、\n]'
    # テキストを区切り文字で分割
    try:
        phrases = re.split(delimiters, text)
        # 最大のフレーズの長さを取得
        max_length = max(len(phrase) for phrase in phrases)
    except:
        max_length = 9999
    return max_length


def is_abnormal_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    # 複数の区切り文字をカウント
    period_count = text.count('.') + text.count(',') + \
        text.count('､') + text.count('｡')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold


def is_good_sentence(sentence):
    if get_longest_phrase_length(sentence) > 100:
        return False
    if is_abnormal_text(sentence):
        return False
    return True


# %%
pid = os.getpid()

for i in tqdm(range(10000)):
    seed = int(pid)+int(datetime.datetime.now().timestamp())
    role, command = prepare_initial_command()
    q1 = ask(role, command, temperature=0.7)
    a1 = ask("You are a helpful assistant. You give helpful, detailed, and step-by-step answers to the user's questions.", q1, temperature=0.01)

    with open(save_path, "a") as f:
        # 1st turn
        # if is_good_sentence(q1) and is_good_sentence(a1):
        covnersation_list = []
        covnersation_list.append({"role": "user", "content": q1})
        covnersation_list.append({"role": "assistant", "content": a1})
        record = {}
        record["messages"] = covnersation_list
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# %%
