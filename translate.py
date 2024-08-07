# %%
# %%


"""
conda activate llmeval
export CUDA_VISIBLE_DEVICES=0
python translate.py 

"""

import joblib
from datasets import load_dataset
import os
from datetime import datetime
from vllm import SamplingParams, LLM
import json
import random
from datasets import load_dataset
import glob
import time
import re
import glob
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# %%
# pidを取得
pid = os.getpid()
random.seed(datetime.now().time().microsecond+int(pid))


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


out_dir = "out_translated_ja"
####################

batch_size = 100
################
# メイン

os.system(f"mkdir -p {out_dir}")
rand_num = random.randint(0, 1000000)
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}_{rand_num}.jsonl"


# %%

# %%
model_name = "cyberagent/calm3-22b-chat"

# model_name="nitky/Oumuamua-7b-instruct-v2"

# model_name = "hatakeyama-llm-team/Tanuki-8B-Instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=6000,
          # max_model_len=7000,
          # gpu_memory_utilization=0.4,
          )


# %%

ds_name = "kanhatakeyama/wizardlm8x22b-logical-math-coding-sft"

ds = load_dataset(ds_name, split="train")

# %%
eng_content_list = []
for r in ds:
    for messages in r["messages"]:
        eng_content_list.append(messages["content"])
print(len(eng_content_list), " english contents")

# %%
en_ja_dict_dir = "out_translated_ja/en_ja_dict.jsonl"


def load_dict():
    en_ja_dict = {}
    for en_ja_dict_path in glob.glob("out_translated_ja/*.jsonl"):
        print("loading ", en_ja_dict_path)
        with open(en_ja_dict_path) as f:
            lines = f.readlines()
        for line in lines:
            d = json.loads(line)
            en_ja_dict[list(d.keys())[0]] = list(d.values())[0]
        # en_ja_dict=joblib.load(en_ja_dict_dir)
    # else:
    #    en_ja_dict = {}
    #
    unfinished_eng_content_list = [
        i for i in eng_content_list if i not in en_ja_dict]

    print(len(unfinished_eng_content_list), "unfinished contents")
    print(len(en_ja_dict), "translated contents")
    return en_ja_dict, unfinished_eng_content_list


# %%

# %%

# %%


def question_to_prompt(question):
    return f"""<|im_start|>system
あなたはプロの翻訳家です。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


def llm_gen(llm, prompt_list):

    outputs = llm.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=0.3,
            max_tokens=2048,
            repetition_penalty=1.2,
            # top_k=50,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


# %%

# %%
while True:
    en_ja_dict, unfinished_eng_content_list = load_dict()
    random.seed(datetime.now().time().microsecond+int(pid))
    n_records = len(unfinished_eng_content_list)
    q_list = []
    id_list = []
    eng_list = []
    try:
        for i in range(batch_size):
            record_id = random.randint(0, n_records)
            text = unfinished_eng_content_list[record_id]
            eng_list.append(text)
            text = f"次のテキストを読みやすい日本語に訳しなさい｡必ず全ての情報を網羅し､翻訳文以外は何も出力しないこと｡\n[英文開始]\n{text}[英文終了]\n#日本語の翻訳\n"
            q_list.append(text)
            id_list.append(record_id)
    except:
        time.sleep(10)
        continue

    # %%

    prompt_list = [question_to_prompt(text) for text in q_list]
    answer_list = llm_gen(llm, prompt_list)

    for i in range(len(q_list)):
        reject_flag = False
        if answer_list[i] == "":
            continue
        if get_longest_phrase_length(answer_list[i]) > 100:
            continue

        en_ja_dict[eng_list[i]] = answer_list[i]

        # unfinished_eng_content_list.pop(id_list[i])

        with open(out_path, "a") as f:
            f.write(json.dumps(
                {eng_list[i]: answer_list[i]}, ensure_ascii=False)+"\n")

    # joblib.dump(en_ja_dict,en_ja_dict_dir)

# %%

# %%
