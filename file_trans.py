# from datasets import load_dataset

# # Load dataset (split = "train" by default if the repo has only one split)
# from datasets import Dataset
from tqdm import tqdm
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("/pfs/training-data/hf/models/Qwen/Qwen2.5-7B-Instruct-1M", trust_remote_code=True)


import re
import json
import random
import collections
from pathlib import Path

import random
random.seed(42)

def trans_data(para_list, number, split):
    length = len(para_list)
    start = random.choice([0, 1])
    candidates = list(range(start, length, 2))
    
    # Safety check: ensure we don't request more chunks than exist
    if number > len(candidates):
        number = len(candidates)
        
    # These are the indices in the document that will be missing (e.g., [0, 2, 4])
    missing_indices = sorted(random.sample(candidates, number))
    
    # These are the indices to be used for the Options (A, B, C...)
    # We copy and shuffle them (e.g., [2, 4, 0])
    shuffled_indices = missing_indices[:]
    random.shuffle(shuffled_indices)
    
    # Example: {2: 'A', 4: 'B', 0: 'C'}
    index_to_option = {}
    suffix = "***The missing chunks can be selected from the following options:***\n"
    
    for i, original_idx in enumerate(shuffled_indices):
        option_char = chr(ord("A") + i)
        index_to_option[original_idx] = option_char
        suffix += f"{option_char}. {para_list[original_idx].strip()}\n\n"

    # We need the answer for the 1st missing gap, then the 2nd, etc.
    target = [index_to_option[idx] for idx in missing_indices]
    
    # Construct Prompt
    prompt = """The following document contains missing segments marked as <CHUNK_i>MISSING</CHUNK_i>.
    Please reason about the logical and narrative structure of the document and select appropriate chunks one by one from the given options to reconstruct it.
    Then, output the label for each missing chunk by order in \\boxed{} separated by commas. For example, \\boxed{C, B, A, D}.\n\nThe document is as follows:\n""" 
    
    for i, para in enumerate(para_list):
        if i in missing_indices:
            # Note: Using i+1 preserves the original paragraph number
            prompt += f"<CHUNK_{missing_indices.index(i)+1}>MISSING</CHUNK_{missing_indices.index(i)+1}>\n\n"
        else:
            prompt += para.strip() + "\n\n"
            
    prompt += suffix

    assert isinstance(prompt, str)
    assert isinstance(target, list)
    assert isinstance(split, str)

    data = {
            "data_source": "reconstruct_long",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "long",
            "reward_model": {"style": "rule", "ground_truth": target},
            "extra_info": {"split": split},
            }
    
    return data
    

data_path_book = Path("/pfs/training-data/wanglei/xiaoyao/train_book_data_2_8.json")
data_path_code = Path("/pfs/training-data/wanglei/xiaoyao/train_code_data_2_8.json")
data_path_arxiv = Path("/pfs/training-data/wanglei/xiaoyao/train_arxiv_data_2_8.json")
# book 11014 3.7w
# code 19585 3.6w
# arxiv 15796 3.7w
def load_json_or_jsonl(path: Path):
    """Load either a JSON array/object or JSONL file."""
    with path.open("r") as f:
        raw = f.read().strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        with path.open("r") as f:
            return [json.loads(line) for line in f if line.strip()]

data_book = load_json_or_jsonl(data_path_book)
print(f"Loaded {len(data_book)} records")
data_code = load_json_or_jsonl(data_path_code)
print(f"Loaded {len(data_code)} records")
data_arxiv = load_json_or_jsonl(data_path_arxiv)
print(f"Loaded {len(data_arxiv)} records")

data = []
data.extend(data_book[:10000])
data.extend(data_code[:10000])
data.extend(data_arxiv[:10000])

# data.extend(data_book[-8000:])
# data.extend(data_code[-3000:])
# data.extend(data_arxiv[-3000:])

# data.extend(random.sample(data_book, 8000))
# data.extend(random.sample(data_code, 3000))
# data.extend(random.sample(data_arxiv, 3000))
print(len(data))

if data:
    print(f"Keys of first record: {list(data[0].keys())}")

pattern = re.compile(
    r"<CHUNK_\d+>(.*?)</CHUNK_\d+>",
    re.DOTALL
)
# pretrain_data = []

random.shuffle(data)
res = []
res_test = []
for item in data[0:5000]:
    matches = pattern.findall(item['prompt'][1]['content'])
    # pretrain_data.append({"text":" ".join(matches)})
    res.append(trans_data(matches, 2, "train"))
for item in data[5000:10000]:
    matches = pattern.findall(item['prompt'][1]['content'])
    # pretrain_data.append({"text":" ".join(matches)})
    res.append(trans_data(matches, 4, "train"))
for item in data[10000:20000]:
    matches = pattern.findall(item['prompt'][1]['content'])
    # pretrain_data.append({"text":" ".join(matches)})
    res.append(trans_data(matches, 6, "train"))
for item in data[20000:30000]:
    matches = pattern.findall(item['prompt'][1]['content'])
    # pretrain_data.append({"text":" ".join(matches)})
    res.append(trans_data(matches, 8, "train"))


from datasets import Dataset


# analysis = []
# for item in data[0:5000]:
#     matches = pattern.findall(item['prompt'][1]['content'])
#     # pretrain_data.append({"text":" ".join(matches)})
#     analysis.append(trans_data(matches, 8, "train"))
# ds = Dataset.from_list(analysis)
# ds.to_parquet("./train_5k_8options.parquet")


# for item in data[10000: 10500]:
#     matches = pattern.findall(item['prompt'][1]['content'])
#     res_test.append(trans_data(matches, random.sample([4, 6, 8], 1)[0], "test"))


# save to parquet

ds = Dataset.from_list(res)
# save
ds.to_parquet("./train_long_2468_mix_30k.parquet")
# df = pd.DataFrame(res_test)
# df.to_parquet("./test_long_468.parquet")



# with open("./long_pretrain.jsonl", "w", encoding="utf-8") as f:
#     for item in pretrain_data:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")