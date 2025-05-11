from pathlib import Path
import pandas as pd

path_input_csv = Path("../../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
path_save.mkdir(parents=True, exist_ok=True)
path_model = Path("../../input/gemma-2/")

df = pd.read_csv(path_input_csv)

from evaluation import PerplexityCalculator

calculator = PerplexityCalculator(model_path=str(path_model))

from util import get_path_words_best, save_text
from util import load_score_memo, save_score_memo, get_perplexity_

import copy

import math
import tqdm
import numpy as np
import random


import pickle

score_memo = load_score_memo()


def get_perplexity(text):
    return get_perplexity_(calculator, score_memo, text)


# brute force n_idx=0

n_idx = 0

text = df.iloc[n_idx, 1]
words = text.split()

# all permutations
from itertools import permutations

score_best = np.inf

words_begin = ["reindeer", "mistletoe"]
# if word_begin in words:
#     words.remove(word_begin)
# else:
#     raise ValueError

for word in words_begin:
    if word in words:
        words.remove(word)
    else:
        raise ValueError

n_words = len(words)

for idx, perm in tqdm.tqdm(
    enumerate(permutations(words)), total=math.factorial(n_words)
):
    perm = words_begin + list(perm)
    text = " ".join(perm)
    score = get_perplexity(text)
    if score < score_best:
        score_best = score
        words_best = perm
        save_text(get_perplexity, n_idx, text, verbose=1)

    if idx % 100 == 0:
        save_score_memo(score_memo)


print(f"best score: {score_best:.4f}")

save_score_memo(score_memo)
