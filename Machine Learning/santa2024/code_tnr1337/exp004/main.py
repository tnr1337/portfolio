#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd

path_input_csv = Path("../../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
path_save.mkdir(parents=True, exist_ok=True)
path_model = Path("../../input/gemma-2/")

df = pd.read_csv(path_input_csv)


# In[2]:


from evaluation import PerplexityCalculator

calculator = PerplexityCalculator(model_path=str(path_model))


# In[3]:


from util import get_path_words_best, save_text
from util import load_score_memo, save_score_memo, get_perplexity_

import copy

import numpy as np
import random


import pickle

score_memo = load_score_memo()


def get_perplexity(text):
    return get_perplexity_(calculator, score_memo, text)


n_idx_total = len(df)
list_words_best = []
list_perplexity_best = []

flag_shuffle = True
flag_use_best = False

for idx in range(n_idx_total):
    if flag_use_best:
        score, list_words = get_path_words_best(idx)
    else:
        text = df.iloc[idx, 1]
        list_words = text.split()
        if flag_shuffle:
            random.shuffle(list_words)
    text = " ".join(list_words)
    list_words_best.append(copy.deepcopy(list_words))
    score_new = get_perplexity(text)
    list_perplexity_best.append(score_new)

    print(f"idx:{idx} score:{score_new:.4f}")

# remember the best perplexity
list_words_best_all = copy.deepcopy(list_words_best)
list_perplexity_best_all = copy.deepcopy(list_perplexity_best)


def update_best_all(n_idx, words, perplexity):
    if perplexity < list_perplexity_best_all[n_idx]:
        list_words_best_all[n_idx] = copy.deepcopy(words)
        list_perplexity_best_all[n_idx] = perplexity


def get_best_all(n_idx):
    return list_words_best_all[n_idx], list_perplexity_best_all[n_idx]


# In[4]:


# dicts_perplexity_memo = [{} for _ in range(n_idx_total)]


# In[5]:


import copy

import numpy as np
import random

# n_idx = 2
# text = df.loc[n_idx, "text"]


# select random word and insert random place
def make_neighbor_1(words):
    # ランダムな単語を選択し、ランダムな場所に挿入
    idx = random.randint(0, len(words) - 1)
    word = words[idx]
    # remove
    words.pop(idx)
    # insert
    idx_insert = random.randint(0, len(words))
    words.insert(idx_insert, word)
    return words


# select random word sequence and insert random place
def make_neighbor_2(words):
    idx1 = random.randint(0, len(words))
    idx2 = random.randint(0, len(words))
    if idx1 == idx2:
        return None
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    words_mid = words[idx1:idx2]
    words = words[:idx1] + words[idx2:]

    idx_insert = random.randint(0, len(words))
    words = words[:idx_insert] + words_mid + words[idx_insert:]
    return words


# select random word sequence and insert at the beginning or the end
def make_neighbor_3(words):
    idx1 = random.randint(1, len(words))
    idx2 = random.randint(1, len(words))
    if idx1 == idx2:
        return None
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    words1 = words[:idx1]
    words2 = words[idx1:idx2]
    words3 = words[idx2:]
    coin = random.randint(1, 2)
    if coin == 1:
        words = words1 + words3 + words2
    else:
        words = words2 + words1 + words3
    return words


# rotate entire
def make_neighbor_4(words):
    # rotate entire
    idx = random.randint(1, len(words) - 1)
    words = words[idx:] + words[:idx]
    return words


# swap adjacent
def make_neighbor_5(words):
    idx = random.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return words


# reverse 区間
def make_neighbor_6(words):
    idx1 = random.randint(0, len(words))
    idx2 = random.randint(0, len(words))
    if idx1 == idx2:
        return None
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    words[idx1:idx2] = words[idx1:idx2][::-1]
    return words


list_prob = {
    1: 10,
    2: 30,
    3: 10,
    4: 1,
    5: 1,
    6: 10,
}

prob_total = sum(list_prob.values())
for key in list_prob.keys():
    list_prob[key] = list_prob[key] / prob_total


def make_neighbor(words_input):
    words_return = None
    while words_return is None:
        words = copy.deepcopy(words_input)
        # list_coin = [1, 2, 3, 4, 5]
        # coin = random.randint(1, 5)
        coin = np.random.choice(list(list_prob.keys()), p=list(list_prob.values()))

        if coin == 1:
            func_neighbor = make_neighbor_1
        elif coin == 2:
            func_neighbor = make_neighbor_2
        elif coin == 3:
            func_neighbor = make_neighbor_3
        elif coin == 4:
            func_neighbor = make_neighbor_4
        elif coin == 5:
            func_neighbor = make_neighbor_5
        elif coin == 6:
            func_neighbor = make_neighbor_6
        else:
            raise ValueError

        words_return = func_neighbor(words)

        if words_return == words_input:
            words_return = None

    assert sorted(words_input) == sorted(words_return)

    return words_return


iter_total = 100
iter_now = 0

n_sample = 32

# words_best = text.split(" ")
# perplexity_best = calculator.get_perplexity(" ".join(words_best))

# print(f"original:{perplexity_best:.2f}")
# print(f" words_best:{words_best}")


def hillclimbing(
    n_idx, words_best, perplexity_best, iter_total=100, n_sample=n_sample, verbose=0
):
    iter_now = 0

    while iter_now < iter_total:
        list_words_nxt = []
        list_texts_nxt = []
        for _ in range(n_sample):
            words_nxt = make_neighbor(words_best)
            list_words_nxt.append(words_nxt)
            list_texts_nxt.append(" ".join(words_nxt))

        # list_perplexity_nxt = calculator.get_perplexity(list_texts_nxt)
        list_perplexity_nxt = [get_perplexity(text) for text in list_texts_nxt]
        # take min
        idx_min = np.argmin(list_perplexity_nxt)
        words_nxt = list_words_nxt[idx_min]
        # perplexity_nxt = list_perplexity_nxt[idx_min]
        perplexity_nxt = get_perplexity(" ".join(words_nxt))

        if iter_now % 50 == 0:
            if verbose:
                print(
                    f"iter:{iter_now} best:{perplexity_best:.2f} nxt:{perplexity_nxt:.2f}"
                )
                # print(f" words_best:{words_best}")

        if perplexity_nxt < perplexity_best:
            perplexity_best = perplexity_nxt
            # perplexity_best_recalculated = calculator.get_perplexity(" ".join(words_nxt))
            # print(list_perplexity_nxt)
            # print(
            #     f"iter:{iter_now} best:{perplexity_best:.2f} recalculated:{perplexity_best_recalculated:.2f}"
            # )
            words_best = copy.deepcopy(words_nxt)
            # print("accepted")

        iter_now += 1

    return words_best, perplexity_best


# In[6]:


list_idx_randomize = []
for idx in list_idx_randomize:
    text = df.iloc[idx, 1]
    list_words = text.split()
    random.shuffle(list_words)
    text = " ".join(list_words)
    list_words_best[idx] = copy.deepcopy(list_words)
    list_perplexity_best[idx] = get_perplexity(text)


# In[7]:


# list_batch_size = [64, 100, 80, 40, 30, 16]
# list_batch_size = [5, 5, 5, 5, 5, 5]
list_batch_size = [1, 1, 1, 1, 1, 1]

list_no_update_cnt = [0] * n_idx_total
list_num_kick = [1] * n_idx_total

# max_no_update_cnt = 1
max_no_update_cnt = 10


def calc_n_kick_and_reset(n_idx):
    n_kick = list_num_kick[n_idx]
    i = 1
    while True:
        if n_kick >= i:
            n_kick = n_kick - i
        else:
            break
        i += 1
        if i > 16:  # kick数は 0, 1, 2, 3, 4, 5, 6
            i = 1
    flag_reset = True if (n_kick == 0 and i >= 2) else False
    n_kick = i - n_kick
    # n_kick -= 1  # to make the final hill climbing longer
    # n_kick = n_kick * 1 - 1
    n_kick = int(np.sqrt(n_kick)) - 1
    return n_kick, flag_reset


def ILS_kick(words, n_kick=2):
    for _ in range(n_kick):
        words = copy.deepcopy(words)
        words = make_neighbor(words)
    return words


# In[8]:


# for i in range(100):
#     print(calc_n_kick_and_reset(0))
#     list_num_kick[0] += 1


# In[9]:


import gc
import torch


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


# In[12]:


n_idx = 0
# list_idx_skip = []
# list_idx_skip = [0]
# list_idx_skip = [0, 1, 2, 3, 4]
list_idx_skip = [0, 4, 5]
while True:
    if n_idx == n_idx_total:
        n_idx = 0

    if n_idx in list_idx_skip:
        n_idx += 1
        continue

    free_memory()

    words_best = list_words_best[n_idx]
    perplexity_best_bef = list_perplexity_best[n_idx]

    print(f"#" * 80)
    print(f"n_idx:{n_idx} perplexity_best:{perplexity_best_bef:.2f}")
    words_best, perplexity_best = hillclimbing(
        n_idx,
        words_best,
        perplexity_best_bef,
        iter_total=100,
        n_sample=list_batch_size[n_idx],
        verbose=1,
    )
    print(f"n_idx:{n_idx} perplexity_best:{perplexity_best:.2f}")

    did_kick = False
    if perplexity_best_bef == perplexity_best:
        list_no_update_cnt[n_idx] += 1
        if list_no_update_cnt[n_idx] >= max_no_update_cnt:
            n_kick, flag_reset = calc_n_kick_and_reset(n_idx)
            list_num_kick[n_idx] += 1
            did_kick = True
            list_no_update_cnt[n_idx] = 0
            if flag_reset:
                print("reset words")
                # words_best = get_path_words_best(n_idx)[1]
                words_best = get_best_all(n_idx)[0]
            print(f"apply {n_kick} kick")
            words_best = ILS_kick(words_best, n_kick=n_kick)
            perplexity_best = get_perplexity(" ".join(words_best))
    else:
        list_no_update_cnt[n_idx] = 0

    list_words_best[n_idx] = words_best
    list_perplexity_best[n_idx] = perplexity_best

    update_best_all(n_idx, words_best, perplexity_best)

    if not did_kick:
        save_text(calculator, n_idx, " ".join(words_best), verbose=1)

    save_score_memo(score_memo)

    n_idx += 1
