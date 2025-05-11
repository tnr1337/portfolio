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


# In[15]:

import collections

import numpy as np
import tqdm

from util import load_score_memo, save_score_memo, get_perplexity_

score_memo = load_score_memo()


def get_perplexity(text):
    return get_perplexity_(calculator, score_memo, text)


def save_text(n_idx, text, verbose=0):
    path_save_idx = path_save / f"{n_idx:04d}"
    if not path_save_idx.exists():
        path_save_idx.mkdir()
    score = calculator.get_perplexity(text)
    if verbose:
        print(f"score:{score:.4f}")
    path_save_text = path_save_idx / f"{score:.4f}.txt"

    with path_save_text.open("w") as f:
        f.write(text)


def greedy(n_idx):
    text = df.iloc[n_idx, 1]
    words = text.split()

    print(f"number of words: {len(words)}")
    n_words = len(words)

    # words_unused = set(words)
    # words_used = set()

    words_unused = collections.Counter(words)

    state = []

    for i in tqdm.trange(n_words):
        # best word and best place to insert
        score_best = np.inf
        state_best = None
        word_best = None

        for word in words_unused.keys():
            assert words_unused[word] > 0
            for i in range(len(state) + 1):
                state_new = state[:i] + [word] + state[i:]
                score = get_perplexity(" ".join(state_new))

                if score < score_best:
                    score_best = score
                    state_best = state_new
                    word_best = word

        assert state_best is not None
        print(f"best score: {score_best}")
        print(f"added word: {word_best}")

        state = state_best
        words_unused[word_best] -= 1
        if words_unused[word_best] == 0:
            del words_unused[word_best]
        # words_used.add(word_best)

        print(state)

        save_score_memo(score_memo)

    score = get_perplexity(" ".join(state))

    return state, score


for n_idx in range(0, len(df)):
    words, score = greedy(n_idx)
    save_text(n_idx, " ".join(words), verbose=1)
