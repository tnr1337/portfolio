#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import pandas as pd

path_input_csv = Path("../../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
path_save.mkdir(parents=True, exist_ok=True)
path_model = Path("../../input/gemma-2/")

df = pd.read_csv(path_input_csv)


# In[ ]:


from evaluation import PerplexityCalculator

calculator = PerplexityCalculator(model_path=str(path_model))


# In[ ]:


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

    n_words = len(words)
    print(f"number of words: {n_words}")

    state = [([word], get_perplexity(word)) for word in words]

    for _ in tqdm.trange(n_words - 1):
        assert 1 <= len(state) <= n_words
        gain_best = -np.inf
        size_state = len(state)
        idx_pair_best = None
        words_new = None
        score_new = None
        for i in range(size_state):
            for j in range(size_state):
                if i == j:
                    continue

                words1, score1 = state[i]
                words2, score2 = state[j]

                for k in range(len(words1)):
                    words12 = words1[:k] + words2 + words1[k:]
                    score12 = get_perplexity(" ".join(words12))

                    # gain = score1 + score2 - score12
                    gain = (
                        score1 * len(words1)
                        + score2 * len(words2)
                        - score12 * len(words12)
                    ) / len(words12)

                    if gain > gain_best:
                        gain_best = gain
                        idx_pair_best = (i, j)
                        words_new = words12
                        score_new = score12

        assert idx_pair_best is not None

        state_nxt = []
        for i in range(size_state):
            if i in idx_pair_best:
                continue
            state_nxt.append(state[i])
        state_nxt.append((words_new, score_new))
        state = state_nxt

        score_total = sum(score * len(words) for words, score in state) / n_words

        print(f"gain: {gain_best}")
        print(f"total score: {score_total:.4f}")
        print(f"{words1} + {words2}")

        save_score_memo(score_memo)

    assert len(state) == 1
    words, score = state[0]

    # for i in tqdm.trange(n_words):
    #     # best word and best place to insert
    #     score_best = np.inf
    #     state_best = None
    #     word_best = None

    #     for word in words_unused:
    #         for i in range(len(state) + 1):
    #             state_new = state[:i] + [word] + state[i:]
    #             score = calculator.get_perplexity(" ".join(state_new))

    #             if score < score_best:
    #                 score_best = score
    #                 state_best = state_new
    #                 word_best = word

    #     assert state_best is not None
    #     print(f"best score: {score_best}")
    #     print(f"added word: {word_best}")

    #     state = state_best
    #     words_unused.remove(word_best)
    #     words_used.add(word_best)

    #     print(state)

    # score = calculator.get_perplexity(" ".join(state))

    return words, score


for n_idx in range(0, len(df)):
    words, score = greedy(n_idx)
    save_text(n_idx, " ".join(words), verbose=1)
