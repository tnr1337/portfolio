from pathlib import Path

import numpy as np
import pandas as pd

path_input_csv = Path("../../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
# path_save.mkdir(parents=True, exist_ok=True)
path_model = Path("../../input/gemma-2/")

df_input = pd.read_csv(path_input_csv)


def get_path_words_best(n_idx):
    path_save_idx = path_save / f"{n_idx:04d}"
    words_original = df_input.loc[n_idx, "text"].split(" ")

    path_txt = path_save_idx.glob("*.txt")
    list_path_txt = list(path_txt)
    if len(list_path_txt) == 0:
        return None, None
    list_scores = [float(path.stem) for path in list_path_txt]
    # print(list_scores)
    # get min score
    idx_min = np.argmin(list_scores)
    score = list_scores[idx_min]
    # get min score path
    path_min = list_path_txt[idx_min]
    # print(path_min)
    # get min score text
    text_min = path_min.read_text()
    # get min score words
    words_min = text_min.split(" ")
    assert sorted(words_min) == sorted(words_original)

    return score, words_min
