from pathlib import Path
import pandas as pd

path_input_csv = Path("../../input/santa-2024/sample_submission.csv")
path_save = Path("./save")
path_save.mkdir(parents=True, exist_ok=True)
path_model = Path("../../input/gemma-2/")


from evaluation import PerplexityCalculator

calculator = PerplexityCalculator(model_path=str(path_model))

from util import load_score_memo, get_perplexity_, save_score_memo

score_memo = load_score_memo()


def get_perplexity(text):
    return get_perplexity_(calculator, score_memo, text)


df = pd.read_csv(path_input_csv)

for n_exp in range(1, 5):
    # n_exp = 1
    # 後で n_exp=1 も計算

    path_save_bef = Path(f"../exp{n_exp:03d}/save")

    from util import save_text

    for n_idx in range(len(df)):
        # n_idx = 3

        path_save_idx = path_save / f"{n_idx:04d}"
        path_save_idx.mkdir(parents=True, exist_ok=True)
        path_save_bef_idx = path_save_bef / f"{n_idx:04d}"

        # get all txt files
        path_txt = path_save_bef_idx.glob("*.txt")
        list_path_txt = list(path_txt)

        # save all text
        for path in list_path_txt:
            text = path.read_text()
            save_text(get_perplexity, n_idx, text, verbose=2)

        save_score_memo(score_memo)

        # 本来であれば break を入れずにforをまわす
        # break

    # break
