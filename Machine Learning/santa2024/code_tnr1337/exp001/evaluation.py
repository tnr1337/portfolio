"""Evaluation metric for Santa 2024."""

import gc
import os
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import transformers
import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    model_path: str = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2",
    load_in_8bit: bool = True,
    clear_mem: bool = False,
) -> float:
    """
    Calculates the mean perplexity of submitted text permutations compared to an original text.

    Parameters
    ----------
    solution : DataFrame
        DataFrame containing the original text in a column named 'text'.
        Includes a row ID column specified by `row_id_column_name`.

    submission : DataFrame
        DataFrame containing the permuted text in a column named 'text'.
        Must have the same row IDs as the solution.
        Includes a row ID column specified by `row_id_column_name`.

    row_id_column_name : str
        Name of the column containing row IDs.
        Ensures aligned comparison between solution and submission.

    model_path : str
        Path to the serialized LLM.

    clear_mem : bool
        Clear GPU memory after scoring by clearing the CUDA cache.
        Useful for testing.

    Returns
    -------
    float
        The mean perplexity score. Lower is better.

    Raises
    ------
    ParticipantVisibleError
        If the submission format is invalid or submitted strings are not valid permutations.

    Examples
    --------
    >>> import pandas as pd
    >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
    >>> solution = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["this is a normal english sentence", "the quick brown fox jumps over the lazy dog"]
    ... })
    >>> submission = pd.DataFrame({
    ...     'id': [0, 1],
    ...     'text': ["sentence english normal a is this", "lazy the over jumps fox brown quick the dog"]
    ... })
    >>> score(solution, submission, 'id', model_path=model_path, clear_mem=True) > 0
    True
    """
    # Check that each submitted string is a permutation of the solution string
    sol_counts = solution.loc[:, "text"].str.split().apply(Counter)
    sub_counts = submission.loc[:, "text"].str.split().apply(Counter)
    invalid_mask = sol_counts != sub_counts
    if invalid_mask.any():
        raise ParticipantVisibleError(
            "At least one submitted string is not a valid permutation of the solution string."
        )

    # Calculate perplexity for the submitted strings (not the solution strings - those are the reference)
    sub_strings = submission["text"].tolist()
    scorer = PerplexityCalculator(
        model_path=model_path,
        load_in_8bit=load_in_8bit,
    )  # Initialize the perplexity calculator with a pre-trained model
    perplexities = scorer.get_perplexity(
        sub_strings
    )  # Calculate perplexity for each submitted string

    if clear_mem:
        # Just move on if it fails. Not essential if we have the score.
        try:
            scorer.clear_gpu_memory()
        except:
            print("GPU memory clearing failed.")

    return float(np.mean(perplexities))


class PerplexityCalculator:
    """
    Calculates perplexity of text using a pre-trained language model.

    Adapted from https://github.com/asahi417/lmppl/blob/main/lmppl/ppl_recurrent_lm.py

    Parameters
    ----------
    model_path : str
        Path to the pre-trained language model

    max_length : int, default=None
        Maximum sequence length for the model. Defaults to None (model's max length).

    load_in_8bit : bool, default=True
        Use 8-bit quantization for the model. Requires CUDA.

    device_map : str, default="auto"
        Device mapping for the model.
    """

    def __init__(
        self,
        model_path: str,
        max_length: Optional[int] = None,
        load_in_8bit: bool = True,
        device_map: str = "auto",
    ):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        # Configure model loading based on quantization setting and device availability
        if load_in_8bit:
            if DEVICE.type != "cuda":
                raise ValueError("8-bit quantization requires CUDA device")
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
                device_map=device_map,
            )

        # Check max_length
        self.max_length = max_length
        if self.max_length is not None:
            assert (
                self.max_length <= self.tokenizer.model_max_length
            ), f"{self.max_length} > {self.tokenizer.model_max_length}"

        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()
        if not load_in_8bit:
            self.model.to(DEVICE)  # Explicitly move the model to the device

    def get_perplexity(
        self, input_texts: Union[str, List[str]], batch_size: Optional[int] = None
    ) -> Union[float, List[float]]:
        """
        Calculates the perplexity of given texts.

        Parameters
        ----------
        input_texts : str or list of str
            A single string or a list of strings.

        batch_size : int, default=None
            Batch size for processing. Defaults to the number of input texts.

        verbose : bool, default=False
            Display progress bar.

        Returns
        -------
        float or list of float
            A single perplexity value if input is a single string,
            or a list of perplexity values if input is a list of strings.

        Examples
        --------
        >>> import pandas as pd
        >>> model_path = "/kaggle/input/gemma-2/transformers/gemma-2-9b/2"
        >>> scorer = PerplexityCalculator(model_path=model_path)

        >>> submission = pd.DataFrame({
        ...     'id': [0, 1, 2],
        ...     'text': ["this is a normal english sentence", "thsi is a slihgtly misspelled zr4g sentense", "the quick brown fox jumps over the lazy dog"]
        ... })
        >>> perplexities = scorer.get_perplexity(submission["text"].tolist())
        >>> perplexities[0] < perplexities[1]
        True
        >>> perplexities[2] < perplexities[0]
        True

        >>> submission = pd.DataFrame({
        ...     'id': [0, 1],
        ...     'text': ["1 + 1 = 2", "1 + 1 = 3"]
        ... })
        >>> perplexities = scorer.get_perplexity(submission["text"].tolist())
        >>> perplexities[0] < perplexities[1]
        True

        >>> perplexities = scorer.get_perplexity(["this is a sentence", "another sentence"])
        >>> all(p > 0 for p in perplexities)
        True

        >>> scorer.clear_gpu_memory()
        """
        # Ensure inputs are always lists
        single_input = isinstance(input_texts, str)
        input_texts = [input_texts] if single_input else input_texts

        # Either use a predefined batch size or else process the entire input as one batch
        batch_size = len(input_texts) if batch_size is None else batch_size

        # Get slicing indices
        batch_id = list(range(0, len(input_texts), batch_size)) + [len(input_texts)]
        batch_id = list(
            zip(batch_id[:-1], batch_id[1:])
        )  # list of (start, end) indices

        loss_list = []
        with torch.no_grad():
            for s, e in batch_id:
                # Tokenize input texts
                if self.max_length is not None:
                    model_inputs = self.tokenizer(
                        input_texts[s:e],
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                else:
                    model_inputs = self.tokenizer(
                        input_texts[s:e],
                        truncation=False,
                        padding=True,
                        return_tensors="pt",
                    )
                if (
                    "token_type_ids" in model_inputs
                ):  # Remove token_type_ids if present (not needed for Causal LMs)
                    model_inputs.pop("token_type_ids")

                model_inputs = {
                    k: v.to(DEVICE) for k, v in model_inputs.items()
                }  # Ensure inputs are on the correct device
                output = self.model(
                    **model_inputs, use_cache=False
                )  # Get model output logits
                logit = output["logits"]

                # Prepare labels and shift for causal language modeling
                label = model_inputs["input_ids"]
                label[label == self.tokenizer.pad_token_id] = (
                    PAD_TOKEN_LABEL_ID  # Mask padding tokens for loss calculation
                )

                shift_logits = logit[..., :-1, :].contiguous()
                shift_label = label[:, 1:].contiguous()

                # Calculate loss, accounting for valid sequence lengths (excluding padding)
                valid_length = (shift_label != PAD_TOKEN_LABEL_ID).sum(dim=-1)
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_label.view(-1)
                )
                loss = loss.view(len(output["logits"]), -1)
                loss = torch.sum(loss, -1) / valid_length
                loss_list += loss.cpu().tolist()

        ppl = [exp(i) for i in loss_list]
        return ppl[0] if single_input else ppl

    def clear_gpu_memory(self) -> None:
        """Clears GPU memory by deleting references and emptying caches."""
        if not torch.cuda.is_available():
            return

        # Delete model and tokenizer if they exist
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        # Run garbage collection
        gc.collect()

        # Clear CUDA cache and reset memory stats
        with DEVICE:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
