#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    get_jobpostings.py
# @Author:      krnj
# @Time:        25/02/2022 11.59
import glob
import json
import os
from typing import List

from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from utils import split_into_sentences

"""The following variable (SITES) need to be modified as the job postings we have are unfortunately proprietary.
Modify to your own needs."""
SITES = ["DUMMY"]


def flatten(t):
    return [item for sublist in t for item in sublist]


def _load_jsonl(path: str, encoding: str) -> List[str]:
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            text = json.loads(line)["text"]
            data.extend(
                    _preprocess(
                            str(text)
                            .encode("ascii", "ignore")
                            .replace(b"\x1a", b"")
                            .replace(b"\x08", b"")
                            .replace(b"/", b" ")
                            .replace(b"\n", b" ")
                            .decode("ascii", "ignore")
                            )
                    )
    return data


def _preprocess(text: str) -> List[str]:
    processed = []
    sentences = split_into_sentences(text)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        filtered_tokens = [
                token for token in tokens if token not in [".", ",", ":", ";"]
                ]
        processed.append(" ".join(filtered_tokens))

    return processed


def load_jobsite(path: str, encoding: str) -> List[str]:
    """This function might not work for you as these should be raw job postings."""
    path = f"{path}/*.jsonl"
    site = path.split("/")[3]
    jsonl = glob.glob(path)
    res = Parallel(n_jobs=os.cpu_count())(
            delayed(_load_jsonl)(j, encoding) for j in tqdm(jsonl, desc=f"Loading {site}")
            )
    postings = flatten(res)

    return postings


if __name__ == "__main__":
    postings = []
    for site in SITES:
        print(f"Loading {site} postings...")
        postings.extend(
                load_jobsite(
                        f"/data/job-postings/{site}/annotations",
                        encoding="us-ascii",
                        )
                )

    print("Total number of sentences :", len(postings))
