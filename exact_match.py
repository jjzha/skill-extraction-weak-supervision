#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    exact_match.py
# @Author:      krnj
# @Time:        18/03/2022 15.56
import os
from typing import Dict, List, TextIO, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm

from get_jobpostings import flatten
from get_representations import _get_word_idx, _load_skill_set


"""The following variable (SITES) need to be modified as the job postings we have are unfortunately proprietary.
Modify to your own needs."""
SITES = ["DUMMY"]


def _load_job_posting(path: str, column: int) -> List[str]:
    sentences = []
    sentence = ""
    with open(path, "r") as in_f:
        for line in in_f:
            l = line.strip().split("\t")
            if len(l) < 2:
                sentences.append(sentence)
                sentence = ""
                continue
            sentence += f"{l[column]} "
    return [sent.strip() for sent in sentences]


def generate_conll(
        postings: List[str], matches: List[Tuple[List, str, str]], out_file: TextIO
        ):
    for posting in postings:
        tokens = posting.split(" ")[:-1]
        match = list(filter(lambda m: m[1] == posting, matches))
        if len(match) == 0:
            for token in tokens:
                out_file.write(f"{token}\tO\n")
        else:
            skill, knowledge = "O", "O"
            for i, token in enumerate(tokens):
                skills = list(filter(lambda m: i in m[0], match))
                if len(skills) == 0:  # If there are no matches
                    skill, knowledge = "O", "O"
                else:
                    skill = "B-Skill" if skill == "O" else "I-Skill"
                out_file.write(f"{token}\t{skill}\n")
        out_file.write("\n")


def word_indicies(posting: str, skills: Dict) -> List:
    indecies = []
    for skill, tax in skills.items():
        idx = _get_word_idx(posting, skill)
        if idx:
            indecies.append(idx + (tax,))
    return indecies


def check_job_postings(
        postings: List[str], skills: Dict, source: str = "house"
        ) -> List[Tuple[List, str, str]]:
    res = Parallel(n_jobs=os.cpu_count())(
            delayed(word_indicies)(posting, skills)
            for posting in tqdm(postings, desc=f"Processing {source}")
            )
    result = [list(filter(None, r)) for r in res]
    result = flatten([res for res in result if len(res) > 0])
    print(f"{source} -", len(result))

    return result


def main():
    print("loading skills...")

    skills = _load_skill_set("esco/skills.txt")
    print("loaded skill...", len(skills))
    print("loading job-postings...")

    for site in SITES:
        postings = _load_job_posting(f"data/{site}_dev.conll", column=1)
        postings_lemma = _load_job_posting(f"data/{site}_dev.conll", column=2)
        print(f"loaded job-postings for", site, len(postings))
        results = check_job_postings(postings, skills, site)
        results_lemma = check_job_postings(postings_lemma, skills, site)
        generate_conll(postings, results, open(f"data/{site}_preds.cleaned.conll", "w"))
        generate_conll(
                postings_lemma,
                results_lemma,
                open(f"data/{site}_preds_lemma.cleaned.conll", "w"),
                )


if __name__ == "__main__":
    main()
