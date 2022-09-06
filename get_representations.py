#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    get_representations.py
# @Author:      mikz
# @Time:        23/02/2022 17.25
import re
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from get_jobpostings import SITES, load_jobsite

EXAMPLE_SENTENCES = [
        "being proficient in Python and Spark",
        "You need to have experience in Python",
        "At least 5 years of experience in coding in Python",
        "lol",
        "being more proficient in Python",
        ]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = ""  # RobertaTokenizerFast.from_pretrained("roberta-base")
MODEL = ""  # RobertaModel.from_pretrained("roberta-base").to(DEVICE)


# TOKENIZER = BertTokenizerFast.from_pretrained("jjzha/jobbert-base-cased")
# MODEL = BertModel.from_pretrained("jjzha/jobbert-base-cased").to(DEVICE)


def _load_skill_set(path: str) -> Dict:
    skills = dict()
    with open(path, "r") as skill_f:
        for line in skill_f:
            skill, tax = line.strip().split("\t")
            skill = re.sub("[\(\[].*?[\)\]]", "", skill)
            skills[skill] = tax
    return skills


def _get_idf_dict(sents: List[str]) -> Dict[str, float]:
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=False, token_pattern=r"\S+")
    vectorizer.fit_transform(sents)
    idf = vectorizer.idf_
    assert len(vectorizer.get_feature_names()) == len(
            set(vectorizer.get_feature_names())
            ), f"Something odd is going on here: {len(vectorizer.get_feature_names())}; " \
               f"{len(set(vectorizer.get_feature_names()))}"
    return dict(zip(vectorizer.get_feature_names(), idf))


def _get_word_idx(sent: str, span: str) -> Tuple[List, str]:
    if span == "":
        return [], sent
    sent_split, span_split = sent.split(), span.split()
    results = []
    sub_list = len(span_split)

    for ind in (i for i, e in enumerate(sent_split) if e == span_split[0]):
        if sent_split[ind: ind + sub_list] == span_split:
            results.append((ind, ind + sub_list - 1))

    if len(results) >= 1:
        # return list of indices of relevance of span
        return list(range(results[0][0], results[0][1] + 1)), sent


def _get_hidden_states(output, token_ids_word):
    # Get last hidden states
    output = output.last_hidden_state.squeeze()
    # Only select the first sub-word token that constitute the requested word
    word_tokens_output = output[token_ids_word[0][0]]
    return word_tokens_output


def get_word_vector(
        sent: str, indices: List[int], tokenizer=None, model=None
        ) -> List[torch.Tensor]:
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    if not tokenizer:
        tokenizer = TOKENIZER
    if not model:
        model = MODEL
    encoded = tokenizer.encode_plus(
            sent,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            ).to(DEVICE)
    with torch.no_grad():
        output = model(**encoded)

    # get all token idxs that belong to the word of interest, you need *TokenizerFast for .word_ids()
    hidden_states = []
    for idx in indices:
        token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
        if len(token_ids_word[0]) > 0:
            hidden = _get_hidden_states(output, token_ids_word)
            hidden_states.append(hidden)

    return hidden_states


def get_skill_isolation(skill: str, tokenizer=None, model=None) -> torch.Tensor:
    if not tokenizer:
        tokenizer = TOKENIZER
    if not model:
        model = MODEL
    encoded_skills = tokenizer(skill, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**encoded_skills)  # -> (batch_size, max_len, emb_dim)

    return output.last_hidden_state[0, 0, :]


def average_over_context(
        sentences: List[str], span: str, _: Dict
        ) -> Tuple[torch.Tensor, int]:
    hits = [_get_word_idx(sent, span) for sent in sentences]
    hits = list(filter(None, hits))
    # average the list of tensors
    tensors = []
    if len(hits) > 0:
        for indices, sent in hits:
            tensors.append(
                    torch.mean(
                            torch.stack(get_word_vector(sent, indices)),
                            dim=0,
                            )
                    )
        return torch.mean(torch.stack(tensors), dim=0), len(hits)
    return None, 0


def get_weighted_span_embeds(
        sentences: List[str], span: str, idf_dict: Dict
        ) -> Tuple[torch.Tensor, int]:
    hits = [_get_word_idx(sent, span) for sent in sentences]
    hits = list(filter(None, hits))
    sent_embeddings = []

    if len(hits) > 0:
        for idx, sent in hits:
            tensor_list = []
            for word, word_vec in zip(span.split(" "), get_word_vector(sent, idx)):
                tensor_list.append(idf_dict[word] * word_vec)
            sent_embeddings.append(torch.sum(torch.stack(tensor_list), dim=0))

        return torch.mean(torch.stack(sent_embeddings), dim=0), len(hits)
    return None, 0


def get_embeds(
        sentences: List[str], span: str, idf_dict: Dict
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    hits = [_get_word_idx(sent, span) for sent in sentences]
    hits = list(filter(None, hits))
    aoc_vectors = []
    wse_vectors = []
    iso = get_skill_isolation(span)

    if len(hits) > 0:
        for idx, sent in hits:
            word_vectors = get_word_vector(sent, idx)
            if len(word_vectors) < 1:
                continue
            aoc_vectors.append(
                    torch.mean(
                            torch.stack(word_vectors),
                            dim=0,
                            )
                    )
            tensor_list = []
            for word, word_vec in zip(span.split(" "), word_vectors):
                tensor_list.append(idf_dict[word][()] * word_vec)
            wse_vectors.append(torch.sum(torch.stack(tensor_list), dim=0))
        aoc = torch.mean(torch.stack(aoc_vectors), dim=0)
        wse = torch.mean(torch.stack(wse_vectors), dim=0)
        return aoc, wse, iso, len(hits)
    return None, None, iso, 0


def save_representations(
        path: str, dname: str, skills: Dict, sentences: List[str]
        ) -> None:
    with h5py.File(path, mode="a") as dataset:
        if not dname in dataset:
            dataset.create_group(dname)
        d_set = dataset[dname]
        if not "ISO" in d_set:
            d_set.create_group("ISO")
        if not "AOC" in d_set:
            d_set.create_group("AOC")
        if not "WSE" in d_set:
            d_set.create_group("WSE")
        if not "idf" in dataset:
            print("Generating IDF scores...")
            idf_dict = dataset.create_group("idf")
            idf = _get_idf_dict(sentences)
            for word, vec in idf.items():
                idf_dict[word] = vec
        idf_dict = dataset["idf"]

        for skill, tax in tqdm(
                skills.items(), desc="Saving representations for skills in esco"
                ):
            if skill.split("/")[0] in d_set["AOC"]:
                continue
            aoc, wse, iso, hits = get_embeds(sentences, skill.split("/")[0], idf_dict)
            if hits > 0:
                d_set["AOC"][skill.split("/")[0]] = aoc.cpu()
                d_set["WSE"][skill.split("/")[0]] = wse.cpu()
            else:
                d_set["AOC"][skill.split("/")[0]] = torch.empty(
                        (1,), dtype=torch.float16
                        )
                d_set["WSE"][skill.split("/")[0]] = torch.empty(
                        (1,), dtype=torch.float16
                        )
            d_set["ISO"][skill.split("/")[0]] = iso.cpu()
            d_set["ISO"][skill.split("/")[0]].attrs["hits"] = hits
            d_set["ISO"][skill.split("/")[0]].attrs["esco_tax"] = tax
            d_set["AOC"][skill.split("/")[0]].attrs["hits"] = hits
            d_set["AOC"][skill.split("/")[0]].attrs["esco_tax"] = tax
            d_set["WSE"][skill.split("/")[0]].attrs["hits"] = hits
            d_set["WSE"][skill.split("/")[0]].attrs["esco_tax"] = tax
    return


def main():
    # words in isolation
    print("loading skills...")
    skills = _load_skill_set("data/skills.txt")  # esco skills
    print("done loading skills...", len(skills))
    print("loading sentences...")
    sentences = []

    # This part might not work for you as the job postings we extract sentences from for AOC and WSE are proprietary.
    for site in SITES:
        sentences.extend(
                load_jobsite(
                        f"/data/job-postings/{site}/annotations",
                        encoding="utf-8",
                        )
                )
    print("done loading sentences...", len(sentences))
    ############################

    save_representations(
            "/data/job-postings/representations/skills.h5",
            "RoBERTa",
            skills,
            sentences,
            )


if __name__ == "__main__":
    main()
