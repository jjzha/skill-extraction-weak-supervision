#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    weak_supervision.py
# @Author:      krnj
# @Time:        30/03/2022 11.59

import argparse
import os
from functools import partial
from typing import Any, List, Optional, Tuple

import h5py
import numpy as np
import ray
import torch
from h5py import Group
from nltk.util import ngrams
from numpy.linalg import norm
from rich.console import Console
from rich.progress import track
from transformers import (BertModel, BertTokenizerFast, RobertaModel, RobertaTokenizerFast)

from exact_match import _load_job_posting
from get_jobpostings import flatten
from get_representations import (DEVICE, _get_word_idx, get_skill_isolation, get_word_vector)

# E_TYPE = Literal["ISO", "AOC", "WSE"]
# M_TYPE = Literal["RoBERTa", "JobBERT"]

ray.init()


def get_model(model_type: str):
    if model_type == "RoBERTa":
        token = RobertaTokenizerFast.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base").to(DEVICE)
    elif model_type == "JobBERT":
        token = BertTokenizerFast.from_pretrained("jjzha/jobbert-base-cased)
        model = BertModel.from_pretrained("jjzha/jobbert-base-cased").to(DEVICE)
        else:
        raise ValueError(f"{model_type} if not a supported model")
    return token, model


def get_cos_sim(a: Any, b: Any) -> float:
    return np.dot(a, b) / (norm(a) * norm(b))


def load_skill_embeddings(path: str, model_type: str, embedding_type: str) -> Group:
    dataset = h5py.File(path, mode="r")
    return {skill: embed[()] for skill, embed in dataset[model_type][embedding_type].items()}


def get_ngrams(sent: str) -> List[str]:
    text_grams = flatten(
            [list(ngrams(sequence=sent.split(), n=i)) for i in range(1, 5)]
            )
    text_grams = [" ".join(ngram) for ngram in text_grams]
    return text_grams


def embed_ngrams(
        ngrams: List[str], context: Optional[str], tokenizer, model
        ) -> List[torch.Tensor]:
    tensors = []
    for ngram in ngrams:
        if context:
            idx, _ = _get_word_idx(context, ngram)
            word_vectors = get_word_vector(context, idx, tokenizer, model)
            if len(word_vectors) < 1:
                tensors.append(torch.empty((1,), dtype=torch.float16))
                continue
            tensors.append(
                    torch.mean(
                            torch.stack(word_vectors),
                            dim=0,
                            ).cpu()
                    )
        else:
            tensors.append(get_skill_isolation(ngram, tokenizer, model).cpu())

    return tensors


@ray.remote
def find_skill(
        skill_embeds: Group, embed: torch.Tensor, ngram: str
        ) -> Tuple[str, Tuple[str, float]]:
    if embed.size(0) == 1:
        return (ngram, ("", -1))

    cosSim = []
    for skill, s_embed in skill_embeds.items():
        # print(skill, type(s_embed))
        cosSim.append(
                (skill, get_cos_sim(embed, s_embed)) if s_embed.size > 1 else (skill, None)
                )

    cosSim = filter(lambda x: x[1] != None, cosSim)
    return (ngram, max(cosSim, key=lambda x: x[1]))


def write_to_file(out_f, sentence: str, prediction: str) -> None:
    tokens = sentence.split()
    idx, _ = _get_word_idx(sentence, prediction)
    for i, token in enumerate(tokens):
        if len(idx) > 0:
            if i == idx[0]:
                out_f.write(f"{token}\tB-Skill\n")
            elif i in idx[1:]:
                out_f.write(f"{token}\tI-Skill\n")
            else:
                out_f.write(f"{token}\tO\n")
        else:
            out_f.write(f"{token}\tO\n")
    out_f.write("\n")


def main(args) -> None:
    console = Console()
    console.log(f"Running script with: {args}")
    path = f"data/{args.site}_{args.set}.{args.model}.{args.embed_type}." \
           f"{'contextual' if args.contextual else 'isolated'}.{args.threshold}.preds.conll"
    if os.path.exists(path):
        console.log("This has already been run")
        exit()
    with console.status("[bold green]Loading data..."):
        postings = _load_job_posting(f"data/{args.site}_dev.conll", column=1)
        console.log(f"Loaded Job postings")
        skill_embeds = ray.put(
                load_skill_embeddings(
                        "/data/job-postings/representations/skills.h5",
                        args.model,
                        args.embed_type,
                        )
                )
        console.log("Loaded skill embeddings")
        pred_file = partial(
                write_to_file,
                out_f=open(
                        path,
                        "w",
                        ),
                )
        tokenizer, model = get_model(model_type=args.model)
        console.log("Loaded model")
        embed_gram = partial(embed_ngrams, tokenizer=tokenizer, model=model)

    for sent in track(
            postings, description="Analysing job postings...", console=console
            ):
        ngrams = get_ngrams(sent)
        ngram_embeds = embed_gram(ngrams, sent if args.contextual else None)
        sent_sim = ray.get(
                [
                        find_skill.remote(skill_embeds, embed, ngram)
                        for ngram, embed in zip(ngrams, ngram_embeds)
                        ]
                )
        best_res = max(sent_sim, key=lambda x: x[1][1])
        if best_res[1][1] > args.threshold:
            pred_file(sentence=sent, prediction=best_res[0])
        else:
            pred_file(sentence=sent, prediction="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find skills by similarity")
    parser.add_argument(
            "--site",
            required=True,
            choices=["house", "tech", "sayfullina"],
            help="Jobposting site to eval on",
            )
    parser.add_argument(
            "--set",
            default="dev",
            choices=["train", "dev", "test"],
            help="What set should be predicted on",
            )
    parser.add_argument(
            "--model",
            required=True,
            default="RoBERTa",
            choices=["RoBERTa", "JobBERT"],
            help="What model embeddings to use",
            )
    parser.add_argument(
            "--embed_type",
            required=True,
            default="ISO",
            choices=["ISO", "AOC", "WSE"],
            help="What kind of skill embeddings should be used as base knowledge",
            )
    parser.add_argument(
            "--threshold",
            default=0,
            type=float,
            help="Threshold to select best similarity score. Set to 0 to always predict a skill.",
            )
    parser.add_argument(
            "--contextual",
            action="store_true",
            help="Contextualize the ngram embeddings",
            )

    main(args=parser.parse_args())
