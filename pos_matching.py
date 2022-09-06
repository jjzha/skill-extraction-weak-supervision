#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
from collections import Counter


def get_pos_esco():
    pos_sequences = []

    # file consisting of parsed esco skills
    with open("esco/skills_preds.conll", "r") as fp:
        current_pos_seq = ""
        for p in fp:
            if len(p.split("\t")) > 1:
                tok_id, tok_p, lemma, pos, _, _, _, _, _, _ = p.rstrip().split("\t")
                current_pos_seq += f"{pos} "
            else:
                pos_sequences.append(current_pos_seq.rstrip())
                current_pos_seq = ""

    k, v = [], []

    for seq, val in Counter(pos_sequences).most_common(int(sys.argv[1])):
        k.append(seq)
        v.append(val)

    return k, v


def find_sub_list(l, sl):
    sll = len(sl)

    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind: ind + sll] == sl:
            return ind, ind + sll - 1


def main():
    """For the POS matching, we give an example with the Sayfullina dataset."""
    parsed_say = ["sayfullina_baseline/sayfullina_test_parsed.conll"]

    pos_seqs_esco, _ = get_pos_esco()

    for preds in parsed_say:
        dataset = preds.split("/")[1].split("_")[0]
        with open(preds, "r") as fp, open(
                f"pos_matching/pos_matching_{dataset}_k{sys.argv[1]}.conll", "w"
                ) as fw:
            current_string = ""
            current_pos_seq = ""
            for p in fp:
                if len(p.split("\t")) > 1:
                    tok_id, tok_p, lemma, pos, _, _, _, _, _, _ = p.rstrip().split("\t")
                    current_string += f"{tok_p} "
                    current_pos_seq += f"{pos} "
                else:
                    current_string = current_string.rstrip().split(" ")
                    current_pos_seq = current_pos_seq.rstrip().split(" ")

                    label_sequence = []
                    for esco_pos in pos_seqs_esco:
                        result = find_sub_list(current_pos_seq, esco_pos.split(" "))
                        if result:
                            seq = list(range(result[0], result[1] + 1))
                            for idx, word in enumerate(current_string):
                                if idx in seq:
                                    if "B-Skill" not in label_sequence:
                                        label_sequence.append("B-Skill")
                                    else:
                                        label_sequence.append("I-Skill")
                                else:
                                    label_sequence.append("O")

                            break

                    if not label_sequence:
                        no_skills = "O " * len(current_string)
                        label_sequence.extend(no_skills.rstrip().split(" "))

                    for c, l in zip(current_string, label_sequence):
                        fw.write(f"{c}\t{l}\n")
                    fw.write("\n")

                    current_string = ""
                    current_pos_seq = ""


if __name__ == "__main__":
    main()
