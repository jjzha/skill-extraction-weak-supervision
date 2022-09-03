#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    nlueval.py
# Derived from https://bitbucket.org/robvanderg/xsid/src/master/

import sys


def readNlu(path):
    slots = []
    intents = []
    curSlots = []
    for line in open(path):
        line = line.strip()
        if line.startswith("# intent: "):
            intents.append(line[10:])
        if line == "":
            slots.append(curSlots)
            curSlots = []
        elif line[0] == "#" and len(line.split("\t")) == 1:
            continue
        else:
            curSlots.append(line.split("\t")[-1])
    return slots, intents


def toSpans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] == "B":
            end = beg
            for end in range(beg + 1, len(tags)):
                if tags[end][0] != "I":
                    break
            spans.add(str(beg) + "-" + str(end) + ":" + tags[beg][2:])
    return spans


def getBegEnd(span):
    return [int(x) for x in span.split(":")[0].split("-")]


def getLooseOverlap(spans1, spans2):
    found = 0
    for spanIdx, span in enumerate(spans1):
        spanBeg, spanEnd = getBegEnd(span)
        label = span.split(":")[1]
        match = False
        for span2idx, span2 in enumerate(spans2):
            span2Beg, span2End = getBegEnd(span2)
            label2 = span2.split(":")[1]
            if label == label2:
                if span2Beg >= spanBeg and span2Beg <= spanEnd:
                    match = True
                if span2End <= spanEnd and span2End >= spanBeg:
                    match = True
        if match:
            found += 1

    return found


def getUnlabeled(spans1, spans2):
    return len(
        set([x.split("-")[0] for x in spans1]).intersection(
            [x.split("-")[0] for x in spans2]
        )
    )


def getInstanceScores(predPath, goldPath):
    goldSlots, goldIntents = readNlu(goldPath)
    predSlots, predIntents = readNlu(predPath)
    intentScores = []
    slotScores = []
    for goldSlot, goldIntent, predSlot, predIntent in zip(
        goldSlots, goldIntents, predSlots, predIntents
    ):
        if goldIntent == predIntent:
            intentScores.append(100.0)
        else:
            intentScores.append(0.0)

        goldSpans = toSpans(goldSlot)
        predSpans = toSpans(predSlot)
        overlap = len(goldSpans.intersection(predSpans))
        tp = overlap
        fp = len(predSpans) - overlap
        fn = len(goldSpans) - overlap

        prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
        rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if prec + rec == 0.0 else 2 * (prec * rec) / (prec + rec)
        slotScores.append(f1)
    return slotScores, intentScores


def calculate_metrics(gold_path, pred_path):
    goldSlots, _ = readNlu(gold_path)
    predSlots, _ = readNlu(pred_path)

    tp = 0
    fp = 0
    fn = 0
    fullyCor = 0

    recall_loose_tp = 0
    recall_loose_fn = 0
    precision_loose_tp = 0
    precision_loose_fp = 0

    tp_ul = 0
    fp_ul = 0
    fn_ul = 0

    for goldSlot, predSlot in zip(goldSlots, predSlots):
        # slots
        goldSpans = toSpans(goldSlot)
        predSpans = toSpans(predSlot)
        overlap = len(goldSpans.intersection(predSpans))
        tp += overlap
        fp += len(predSpans) - overlap
        fn += len(goldSpans) - overlap

        overlap_ul = getUnlabeled(goldSpans, predSpans)
        tp_ul += overlap_ul
        fp_ul += len(predSpans) - overlap_ul
        fn_ul += len(goldSpans) - overlap_ul

        overlapLoose = getLooseOverlap(goldSpans, predSpans)
        recall_loose_tp += overlapLoose
        recall_loose_fn += len(goldSpans) - overlapLoose

        overlapLoose = getLooseOverlap(predSpans, goldSpans)
        precision_loose_tp += overlapLoose
        precision_loose_fp += len(predSpans) - overlapLoose

        # fully correct sentences
        if overlap == len(goldSpans) and len(goldSpans) == len(predSpans):
            fullyCor += 1

    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if prec + rec == 0.0 else 2 * (prec * rec) / (prec + rec)

    loose_prec = (
        0.0
        if precision_loose_tp + precision_loose_fp == 0
        else precision_loose_tp / (precision_loose_tp + precision_loose_fp)
    )
    loose_rec = (
        0.0
        if recall_loose_tp + recall_loose_fn == 0
        else recall_loose_tp / (recall_loose_tp + recall_loose_fn)
    )
    loose_f1 = (
        0.0
        if loose_prec + loose_rec == 0.0
        else 2 * (loose_prec * loose_rec) / (loose_prec + loose_rec)
    )

    return {
        "strict": {"precision": prec, "recall": rec, "f1": f1},
        "loose": {"precision": loose_prec, "recall": loose_rec, "f1": loose_f1},
    }


if __name__ == "__main__":
    print(calculate_metrics(sys.argv[1], sys.argv[2]))
