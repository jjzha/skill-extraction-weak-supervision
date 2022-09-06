import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from sklearn.feature_extraction.text import CountVectorizer


def identity(x):
    return x


def mosaic(unigrams, bigrams, trigrams, unicnts, bicnts, tricnts, len_dict, pos_tags, pos_cnts):
    fig, axs = plt.subplot_mosaic([['A', 'A', 'B'], ['C', 'D', 'E'], ['F', 'G']],
                                  figsize=(9, 5), constrained_layout=True)

    for label, ax in axs.items():
        ax.grid(visible=True, axis="both", linestyle=":", color="grey")
        trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='medium', va='bottom', color='blue')

    axs['A'].bar(list(len_dict.keys()), list(len_dict.values()), color='royalblue', width=0.7)
    axs['A'].set_title('Distribution of Length ESCO Skills in Tokens', fontsize=9, alpha=0.7)
    axs['A'].set_ylabel("Frequency", fontsize=10, alpha=0.6)
    axs['A'].set_xlabel("Token Length", fontsize=10, alpha=0.6)
    axs['A'].set_xticks(list(len_dict.keys()))

    axs['B'].bar(unigrams, unicnts, color='darkorange', width=0.7)  # tan #teal
    axs['B'].set_title('Distribution of ESCO Skills (Unigrams)', fontsize=9, alpha=0.7)
    for i, (name, height) in enumerate(zip(unigrams, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
        axs['B'].text(i, height, ' ' + name, color='black',
                      ha='center', va='bottom', rotation=90, fontsize=9)
    axs['B'].set_ylabel("Frequency", fontsize=10, alpha=0.6)
    axs['B'].set_xlabel("Unigram", fontsize=10, alpha=0.6)
    axs['B'].set_xticks([])
    # axs['B'].set_xticklabels(unigrams, rotation=30, ha="right", fontsize=8)

    axs['C'].bar(bigrams, bicnts, color='lightsteelblue', width=0.7)  # tan #teal
    axs['C'].set_title('Distribution of ESCO Skills (Bigrams)', fontsize=9, alpha=0.7)
    for i, (name, height) in enumerate(zip(bigrams, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
        axs['C'].text(i, height, ' ' + name, color='black',
                      ha='center', va='bottom', rotation=90, fontsize=9)
    axs['C'].set_ylabel("Frequency", fontsize=10, alpha=0.6)
    axs['C'].set_xlabel("Bigram", fontsize=10, alpha=0.6)
    # axs['C'].set_xticklabels(bigrams, rotation=30, ha="right", fontsize=8)
    axs['C'].set_xticks([])

    axs['D'].bar(trigrams, tricnts, color='lightskyblue', width=0.7)  # tan #teal
    axs['D'].set_title('Distribution of ESCO Skills (Trigrams)', fontsize=9, alpha=0.7)
    for i, (name, height) in enumerate(zip(trigrams, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
        axs['D'].text(i, height, ' ' + name, color='black',
                      ha='center', va='bottom', rotation=90, fontsize=8)
    axs['D'].set_ylabel("Frequency", fontsize=10, alpha=0.6)
    axs['D'].set_xlabel("Trigram", fontsize=10, alpha=0.6)
    # axs['C'].set_xticklabels(bigrams, rotation=30, ha="right", fontsize=8)
    axs['D'].set_xticks([])

    axs['E'].bar(pos_tags, pos_cnts, color='orchid', width=0.7)
    axs['E'].set_title('Skills POS Sequences in ESCO', fontsize=9, alpha=0.7)
    for i, (name, height) in enumerate(zip(pos_tags, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])):
        axs['E'].text(i, height, ' ' + name, color='black',
                      ha='center', va='bottom', rotation=90, fontsize=9)
    axs['E'].set_ylabel("Frequency", fontsize=10, alpha=0.6)
    axs['E'].set_xlabel("POS Sequence", fontsize=10, alpha=0.6)
    axs['E'].set_xticklabels(pos_tags, rotation=30, ha="right", fontsize=8)
    axs['E'].set_xticks([])

    # plt.show()
    plt.savefig("plots/mosaic.pdf", dpi=300)


def ngram_dist(grams: list, cnts: list) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(visible=True, axis="both", linestyle=":", color="grey")

    ax.bar(grams, cnts, color='teal', width=0.7)  # tan #teal
    ax.set_title('Distribution of N-grams ESCO Skills', fontsize=14, alpha=0.7)
    ax.set_ylabel("Frequency", fontsize=12, alpha=0.6)
    ax.set_xlabel("N-gram", fontsize=12, alpha=0.6)
    plt.yticks(fontsize=12)
    plt.xticks(grams, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/2gram.pdf", dpi=300)


def len_dist(len_dict: Counter) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.grid(visible=True, axis="both", linestyle=":", color="grey")

    ax.bar(list(len_dict.keys()), list(len_dict.values()), color='mediumslateblue', width=0.7)
    ax.set_title('Distribution of Length ESCO Skills in Tokens', fontsize=14, alpha=0.7)
    ax.set_ylabel("Frequency", fontsize=12, alpha=0.6)
    ax.set_xlabel("Token Length", fontsize=12, alpha=0.6)
    plt.yticks(fontsize=12)
    plt.xticks(list(len_dict.keys()), fontsize=12)
    plt.setp(ax.get_xticklabels(), ha="center")
    plt.tight_layout()
    # plt.show()
    plt.savefig("plots/distribution_tokens_new.pdf", dpi=300)


def get_ngrams(gram, corpus):
    vectorizer = CountVectorizer(ngram_range=(gram, gram), stop_words="english")

    ngrams = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()

    count_values = ngrams.toarray().sum(axis=0)
    # output n-grams
    ngram_count = Counter()
    for ng_count, ng_text in zip(count_values, vocab):
        ngram_count[ng_text] = ng_count

    grams = []
    cnts = []
    for ngram, count in ngram_count.most_common(10):
        grams.append(ngram)
        cnts.append(count)

    return grams, cnts


def get_pos_esco():
    pos_sequences = []
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

    for seq, val in Counter(pos_sequences).most_common(10):
        k.append(seq)
        v.append(val)

    return k, v


if __name__ == "__main__":
    classes = defaultdict(list)
    corpus = []
    labels = []
    num_tokens = []

    with open("esco/skills.json", "r") as f:
        data = json.load(f)

        for obj in data:
            if obj.get("title") and (obj.get("className") == "Skill"):
                classes[obj.get("className")].append(obj.get("title"))
                corpus.append(obj.get("title"))
                num_tokens.append(len(obj.get("title").split()))

                if obj.get("hasSkillType") and obj.get("broaderHierarchyConcept"):
                    type = "K" if obj.get("hasSkillType")[0].split("/")[-1] == "knowledge" else ""
                    label = obj.get("broaderHierarchyConcept")[0].split("/")[-1]
                    if label.startswith("L"):
                        labels.append(label)
                    else:
                        labels.append(type + label)
                else:
                    labels.append("None")

    # ngram counts
    unigrams, unicnts = get_ngrams(1, corpus)
    bigrams, bicnts = get_ngrams(2, corpus)
    trigrams, tricnts = get_ngrams(3, corpus)

    # pos tags esco
    pos_tags_esco, pos_cnts_esco = get_pos_esco()

    mosaic(unigrams, bigrams, trigrams, unicnts, bicnts, tricnts, Counter(num_tokens), pos_tags_esco,
           pos_cnts_esco)
