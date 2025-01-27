import argparse
import json
import math
import os
import random
import string

import nltk
import nltk.data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", "-i", type=str, default="data/topicalchat_subset.json"
    )
    parser.add_argument("--output_path", "-o", type=str, default="data/tc_attacks.json")
    return parser.parse_args()


def download_nltk_data():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)

    resources = [
        "stopwords",
        "averaged_perceptron_tagger_eng",
        "punkt",
        "maxent_ne_chunker",
        "words",
        "punkt_tab",
    ]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)


def make_attacks(
    prediction, context=None, grounded_context=None, seperator="<|endoftext|>"
):
    download_nltk_data()

    adversaries = []
    adversaries.append(("spktag_teacher", "teacher: " + prediction))
    adversaries.append(("spktag_agent", "agent: " + prediction))
    adversaries.append(("spktag_user", "user: " + prediction))

    adversaries.append(("static_hello", "Hello"))
    adversaries.append(("static_idk", "I don't know"))
    adversaries.append(("static_idkwdyt", "I don't know, what do you think?"))
    adversaries.append(("static_idkwdytit", "I don't know, what do you think? I think"))
    adversaries.append(("static_sorry", "I'm sorry, can you repeat?"))
    adversaries.append(("static_willdo", "I will do"))
    adversaries.append(("static_fantastic", "fantastic! how are you?"))

    adversaries.append(("ungram_nopunc", remove_punctuation(prediction)))
    adversaries.append(("ungram_nostop", remove_nltk_stopwords(prediction)))
    adversaries.append(("ungram_nounverb", keep_only_nouns_and_verbs(prediction)))
    adversaries.append(("ungram_noun", keep_only_nouns(prediction)))
    adversaries.append(("ungram_scramble", jumble_words_in_sentence(prediction)))
    adversaries.append(("ungram_reverse", reverse_words_in_sentence(prediction)))
    adversaries.append(("ungram_repeat", repeat_some_words_in_response(prediction)))

    if context is not None:
        adversaries.append(("contextrep_contextonly", context.split(seperator)[-1]))
        adversaries.append(
            ("contextrep_contextplus", context.split(seperator)[-1] + " " + prediction)
        )

    if grounded_context is not None:
        grounded_context_sentences = nltk.sent_tokenize(grounded_context)
        adversaries.append(
            ("contextrep_factonly", random.choice(grounded_context_sentences))
        )

    return adversaries


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_nltk_stopwords(text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)


def keep_only_nouns_and_verbs(text):
    word_tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(word_tokens)
    filtered_sentence = [
        w[0]
        for w in tagged
        if w[1]
        in [
            "NN",
            "NNS",
            "NNP",
            "NNPS",
            "PRP",
            "PPZ",
            "VB",
            "VBD",
            "VBG",
            "VBN",
            "VBP",
            "VBZ",
            "WP",
            "WP$",
        ]
    ]
    return " ".join(filtered_sentence)


def keep_only_nouns(text):
    word_tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(word_tokens)
    filtered_sentence = [w[0] for w in tagged if w[1] in ["NN", "NNS", "NNP", "NNPS"]]
    return " ".join(filtered_sentence)


def jumble_words_in_sentence(text):
    if len(text.split()) < 2:
        return text

    word_tokens = nltk.word_tokenize(text)
    random.shuffle(word_tokens)
    shuffled = " ".join(word_tokens)

    max_tries = 10
    while shuffled == text:
        random.shuffle(word_tokens)
        shuffled = " ".join(word_tokens)
        max_tries -= 1
        if max_tries == 0:
            return text

    return shuffled


def reverse_words_in_sentence(text):
    if len(text.split()) < 2:
        return text

    word_tokens = nltk.word_tokenize(text)
    word_tokens.reverse()

    return " ".join(word_tokens)


def repeat_some_words_in_response(text, percentage=0.2):
    word_tokens = text.split()
    num_words = len(word_tokens)
    num_words_to_repeat = math.ceil(num_words * percentage)
    if num_words_to_repeat == 0:
        return text

    random_indices = random.sample(range(num_words), num_words_to_repeat)
    for index in random_indices:
        word_tokens.insert(index, word_tokens[index])

    return " ".join(word_tokens)


def compile_attacks_with_context(prediction, context, grounded_context, attacks):
    reference = {
        "system_id": "reference",
        "system_output": prediction,
        "source": context,
        "context": grounded_context,
    }
    attack_entries = [
        {
            "system_id": attack_type,
            "system_output": attack,
            "source": context,
            "context": grounded_context,
        }
        for attack_type, attack in attacks
    ]
    return [reference] + attack_entries


def compile_attacks_without_context(prediction, context, attacks):
    reference = {
        "system_id": "reference",
        "system_output": prediction,
        "source": context,
    }
    attack_entries = [
        {
            "system_id": attack_type,
            "system_output": attack,
            "source": context,
        }
        for attack_type, attack in attacks
    ]
    return [reference] + attack_entries


def make_attacks_from_data(data):
    attack_entries = []
    for entry in data:
        if entry.get("system_id", None) == "Original Ground Truth":
            prediction = entry["system_output"]
            context = entry["source"]
            grounded_context = entry.get("context", None)
            attacks = make_attacks(
                prediction, context, grounded_context, seperator="<|endoftext|>"
            )
            if grounded_context:
                attack_entries.append(
                    compile_attacks_with_context(
                        prediction, context, grounded_context, attacks
                    )
                )
            else:
                attack_entries.append(
                    compile_attacks_without_context(prediction, context, attacks)
                )

    return attack_entries


def main():
    args = parse_args()
    with open(args.input_path, "r") as f:
        data = json.load(f)

    attacks = make_attacks_from_data(data)

    with open(args.output_path, "w") as f:
        json.dump(attacks, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
