from nltk.stem.lancaster import LancasterStemmer

sno = LancasterStemmer()


def get_event_regex(
    general_keywords, event_keywords, operator, stem=True, capture=False
):
    if operator == "or":
        regex_keywords_list = []
        for keyword in general_keywords + event_keywords:
            regex_keywords_list.append(
                f"{' '.join([sno.stem(word) for word in keyword.split()])}"
                if stem
                else keyword
            )

        keywords_regex = "|".join(regex_keywords_list)

        if capture:
            regex = rf"\b({keywords_regex})\b"
        else:
            regex = rf"\b(?:{keywords_regex})\b"

    elif operator == "and":
        regex_general_keywords_list = []
        for keyword in general_keywords:
            regex_general_keywords_list.append(
                f"{' '.join([sno.stem(word) for word in keyword.split()])}"
            )

        regex_general_keywords = "|".join(regex_general_keywords_list)

        regex_event_keywords_list = []
        for keyword in event_keywords:
            regex_event_keywords_list.append(
                f"{' '.join([sno.stem(word) for word in keyword.split()])}"
            )

        regex_event_keywords = "|".join(regex_event_keywords_list)

        if capture:
            regex = rf"\b({regex_general_keywords})\b.*\b({regex_event_keywords})\b|\b({regex_event_keywords})\b.*\b({regex_general_keywords})\b"
        else:
            regex = rf"\b(?:{regex_general_keywords})\b.*\b(?:{regex_event_keywords})\b|\b(?:{regex_event_keywords})\b.*\b(?:{regex_general_keywords})\b"

    else:
        raise ValueError("Operator must be 'or' or 'and'")

    return regex
