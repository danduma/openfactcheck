classifier = None


def _load_classifier():
    global classifier
    if classifier is not None:
        return classifier

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except Exception as e:
        raise ImportError(
            "NLI stance mode requires optional dependency `transformers` "
            "and a local model download (`roberta-large-mnli`)."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier

nli_labelmap = {
    "NEUTRAL": 3,
    "CONTRADICTION":2,
    "ENTAILMENT": 1   
}

nli2stance = {
    "NEUTRAL": 0,
    "CONTRADICTION": -1,
    "ENTAILMENT": 1   
}

stance_map = {
    'irrelevant': 3,
    'refute': 2,
    'partially-support': 1,
    'completely-support': 1
}

def nli_infer(premise, hypothesis):
    local_classifier = _load_classifier()
    # predict one example by nli model
    try: 
        input = "<s>{}</s></s>{}</s></s>".format(premise, hypothesis)
        pred = local_classifier(input)
        # print(pred)
    except:
        # token length > 514
        L = len(premise)
        premise = premise[:int(L/2)]
        input = "<s>{}</s></s>{}</s></s>".format(premise, hypothesis)
        pred = local_classifier(input)
        # print(pred) 
        # [{'label': 'CONTRADICTION', 'score': 0.9992701411247253}]
    return nli2stance[pred[0]['label']]
