import copy

import torch


def format_output(prediction, labels_to_indice, output_template, top_class, sub_class, temp: float = 0.1):
    result = copy.deepcopy(output_template)
    labels = prediction['labels']
    scores = prediction['scores']

    top_class_scores = torch.zeros(len(top_class))
    sub_class_scores = [torch.zeros(len(sub)) for sub in sub_class]

    for label, score in zip(labels, scores):
        idx = labels_to_indice[label]

        # if top class (top class' indice is integer)
        if isinstance(idx, int):
            top_class_scores[idx] = score
        else:
            top_idx, sub_idx = idx
            sub_class_scores[top_idx][sub_idx] = score

    # Score Normalization
    top_class_scores = (top_class_scores/temp).softmax(dim=-1)

    for top_idx, top in enumerate(top_class):
        top_event_score = top_class_scores[top_idx]

        if top != "Miscellaneous Policies":  # skip Miscellaneous Policies
            sub_class_scores[top_idx] = top_event_score*((sub_class_scores[top_idx]/temp).softmax(dim=-1))

    # Store scores
    for top_idx, top in enumerate(top_class):
        top_event = result[0]['children'][top_idx]
        top_event['percentage'] = top_class_scores[top_idx].item()

        if top != "Miscellaneous Policies":  # skip Miscellaneous Policies
            for sub_idx in range(len(sub_class[top_idx])):
                top_event['children'][sub_idx]['percentage'] = sub_class_scores[top_idx][sub_idx].item()

    return result