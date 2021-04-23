def format_output(prediction, labels_to_indice, output_template, top_class, sub_class, temp: float = 0.1):
    result = copy.deepcopy(output_template)
    labels = prediction['labels']
    scores = prediction['scores']

    num_top_event = len(top_class)
    num_sub_event = len(sub_class)

    top_event_scores = torch.zeros(num_top_event)
    sub_event_scores = [torch.zeros(len(sub)) for sub in sub_class]

    for label, score in zip(labels, scores):
        idx = labels_to_indice[label]

        # if top class (top class' indice is integer)
        if isinstance(idx, int):
            top_event_scores[idx] = score
        else:
            top_idx, sub_idx = idx
            sub_event_scores[top_idx][sub_idx] = score

    # Score Normalization
    top_event_scores = (top_event_scores/temp).softmax(dim=-1)

    for top_idx in range(num_top_event):
        top_event_score = top_event_scores[top_idx]

        if top_idx < 3:  # skip Miscellaneous Policies
            sub_event_scores[top_idx] = top_event_score*((sub_event_scores[top_idx]/temp).softmax(dim=-1))

    # Store scores
    for top_idx in range(len(top_class)):
        top_event = result[0]['children'][top_idx]
        top_event['percentage'] = top_event_scores[top_idx].item()

        if top_idx < 3:  # skip Miscellaneous Policies
            for sub_idx in range(len(sub_class[top_idx])):
                top_event['children'][sub_idx]['percentage'] = sub_event_scores[top_idx][sub_idx].item()

    return result