"""
    An REST API for news hierarchical classification
"""
import os
import sys
import warnings

from flask import Flask, jsonify, request

import torch
from transformers import pipeline

import config
from utils import format_output


MAX_LENGTH = 1024
DEVICEIDX = 0 if torch.cuda.is_available() else -1
classifier = None
app = Flask(__name__)


@app.route("/classify", methods=["POST"])
def classify():
    data = {"success": False}

    if request.method == "POST":
        # assume there is only one news here
        news = request.get_json()
        summary = news["summary"]

        prediction = classifier(
            summary, 
            config.hypothesis_candidate, 
            config.hypothesis_template, 
            multi_class=True,
        )

        prediction = format_output(
            prediction, 
            config.labels_to_indice, 
            config.output_template, 
            config.top_class, 
            config.sub_class,
            config.temperature
        )

        data["id"] = news["id"]
        data["prediction"] = prediction
        data["too_long"] = True if len(summary) > MAX_LENGTH else False
        data["success"] = True

    return jsonify(data)


def load_model(model_path):
    global classifier
    
    classifier = pipeline(
        'zero-shot-classification',
        tokenizer='facebook/bart-large-mnli',
        model=model_path,
        device=DEVICEIDX,
        framework='pt',
    )


if __name__ == '__main__':

    try:
        model_path = os.path.join("checkpoints", sys.argv[1])
    except IndexError:
        warnings.warn("Please specify the checkpoint's name as an argument in the command if any. \nUse 'facebook/bart-large-mnli' from HuggingFace now...")
        model_path = "facebook/bart-large-mnli"

    load_model(model_path)
    app.run()
