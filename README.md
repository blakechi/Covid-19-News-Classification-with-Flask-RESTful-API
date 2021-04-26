# Few-shot Covid-19 News Hierarchical Classification

## Introduction

The objective is to classify Covid-19 related news into hierarchical classes defined from [Oxford Covid-19 Government Response Tracker](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md#containment-and-closure-policies). We use **Google Colaboratory** for training.

## Method

### Dataset

The news we use in this project is fetched from a private database with the timeline from Apr. 2020 to Jun. 2020.

### News Summarization

Typically, news related articles are pretty long to read when we want to label them for downstream tasks, such as text classification or summarization. Therefore, we summarize news first to alleviate the labelling workload and efficiently accelerate the whole labeling process. Moreover, given that summarization is shorter than raw articles in terms of numbers of text, using summarizations instead for downstream tasks can decline the computational cost. Please check out [this repository](https://github.com/blakechi/news_summarization) for details.

### Data Labelling

To make the labeling efficient, we set up [Label Studio](https://labelstud.io), which provides intuitive and concise UI interfaces, remote labeling, and online model training and deployment, on our server. We classify news into 4 top categories from their summaries instead of raw news and follow the guideline from Oxford Covid-19 Government Response Tracker as well. We have labelled around 500 news and we got 74, 39, 90, and 353 data points for Containment and Closure, Economic, Health System, and Miscellaneous Policies respectively. Note that each news might contain multiple categories. To get a balanced number of labels across all categories and given the rarity of Economic Policy, we randomly pick 30 for each of them and end up getting 120 labelled data points for training in total. The rest of labelled news is used for testing.

### News Hierarchical Classification

Due to the ongoing events about Covid-19 everyday, Oxford Covid-19 Government Response Tracker keeps continuously updating their criterions and adding new categories as well. Therefore, instead of training a language model on a fixed set of pre-defined categories for classification that loses flexibility when new categories added in the future, we focus on models that perform outstandingly in natural language inference (NLI) field. In the NLI field, we ask models to predict the similarity of semantic meaning in two sentences called premise (input data) and hypothesis (categories for classification) and output the similarity into three categories: entailment, neutral, and contraction. Framing classification in this way has multiple advantages: i) Categories for classification are now flexible by predicting the similarity of input texts and given categories, ii) the classification is not limited to domain specific data, and iii) we now can only label a few or even no data for downstream tasks once our language model learns semantic meanings in language, also known as few-shot or zero-shot learning.

Since our objective is to predict the semantic similarity between two sentences, we can easily augment our labeled data by i) choosing multiple different descriptions (hypothesis) for categories to generate positive data pairs (entailment) and ii) randomly pairing the data with other categories’ descriptions to generate negative ones (contraction).

In our project, the premise is the news summaries and the hypothesis is the descriptions for each category in Oxford Covid-19 Government Response Tracker. We augment hypothesis into six folds by inserting the name of the categories into two templates that have similar meaning but different keywords, and combining all the descriptions of second-level categories into one sentence for positive pairs, and randomly pairing for negative pairs. Note that since we only fine tune the model on top-level categories, the data augmentation methods don’t apply to second-level categories. On [Hugging Face](https://huggingface.co), we choose a pretrained [Bart-large for zero-shot classification](https://huggingface.co/facebook/bart-large-mnli).

In the training phase, we fine tune the Bart by the augmented data. In the inference phase, we reuse the hypothesis we applied in training for top categories, and for second-level ones, we use their descriptions directly from Oxford Covid-19 Government Response Tracker. We flatten the hierarchical structure of top and second-level categories as normal multi-class classification tasks. However, to reserve the hierarchical structure, we normalize the predictions separately by softmax on top and second-level categories such that the sum of top categories and their respective second-level ones equals 1. Then, like conditional probability, each second-level categories’ predictions is multiplied by their top one.

## Install

1. Install packages

   > Note: Use python3.8 and update pip in virtual enviroment

   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Create a folder called `checkpoints` under this repo and put your checkpoint(s) inside it.

3. Create `server.json` for query (Only needed for **Usage 2.** and **3.**)

   - Under the repository directory

     ```bash
     vim server.json
     ```

   - Inside server.json
     ```json
     {
       "url": "your/server/url/for/query"
     }
     ```

## Usage

1. Deploy REST API \
   Use the pretrained checkpoint - `facebook/bart-large-mnli` from [HuggingFace](https://huggingface.co/facebook/bart-large-mnli)
   ```bash
   python src/classifier.py
   ```
   or use yours under `checkpoints`
   ```bash
   python src/classifier.py your_checkpoint_name
   ```
   
2. Test the API
   - Go through **1.** first.
   - Modify the query in `test_classifier.py` or add any news you want as string without using query.
   - Then:
     `bash python test/test_classifier.py `
     > **If**: ModuleNotFoundError: No module named 'gql.transport.aiohttp' \
     > **Solution**:
     >
     > ```bash
     > pip uninstall gql
     > pip install --pre gql[all]
     > ```

## Used Packages

- transformers (Hugging Face)
- torch
- gql (GraphQL)
- tqdm

## Future Works

- [ ] Prompt Engineering

## Citation

```bibtex
@article{lewis2019bart,
    title = {BART: Denoising Sequence-to-Sequence Pre-training for Natural
Language Generation, Translation, and Comprehension},
    author = {Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad and
              Abdelrahman Mohamed and Omer Levy and Veselin Stoyanov
              and Luke Zettlemoyer },
    journal={arXiv preprint arXiv:1910.13461},
    year = {2019},
}
```
