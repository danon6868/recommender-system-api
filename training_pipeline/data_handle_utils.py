from pathlib import Path
import re
import string
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import RobertaModel
from transformers import DistilBertModel
from transformers import DataCollatorWithPadding
from common_utils import TRAINING_CONFIG


@dataclass
class LoadedData:
    feed_data: pd.DataFrame
    user_info: pd.DataFrame
    posts_info: pd.DataFrame


def get_clusters_features(embeddings: np.ndarray) -> pd.DataFrame:
    centered = embeddings - embeddings.mean()
    pca = PCA(n_components=TRAINING_CONFIG["n_pca_components"])
    pca_decomp = pca.fit_transform(centered)

    kmeans = KMeans(
        n_clusters=TRAINING_CONFIG["n_kmeans_clusters"],
        random_state=TRAINING_CONFIG["random_state"],
    ).fit(pca_decomp)

    dists_columns = [
        f"distance_to_{i}_cluster"
        for i in range(1, TRAINING_CONFIG["n_kmeans_clusters"] + 1)
    ]
    cluster_features = pd.DataFrame(
        data=kmeans.transform(pca_decomp), columns=dists_columns
    )
    cluster_features["text_cluster"] = kmeans.labels_

    return cluster_features


def extract_text_features(texts: pd.Series, method: str = "tf_idf"):
    available_methods = {
        "tf_idf": extract_text_features_tf_idf,
        "bert": extract_text_features_transformers,
        "roberta": extract_text_features_transformers,
        "distilbert": extract_text_features_transformers,
    }

    assert method in available_methods, f"Unknown feature extraction method: {method}"
    logger.info(f"Using {method} as text feature extraction method.")

    text_features = available_methods[method](texts, method)

    return text_features


def extract_text_features_tf_idf(texts: pd.Series, model_name=None):
    wnl = WordNetLemmatizer()

    def preprocessing(line, token=wnl):
        line = line.lower()
        line = re.sub(r"[{}]".format(string.punctuation), " ", line)
        line = line.replace("\n\n", " ").replace("\n", " ")
        line = " ".join([token.lemmatize(x) for x in line.split(" ")])
        return line

    tfidf = TfidfVectorizer(stop_words="english", preprocessor=preprocessing)
    embeddings = tfidf.fit_transform(texts).toarray()

    cluster_features = get_clusters_features(embeddings)
    cluster_features["total_tfidf"] = embeddings.sum(axis=1)
    cluster_features["max_tfidf"] = embeddings.max(axis=1)
    cluster_features["mean_tfidf"] = embeddings.mean(axis=1)

    return cluster_features


def get_model(model_name: str) -> nn.Module:
    assert model_name in [
        "bert",
        "roberta",
        "distilbert",
    ], f"Unknown model name: {model_name}"

    checkpoint_names = {
        "bert": "bert-base-cased",
        "roberta": "roberta-base",
        "distilbert": "distilbert-base-cased",
    }

    model_classes = {
        "bert": BertModel,
        "roberta": RobertaModel,
        "distilbert": DistilBertModel,
    }

    return AutoTokenizer.from_pretrained(checkpoint_names[model_name]), model_classes[
        model_name
    ].from_pretrained(checkpoint_names[model_name])


class PostDataset(Dataset):
    def __init__(self, texts, tokenizer):
        super().__init__()

        self.texts = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return {
            "input_ids": self.texts["input_ids"][idx],
            "attention_mask": self.texts["attention_mask"][idx],
        }

    def __len__(self):
        return len(self.texts["input_ids"])


@torch.inference_mode()
def get_embeddings_labels(model, loader, device):
    model.eval()

    total_embeddings = []

    for batch in tqdm(loader):
        batch = {key: batch[key].to(device) for key in ["attention_mask", "input_ids"]}
        embeddings = model(**batch)["last_hidden_state"][:, 0, :]
        total_embeddings.append(embeddings.cpu())

    return torch.cat(total_embeddings, dim=0)


def extract_text_features_transformers(
    texts: pd.Series, model_name: str
) -> pd.DataFrame:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using {device} as a device.")

    logger.info(f"Start loading {model_name} model")
    tokenizer, model = get_model(model_name)
    model.to(device)
    dataset = PostDataset(texts.values.tolist(), tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        dataset, batch_size=8, collate_fn=data_collator, pin_memory=True, shuffle=False
    )
    embeddings = get_embeddings_labels(model, loader, device).numpy()

    cluster_features = get_clusters_features(embeddings)

    return cluster_features
