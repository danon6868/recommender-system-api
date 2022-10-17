import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertModel,
    DataCollatorWithPadding,
    DistilBertModel,
    RobertaModel,
)

from common_utils import TRAINING_CONFIG


@dataclass
class LoadedData:
    """Class for loaded data representation."""

    feed_data: pd.DataFrame
    user_info: pd.DataFrame
    posts_info: pd.DataFrame


def get_clusters_features(embeddings: np.ndarray) -> pd.DataFrame:
    """Create cluster features based on given embeddings, e.g.,
    for each object distances to each cluster and cluster label.

    Args:
        embeddings (np.ndarray): Text embeddings.

    Returns:
        pd.DataFrame: The table with cluster features.
    """

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


def extract_text_features(texts: pd.Series, method: str = "tf_idf") -> pd.DataFrame:
    """Extract text features using one of the following methods: TF-IDF, BERT, ROBERTA, DISTILBERT.
    For TF-IDF also calculate mean, max and total TF-IDF.

    Args:
        texts (pd.Series): The set of text to be processed.
        method (str, optional): Method for text embeddings creation. Defaults to "tf_idf".

    Returns:
        pd.DataFrame: The table with text features.
    """

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


def extract_text_features_tf_idf(texts: pd.Series, model_name=None) -> pd.DataFrame:
    """Extract text features using TF-IDF.

    Args:
        texts (pd.Series): The set of text to be processed.
        model_name (_type_, optional): Here it is a dummy argument so that the interface
        will be similar with the `extract_text_features_transformers`. Defaults to None.

    Returns:
        pd.DataFrame: The table with text features.
    """

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


def get_model(model_name: str) -> Tuple[AutoTokenizer, nn.Module]:
    """Load `model_name` transformer and tokenizer for it.

    Args:
        model_name (str): The name of using transformer.

    Returns:
        Tuple[AutoTokenizer, nn.Module]: Tokenizer and model for text processing.
    """

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
    """PyTorch dataset for Posts dataset representation."""

    def __init__(self, texts: pd.Series, tokenizer: AutoTokenizer) -> None:
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.texts["input_ids"][idx],
            "attention_mask": self.texts["attention_mask"][idx],
        }

    def __len__(self) -> int:
        return len(self.texts["input_ids"])


@torch.inference_mode()
def get_embeddings_labels(
    model: nn.Module, loader: DataLoader, device: str
) -> torch.Tensor:
    """Calculate embeddings for a given set of texts.

    Args:
        model (nn.Module): Transformer model.
        loader (DataLoader): Dataloader.
        device (str): Device (cpu or cuda).

    Returns:
        torch.Tensor: Text embeddings.
    """

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
    """Extract text features using transformers.

    Args:
        texts (pd.Series): The set of text to be processed.
        model_name (str): The name of using transformer.

    Returns:
        pd.DataFrame: The table with text features.
    """

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
