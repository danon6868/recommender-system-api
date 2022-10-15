# recommender-system-api

# Plans

1. Pipeline for feature extraction (download data, process it, text embeddings, optuna for hp selection of catboost, retrain model using whole data, save model, save features)
2. Based on saved models and saved data, create endpoint for post recommendations
3. A/B testing system...

For 1: create config with source of date (e.g. pgsql connection or local files). Take argument, which alg use to create text embeddings