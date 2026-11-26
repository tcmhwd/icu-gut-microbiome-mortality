"""dl_models.py — MLP, FT-Transformer, TabNet (Section 31).

Only tabular architectures are used (no LSTM/GRU/sequence Transformer):
predictors here are first-24h aggregated tabular features, not hourly
longitudinal sequences (Section 31).

FT-Transformer note: the `rtdl` package is not installed in the 'gpt' conda
env (only `rtdl_num_embeddings`, a numeric-embedding-only helper package was
found). Rather than add an unvetted new dependency, FT-Transformer (Gorishniy
et al. 2021) is implemented directly with PyTorch here: every input feature
(already numeric after the Universal preprocessing pipeline: continuous
values, missingness indicators, and binary categoricals all live on a common
numeric/0-1 scale) is tokenized with its own per-feature linear embedding
(weight+bias -> d_token), a learned [CLS] token is prepended, and a standard
`nn.TransformerEncoder` attends over the token sequence; the CLS output feeds
a linear classification head. This is a documented, faithful simplification
of FT-Transformer for an all-numeric/binary tabular input (no distinct
high-cardinality categorical columns exist in this predictor set).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from . import config


def set_seed(seed: int = config.RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. MLP
# ---------------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}[activation]
        layers = []
        d_in = input_dim
        for d_out in hidden_dims:
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), act(), nn.Dropout(dropout)]
            d_in = d_out
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp_optuna_space(trial):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = [trial.suggest_int(f"hidden_dim_{i}", 16, 256, log=True) for i in range(n_layers)]
    return {
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu"]),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "pos_weight_mode": trial.suggest_categorical("pos_weight_mode", ["balanced", "none"]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
    }


def mlp_reconstruct_params(raw_best_params: dict) -> dict:
    """'hidden_dims' is assembled from per-layer trial.suggest_int calls
    (hidden_dim_0, hidden_dim_1, ...) inside mlp_optuna_space(), so it never
    appears as its own key in study.best_params -- must be rebuilt here
    before build_mlp() can use it for the final refit."""
    params = dict(raw_best_params)
    n_layers = params["n_layers"]
    params["hidden_dims"] = [params[f"hidden_dim_{i}"] for i in range(n_layers)]
    return params


def build_mlp(input_dim, params):
    return SimpleMLP(input_dim, params["hidden_dims"], params["dropout"], params["activation"])


# ---------------------------------------------------------------------------
# 2. FT-Transformer (simplified, see module docstring)
# ---------------------------------------------------------------------------
class FTTransformer(nn.Module):
    def __init__(self, input_dim, d_token=32, n_heads=4, n_layers=2, ffn_dim=64, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(input_dim, d_token))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        # x: (batch, input_dim) -> tokens: (batch, input_dim, d_token)
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls = self.cls_token.expand(x.size(0), -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        out = self.encoder(seq)
        cls_out = self.norm(out[:, 0, :])
        return self.head(cls_out).squeeze(-1)


def fttransformer_optuna_space(trial):
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    d_token = trial.suggest_categorical("d_token", [16, 32, 64])
    return {
        "d_token": d_token,
        "n_heads": n_heads,
        "n_layers": trial.suggest_int("n_layers", 1, 4),
        "ffn_dim": trial.suggest_int("ffn_dim", 32, 256, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "pos_weight_mode": trial.suggest_categorical("pos_weight_mode", ["balanced", "none"]),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
    }


def build_fttransformer(input_dim, params):
    return FTTransformer(
        input_dim, d_token=params["d_token"], n_heads=params["n_heads"],
        n_layers=params["n_layers"], ffn_dim=params["ffn_dim"], dropout=params["dropout"],
    )


# ---------------------------------------------------------------------------
# Shared training loop for MLP / FT-Transformer (both take a plain
# (batch, input_dim) float tensor and emit a logit)
# ---------------------------------------------------------------------------
def train_torch_classifier(model, X_train, y_train, X_val, y_val, params,
                            max_epochs=200, patience=15, seed=config.RANDOM_SEED,
                            trial=None, step_offset=0):
    """step_offset lets callers doing k-fold CV within a single Optuna trial
    (e.g. build_torch_objective) keep the (trial, step) pairs passed to
    trial.report() strictly increasing across folds -- e.g. fold i uses
    step_offset=i*max_epochs -- so Optuna doesn't silently drop/overwrite a
    later fold's report at a step number an earlier fold already used."""
    set_seed(seed)
    device = get_device()
    model = model.to(device)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv_np = np.asarray(y_val)

    if params.get("pos_weight_mode") == "balanced":
        n_pos = max((yt == 1).sum().item(), 1)
        n_neg = max((yt == 0).sum().item(), 1)
        pos_weight = torch.tensor(n_neg / n_pos, device=device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt_cls = torch.optim.AdamW if params.get("optimizer") == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    ds = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=params["batch_size"], shuffle=True,
                                          generator=torch.Generator().manual_seed(seed))

    best_auc = -np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv).cpu().numpy()
        val_prob = 1 / (1 + np.exp(-val_logits))
        if len(np.unique(yv_np)) > 1:
            val_auc = roc_auc_score(yv_np, val_prob)
        else:
            val_auc = np.nan

        if trial is not None:
            import optuna
            trial.report(val_auc if not np.isnan(val_auc) else 0.0, step_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


def predict_proba_torch(model, X, device=None):
    device = device or get_device()
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return 1 / (1 + np.exp(-logits))


# ---------------------------------------------------------------------------
# 3. TabNet (pytorch-tabnet, verified installed in 'gpt' env)
# ---------------------------------------------------------------------------
def tabnet_optuna_space(trial):
    n_d = trial.suggest_int("n_d", 8, 64)
    return {
        "n_d": n_d,
        "n_a": n_d,  # conventionally tied, per pytorch-tabnet docs
        "n_steps": trial.suggest_int("n_steps", 3, 8),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 5e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        "virtual_batch_size": trial.suggest_categorical("virtual_batch_size", [64, 128, 256]),
    }


def build_and_train_tabnet(X_train, y_train, X_val, y_val, params, seed=config.RANDOM_SEED,
                            max_epochs=200, patience=15):
    from pytorch_tabnet.tab_model import TabNetClassifier
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    clf = TabNetClassifier(
        n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
        gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
        optimizer_params={"lr": params["lr"]},
        seed=seed, verbose=0, device_name=device_name,
    )
    clf.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)], eval_metric=["auc"],
        max_epochs=max_epochs, patience=patience,
        batch_size=params["batch_size"], virtual_batch_size=params["virtual_batch_size"],
    )
    return clf


def tabnet_predict_proba(clf, X):
    return clf.predict_proba(X)[:, 1]
