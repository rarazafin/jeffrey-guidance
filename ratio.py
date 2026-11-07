import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


def log_ratio_from_classifier(X, model, T=1, eps=None):
    """
    log(p(x)/q(x)) = log(D(x)) - log(1-D(x))
    avec D(x) = sigmoid(logit)
    """
    logits = model(X.float())

    # shape safety: (B,1) -> (B,)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)

    log_D = F.logsigmoid(logits/T)
    log_1mD = F.logsigmoid(-logits/T)

    return log_D - log_1mD

def log_regressor(X, y, sigma):
    '''
    X: embedding f(x)
    y: target values
    sigma: standard deviation

    '''
    return - ((X - y)**2).sum(dim=1) / (2 * sigma**2)


class MLP(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)  # logit
    
def get_classifier(embedder, dataset_name, device, ddpm=False):
    if embedder == "inception":
        mlp = MLP(dim=2048)
        if dataset_name == "cifar10":
            if not ddpm:
                mlp.load_state_dict(torch.load("mlp_weights_calibrated.pt"))
                mlp.to(device)
                T = 1.0
            else:
                checkpoint = torch.load("mlp_cifar10_ddpm_with_temperature.pt", map_location=device)
                mlp.load_state_dict(checkpoint["mlp_state_dict"])
                mlp.to(device)
                T = checkpoint["temperature"]
        elif dataset_name == "ffhq":
            checkpoint = torch.load("mlp_ffhq_with_temperature.pt", map_location=device)
            mlp.load_state_dict(checkpoint["mlp_state_dict"])
            mlp.to(device)
            T = checkpoint["temperature"]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    elif embedder == "clip":
        mlp = MLP(dim=7)
        checkpoint = torch.load("mlp_CLIP_with_temperature.pt", map_location=device)
        mlp.load_state_dict(checkpoint["mlp_state_dict"])
        mlp.to(device)
        T = checkpoint["temperature"]
    else:
        raise ValueError(f"Unknown embedder: {embedder}")
    
    
    return mlp, T


# ----------------------------------------
# ------------ Fairness ------------------
# ----------------------------------------

def get_attribute_classifier(ckpt_path,fairness,device):
    if fairness == 'base':
        out = 2
    elif fairness in ['joint', 'match','joint_makeup_lipstick', 'joint_linear_probe']:
        out = 4
    elif fairness == "joint_three":
        out = 8
    elif 'balance' in fairness:
        out = 1
    else:
        raise ValueError(f"Unknown fairness setting: {fairness}")
    
    classifier_model = torchvision.models.resnet18(pretrained=True)

    classifier_model.fc = nn.Linear(classifier_model.fc.in_features, out)
    #classifier_model.load_state_dict(torch.load(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    try:
        state_dict = checkpoint["model_state_dict"]
        alpha = checkpoint["alpha"]
        classifier_model.load_state_dict(state_dict)
    except Exception:
        print("No alpha")
        alpha = 1.0
        classifier_model.load_state_dict(checkpoint)
    classifier_model.to(device)
    return classifier_model, alpha


def log_ratio_from_attribute_classifier(x, classifier_fn, rho_table):
    """
    Log ratio pour deux attributs mais en assumant que le 
    classifieur est factorisé en deux classifieurs binaires indépendants
    (pas distribution jointe).
    x: (B, C, H, W)
    classifier_fn: retourne logits (B, 2)
    rho_table: dict {(a1, a2): rho}
    """

    logits = classifier_fn(x)           # (B,2)
    probs = torch.sigmoid(logits)       # (B,2)

    p1 = probs[:, 0]  # P(a1=1|x)
    p2 = probs[:, 1]  # P(a2=1|x)

    p00 = (1 - p1) * (1 - p2)
    p01 = (1 - p1) * p2
    p10 = p1 * (1 - p2)
    p11 = p1 * p2

    rho00 = rho_table[(0,0)]
    rho01 = rho_table[(0,1)]
    rho10 = rho_table[(1,0)]
    rho11 = rho_table[(1,1)]

    rho_x = (
        p00 * rho00 +
        p01 * rho01 +
        p10 * rho10 +
        p11 * rho11
    )

    log_rho = torch.log(rho_x + 1e-12)

    return log_rho

def log_ratio_logsumexp_from_attribute_classifier(x, classifier_fn, rho_table):
    """
    Fonction précédente mais en logsumexp.
    x: (B, C, H, W)
    classifier_fn: retourne logits (B, 2)
    rho_table: dict {(a1, a2): rho}
    """

    logits = classifier_fn(x)  # (B,2)

    # log-probabilités Bernoulli stables
    log_p1 = F.logsigmoid(logits[:, 0])        # log P(a1=1|x)
    log_1mp1 = F.logsigmoid(-logits[:, 0])     # log P(a1=0|x)

    log_p2 = F.logsigmoid(logits[:, 1])        # log P(a2=1|x)
    log_1mp2 = F.logsigmoid(-logits[:, 1])     # log P(a2=0|x)

    # log p(a|x)
    log_p00 = log_1mp1 + log_1mp2
    log_p01 = log_1mp1 + log_p2
    log_p10 = log_p1 + log_1mp2
    log_p11 = log_p1 + log_p2

    # log rho(a)
    log_rho00 = torch.log(torch.tensor(rho_table[(0,0)], device=x.device))
    log_rho01 = torch.log(torch.tensor(rho_table[(0,1)], device=x.device))
    log_rho10 = torch.log(torch.tensor(rho_table[(1,0)], device=x.device))
    log_rho11 = torch.log(torch.tensor(rho_table[(1,1)], device=x.device))

    # stack (B,4)
    terms = torch.stack([
        log_p00 + log_rho00,
        log_p01 + log_rho01,
        log_p10 + log_rho10,
        log_p11 + log_rho11
    ], dim=1)

    # logsumexp
    log_rho = torch.logsumexp(terms, dim=1)

    return log_rho

def log_ratio_joint_from_attribute_classifier(x, classifier_fn, rho_table):
    """
    Deux attributs mais assume qu'on a un classifieur joint (pas factorisé).
    x: (B, C, H, W)
    classifier_fn: retourne logits (B, 4) pour (a1,a2) ∈ {(0,0),(0,1),(1,0),(1,1)}
    rho_table: dict {(a1, a2): rho}
    """

    logits = classifier_fn(x)  # (B,4)

    # log p(a|x) via log-softmax
    log_probs = F.log_softmax(logits, dim=1)  # (B,4)

    # ordre IMPORTANT : doit matcher celui du classifier
    # ici on suppose :
    # index 0 -> (0,0)
    # index 1 -> (0,1)
    # index 2 -> (1,0)
    # index 3 -> (1,1)

    log_rhos = torch.tensor([
        torch.log(torch.tensor(rho_table[(0,0)])),
        torch.log(torch.tensor(rho_table[(0,1)])),
        torch.log(torch.tensor(rho_table[(1,0)])),
        torch.log(torch.tensor(rho_table[(1,1)])),
    ], device=x.device)  # (4,)

    # broadcast (B,4)
    terms = log_probs + log_rhos

    # logsumexp
    log_rho = torch.logsumexp(terms, dim=1)

    return log_rho

def log_ratio_joint_three_from_attribute_classifier(x, classifier_fn, rho_table):
    """
    Trois attributs en assumant qu'on a un classifieur joint (pas factorisé).
    x: (B, C, H, W)
    classifier_fn: retourne logits (B, 8) pour (a1,a2,a3) ∈ {0,1}^3
    rho_table: dict {(a1, a2, a3): rho}
    """

    logits = classifier_fn(x)  # (B,8)

    # log p(a|x) via log-softmax
    log_probs = F.log_softmax(logits, dim=1)  # (B,8)

    # ordre IMPORTANT : doit matcher celui du classifier
    # index = a1*4 + a2*2 + a3  (Male, Young, Blond)
    log_rhos = torch.tensor([
        torch.log(torch.tensor(rho_table[(0,0,0)])),
        torch.log(torch.tensor(rho_table[(0,0,1)])),
        torch.log(torch.tensor(rho_table[(0,1,0)])),
        torch.log(torch.tensor(rho_table[(0,1,1)])),
        torch.log(torch.tensor(rho_table[(1,0,0)])),
        torch.log(torch.tensor(rho_table[(1,0,1)])),
        torch.log(torch.tensor(rho_table[(1,1,0)])),
        torch.log(torch.tensor(rho_table[(1,1,1)])),
    ], device=x.device)  # (8,)

    # broadcast (B,8)
    terms = log_probs + log_rhos

    # logsumexp
    log_rho = torch.logsumexp(terms, dim=1)

    return log_rho

def log_ratio_match_from_attribute_classifier(x, classifier_fn, rho_table):
    logits = classifier_fn(x)
    log_probs = torch.log_softmax(logits, dim=1)

    rho = torch.tensor(
        [rho_table[(0,0)], rho_table[(0,1)], rho_table[(1,0)], rho_table[(1,1)]],
        device=logits.device,
        dtype=logits.dtype
    )

    log_rho = torch.log(rho + 1e-8).unsqueeze(0)

    return torch.logsumexp(log_probs + log_rho, dim=1)

def log_ratio_balance_from_attribute_classifier(x, classifier_fn, rho_table):

    logits = classifier_fn(x).view(-1)

    log_p1 = torch.nn.functional.logsigmoid(logits)
    log_p0 = torch.nn.functional.logsigmoid(-logits)

    p_theta_0 = rho_table["0"]
    p_theta_1 = rho_table["1"]

    log_p_theta_0 = torch.log(torch.tensor(p_theta_0 + 1e-8, device=logits.device))
    log_p_theta_1 = torch.log(torch.tensor(p_theta_1 + 1e-8, device=logits.device))

    return torch.logaddexp(
        log_p0 - log_p_theta_0,
        log_p1 - log_p_theta_1
    ) - math.log(2)

def log_ratio_balance_baseline_from_attribute_classifier(x, classifier_fn, a):
    logits = classifier_fn(x).view(-1)

    log_p1 = torch.nn.functional.logsigmoid(logits)
    log_p0 = torch.nn.functional.logsigmoid(-logits)

    logS = torch.where(a == 1, log_p1, log_p0)

    return logS
