import math

import numpy as np

from _utils import check, close, load, need


WORDS = ["chat", "chien", "cheval", "lapin", "loup"]


def ex00(d):
    m = load(d, "bigram.py")
    need(m, "extract_words", "build_vocab", "bigram_counts", "bigram_probs", "sample_word", "nll")
    words = m.extract_words("Le Chat, le chat ! L'été... Où est-il ? 42 ÉTÉ çà")
    check(words == ["chat", "est", "été"], f"extract_words incorrect : reçu {words} (minuscules, lettres seulement, longueur >= 3)")
    stoi, itos = m.build_vocab(WORDS)
    check(stoi["."] == 0 and itos[0] == ".", "le caractère spécial '.' doit avoir l'indice 0")
    check(list(itos[i] for i in range(1, len(itos))) == sorted(set("".join(WORDS))), "les autres caractères doivent être triés")
    check(all(itos[stoi[c]] == c for c in stoi), "stoi et itos doivent être inverses l'un de l'autre")
    N = m.bigram_counts(WORDS, stoi)
    V = len(stoi)
    check(N.shape == (V, V), f"bigram_counts doit renvoyer une matrice ({V}, {V})")
    check(N[stoi["."], stoi["c"]] == 3, "N['.', 'c'] doit valoir 3 (chat, chien, cheval commencent par c)")
    check(N[stoi["h"], stoi["e"]] == 1 and N[stoi["n"], stoi["."]] == 2, "bigram_counts incorrect")
    check(N.sum() == sum(len(w) + 1 for w in WORDS), "il doit y avoir len(mot) + 1 bigrammes par mot")
    P = m.bigram_probs(N, smoothing=1)
    check(close(P.sum(axis=1), 1), "chaque ligne de bigram_probs doit sommer à 1")
    check(close(P[stoi["."], stoi["c"]], 4 / (5 + V)), "bigram_probs : lissage incorrect (ajoute smoothing à chaque case)")
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = m.sample_word(P, itos, rng)
        check(isinstance(w, str) and "." not in w, "sample_word doit renvoyer une chaîne sans '.'")
    Pdet = np.zeros((V, V))
    for w in ["chat"]:
        seq = ["."] + list(w) + ["."]
        for a, b in zip(seq, seq[1:]):
            Pdet[stoi[a], stoi[b]] = 1
    check(m.sample_word(Pdet, itos, rng) == "chat", "sample_word doit partir de '.' et s'arrêter au prochain '.'")
    expected = -np.mean([math.log(P[stoi[a], stoi[b]]) for w in WORDS for a, b in zip("." + w, w + ".")])
    check(close(m.nll(WORDS, P, stoi), expected), "nll incorrect : moyenne de -log P sur tous les bigrammes")


def _torch():
    try:
        import torch

        return torch
    except ImportError:
        raise AssertionError("PyTorch n'est pas installé : voir le module 07")


def ex01(d):
    torch = _torch()
    m = load(d, "mlp_lm.py")
    need(m, "build_vocab", "build_dataset", "MLPLM", "generate")
    stoi, itos = m.build_vocab(["chat", "chien"])
    X, Y = m.build_dataset(["chat"], stoi, block_size=3)
    check(X.dtype == torch.long and Y.dtype == torch.long, "X et Y doivent être des tenseurs d'entiers (torch.long)")
    check(X.shape == (5, 3) and Y.shape == (5,), f"pour 'chat' avec block_size=3, attendu X (5, 3) et Y (5,), reçu {tuple(X.shape)}")
    s = lambda t: "".join(itos[int(i)] for i in t)
    check([s(x) for x in X] == ["...", "..c", ".ch", "cha", "hat"], "contexts incorrects : commence avec '...' (padding de 0)")
    check(s(Y) == "chat.", "cibles Y incorrectes")
    model = m.MLPLM(vocab_size=27, block_size=3, emb_dim=10, hidden=64)
    check(isinstance(model, torch.nn.Module), "MLPLM doit hériter de nn.Module")
    out = model(torch.zeros(8, 3, dtype=torch.long))
    check(out.shape == (8, 27), "MLPLM doit renvoyer des logits de forme (batch, vocab_size)")
    check(any(isinstance(x, torch.nn.Embedding) for x in model.modules()), "utilise un nn.Embedding")
    small = m.MLPLM(vocab_size=len(itos), block_size=3, emb_dim=10, hidden=64)
    g = torch.Generator().manual_seed(0)
    words = m.generate(small, itos, block_size=3, n=5, generator=g)
    check(isinstance(words, list) and len(words) == 5 and all(isinstance(w, str) for w in words), "generate doit renvoyer n mots")


def _ref_attention(torch, x, Wq, Wk, Wv):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    att = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    T = x.shape[1]
    att = att.masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
    return torch.softmax(att, dim=-1) @ v


def ex02(d):
    torch = _torch()
    m = load(d, "attention.py")
    need(m, "causal_self_attention", "MultiHeadAttention")
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8)
    Wq, Wk, Wv = (torch.randn(8, 4) for _ in range(3))
    out = m.causal_self_attention(x, Wq, Wk, Wv)
    check(out.shape == (2, 5, 4), f"forme attendue (2, 5, 4), reçu {tuple(out.shape)}")
    check(torch.allclose(out, _ref_attention(torch, x, Wq, Wk, Wv), atol=1e-5), "causal_self_attention incorrect (scale 1/sqrt(head_size) ?)")
    x2 = x.clone()
    x2[:, 3:] = torch.randn(2, 2, 8)
    out2 = m.causal_self_attention(x2, Wq, Wk, Wv)
    check(torch.allclose(out[:, :3], out2[:, :3], atol=1e-5), "fuite du futur : la position t ne doit voir que les positions <= t")
    mha = m.MultiHeadAttention(n_embd=16, n_head=4, block_size=10)
    check(isinstance(mha, torch.nn.Module), "MultiHeadAttention doit hériter de nn.Module")
    y = mha(torch.randn(3, 7, 16))
    check(y.shape == (3, 7, 16), "MultiHeadAttention doit renvoyer la même forme que l'entrée")


def ex03(d):
    torch = _torch()
    load(d, "attention.py")
    m = load(d, "mini_gpt.py")
    need(m, "MiniGPT")
    torch.manual_seed(0)
    model = m.MiniGPT(vocab_size=50, n_embd=32, n_head=4, n_layer=2, block_size=16)
    idx = torch.randint(0, 50, (4, 16))
    logits, loss = model(idx, idx)
    check(logits.shape == (4, 16, 50), "logits doit être de forme (B, T, vocab_size)")
    check(abs(loss.item() - math.log(50)) < 0.5, f"à l'init, la loss doit être proche de ln(50) = {math.log(50):.2f}, reçu {loss.item():.2f}")
    logits_none, loss_none = model(idx)
    check(loss_none is None, "sans targets, la loss renvoyée doit être None")
    idx2 = idx.clone()
    idx2[:, 10:] = torch.randint(0, 50, (4, 6))
    model.eval()
    a, _ = model(idx)
    b, _ = model(idx2)
    check(torch.allclose(a[:, :10], b[:, :10], atol=1e-5), "fuite du futur dans MiniGPT : vérifie le masque causal")
    out = model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=30)
    check(out.shape == (1, 31), "generate doit ajouter max_new_tokens tokens")


EXERCISES = [("ex00", ex00), ("ex01", ex01), ("ex02", ex02), ("ex03", ex03)]
