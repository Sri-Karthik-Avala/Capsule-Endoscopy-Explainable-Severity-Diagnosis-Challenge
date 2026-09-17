import os, sys, glob, random, math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 1234
IMG = 336
BATCH = 32
EPOCHS = 16
FOLDS = 5
LR = 3e-4
WD = 1e-4
DROP = 0.3
BACKBONES = ["efficientnet_b0", "convnext_nano", "resnet34"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NW = 4 if os.name == "posix" else 0


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def find_first(cands):
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def locate_file(name):
    roots = [".", "public", os.path.join(".", "public"), "..", os.path.join("..", "public"),
             "input", os.path.join(".", "input"), "data", os.path.join(".", "data")]
    hit = find_first([os.path.join(r, name) for r in roots])
    if hit:
        return hit
    for d in [".", "..", "public", "input", "data"]:
        if os.path.isdir(d):
            g = glob.glob(os.path.join(d, "**", name), recursive=True)
            if g:
                return g[0]
    g = glob.glob(os.path.join("**", name), recursive=True)
    return g[0] if g else None


def locate_imgdir(sample_id, names):
    roots = [".", "public", os.path.join(".", "public"), "..", os.path.join("..", "public"),
             "input", "data"]
    for nm in names:
        for r in roots:
            d = os.path.join(r, nm)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, sample_id)):
                return d
    for nm in names:
        g = glob.glob(os.path.join("**", nm, sample_id), recursive=True)
        if g:
            return os.path.dirname(g[0])
    g = glob.glob(os.path.join("**", sample_id), recursive=True)
    return os.path.dirname(g[0]) if g else None


def output_dir():
    for d in ["working", os.path.join(".", "working"), os.path.join("..", "working")]:
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            continue
    return "."


def make_amp(use_amp):
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        def ctx():
            return torch.amp.autocast("cuda", enabled=use_amp)
        return scaler, ctx
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        def ctx():
            return torch.cuda.amp.autocast(enabled=use_amp)
        return scaler, ctx


def rankdata_avg(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = ranks[i:j + 1].mean()
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def auc_score(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    r = rankdata_avg(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def spearman(a, b):
    ra = rankdata_avg(a); rb = rankdata_avg(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def macro_f1(y_true, y_pred, n_classes):
    fs = []
    for c in range(n_classes):
        at = (y_true == c); ap = (y_pred == c)
        if at.sum() == 0:
            continue
        tp = int((at & ap).sum()); fp = int((~at & ap).sum()); fn = int((at & ~ap).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fs.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return float(np.mean(fs)) if fs else 0.0


def strat_kfold(y, k, seed):
    rng = np.random.RandomState(seed)
    folds = np.full(len(y), -1, dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            folds[ix] = i % k
    return folds


def tune_bias(probs, y, n_classes, rounds=4):
    logp = np.log(np.clip(probs, 1e-9, 1.0))
    b = np.zeros(n_classes)
    grid = np.linspace(-5.0, 5.0, 51)
    best = macro_f1(y, logp.argmax(1), n_classes)
    for _ in range(rounds):
        improved = False
        for c in range(n_classes):
            best_d = b[c]; best_f = best
            for d in grid:
                b[c] = d
                f = macro_f1(y, (logp + b).argmax(1), n_classes)
                if f > best_f:
                    best_f = f; best_d = d
            b[c] = best_d
            if best_f > best + 1e-9:
                improved = True
            best = best_f
        if not improved:
            break
    return b


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        self.body = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = 256

    def forward(self, x):
        return self.pool(self.body(x)).flatten(1)


def build_backbone(pref):
    try:
        import timm
        m = timm.create_model(pref, pretrained=True, num_classes=0, global_pool="avg")
        return m, int(m.num_features)
    except Exception:
        pass
    try:
        import torchvision.models as tvm
        if "efficientnet" in pref:
            try:
                net = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
            except Exception:
                net = tvm.efficientnet_b0(weights=None)
            fd = net.classifier[1].in_features
            net.classifier = nn.Identity()
            return net, fd
        else:
            try:
                net = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
            except Exception:
                net = tvm.resnet34(weights=None)
            fd = net.fc.in_features
            net.fc = nn.Identity()
            return net, fd
    except Exception:
        pass
    return SmallCNN(), 256


class Net(nn.Module):
    def __init__(self, pref, n_classes):
        super().__init__()
        self.backbone, fd = build_backbone(pref)
        self.drop = nn.Dropout(DROP)
        self.head_cls = nn.Linear(fd, n_classes)
        self.head_urg = nn.Linear(fd, 1)
        self.head_sev = nn.Linear(fd, 1)

    def forward(self, x):
        f = self.backbone(x)
        if f.ndim > 2:
            f = f.mean(dim=(-2, -1))
        f = self.drop(f)
        return self.head_cls(f), self.head_urg(f).squeeze(1), self.head_sev(f).squeeze(1)


class ImgDS(Dataset):
    def __init__(self, arrs, idx, train, y=None, yu=None, ys=None):
        self.arrs = arrs; self.idx = idx; self.train = train
        self.y = y; self.yu = yu; self.ys = ys
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.idx)

    def _aug(self, arr):
        if random.random() < 0.5:
            arr = arr[:, ::-1, :]
        if random.random() < 0.5:
            arr = arr[::-1, :, :]
        k = random.randint(0, 3)
        if k:
            arr = np.rot90(arr, k, axes=(0, 1))
        h, w = arr.shape[:2]
        s = random.uniform(0.7, 1.0)
        ch = max(8, int(round(h * s))); cw = max(8, int(round(w * s)))
        y0 = random.randint(0, h - ch); x0 = random.randint(0, w - cw)
        arr = arr[y0:y0 + ch, x0:x0 + cw, :]
        im = Image.fromarray(np.ascontiguousarray(arr)).resize((IMG, IMG), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if random.random() < 0.5:
            t = torch.clamp(t * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)
        if random.random() < 0.3:
            t = torch.clamp(t + torch.randn_like(t) * random.uniform(0.01, 0.05), 0, 1)
        return t

    def __getitem__(self, k):
        i = self.idx[k]
        arr = self.arrs[i]
        if self.train:
            t = self._aug(arr)
        else:
            im = Image.fromarray(arr).resize((IMG, IMG), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        t = (t - self.mean) / self.std
        if self.train:
            return t, self.y[i], self.yu[i], self.ys[i]
        return t, i


def load_arrs(ids, d):
    out = []
    for f in ids:
        im = Image.open(os.path.join(d, f)).convert("RGB")
        if im.size != (336, 336):
            im = im.resize((336, 336), Image.BILINEAR)
        out.append(np.asarray(im, dtype=np.uint8))
    return out


def tta_batch(x):
    return [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]), torch.rot90(x, 1, [2, 3])]


def main():
    set_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_csv = locate_file("train.csv")
    test_csv = locate_file("test.csv")
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    tr_dir = locate_imgdir(tr.id.iloc[0], ["train_images", "train"])
    te_dir = locate_imgdir(te.id.iloc[0], ["test_images", "test"])

    classes = sorted(tr.label.unique().tolist())
    cls2i = {c: i for i, c in enumerate(classes)}
    NC = len(classes)
    y = tr.label.map(cls2i).values.astype(np.int64)
    yu = tr.urgent_prob.values.astype(np.float32)
    ys = tr.severity_prob.values.astype(np.float32)

    lab_urg = tr.groupby("label").urgent_prob.mean()
    lab_sev = tr.groupby("label").severity_prob.mean()
    g_urg = float(tr.urgent_prob.mean()); g_sev = float(tr.severity_prob.mean())

    rt_classes = tr.reason_tag.value_counts().index.tolist()
    rt_freq = tr.reason_tag.value_counts().values.astype(float)
    sr_classes = tr.severity_reason.value_counts().index.tolist()
    sr_freq = tr.severity_reason.value_counts().values.astype(float)

    cnt = np.array([(y == c).sum() for c in range(NC)], dtype=float)
    cw = (1.0 / np.sqrt(cnt))
    cw = cw / cw.mean()
    cw_t = torch.tensor(cw, dtype=torch.float32, device=dev)

    tr_arrs = load_arrs(tr.id.values, tr_dir)
    te_arrs = load_arrs(te.id.values, te_dir)
    NT = len(tr_arrs); NE = len(te_arrs)

    folds = strat_kfold(y, FOLDS, SEED)

    oof_p = np.zeros((NT, NC), dtype=np.float64)
    oof_u = np.zeros(NT, dtype=np.float64)
    oof_s = np.zeros(NT, dtype=np.float64)
    oof_c = np.zeros(NT, dtype=np.float64)
    test_p = np.zeros((NE, NC), dtype=np.float64)
    test_u = np.zeros(NE, dtype=np.float64)
    test_s = np.zeros(NE, dtype=np.float64)
    test_c = 0.0

    use_amp = (dev == "cuda")
    pin = (dev == "cuda")

    for bb in BACKBONES:
        for fold in range(FOLDS):
            set_seed(SEED + fold * 17 + (abs(hash(bb)) % 1000))
            trn_idx = np.where(folds != fold)[0]
            val_idx = np.where(folds == fold)[0]
            ds_tr = ImgDS(tr_arrs, trn_idx, True, y, yu, ys)
            ds_va = ImgDS(tr_arrs, val_idx, False)
            dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=NW,
                               pin_memory=pin, drop_last=False)
            dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=NW, pin_memory=pin)

            model = Net(bb, NC).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            steps = max(1, len(dl_tr)) * EPOCHS
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
            scaler, amp_ctx = make_amp(use_amp)

            for ep in range(EPOCHS):
                model.train()
                for xb, yb, ub, sb in dl_tr:
                    xb = xb.to(dev, non_blocking=True); yb = yb.to(dev)
                    ub = ub.to(dev); sb = sb.to(dev)
                    opt.zero_grad()
                    with amp_ctx():
                        lo, uo, so = model(xb)
                        loss = (F.cross_entropy(lo, yb, weight=cw_t)
                                + 0.5 * F.mse_loss(torch.sigmoid(uo), ub)
                                + 0.5 * F.mse_loss(torch.sigmoid(so), sb))
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update(); sched.step()

            model.eval()
            with torch.no_grad():
                vp = []; vu = []; vs = []
                for xb, _ in dl_va:
                    xb = xb.to(dev)
                    views = tta_batch(xb); nv = len(views)
                    pl = torch.zeros(xb.size(0), NC, device=dev)
                    pu = torch.zeros(xb.size(0), device=dev)
                    pss = torch.zeros(xb.size(0), device=dev)
                    for xt in views:
                        with amp_ctx():
                            lo, uo, so = model(xt)
                        pl += torch.softmax(lo.float(), 1)
                        pu += torch.sigmoid(uo.float()); pss += torch.sigmoid(so.float())
                    vp.append((pl / nv).cpu().numpy())
                    vu.append((pu / nv).cpu().numpy()); vs.append((pss / nv).cpu().numpy())
                vp = np.concatenate(vp); vu = np.concatenate(vu); vs = np.concatenate(vs)
                oof_p[val_idx] += vp; oof_u[val_idx] += vu; oof_s[val_idx] += vs
                oof_c[val_idx] += 1.0

                ds_te = ImgDS(te_arrs, np.arange(NE), False)
                dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, num_workers=NW, pin_memory=pin)
                for xb, ii in dl_te:
                    xb = xb.to(dev)
                    views = tta_batch(xb); nv = len(views)
                    pl = torch.zeros(xb.size(0), NC, device=dev)
                    pu = torch.zeros(xb.size(0), device=dev)
                    pss = torch.zeros(xb.size(0), device=dev)
                    for xt in views:
                        with amp_ctx():
                            lo, uo, so = model(xt)
                        pl += torch.softmax(lo.float(), 1)
                        pu += torch.sigmoid(uo.float()); pss += torch.sigmoid(so.float())
                    ii = ii.numpy()
                    test_p[ii] += (pl / nv).cpu().numpy()
                    test_u[ii] += (pu / nv).cpu().numpy(); test_s[ii] += (pss / nv).cpu().numpy()
                test_c += 1.0
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

    oof_p /= np.clip(oof_c, 1, None)[:, None]
    oof_u /= np.clip(oof_c, 1, None)
    oof_s /= np.clip(oof_c, 1, None)
    test_p /= test_c; test_u /= test_c; test_s /= test_c

    bias = tune_bias(oof_p, y, NC)
    oof_pred = (np.log(np.clip(oof_p, 1e-9, 1)) + bias).argmax(1)
    test_pred = (np.log(np.clip(test_p, 1e-9, 1)) + bias).argmax(1)

    raw_f1 = macro_f1(y, oof_p.argmax(1), NC)
    cal_f1 = macro_f1(y, oof_pred, NC)

    oof_lm_u = np.array([lab_urg.get(classes[c], g_urg) for c in y])
    oof_lm_s = np.array([lab_sev.get(classes[c], g_sev) for c in y])
    best_wu, best_su = 1.0, -1
    for w in np.linspace(0, 1, 21):
        sc = spearman(w * oof_u + (1 - w) * oof_lm_u, yu)
        if sc > best_su:
            best_su = sc; best_wu = w
    best_ws, best_rs = 1.0, 1e9
    for w in np.linspace(0, 1, 21):
        bl = w * oof_s + (1 - w) * oof_lm_s
        rmse = math.sqrt(float(np.mean((bl - ys) ** 2)))
        if rmse < best_rs:
            best_rs = rmse; best_ws = w

    test_lm_u = np.array([lab_urg.get(classes[c], g_urg) for c in test_pred])
    test_lm_s = np.array([lab_sev.get(classes[c], g_sev) for c in test_pred])
    urg_final = np.clip(best_wu * test_u + (1 - best_wu) * test_lm_u, 0, 1)
    sev_final = np.clip(best_ws * test_s + (1 - best_ws) * test_lm_s, 0, 1)

    oof_u_bl = best_wu * oof_u + (1 - best_wu) * oof_lm_u
    oof_s_bl = best_ws * oof_s + (1 - best_ws) * oof_lm_s
    pri_truth = tr.review_priority.values
    best_a, best_psp = 0.5, -1
    for a in np.linspace(0, 1, 21):
        sc = spearman(a * oof_u_bl + (1 - a) * oof_s_bl, pri_truth)
        if sc > best_psp:
            best_psp = sc; best_a = a
    test_pri_score = best_a * urg_final + (1 - best_a) * sev_final
    pr = rankdata_avg(test_pri_score) / len(test_pri_score)
    review_priority = np.clip((pr * 5).astype(int) + 1, 1, 5)

    auc_oof = auc_score((yu > 0.5).astype(int), oof_u_bl)
    sev_score = max(0.0, 1.0 - best_rs)
    pri_sp = best_psp

    def assign_marginal(n, cls, freq, seed):
        rng = np.random.RandomState(seed)
        p = freq / freq.sum()
        counts = np.floor(p * n).astype(int)
        rem = n - int(counts.sum())
        order = np.argsort(-(p * n - np.floor(p * n)))
        for j in range(rem):
            counts[order[j % len(order)]] += 1
        arr = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        rng.shuffle(arr)
        return [cls[a] for a in arr[:n]]

    reason_pred = assign_marginal(NE, rt_classes, rt_freq, SEED + 1)
    sevr_pred = assign_marginal(NE, sr_classes, sr_freq, SEED + 2)

    sub = pd.DataFrame({
        "id": te.id.values,
        "label": [classes[c] for c in test_pred],
        "reason_tag": reason_pred,
        "severity_reason": sevr_pred,
        "urgent_prob": np.round(urg_final, 6),
        "severity_prob": np.round(sev_final, 6),
        "review_priority": review_priority.astype(int),
    })
    od = output_dir()
    sub.to_csv(os.path.join(od, "submission.csv"), index=False)
    sub.to_csv("submission.csv", index=False)

    est = (0.40 * cal_f1 + 0.15 * (1.0 / len(rt_classes)) + 0.15 * (1.0 / len(sr_classes))
           + 0.10 * auc_oof + 0.10 * sev_score + 0.10 * max(0.0, pri_sp))
    print("classes:", NC, classes)
    print("label macroF1 raw=%.4f cal=%.4f" % (raw_f1, cal_f1))
    print("urgent AUC(oof,>0.5)=%.4f wu=%.2f" % (auc_oof, best_wu))
    print("severity score=%.4f rmse=%.4f ws=%.2f" % (sev_score, best_rs, best_ws))
    print("priority spearman=%.4f a=%.2f" % (pri_sp, best_a))
    print("reason 1/K=%.4f sevreason 1/K=%.4f" % (1.0 / len(rt_classes), 1.0 / len(sr_classes)))
    print("EST final=%.4f" % est)
    print("wrote", os.path.join(od, "submission.csv"), sub.shape)


if __name__ == "__main__":
    main()
