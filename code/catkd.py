"""
CAT-KD reproduction (CVPR 2023, Guo et al.) - lightweight single-file version.
Teacher: ResNet32x4 (pretrained, from mdistiller)  ->  Student: ResNet8x4, CIFAR-100.

Usage (Colab):
  python catkd.py --method catkd               # CAT-KD  (paper: 76.91)
  python catkd.py --method kd                  # vanilla KD baseline (paper: 73.33)
  python catkd.py --method ce                  # student alone (paper: 72.5)
  python catkd.py --method catkd --epochs 120  # lighter run (schedule scaled)
"""
import argparse, os, time
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T

# ----------------------------------------------------------------------------
# 1. CIFAR ResNet (same layer names as mdistiller, so the teacher ckpt loads)
# ----------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.downsample = None
        if stride != 1 or inp != out:
            self.downsample = nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False),
                                            nn.BatchNorm2d(out))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + idt)


class ResNet(nn.Module):
    def __init__(self, depth, filters=(32, 64, 128, 256), num_classes=100):
        super().__init__()
        n = (depth - 2) // 6                      # blocks per stage
        self.conv1 = nn.Conv2d(3, filters[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.inp = filters[0]
        self.layer1 = self._make(filters[1], n, 1)
        self.layer2 = self._make(filters[2], n, 2)
        self.layer3 = self._make(filters[3], n, 2)
        self.fc = nn.Linear(filters[3], num_classes)

    def _make(self, out, n, stride):
        layers = [BasicBlock(self.inp, out, stride)]
        self.inp = out
        layers += [BasicBlock(out, out) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        feat = self.layer3(self.layer2(self.layer1(x)))          # B x C x 8 x 8
        # KEY IDEA OF THE PAPER: FC -> 1x1 conv, applied BEFORE global avg pool.
        # cam[b,k] = W_k . feat[b,:,i,j] + b_k  -> one Class Activation Map per class.
        w = self.fc.weight.view(*self.fc.weight.shape, 1, 1)     # K x C x 1 x 1
        cam = F.conv2d(feat, w, self.fc.bias)                    # B x K x 8 x 8
        logits = cam.mean(dim=(2, 3))                            # == fc(GAP(feat)), identical
        return logits, cam


def resnet32x4(): return ResNet(32)   # teacher, ~7.4M params
def resnet8x4():  return ResNet(8)    # student, ~1.2M params

# ----------------------------------------------------------------------------
# 2. Losses
# ----------------------------------------------------------------------------
def cat_loss(cam_s, cam_t, res=2, normalize=False):
    """Class Attention Transfer loss (paper Eq. 5).
    Average-pool CAMs of ALL classes to res x res, optionally L2-normalise each
    class map, then mean squared distance between student and teacher."""
    s = F.adaptive_avg_pool2d(cam_s, res)                         # B x K x res x res
    t = F.adaptive_avg_pool2d(cam_t, res)
    if normalize:   # paper: use when teacher/student architectures differ
        s = F.normalize(s.flatten(2), dim=2)
        t = F.normalize(t.flatten(2), dim=2)
    return F.mse_loss(s, t)


def kd_loss(logit_s, logit_t, T_=4.0):
    """Hinton KD: KL between softened distributions, scaled by T^2."""
    return F.kl_div(F.log_softmax(logit_s / T_, 1), F.softmax(logit_t / T_, 1),
                    reduction="batchmean") * T_ * T_

# ----------------------------------------------------------------------------
# 3. Data
# ----------------------------------------------------------------------------
def loaders(bs, workers, subset=1.0):
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    tr = T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    te = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    dtr = torchvision.datasets.CIFAR100("./data", True, tr, download=True)
    dte = torchvision.datasets.CIFAR100("./data", False, te, download=True)
    if subset < 1.0:   # own experiment: keep the same fraction of EVERY class (fixed seed)
        g = torch.Generator().manual_seed(0)
        targets = torch.tensor(dtr.targets)
        keep = []
        for c in range(100):
            idx = (targets == c).nonzero().flatten()
            idx = idx[torch.randperm(len(idx), generator=g)]
            keep += idx[: int(len(idx) * subset)].tolist()
        dtr = torch.utils.data.Subset(dtr, keep)
        print(f"Training on {len(keep)} images ({subset:.0%} of CIFAR-100 train)")
    return (torch.utils.data.DataLoader(dtr, bs, shuffle=True, num_workers=workers,
                                        pin_memory=True, drop_last=True, persistent_workers=True),
            torch.utils.data.DataLoader(dte, 256, num_workers=workers, pin_memory=True))


@torch.no_grad()
def evaluate(model, loader, dev, amp=False):
    model.eval(); correct = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev == "cuda"):
            correct += (model(x)[0].argmax(1) == y).sum().item()
    return 100.0 * correct / len(loader.dataset)

# ----------------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["ce", "kd", "catkd"], default="catkd")
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=15.0, help="CAT loss weight (authors' yaml: 15)")
    p.add_argument("--cam_res", type=int, default=2, help="CAM pool size (paper: 2)")
    p.add_argument("--normalize", action="store_true", help="L2-normalise CAMs")
    p.add_argument("--binarize", action="store_true", help="binarize teacher CAMs (paper Table 3)")
    p.add_argument("--subset", type=float, default=1.0, help="fraction of training images per class (own experiment)")
    p.add_argument("--teacher_ckpt", default="download_ckpts/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth")
    p.add_argument("--out", default="runs")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", help="FP16 mixed precision (faster, can overflow)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    use_amp = a.amp and dev == "cuda"
    tag = f"{a.method}_ep{a.epochs}_b{a.beta:g}_r{a.cam_res}{'_norm' if a.normalize else ''}{'_bin' if a.binarize else ''}{f'_sub{a.subset:g}' if a.subset < 1 else ''}{'_amp' if use_amp else ''}"
    os.makedirs(a.out, exist_ok=True)
    ckpt_path, log_path = f"{a.out}/{tag}_last.pth", f"{a.out}/{tag}_log.csv"
    train_dl, test_dl = loaders(a.bs, a.workers, a.subset)

    teacher = None
    if a.method != "ce":
        teacher = resnet32x4().to(dev)
        # weights_only=False: the mdistiller ckpt also stores numpy values (trusted source)
        sd = torch.load(a.teacher_ckpt, map_location=dev, weights_only=False)
        teacher.load_state_dict(sd.get("model", sd))   # strict: fails loudly on mismatch
        teacher.eval()
        for q in teacher.parameters(): q.requires_grad_(False)
        print(f"Teacher acc: {evaluate(teacher, test_dl, dev, use_amp):.2f}  (expected ~79.42)")

    student = resnet8x4().to(dev)
    opt = torch.optim.SGD(student.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    # paper: /10 at epochs 150,180,210 of 240 -> scaled proportionally if --epochs changes
    ms = [int(a.epochs * f) for f in (150 / 240, 180 / 240, 210 / 240)]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, ms, 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, best = 0, 0.0
    if os.path.exists(ckpt_path):  # auto-resume after a Colab disconnect
        c = torch.load(ckpt_path, map_location=dev, weights_only=False)
        student.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); start, best = c["epoch"] + 1, c["best"]
        print(f"Resumed from epoch {start}, best {best:.2f}")
    elif not os.path.exists(log_path):
        open(log_path, "w").write("epoch,lr,loss,ce,aux,test_acc,best,sec\n")

    for ep in range(start, a.epochs):
        student.train(); t0 = time.time(); tot = ce_sum = aux_sum = 0.0
        for it, (x, y) in enumerate(train_dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logit_s, cam_s = student(x)
                ce = F.cross_entropy(logit_s, y)
                if a.method == "ce":
                    aux = torch.zeros((), device=dev); loss = ce
                else:
                    with torch.no_grad():
                        logit_t, cam_t = teacher(x)
                        if a.binarize:   # as in authors' code: 1 where CAM > its mean |value|, else 0
                            thr = cam_t.abs().mean(dim=(2, 3), keepdim=True)
                            cam_t = (cam_t > thr).float()
                    if a.method == "kd":        # mdistiller KD weights: 0.1 CE + 0.9 KD, T=4
                        aux = kd_loss(logit_s.float(), logit_t.float())
                        loss = 0.1 * ce + 0.9 * aux
                    else:                       # CAT-KD: L = CE + beta * L_CAT  (Eq. 6)
                        aux = cat_loss(cam_s.float(), cam_t.float(), a.cam_res, a.normalize)
                        loss = ce + a.beta * aux
            # Safety check: stop immediately instead of training on NaN for hours
            if not torch.isfinite(loss):
                print(f"STOP: loss is {loss.item()} at epoch {ep}, iter {it} | ce={ce.item():.4f} "
                      f"aux={aux.item():.6f} beta*aux={a.beta * aux.item():.4f}")
                raise SystemExit(1)
            if ep == start and it % 200 == 0:   # show loss scale early on
                print(f"  iter {it}: ce={ce.item():.4f} aux={aux.item():.6f} "
                      f"weighted aux={(a.beta if a.method == 'catkd' else 0.9) * aux.item():.4f}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); ce_sum += ce.item(); aux_sum += aux.item()
        sched.step()
        acc = evaluate(student, test_dl, dev, use_amp); best = max(best, acc); nb = len(train_dl)
        line = f"{ep},{opt.param_groups[0]['lr']:.5f},{tot/nb:.4f},{ce_sum/nb:.4f},{aux_sum/nb:.6f},{acc:.2f},{best:.2f},{time.time()-t0:.0f}"
        print(line); open(log_path, "a").write(line + "\n")
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best}, ckpt_path)
    print(f"FINAL [{tag}] best top-1 = {best:.2f}")


if __name__ == "__main__":
    main()
