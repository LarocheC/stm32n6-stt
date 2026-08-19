#!/usr/bin/env python3
"""Score a board UART capture of the multi-utterance corpus against the host.

    python firmware/test/score_corpus.py --log board/traces/corpus_run.log \
                                         --sidecar artifacts/corpus/corpus_sidecar.json

The one question this answers: is the device close enough to the laptop?  It
reports per-frame agreement, WER three ways (host vs reference, device vs
reference, device vs host) with a Levenshtein alignment implemented here, the
paired per-utterance comparison with a bootstrap and a sign test, whether the
device's disagreements sit at tight host logit margins, and whether they are
blank-placement shifts that CTC collapses away or real substitutions that reach
the transcript.

INPUT 1 -- the capture.  Any text containing lines of the form

    # u <i> ids: <n_out_frames integers separated by single spaces>

ANSI escapes are stripped.  If the firmware loops, the capture holds several
passes over the corpus; the scorer splits them, checks they are identical,
says so, and scores one.

INPUT 2 -- the sidecar written by firmware/tools/gen_corpus.py, which carries
the same utterances in the same order as the flashed blob, plus the host argmax
ids, the runner-up id and the top1-top2 logit margin for every frame.

Self-test (no board needed):

    python firmware/test/score_corpus.py --self-test

builds synthetic captures out of the sidecar's host ids with a known number of
known perturbations and checks the scorer reports exactly those.
"""
import argparse, json, math, os, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Z95 = 1.959963984540054

# ---------------------------------------------------------------- primitives

ANSI_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
ANSI_2CH = re.compile(r"\x1b[@-Z\\-_]")
# The firmware also prints a per-utterance PMU cycle count between the index
# and the ids ("# u 26 cyc 83995666 ids: ..."), so accept an optional cyc
# field and capture it -- it is the only per-utterance latency measurement
# there is, and dropping it would mean re-flashing to get it back.
LINE_RE = re.compile(r"#\s*u\s+(\d+)\s+(?:cyc\s+(\d+)\s+)?ids\s*:\s*([-\d][\d\s\-]*)")
DONE_RE = re.compile(r"#\s*corpus\s+done")
CYCLE_RE = re.compile(r"#\s*invoke\s+(\d+)\s+cycles")


def strip_ansi(s):
    s = ANSI_OSC.sub("", s)
    s = ANSI_CSI.sub("", s)
    s = ANSI_2CH.sub("", s)
    return s.replace("\r\n", "\n").replace("\r", "\n")


def wilson(k, n, z=Z95):
    """Wilson score interval for k successes in n trials.  Golden value:
    wilson(4, 17) == (0.09555, 0.47262), the interval quoted in board/GATE4.md."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def wilson_ratio(errors, words):
    """Wilson on a word error rate.  Word errors are not independent Bernoulli
    trials and insertions can push errors above the reference length, so this is
    an approximation; it is clamped and flagged when it happens.  The bootstrap
    over utterances, reported alongside, does not make that assumption."""
    if words <= 0:
        return (float("nan"), float("nan"), False)
    k = min(errors, words)
    lo, hi = wilson(k, words)
    return (lo, hi, errors > words)


def levenshtein(ref, hyp):
    """Word-level alignment implemented here, not imported.  Returns
    (substitutions, insertions, deletions).  Insertion = a word in hyp with no
    counterpart in ref; deletion = a word in ref missing from hyp."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != hyp[j - 1]))
    i, j, S, I, D = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                S += 1
            i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    assert S + I + D == int(d[n][m]), (S, I, D, int(d[n][m]))
    assert m == n - D + I, (n, m, D, I)
    return S, I, D


def ctc_greedy(ids, blank=1024):
    """Collapse consecutive repeats, then drop blank."""
    out, prev = [], -1
    for t in ids:
        t = int(t)
        if t != prev and t != blank:
            out.append(t)
        prev = t
    return out


def ids_to_text(collapsed, vocab):
    return "".join(vocab[t] for t in collapsed).replace("▁", " ").strip()


def regions_of(flags, gap=0):
    """Maximal runs of disagreeing frames, merging runs separated by <= gap
    agreeing frames.  A blank-placement shift moves a whole token run by one
    frame and therefore shows up as several adjacent disagreeing frames that are
    only collapse-neutral together -- reverting any one of them alone changes the
    output.  Regions are the unit that answers 'does this reach the transcript'."""
    idx = [int(i) for i in np.nonzero(flags)[0]]
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= gap + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


# ---------------------------------------------------------------- the sidecar

def load_sidecar(path):
    """Normalise either sidecar schema into one shape.

    AUTHORITATIVE: artifacts/corpus/corpus_ref.json from the blob task -- top level
    N/T/NMEL/BLANK/model/blob{}, utterances[] with index/key/duration_s/n_words_ref/
    ref/host_ids/host_margin/host_text/truncated.
    Also accepted: corpus_alt_sidecar.json ({"magic":"STTC-sidecar"}, utts[] with
    i/k/d/nw/ref/host_ids/host_margin/host_top2/host_text), which additionally
    carries the host's runner-up token per frame.

    Returns a dict with: blank, N, n_out_frames, model, blob (path/digest),
    utts[{i,k,d,nw,ref,host_ids,host_margin,host_top2|None,host_text,truncated}].
    """
    raw = json.load(open(path))
    out = dict(path=path)
    if raw.get("magic") == "STTC-sidecar":
        out.update(schema="corpus_sidecar", blank=raw["blank"], N=raw["N"],
                   n_out_frames=raw["n_out_frames"], model=raw["model"],
                   blob_path=raw.get("blob"),
                   digest=("sha256", raw.get("blob_sha256")))
        out["utts"] = [dict(i=u["i"], k=u["k"], d=u["d"], nw=u["nw"], ref=u["ref"],
                            host_ids=u["host_ids"], host_margin=u["host_margin"],
                            host_top2=u.get("host_top2"), host_text=u["host_text"],
                            truncated=bool(u.get("truncated", False)))
                       for u in raw["utts"]]
    elif "utterances" in raw and "BLANK" in raw:
        b = raw.get("blob", {})
        digest = ("md5", b["md5"]) if b.get("md5") else ("sha256", b.get("sha256"))
        out.update(schema="corpus_ref", blank=raw["BLANK"], N=raw["N"],
                   n_out_frames=len(raw["utterances"][0]["host_ids"]),
                   model=raw["model"], blob_path=b.get("path"), digest=digest)
        out["utts"] = [dict(i=u["index"], k=u["key"], d=u["duration_s"],
                            nw=u["n_words_ref"], ref=u["ref"],
                            host_ids=u["host_ids"], host_margin=u["host_margin"],
                            host_top2=u.get("host_top2"), host_text=u["host_text"],
                            truncated=bool(u.get("truncated", False)))
                       for u in raw["utterances"]]
    else:
        raise SystemExit(f"{path}: unrecognised sidecar schema "
                         f"(keys {sorted(raw)[:12]})")
    n = out["n_out_frames"]
    for u in out["utts"]:
        if len(u["host_ids"]) != n or len(u["host_margin"]) != n:
            raise SystemExit(f"{path}: utterance {u['i']} has "
                             f"{len(u['host_ids'])} ids / {len(u['host_margin'])} margins, "
                             f"expected {n}")
    if len(out["utts"]) != out["N"]:
        raise SystemExit(f"{path}: N={out['N']} but {len(out['utts'])} utterances")
    out["raw"] = raw
    return out


# ---------------------------------------------------------------- log parsing

def parse_log(text, n_out_frames):
    """Split a capture into passes over the corpus.  A new pass starts when an
    index repeats or after a '# corpus done' marker."""
    text = strip_ansi(text)
    lines = text.split("\n")
    passes, cur, bad, cycles = [], {}, [], []
    n_lines = 0
    li = -1
    while li + 1 < len(lines):
        li += 1
        raw = lines[li]
        if CYCLE_RE.search(raw):
            cycles.append(int(CYCLE_RE.search(raw).group(1)))
        if DONE_RE.search(raw):
            if cur:
                passes.append(cur); cur = {}
            continue
        m = LINE_RE.search(raw)
        if not m:
            continue
        n_lines += 1
        idx = int(m.group(1))
        if m.group(2):
            cycles.append(int(m.group(2)))
        try:
            ids = [int(x) for x in m.group(3).split()]
        except ValueError:
            bad.append((idx, "unparsable ids", raw.strip()[:100])); continue
        # a terminal capture can hard-wrap a 100-number line; absorb following
        # lines that are nothing but integers until the count is satisfied.
        while len(ids) < n_out_frames and li + 1 < len(lines):
            j = li + 1
            while j < len(lines) and not lines[j].strip():
                j += 1                      # blank lines carry nothing; step over them
            if j >= len(lines) or not re.fullmatch(r"[\d\s]+", lines[j].strip()):
                break
            ids += [int(x) for x in lines[j].split()]
            li = j
        if len(ids) != n_out_frames:
            bad.append((idx, f"{len(ids)} ids, expected {n_out_frames}", raw.strip()[:100]))
            continue
        if min(ids) < 0 or max(ids) > 1024:
            bad.append((idx, f"id out of range [{min(ids)},{max(ids)}]", raw.strip()[:100]))
            continue
        if idx in cur:
            passes.append(cur); cur = {}
        cur[idx] = ids
    if cur:
        passes.append(cur)
    return dict(passes=passes, bad=bad, n_id_lines=n_lines, cycles=cycles)


def compare_passes(passes):
    """Are the repeats identical?  Returns (identical, notes)."""
    notes = []
    if len(passes) < 2:
        return True, notes
    ref = passes[0]
    identical = True
    for p, cur in enumerate(passes[1:], start=1):
        if set(cur) != set(ref):
            identical = False
            miss = sorted(set(ref) - set(cur)); extra = sorted(set(cur) - set(ref))
            notes.append(f"pass {p}: index set differs (missing {miss[:8]}, extra {extra[:8]})")
        diff = [i for i in sorted(set(cur) & set(ref)) if cur[i] != ref[i]]
        if diff:
            identical = False
            nfr = {i: sum(a != b for a, b in zip(ref[i], cur[i])) for i in diff}
            notes.append(f"pass {p}: {len(diff)} utterances differ from pass 0 "
                         f"({', '.join(f'u{i}:{nfr[i]}fr' for i in diff[:8])})")
    return identical, notes


# ---------------------------------------------------------------- statistics

def boot_indices(n, B, rng):
    return rng.integers(0, n, size=(B, n))


def paired_bootstrap(dev_err, host_err, words, B, rng):
    """Percentile bootstrap over utterances (the resampling unit) on the
    corpus-level WER difference (total device errors - total host errors) /
    total reference words."""
    dev_err = np.asarray(dev_err, float); host_err = np.asarray(host_err, float)
    words = np.asarray(words, float)
    idx = boot_indices(len(words), B, rng)
    w = words[idx].sum(1)
    d = (dev_err[idx].sum(1) - host_err[idx].sum(1)) / np.maximum(w, 1)
    obs = (dev_err.sum() - host_err.sum()) / max(words.sum(), 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, lo, hi, min(1.0, float(p)), d


def boot_wer(err, words, B, rng):
    err = np.asarray(err, float); words = np.asarray(words, float)
    idx = boot_indices(len(words), B, rng)
    r = err[idx].sum(1) / np.maximum(words[idx].sum(1), 1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def signflip_test(diff, words, B, rng):
    """Paired permutation test: under the null the sign of each utterance's
    (device errors - host errors) is exchangeable."""
    diff = np.asarray(diff, float)
    obs = diff.sum() / max(np.asarray(words, float).sum(), 1)
    signs = rng.choice([-1.0, 1.0], size=(B, len(diff)))
    stat = (signs * diff).sum(1) / max(np.asarray(words, float).sum(), 1)
    p = (np.abs(stat) >= abs(obs) - 1e-12).mean()
    return float(obs), float(p)


def sign_test(diff):
    """Exact two-sided binomial sign test on the utterances where the device and
    the host differ."""
    nz = [d for d in diff if d != 0]
    n = len(nz); k = sum(1 for d in nz if d > 0)
    if n == 0:
        return n, k, 1.0
    def tail(x):
        return sum(math.comb(n, i) for i in range(0, x + 1)) / 2.0 ** n
    p = 2 * min(tail(min(k, n - k)), 1.0)
    return n, k, min(1.0, p)


def auc_less(a, b):
    """P(a < b) + 0.5 P(a == b) over all pairs, exactly, via searchsorted.
    auc_less([0,0],[1,1]) == 1.0 ; auc_less([1,1],[0,0]) == 0.0 ; ties give 0.5."""
    a = np.asarray(a, float); b = np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    lo = np.searchsorted(b, a, "left")          # count of b strictly < a
    hi = np.searchsorted(b, a, "right")         # count of b <= a
    greater = len(b) - hi                       # count of b strictly > a
    ties = hi - lo
    return float((greater + 0.5 * ties).sum() / (len(a) * len(b)))


# ---------------------------------------------------------------- the scoring

def score(side, dev, B=10000, seed=1, gap=0):
    """dev: {utterance index -> list of ids}.  Returns a dict of everything."""
    rng = np.random.default_rng(seed)
    blank = side["blank"]
    vocab = [l.rsplit(" ", 1)[0] for l in
             open(os.path.join(REPO, "tokenizer", "vocab.txt"), encoding="utf-8")
             .read().split("\n") if l.strip()]
    utts = {u["i"]: u for u in side["utts"]}
    have_top2 = all(u.get("host_top2") for u in side["utts"])
    idxs = sorted(i for i in dev if i in utts)
    missing = sorted(set(utts) - set(dev))
    unknown = sorted(set(dev) - set(utts))

    rows, all_margin, dis_margin, dis_frames, dis_regions = [], [], [], [], []
    for i in idxs:
        u = utts[i]
        h = np.asarray(u["host_ids"]); d = np.asarray(dev[i])
        mg = np.asarray(u["host_margin"], float)
        t2 = np.asarray(u["host_top2"]) if have_top2 else None
        ne = h != d
        all_margin.append(mg)
        dis_margin.append(mg[ne])

        hc, dc = ctc_greedy(h, blank), ctc_greedy(d, blank)
        htext, dtext = ids_to_text(hc, vocab), ids_to_text(dc, vocab)
        ref = u["ref"].lower().split()
        hS, hI, hD = levenshtein(ref, htext.split())
        dS, dI, dD = levenshtein(ref, dtext.split())
        xS, xI, xD = levenshtein(htext.split(), dtext.split())

        regs = regions_of(ne, gap)
        frame_region = {}
        for rg in regs:
            rev = d.copy()
            for f in rg:
                rev[f] = h[f]
            r_neutral = ctc_greedy(rev, blank) == dc
            # compare the collapsed content over the region PLUS one agreeing frame
            # of context on each side: that context is what makes an extension of a
            # token run into an adjacent blank a shift rather than a new token.
            a0, a1 = max(0, rg[0] - 1), min(len(h), rg[-1] + 2)
            hs = ctc_greedy(h[a0:a1], blank)
            ds = ctc_greedy(d[a0:a1], blank)
            if r_neutral and hs == ds:
                cls = "blank-shift"          # same tokens, moved against the blanks
            elif r_neutral:
                cls = "neutral-other"        # collapsed away, but not a pure shift
            else:
                cls = "visible"              # reaches the transcript
            rec = dict(u=i, frames=[int(f) for f in rg], n=len(rg), cls=cls,
                       neutral=bool(r_neutral),
                       host=[int(x) for x in h[rg]], dev=[int(x) for x in d[rg]],
                       min_margin=float(mg[rg].min()))
            dis_regions.append(rec)
            for f in rg:
                frame_region[int(f)] = cls

        for f in np.nonzero(ne)[0]:
            f = int(f)
            rev = d.copy(); rev[f] = h[f]
            neutral = ctc_greedy(rev, blank) == dc
            if h[f] == blank:
                kind = "blank->token"
            elif d[f] == blank:
                kind = "token->blank"
            else:
                kind = "token->token"
            dis_frames.append(dict(u=i, f=f, host=int(h[f]), dev=int(d[f]),
                                   margin=float(mg[f]),
                                   is_top2=(bool(d[f] == t2[f]) if have_top2 else None),
                                   kind=kind, neutral=bool(neutral),
                                   region_cls=frame_region[f]))
        rows.append(dict(i=i, k=u["k"], nw=len(ref), ref=u["ref"], d=u["d"],
                         nw_sidecar=u.get("nw"), truncated=bool(u.get("truncated")),
                         host_text=htext, dev_text=dtext,
                         sidecar_text=u["host_text"],
                         n_frames=len(h), n_diff=int(ne.sum()),
                         host_S=hS, host_I=hI, host_D=hD, host_err=hS + hI + hD,
                         dev_S=dS, dev_I=dI, dev_D=dD, dev_err=dS + dI + dD,
                         xS=xS, xI=xI, xD=xD, x_err=xS + xI + xD,
                         host_ids_len=len(hc), same_ids=bool(hc == dc),
                         same_text=bool(htext == dtext),
                         n_host_words=len(htext.split())))

    all_margin = np.concatenate(all_margin) if all_margin else np.zeros(0)
    dis_margin = np.concatenate(dis_margin) if dis_margin else np.zeros(0)
    nfr = sum(r["n_frames"] for r in rows)
    ndf = sum(r["n_diff"] for r in rows)

    words = [r["nw"] for r in rows]
    hw = [r["n_host_words"] for r in rows]
    he = [r["host_err"] for r in rows]
    de = [r["dev_err"] for r in rows]
    xe = [r["x_err"] for r in rows]
    diff = [a - b for a, b in zip(de, he)]

    obs, blo, bhi, bp, _ = paired_bootstrap(de, he, words, B, rng)
    perm_obs, perm_p = signflip_test(diff, words, B, rng)
    sn, sk, sp = sign_test(diff)

    def agg(errs, wds, S, I, D):
        tw = sum(wds); te = sum(errs)
        lo, hi, over = wilson_ratio(te, tw)
        blo_, bhi_ = boot_wer(errs, wds, B, rng)
        return dict(S=sum(S), I=sum(I), D=sum(D), err=te, words=tw,
                    wer=te / tw if tw else float("nan"),
                    wilson=(lo, hi), over=over, boot=(blo_, bhi_))

    res = dict(
        rows=rows, missing=missing, unknown=unknown,
        n_utt=len(rows), n_frames=nfr, n_diff=ndf,
        frame_rate=ndf / nfr if nfr else float("nan"),
        frame_wilson=wilson(ndf, nfr),
        host=agg(he, words, [r["host_S"] for r in rows], [r["host_I"] for r in rows],
                 [r["host_D"] for r in rows]),
        dev=agg(de, words, [r["dev_S"] for r in rows], [r["dev_I"] for r in rows],
                [r["dev_D"] for r in rows]),
        xhost=agg(xe, hw, [r["xS"] for r in rows], [r["xI"] for r in rows],
                  [r["xD"] for r in rows]),
        paired=dict(obs=obs, lo=blo, hi=bhi, p=bp, B=B,
                    mean_diff=float(np.mean(diff)) if diff else float("nan"),
                    worse=sum(1 for x in diff if x > 0),
                    better=sum(1 for x in diff if x < 0),
                    tied=sum(1 for x in diff if x == 0),
                    perm_p=perm_p, sign_n=sn, sign_k=sk, sign_p=sp,
                    diff=diff),
        margins=dict(all=all_margin, dis=dis_margin),
        dis_frames=dis_frames, dis_regions=dis_regions, gap=gap,
        have_top2=have_top2,
    )

    # margin enrichment: deciles of the all-frame margin distribution
    if len(all_margin) and len(dis_margin):
        edges = np.percentile(all_margin, np.arange(0, 101, 10))
        edges[0] = -np.inf; edges[-1] = np.inf
        cnt_all, _ = np.histogram(all_margin, bins=edges)
        cnt_dis, _ = np.histogram(dis_margin, bins=edges)
        res["margin_deciles"] = dict(edges=edges, all=cnt_all, dis=cnt_dis)
        res["auc"] = auc_less(dis_margin, all_margin)
        n1 = len(dis_margin)
        draws = rng.choice(all_margin, size=(min(B, 2000), n1), replace=True)
        null = np.array([auc_less(row, all_margin) for row in draws])
        res["auc_p"] = float((np.abs(null - 0.5) >= abs(res["auc"] - 0.5) - 1e-12).mean())
        res["auc_B"] = len(null)
        for q in (1, 5, 10):
            thr = np.percentile(all_margin, q)
            res[f"dis_below_p{q}"] = float((dis_margin <= thr).mean())
    return res


# ---------------------------------------------------------------- the report

def pct(x):
    return "nan" if x != x else f"{100*x:.2f} %"


def report(res, side, meta, out=sys.stdout):
    P = lambda *a: print(*a, file=out)
    rows = res["rows"]
    P("=" * 78)
    P("CORPUS SCORE -- device vs host, %d utterances" % res["n_utt"])
    P("=" * 78)
    for k, v in meta.items():
        P(f"  {k:<22} {v}")
    P(f"  {'model (host ref)':<22} {side['model']}")
    P(f"  {'sidecar schema':<22} {side['schema']}  ({side['path']})")
    P(f"  {'blob':<22} {side['blob_path']}  {side['digest'][0]} {side['digest'][1]}")
    P(f"  {'sidecar N / frames':<22} {side['N']} / {side['n_out_frames']}")
    trunc = [r["k"] for r in rows if r["truncated"]]
    if trunc:
        P(f"  {'truncated utterances':<22} {len(trunc)} -- {trunc}")
        P(f"  {'':<22} (longer than the 800-frame window; host and device see the")
        P(f"  {'':<22}  same truncated features, so device-vs-host is unaffected,")
        P(f"  {'':<22}  but their WER against the reference is inflated for both)")
    bad_nw = [r["k"] for r in rows if r["nw_sidecar"] not in (None, r["nw"])]
    if bad_nw:
        P(f"  !! reference word count disagrees with the sidecar on {len(bad_nw)}: {bad_nw[:5]}")
    if res["missing"]:
        P(f"  !! MISSING from capture: {res['missing']}")
    if res["unknown"]:
        P(f"  !! indices not in sidecar: {res['unknown']}")

    # ---- 1
    P("\n" + "-" * 78)
    P("1. PER-FRAME AGREEMENT (argmax id, device vs host onnxruntime)")
    P("-" * 78)
    lo, hi = res["frame_wilson"]
    P(f"  frames compared        {res['n_frames']}")
    P(f"  frames disagreeing     {res['n_diff']}")
    P(f"  disagreement rate      {pct(res['frame_rate'])}   Wilson 95% [{pct(lo)}, {pct(hi)}]")
    per = sorted(r["n_diff"] for r in rows)
    nz = [r for r in rows if r["n_diff"]]
    P(f"  utterances with >=1 disagreeing frame   {len(nz)} / {len(rows)}")
    if per:
        P(f"  per-utterance disagreeing frames: min {per[0]}  median {int(np.median(per))} "
          f" p90 {int(np.percentile(per,90))}  max {per[-1]}")
    P("  NOTE adjacent frames are not independent, so the Wilson interval here is")
    P("       an approximation; it is reported because the brief asks for it.")

    # ---- 2
    P("\n" + "-" * 78)
    P("2. WER  (corpus-level = total errors / total reference words, the standard")
    P("   definition -- NOT the mean of per-utterance rates)")
    P("-" * 78)
    P(f"  {'comparison':<26}{'S':>5}{'I':>5}{'D':>5}{'err':>6}{'words':>7}   WER      Wilson 95%           bootstrap 95%")
    for name, key in (("host vs reference", "host"), ("device vs reference", "dev"),
                      ("device vs HOST", "xhost")):
        a = res[key]
        P(f"  {name:<26}{a['S']:>5}{a['I']:>5}{a['D']:>5}{a['err']:>6}{a['words']:>7}   "
          f"{pct(a['wer']):<9}[{pct(a['wilson'][0])}, {pct(a['wilson'][1])}]  "
          f"[{pct(a['boot'][0])}, {pct(a['boot'][1])}]")
        if a["over"]:
            P("      (errors exceeded reference words; Wilson clamped -- trust the bootstrap)")
    P("  'device vs HOST' uses the host transcript as the reference: it is the")
    P("  'close enough to the laptop' number, and it is the only one of the three")
    P("  that isolates the device from the model's own errors.")

    P("\n  per-utterance distribution of WER (errors / reference words):")
    for name, key, fld in (("host  ", "host", "host_err"), ("device", "dev", "dev_err"),
                           ("dev|host", "xhost", "x_err")):
        base = [r["nw"] if key != "xhost" else max(r["n_host_words"], 1) for r in rows]
        v = np.array([r[fld] / b for r, b in zip(rows, base)])
        qs = np.percentile(v, [0, 25, 50, 75, 90, 100])
        P(f"    {name:<9} min {pct(qs[0])}  q1 {pct(qs[1])}  med {pct(qs[2])}  "
          f"q3 {pct(qs[3])}  p90 {pct(qs[4])}  max {pct(qs[5])}   "
          f"perfect {int((v==0).sum())}/{len(v)}")

    P("\n  per-utterance table (dfr = disagreeing frames; err = S+I+D vs reference):")
    P(f"  {'i':>3} {'key':<18}{'nw':>4}{'dfr':>5}{'hErr':>5}{'dErr':>5}{'dif':>5}"
      f"{'x(S/I/D)':>10}  same_ids")
    for r in rows:
        sid = "%d/%d/%d" % (r["xS"], r["xI"], r["xD"])
        P(f"  {r['i']:>3} {r['k']:<18}{r['nw']:>4}{r['n_diff']:>5}{r['host_err']:>5}"
          f"{r['dev_err']:>5}{r['dev_err']-r['host_err']:>+5}{sid:>10}"
          f"  {'yes' if r['same_ids'] else 'NO'}")
    worst = sorted(rows, key=lambda r: -(r["dev_err"] - r["host_err"]))[:5]
    P("\n  utterances where the device costs the most relative to the host:")
    for r in worst:
        if r["dev_err"] - r["host_err"] <= 0:
            continue
        P(f"    u{r['i']} {r['k']}  +{r['dev_err']-r['host_err']} errors, {r['n_diff']} frames differ")
        P(f"      ref  : {r['ref'].lower()}")
        P(f"      host : {r['host_text']}")
        P(f"      dev  : {r['dev_text']}")

    # ---- 3
    P("\n" + "-" * 78)
    P("3. PAIRED COMPARISON (per utterance: device errors - host errors)")
    P("-" * 78)
    p = res["paired"]
    P(f"  mean difference per utterance   {p['mean_diff']:+.4f} errors")
    P(f"  device worse on {p['worse']} utterances, better on {p['better']}, tied on {p['tied']}")
    P(f"  corpus WER difference           {100*p['obs']:+.3f} points")
    P(f"    percentile bootstrap over utterances (B={p['B']}, resampling unit = utterance)")
    P(f"    95% CI [{100*p['lo']:+.3f}, {100*p['hi']:+.3f}] points,  two-sided p = {p['p']:.4f}")
    P(f"  paired sign-flip permutation test (B={p['B']}): p = {p['perm_p']:.4f}")
    P(f"  exact two-sided sign test on the {p['sign_n']} non-tied utterances "
      f"({p['sign_k']} worse): p = {p['sign_p']:.4f}")
    d = np.array(p["diff"])
    if len(d):
        vals, cnts = np.unique(d, return_counts=True)
        P("  distribution of the per-utterance difference (errors: count): " +
          ", ".join(f"{int(v):+d}: {int(c)}" for v, c in zip(vals, cnts)))

    # ---- 4
    P("\n" + "-" * 78)
    P("4. DO DISAGREEMENTS SIT AT TIGHT HOST MARGINS?  (margin = host top1 - top2 logit)")
    P("-" * 78)
    am, dm = res["margins"]["all"], res["margins"]["dis"]
    if len(dm) == 0:
        P("  no disagreeing frames -- nothing to test.")
    else:
        qs = [1, 5, 10, 25, 50, 75, 90]
        P(f"  {'percentile':<14}" + "".join(f"{q:>8}" for q in qs))
        P(f"  {'all frames':<14}" + "".join(f"{np.percentile(am,q):>8.3f}" for q in qs))
        P(f"  {'disagreeing':<14}" + "".join(f"{np.percentile(dm,q):>8.3f}" for q in qs))
        P(f"  median margin: all {np.median(am):.3f}   disagreeing {np.median(dm):.3f}")
        P(f"  P(margin at a disagreeing frame < margin at a random frame) = {res['auc']:.4f}")
        P(f"    (0.5 = no relation; permutation p = {res['auc_p']:.4f}, B={res['auc_B']})")
        for q in (1, 5, 10):
            P(f"    {pct(res[f'dis_below_p{q}']):>8} of disagreements fall in the tightest "
              f"{q} % of margins (expected {q} % if unrelated)")
        md = res["margin_deciles"]
        P(f"  {'decile of margin':<20}{'all frames':>12}{'disagree':>10}{'rate':>10}{'enrichment':>12}")
        base = res["frame_rate"]
        for k in range(10):
            a, b = int(md["all"][k]), int(md["dis"][k])
            r = b / a if a else float("nan")
            P(f"  {k+1:>2} [{md['edges'][k]:>7.3f},{md['edges'][k+1]:>7.3f}]{a:>10}{b:>10}"
              f"{pct(r):>10}{(r/base if base else float('nan')):>11.2f}x")
        if res["have_top2"]:
            top2 = sum(1 for f in res["dis_frames"] if f["is_top2"])
            P(f"  device chose the host's runner-up token on {top2} of "
              f"{len(res['dis_frames'])} disagreeing frames "
              f"({pct(top2/len(res['dis_frames']))})")
        else:
            P("  the sidecar carries no per-frame runner-up id (host_top2), so the")
            P("  'did the device pick the host's second choice' split is not available.")

    # ---- 5
    P("\n" + "-" * 78)
    P("5. BLANK-PLACEMENT SHIFT THAT CTC COLLAPSES AWAY, OR A REAL SUBSTITUTION?")
    P("-" * 78)
    P("  A disagreement is collapse-neutral if reverting it to the host ids leaves")
    P("  the device's collapsed id sequence unchanged -- CTC swallows it.")
    df = res["dis_frames"]; rg = res["dis_regions"]
    if not df:
        P("  no disagreeing frames.")
    else:
        P(f"  A blank-placement shift moves a whole token run by a frame, so it appears")
        P(f"  as several ADJACENT disagreeing frames that are only collapse-neutral")
        P(f"  together -- reverting any one of them alone changes the output.  The unit")
        P(f"  of the question is therefore the region: a maximal run of adjacent")
        P(f"  disagreeing frames (gap tolerance {res['gap']}).")
        P("")
        P(f"  {'region class':<24}{'regions':>9}{'frames':>9}")
        for cls, what in (("blank-shift", "same tokens, re-aligned against the blanks"),
                          ("neutral-other", "collapsed away, but not a pure shift"),
                          ("visible", "REACHES THE TRANSCRIPT")):
            n = sum(1 for r in rg if r["cls"] == cls)
            f = sum(r["n"] for r in rg if r["cls"] == cls)
            P(f"  {cls:<24}{n:>9}{f:>9}   {what}")
        P(f"  {'TOTAL':<24}{len(rg):>9}{len(df):>9}")
        neu = sum(r["n"] for r in rg if r["neutral"])
        P(f"  {pct(neu/len(df))} of disagreeing frames ({neu}/{len(df)}) are collapsed away")
        P(f"  by CTC; {len(df)-neu} reach the transcript.")
        P("")
        P("  single-frame test, for reference (revert one frame at a time):")
        n1 = sum(1 for f in df if f["neutral"])
        P(f"    {n1} neutral / {len(df)-n1} visible -- this UNDERCOUNTS shifts, see above.")
        P("")
        kinds = ["blank->token", "token->blank", "token->token"]
        P(f"  per-frame id change, split by the verdict of the region it belongs to:")
        P(f"  {'kind':<16}{'in neutral':>12}{'in visible':>12}{'total':>7}")
        for k in kinds:
            n = sum(1 for f in df if f["kind"] == k and f["region_cls"] != "visible")
            v = sum(1 for f in df if f["kind"] == k and f["region_cls"] == "visible")
            P(f"  {k:<16}{n:>12}{v:>12}{n+v:>7}")
        blanky = sum(1 for f in df if f["kind"] != "token->token")
        P(f"  {blanky} of {len(df)} frames involve a blank on one side "
          f"({pct(blanky/len(df))}); {len(df)-blanky} are token->token.")
        nz = [r for r in rows if r["n_diff"]]
        same = sum(1 for r in nz if r["same_ids"])
        P(f"  utterances with disagreeing frames whose collapsed id sequence is still")
        P(f"  identical to the host's: {same} / {len(nz)}   "
          f"(identical transcript text: {sum(1 for r in nz if r['same_text'])} / {len(nz)})")
        vis = [r for r in rg if r["cls"] == "visible"]
        if vis and len(vis) <= 40:
            P("\n  every region that reaches the transcript:")
            for r in sorted(vis, key=lambda r: (r["u"], r["frames"][0])):
                P(f"    u{r['u']:<3} frames {r['frames'][0]}..{r['frames'][-1]}  "
                  f"host {r['host']} -> dev {r['dev']}  min margin {r['min_margin']:.3f}")

    # ---- 6
    P("\n" + "-" * 78)
    P("6. VERDICT")
    P("-" * 78)
    x = res["xhost"]; h = res["host"]; dv = res["dev"]
    P(f"  Device-vs-host WER is {pct(x['wer'])} (bootstrap 95% [{pct(x['boot'][0])}, {pct(x['boot'][1])}])")
    P(f"  over {x['words']} host words in {res['n_utt']} utterances.  Against the reference,")
    P(f"  host {pct(h['wer'])} vs device {pct(dv['wer'])}; the paired difference is "
      f"{100*p['obs']:+.3f} points")
    P(f"  (bootstrap 95% [{100*p['lo']:+.3f}, {100*p['hi']:+.3f}], p = {p['p']:.4f}; "
      f"sign test p = {p['sign_p']:.4f}).")
    supported, unsupported = [], []
    if p["lo"] <= 0 <= p["hi"]:
        supported.append("the paired 95 % CI contains 0: this corpus does not show the "
                         "device to be worse than the host")
        if res["n_diff"] and max(abs(p["lo"]), abs(p["hi"])) > 5e-5:
            unsupported.append(f"'no difference' is NOT established -- the same data are "
                               f"consistent with the device being up to "
                               f"{100*max(abs(p['lo']), abs(p['hi'])):.2f} WER points worse")
        elif res["n_diff"]:
            supported.append("every bootstrap resample gives a paired difference of zero: "
                             "no disagreement on this corpus reached the transcript")
        else:
            supported.append("the device output is identical to the host on every frame "
                             "of every utterance, so there is nothing left to test")
    else:
        supported.append(f"the paired 95 % CI excludes 0: the device is "
                         f"{'worse' if p['obs']>0 else 'better'} than the host on this corpus")
    supported.append(f"the device reproduces the host transcript exactly on "
                     f"{sum(1 for r in rows if r['same_text'])} of {len(rows)} utterances")
    if df:
        neu = sum(r["n"] for r in rg if r["neutral"])
        supported.append(f"{pct(neu/len(df))} of frame disagreements are collapsed away "
                         f"by CTC and never reach the transcript")
    unsupported.append(f"{res['n_utt']} utterances / {x['words']} words is a small sample: "
                       f"the bootstrap CI on the paired difference is "
                       f"{100*(p['hi']-p['lo']):.2f} points wide, so any true difference "
                       f"smaller than that is invisible here")
    unsupported.append("one corpus (LibriSpeech dev-clean, clean read speech, <= 7.69 s) "
                       "and one decode (greedy CTC); nothing here speaks to noisy audio, "
                       "other speakers, longer utterances, or a beam search")
    unsupported.append("the frames come from host-computed features fed to the device: this "
                       "isolates the NPU, and says nothing about the on-device front end")
    P("\n  SUPPORTED by these numbers:")
    for s in supported:
        P(f"    - {s}")
    P("\n  NOT supported by these numbers:")
    for s in unsupported:
        P(f"    - {s}")
    P("=" * 78)


# ---------------------------------------------------------------- self-test

def _fmt_log(dev, ansi=False, done=True):
    out = []
    for i in sorted(dev):
        line = f"# u {i} ids: " + " ".join(str(int(x)) for x in dev[i])
        if ansi and i % 3 == 0:
            line = "\x1b[32m" + line + "\x1b[0m"
        out.append(line)
        out.append(f"# invoke {74421588 + i} cycles = 124.035 ms at 600000000 Hz")
    if done:
        out.append("# corpus done")
    return "\n".join(out) + "\n"


def _check(name, got, want, fails):
    ok = got == want
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, want {want!r}")
    if not ok:
        fails.append(name)
    return ok


def self_test(side_path, B=2000):
    fails = []
    print("=" * 78)
    print("SELF-TEST -- the scorer is shown wrong answers it already knows")
    print("=" * 78)

    print("\n  A. primitives")
    lo, hi = wilson(4, 17)
    _check("wilson(4,17) low  == board/GATE4.md 9.6 %", round(100 * lo, 1), 9.6, fails)
    _check("wilson(4,17) high == board/GATE4.md 47.3 %", round(100 * hi, 1), 47.3, fails)
    _check("wilson(0,100) low", round(wilson(0, 100)[0], 6), 0.0, fails)
    _check("ctc collapse+blank", ctc_greedy([1024, 5, 5, 1024, 5, 7, 7, 1024], 1024), [5, 5, 7], fails)
    _check("ctc all blank", ctc_greedy([1024] * 10, 1024), [], fails)
    _check("levenshtein identical", levenshtein("a b c".split(), "a b c".split()), (0, 0, 0), fails)
    _check("levenshtein 1 sub", levenshtein("a b c".split(), "a x c".split()), (1, 0, 0), fails)
    _check("levenshtein 1 del", levenshtein("a b c".split(), "a c".split()), (0, 0, 1), fails)
    _check("levenshtein 1 ins", levenshtein("a b c".split(), "a b x c".split()), (0, 1, 0), fails)
    _check("levenshtein empty hyp", levenshtein("a b c".split(), []), (0, 0, 3), fails)
    _check("levenshtein empty ref", levenshtein([], "a b".split()), (0, 2, 0), fails)
    _check("auc_less all-lower", auc_less([0.0, 0.0], [1.0, 1.0]), 1.0, fails)
    _check("auc_less all-higher", auc_less([1.0, 1.0], [0.0, 0.0]), 0.0, fails)
    _check("auc_less all-tied", auc_less([1.0, 1.0], [1.0, 1.0]), 0.5, fails)
    _check("levenshtein 2s1d1i",
           levenshtein("the quick brown fox jumps".split(), "the quik brown red fox jump".split()),
           (2, 1, 0), fails)
    # cross-check total edit distance against model/fe.py's independent DP
    sys.path.insert(0, os.path.join(REPO, "model"))
    import fe as _fe
    rng = np.random.default_rng(7)
    worst = 0
    for _ in range(300):
        r = [str(x) for x in rng.integers(0, 5, rng.integers(0, 9))]
        hh = [str(x) for x in rng.integers(0, 5, rng.integers(0, 9))]
        mine = sum(levenshtein(r, hh))
        theirs = int(_fe.wer(" ".join(r), " ".join(hh))[0]) if r or hh else 0
        worst = max(worst, abs(mine - theirs))
    _check("edit distance == model/fe.py wer() on 300 random pairs (max |diff|)", worst, 0, fails)

    side = load_sidecar(side_path)
    blank = side["blank"]
    vocab = [l.rsplit(" ", 1)[0] for l in
             open(os.path.join(REPO, "tokenizer", "vocab.txt"), encoding="utf-8")
             .read().split("\n") if l.strip()]
    utts = {u["i"]: u for u in side["utts"]}
    have_top2 = all(u.get("host_top2") for u in side["utts"])
    host = {i: list(u["host_ids"]) for i, u in utts.items()}
    print(f"     sidecar: {side_path}")
    print(f"     schema {side['schema']}  N={side['N']}  frames={side['n_out_frames']}  "
          f"host_top2 {'present' if have_top2 else 'ABSENT (a stand-in token is used)'}")

    def alt(u, f):
        """The token to substitute at frame f: the host's runner-up when the sidecar
        carries it, otherwise a deterministic different non-blank token."""
        if u.get("host_top2"):
            return u["host_top2"][f]
        return (u["host_ids"][f] + 1) % blank

    # ---------------- construct the injected device output
    dev = {i: list(v) for i, v in host.items()}
    inj = {}

    # u0: pure blank shift -- extend a token run one frame earlier into a blank.
    h0 = host[0]; shifted = []
    for f in range(1, len(h0) - 1):
        if h0[f] == blank and h0[f + 1] != blank and h0[f - 1] != h0[f + 1]:
            dev[0][f] = h0[f + 1]; shifted.append(f)
        if len(shifted) == 3:
            break
    inj["u0_blank_shift_frames"] = len(shifted)

    # u1: substitution at the tightest host margin among non-blank frames.
    u1 = utts[1]; mg = np.asarray(u1["host_margin"], float)
    nb = [f for f in range(len(mg)) if u1["host_ids"][f] != blank]
    f1 = min(nb, key=lambda f: mg[f])
    dev[1][f1] = alt(u1, f1)
    inj["u1_tight_margin_frame"] = f1
    inj["u1_margin"] = float(mg[f1])

    # u2: catastrophic -- every frame blank, so the device emits nothing.  Pick the
    # utterance with the most non-blank frames so the injection is as large as the
    # corpus allows (index 2 can be nearly all blank).
    U2 = max(host, key=lambda i: (sum(1 for t in host[i] if t != blank), -i))
    dev[U2] = [blank] * len(host[U2])
    inj["u2_index"] = U2
    inj["u2_blanked_frames"] = sum(1 for a, b in zip(host[U2], dev[U2]) if a != b)

    # u3: delete one whole token run -> a real deletion that reaches the transcript.
    # Prefer an utterance that has a run of >= 2 frames, so the region has width.
    U3 = next((i for i in sorted(host) if i not in (0, 1, U2) and
               any(host[i][f] != blank and f + 1 < len(host[i]) and
                   host[i][f + 1] == host[i][f] for f in range(len(host[i])))),
              3 if 3 not in (0, 1, U2) else 4)
    inj["u3_index"] = U3
    h3 = host[U3]; runs = []
    f = 0
    while f < len(h3):
        if h3[f] != blank:
            g = f
            while g + 1 < len(h3) and h3[g + 1] == h3[f]:
                g += 1
            runs.append((f, g))
            f = g + 1
        else:
            f += 1
    tgt = max(runs, key=lambda r: (r[1] - r[0], -r[0]))
    for f in range(tgt[0], tgt[1] + 1):
        dev[U3][f] = blank
    inj["u3_run_blanked_frames"] = tgt[1] - tgt[0] + 1
    inj["u3_token_removed"] = int(h3[tgt[0]])

    total_inj = sum(sum(1 for a, b in zip(host[i], dev[i]) if a != b) for i in dev)
    print(f"\n  B. injected: {json.dumps(inj)}")
    print(f"     total perturbed frames = {total_inj}")

    # ---------------- parsing
    print("\n  C. log parsing")
    log = _fmt_log(dev, ansi=True)
    pr = parse_log(log, side["n_out_frames"])
    _check("passes found", len(pr["passes"]), 1, fails)
    _check("bad lines", len(pr["bad"]), 0, fails)
    _check("utterances parsed", len(pr["passes"][0]), side["N"], fails)
    _check("ANSI stripped, ids intact", pr["passes"][0][0], dev[0], fails)

    rep3 = parse_log(log * 3, side["n_out_frames"])
    ident, notes = compare_passes(rep3["passes"])
    _check("3 identical repeats detected", (len(rep3["passes"]), ident), (3, True), fails)

    bad = {i: list(v) for i, v in dev.items()}
    bad[5] = list(bad[5]); bad[5][10] = (bad[5][10] + 1) % 1024
    mixed = parse_log(log + _fmt_log(bad), side["n_out_frames"])
    ident2, notes2 = compare_passes(mixed["passes"])
    _check("differing repeat flagged", ident2, False, fails)
    print(f"      note: {notes2[0] if notes2 else '(none)'}")

    partial = {i: v for i, v in host.items() if i not in (9, 10)}
    resp = score(side, partial, B=200, seed=1)
    _check("dropped utterances reported as missing", resp["missing"], [9, 10], fails)
    _check("scoring falls back to the utterances present", resp["n_utt"], side["N"] - 2, fails)
    resu = score(side, {**host, 999: list(host[0])}, B=200, seed=1)
    _check("index not in sidecar reported", resu["unknown"], [999], fails)

    trunc = parse_log("# u 7 ids: 1 2 3\n", side["n_out_frames"])
    _check("short line rejected", (len(trunc["passes"]), len(trunc["bad"])), (0, 1), fails)
    oob = parse_log("# u 7 ids: " + " ".join(["9999"] * side["n_out_frames"]) + "\n",
                    side["n_out_frames"])
    _check("out-of-range id rejected", len(oob["bad"]), 1, fails)
    nopass = parse_log("garbage\n# invoke 5 cycles = 1 ms\n", side["n_out_frames"])
    _check("no id lines -> no passes", len(nopass["passes"]), 0, fails)

    # ---------------- scoring
    print("\n  D. scoring the injected capture")
    res = score(side, pr["passes"][0], B=B, seed=1)
    by = {r["i"]: r for r in res["rows"]}
    _check("total disagreeing frames == injected", res["n_diff"], total_inj, fails)
    _check("u0 disagreeing frames", by[0]["n_diff"], inj["u0_blank_shift_frames"], fails)
    _check("u1 disagreeing frames", by[1]["n_diff"], 1, fails)
    _check("u2 disagreeing frames", by[U2]["n_diff"], inj["u2_blanked_frames"], fails)
    _check("u3 disagreeing frames", by[U3]["n_diff"], inj["u3_run_blanked_frames"], fails)
    _check("untouched utterances have 0 disagreements",
           sum(1 for r in res["rows"] if r["i"] not in (0, 1, U2, U3) and r["n_diff"]),
           0, fails)

    r0 = [r for r in res["dis_regions"] if r["u"] == 0]
    _check("u0 every region classified blank-shift",
           sorted({r["cls"] for r in r0}), ["blank-shift"], fails)
    r3 = [r for r in res["dis_regions"] if r["u"] == U3]
    _check("u3 region classified visible", sorted({r["cls"] for r in r3}), ["visible"], fails)
    _check("region frames partition the disagreeing frames",
           sum(r["n"] for r in res["dis_regions"]), res["n_diff"], fails)
    d0 = [f for f in res["dis_frames"] if f["u"] == 0]
    _check("u0 frames all classified blank->token",
           sorted({f["kind"] for f in d0}), ["blank->token"], fails)
    _check("u0 frames all collapse-neutral", all(f["neutral"] for f in d0), True, fails)
    _check("u0 device transcript == host transcript",
           by[0]["dev_text"] == by[0]["host_text"], True, fails)
    _check("u0 device-vs-host S/I/D all zero",
           (by[0]["xS"], by[0]["xI"], by[0]["xD"]), (0, 0, 0), fails)

    d1 = [f for f in res["dis_frames"] if f["u"] == 1][0]
    _check("u1 frame index", d1["f"], inj["u1_tight_margin_frame"], fails)
    if have_top2:
        _check("u1 device took the host runner-up", d1["is_top2"], True, fails)
    else:
        _check("u1 runner-up split reported as unavailable", d1["is_top2"], None, fails)
    _check("u1 margin recovered from sidecar", round(d1["margin"], 6),
           round(inj["u1_margin"], 6), fails)
    _check("u1 margin is the tightest in that utterance",
           d1["margin"] == min(f["margin"] for f in res["dis_frames"] if f["u"] == 1), True, fails)

    _check("u2 device transcript empty", by[U2]["dev_text"], "", fails)
    _check("u2 vs reference = all deletions",
           (by[U2]["dev_S"], by[U2]["dev_I"], by[U2]["dev_D"]), (0, 0, by[U2]["nw"]), fails)
    _check("u2 device WER vs reference == 100 %", by[U2]["dev_err"] == by[U2]["nw"], True, fails)
    _check("u2 paired difference == host words missed",
           by[U2]["dev_err"] - by[U2]["host_err"], by[U2]["nw"] - by[U2]["host_err"], fails)

    hc3 = ctc_greedy(host[U3], blank); dc3 = ctc_greedy(dev[U3], blank)
    _check("u3 removed exactly one collapsed token", len(hc3) - len(dc3), 1, fails)
    _check("u3 surviving ids == host ids with exactly one element deleted",
           sum(1 for j in range(len(hc3)) if hc3[:j] + hc3[j + 1:] == dc3) > 0, True, fails)
    _check("u3 deleted token id", inj["u3_token_removed"] in hc3, True, fails)
    _check("u3 scorer's device text == text rebuilt from the surviving ids",
           by[U3]["dev_text"], ids_to_text(dc3, vocab), fails)
    _check("u3 has at least one collapse-visible frame",
           any(not f["neutral"] for f in res["dis_frames"] if f["u"] == U3), True, fails)
    _check("u3 device-vs-host edit distance == fe.py DP",
           by[U3]["x_err"], int(_fe.wer(by[U3]["host_text"], by[U3]["dev_text"])[0]), fails)

    # aggregate identities
    tot_err = sum(r["dev_err"] for r in res["rows"])
    tot_w = sum(r["nw"] for r in res["rows"])
    _check("corpus WER == total errors / total ref words (not a mean of rates)",
           round(res["dev"]["wer"], 12), round(tot_err / tot_w, 12), fails)
    _check("device errors == host errors + injected damage",
           res["dev"]["err"] - res["host"]["err"],
           sum(by[i]["dev_err"] - by[i]["host_err"] for i in (0, 1, U2, U3)), fails)
    _check("paired worse/better/tied sums to N",
           res["paired"]["worse"] + res["paired"]["better"] + res["paired"]["tied"],
           res["n_utt"], fails)
    _check("host row text == sidecar host text (host side reproducible)",
           sum(1 for r in res["rows"] if r["host_text"] != r["sidecar_text"]), 0, fails)

    # ---- clean capture: device == host exactly
    print("\n  E. control -- a capture that agrees with the host everywhere")
    res0 = score(side, host, B=B, seed=1)
    _check("0 disagreeing frames", res0["n_diff"], 0, fails)
    _check("device-vs-host WER == 0", res0["xhost"]["err"], 0, fails)
    _check("device WER == host WER", res0["dev"]["wer"] == res0["host"]["wer"], True, fails)
    _check("paired difference == 0", res0["paired"]["obs"], 0.0, fails)
    _check("paired bootstrap CI == [0,0]",
           (round(res0["paired"]["lo"], 12), round(res0["paired"]["hi"], 12)), (0.0, 0.0), fails)

    # ---- power check: the margin test must fire when disagreements are seeded low
    print("\n  F. control -- margin test on disagreements deliberately seeded at low margins")
    devlow = {i: list(v) for i, v in host.items()}
    nlow = 0
    for i, u in utts.items():
        m = np.asarray(u["host_margin"], float)
        for f in np.argsort(m)[:3]:
            devlow[i][int(f)] = alt(u, int(f))
            nlow += 1
    reslow = score(side, devlow, B=B, seed=1)
    _check("seeded low-margin disagreement count", reslow["n_diff"], nlow, fails)
    print(f"      AUC P(disagree margin < random margin) = {reslow['auc']:.4f} "
          f"(permutation p = {reslow['auc_p']:.4f})")
    _check("AUC detects the low-margin seeding (>0.9)", reslow["auc"] > 0.9, True, fails)
    _check("AUC permutation p < 0.01", reslow["auc_p"] < 0.01, True, fails)
    devhi = {i: list(v) for i, v in host.items()}
    for i, u in utts.items():
        m = np.asarray(u["host_margin"], float)
        for f in np.argsort(-m)[:3]:
            devhi[i][int(f)] = alt(u, int(f))
    reshi = score(side, devhi, B=B, seed=1)
    print(f"      the same test on HIGH-margin seeding gives AUC {reshi['auc']:.4f} "
          f"(p = {reshi['auc_p']:.4f}) -- the instrument is not stuck on 'yes'")
    _check("AUC does not fire on high-margin seeding (<0.1)", reshi["auc"] < 0.1, True, fails)

    # ---- G. the one real device output in the tree, with a published answer
    trace = os.path.join(REPO, "board", "traces", "round19_relu4d_pass.log")
    print("\n  G. golden -- the real board output for utterance 0 "
          "(board/traces/round19_relu4d_pass.log)")
    if not os.path.exists(trace) or utts[0]["k"] != "1272-128104-0000":
        print("     skipped: trace missing or utterance 0 is not the canned utterance")
    else:
        txt = strip_ansi(open(trace, "rb").read().decode("utf-8", "replace"))
        vecs = [[int(x) for x in b.split()]
                for b in re.findall(r"#\s*ids:\s*([\d\s]+)", txt)]
        vecs = [v for v in vecs if len(v) == side["n_out_frames"]]
        _check("complete id vectors recovered from the raw ANSI-laden trace", len(vecs), 14, fails)
        _check("all 14 board runs identical (GATE4.md: one distinct output)",
               len({tuple(v) for v in vecs}), 1, fails)
        # replay it through the whole pipeline as if it were a corpus capture
        glog = "# u 0 ids: " + " ".join(str(x) for x in vecs[0]) + "\n"
        for i in sorted(host):
            if i:
                glog += f"# u {i} ids: " + " ".join(str(x) for x in host[i]) + "\n"
        gp = parse_log(glog, side["n_out_frames"])
        gres = score(side, gp["passes"][0], B=B, seed=1)
        g0 = {r["i"]: r for r in gres["rows"]}[0]
        _check("board vs host: 5 of 100 frames (GATE4.md '# mismatches 5 / 100')",
               g0["n_diff"], 5, fails)
        _check("host vs reference: 1 substitution on 17 words (5.88 %)",
               (g0["host_S"], g0["host_I"], g0["host_D"], g0["nw"]), (1, 0, 0, 17), fails)
        _check("device vs reference: 2 substitutions + 2 deletions on 17 words "
               "(GATE4.md 23.53 %)",
               (g0["dev_S"], g0["dev_I"], g0["dev_D"], g0["nw"]), (2, 0, 2, 17), fails)
        _check("device WER on that utterance == 23.53 %",
               round(100 * g0["dev_err"] / g0["nw"], 2), 23.53, fails)
        _check("host WER on that utterance == 5.88 %",
               round(100 * g0["host_err"] / g0["nw"], 2), 5.88, fails)
        lo, hi = wilson(g0["dev_err"], g0["nw"])
        _check("its Wilson interval == the [9.6 %, 47.3 %] quoted in the brief",
               (round(100 * lo, 1), round(100 * hi, 1)), (9.6, 47.3), fails)
        gr = [r for r in gres["dis_regions"] if r["u"] == 0]
        print(f"      regions: " + "; ".join(
            f"frames {r['frames'][0]}..{r['frames'][-1]} {r['host']}->{r['dev']} {r['cls']}"
            for r in gr))
        _check("3 of the 5 frames are one blank-placement shift CTC collapses away "
               "(the brief's split)",
               sum(r["n"] for r in gr if r["neutral"]), 3, fails)
        _check("the other 2 reach the transcript",
               sum(r["n"] for r in gr if not r["neutral"]), 2, fails)
        print("      NEGATIVE RESULT: the per-frame leave-one-out test calls only "
              f"{sum(1 for f in gres['dis_frames'] if f['u']==0 and f['neutral'])} of 5 "
              "neutral --")
        print("      frames 55/56/57 are a joint shift, neutral only as a group. "
              "Regions, not frames.")
        print(f"      device text : {g0['dev_text']}")
        print(f"      host text   : {g0['host_text']}")

    # ---- H. a hard-wrapped capture
    print("\n  H. a capture whose long id lines were wrapped by the terminal")
    wrapped = ""
    for i in sorted(host):
        toks = [str(x) for x in host[i]]
        wrapped += f"# u {i} ids: " + " ".join(toks[:37]) + "\r\n"
        wrapped += " ".join(toks[37:80]) + "\r\n" + " ".join(toks[80:]) + "\r\n"
    wp = parse_log(wrapped, side["n_out_frames"])
    _check("wrapped lines rejoined", len(wp["passes"][0]) if wp["passes"] else 0,
           side["N"], fails)
    _check("wrapped ids identical to the source",
           wp["passes"][0][3] == host[3] if wp["passes"] else False, True, fails)
    # the board's own colouring, as it appears in board/traces/round19_relu4d_pass.log:
    # an ESC[0m pair between every single number, and CRLF line endings.
    boardish = "".join("\x1b[0m# u %d ids:\x1b[0m" % i +
                       "".join("\x1b[0m %d\x1b[0m" % x for x in host[i]) + "\r\n"
                       for i in sorted(host)) + "\x1b[0m# corpus done\r\n"
    bp = parse_log(boardish, side["n_out_frames"])
    _check("ESC[0m between every number survives parsing",
           (len(bp["passes"]), len(bp["passes"][0]) if bp["passes"] else 0),
           (1, side["N"]), fails)
    _check("board-style ids identical to the source",
           bp["passes"][0][11] == host[11] if bp["passes"] else False, True, fails)

    junk = parse_log("# u 4 ids: 1024 1024 0m 1024\n", side["n_out_frames"])
    _check("a dropped ESC byte mid-line is refused, not silently truncated",
           (len(junk["passes"]), len(junk["bad"])), (0, 1), fails)

    print("\n" + "=" * 78)
    if fails:
        print(f"SELF-TEST FAILED: {len(fails)} checks -- {fails}")
    else:
        print("SELF-TEST PASSED: every injected error was reported exactly as injected.")
    print("=" * 78)
    return 1 if fails else 0


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="captured UART text")
    ap.add_argument("--sidecar", default="artifacts/corpus/corpus_ref.json")
    ap.add_argument("--blob", help="optional: the flashed blob, digest-checked "
                                   "against the sidecar")
    ap.add_argument("--pass", dest="which", type=int, default=0,
                    help="which pass to score when the capture repeats (default 0)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gap", type=int, default=0,
                    help="merge disagreeing frames separated by <= gap agreeing frames "
                         "into one region (default 0 = strictly adjacent)")
    ap.add_argument("--json", help="also dump the numbers here")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    side_path = a.sidecar if os.path.isabs(a.sidecar) else os.path.join(REPO, a.sidecar)
    if a.self_test:
        sys.exit(self_test(side_path, B=min(a.bootstrap, 2000)))
    if not a.log:
        ap.error("--log is required (or use --self-test)")

    side = load_sidecar(side_path)
    meta = {"log": a.log}
    if a.blob:
        import hashlib
        alg, want = side["digest"]
        h = hashlib.new(alg, open(a.blob, "rb").read()).hexdigest()
        meta["blob checked"] = f"{a.blob} {alg} {h[:16]} " + \
            ("MATCHES the sidecar" if h == want else "!! DOES NOT MATCH THE SIDECAR !!")

    raw = open(a.log, "rb").read().decode("utf-8", "replace")
    pr = parse_log(raw, side["n_out_frames"])
    if not pr["passes"]:
        raise SystemExit(f"{a.log}: no '# u <i> ids: ...' lines parsed "
                         f"({len(pr['bad'])} malformed)")
    ident, notes = compare_passes(pr["passes"])
    meta["passes in capture"] = (f"{len(pr['passes'])}, " +
                                 ("identical" if ident else "NOT IDENTICAL"))
    meta["scoring pass"] = a.which
    for j, n in enumerate(notes):
        meta[f"  repeat note {j}"] = n
    if pr["bad"]:
        meta["malformed id lines"] = f"{len(pr['bad'])} -- {pr['bad'][:3]}"
    if pr["cycles"]:
        c = np.array(pr["cycles"])
        meta["measured invoke cycles"] = (f"n={len(c)} min={c.min()} median={int(np.median(c))} "
                                          f"max={c.max()} (per-invoke counter reads, not derived)")
    if a.which >= len(pr["passes"]):
        raise SystemExit(f"--pass {a.which} but only {len(pr['passes'])} passes")

    res = score(side, pr["passes"][a.which], B=a.bootstrap, seed=a.seed, gap=a.gap)
    report(res, side, meta)
    if a.json:
        dump = {k: v for k, v in res.items() if k not in ("margins", "margin_deciles")}
        dump["margin_summary"] = dict(
            all_median=float(np.median(res["margins"]["all"])),
            dis_median=float(np.median(res["margins"]["dis"])) if len(res["margins"]["dis"]) else None,
            auc=res.get("auc"), auc_p=res.get("auc_p"))
        json.dump(dump, open(a.json, "w"), indent=1, default=float)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
