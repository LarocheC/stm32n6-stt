#!/usr/bin/env python3
"""Score an artifacts/compile/<tag>/ build against the Gate 4 bar.

Reports, per build:
  epochs / SW epochs / hybrid epochs        network_generate_report.txt
  ACTIV->CONVACC epochs                     the Round 18b defect: an activation
                                            accelerator's output port driving a
                                            convolutional accelerator's data
                                            input port (dst CONVACC port 0)
                                            through the stream switch.  A
                                            correct build has ZERO.
  cpuRAM2 / npuRAM6 / octoFlash / PSRAM     report memory table
  estimated_tot_cycles                      summed over
                                            ll_atonn_rt_epoch_block_array,
                                            the compiler's own latency model
                                            (2.4 % over the measured 194.0 ms
                                            for the 1064-epoch build,
                                            board/GATE4.md Round 18c)
  unproven links                            any (dst unit, dst port, src unit)
                                            stream-switch combination that no
                                            epoch the board has ever executed
                                            uses.  ACTIV -> CONVACC port 0 was
                                            exactly such a combination.

    compile/score_build.py <tagdir> [<tagdir> ...] [--json] [--links]
"""
import json
import os
import re
import sys

# Stream-switch (dst_unit, dst_port, src_unit) combinations that silicon has
# executed: every epoch below the stall of artifacts/compile/g800_noauto (the
# board ran 2..352 of that build, board/traces/round18_noauto_discriminator.log)
# plus every epoch of artifacts/compile/g800_mem (the board ran all 1064,
# board/traces/round18_forcemem_pass.log).
PROVEN = {
    ("ACTIV", "0", "ARITH"), ("ACTIV", "0", "STRENG"),
    ("ARITH", "0", "CONVACC"), ("ARITH", "0", "POOL"), ("ARITH", "0", "STRENG"),
    ("ARITH", "0", "ARITH"), ("ARITH", "1", "STRENG"),
    ("CONVACC", "0", "STRENG"), ("CONVACC", "1", "STRENG"),
    ("CONVACC", "2", "CONVACC"), ("CONVACC", "2", "STRENG"),
    ("POOL", "0", "STRENG"),
    ("STRENG", "0", "STRENG"), ("STRENG", "0", "ARITH"),
    ("STRENG", "0", "ACTIV"), ("STRENG", "0", "CONVACC"),
    ("STRENG", "0", "POOL"),
}

LINK = re.compile(
    r'ATONN_DSTPORT\(STRSWITCH, 0, (\w+), (\d+), (\d+)\), '
    r'LL_Switch_Init_Source\(0\) = ATONN_SRCPORT\(STRSWITCH, 0, (\w+), (\d+), (\d+)\)')


def epoch_bodies(src):
    parts = re.split(r'(?m)^static void LL_ATON_(Start|End)_EpochBlock_(\d+)\(', src)
    out, i = {}, 1
    while i + 2 <= len(parts):
        if parts[i] == "Start":
            out[int(parts[i + 1])] = parts[i + 2]
        i += 3
    return out


def score(tagdir):
    out = os.path.join(tagdir, "st_ai_output")
    if not os.path.isdir(out):
        out = tagdir
    rep = os.path.join(out, "network_generate_report.txt")
    nc = os.path.join(out, "network.c")
    r = {"tag": os.path.basename(tagdir.rstrip("/"))}
    if not (os.path.exists(rep) and os.path.exists(nc)):
        r["error"] = "no report/network.c in %s" % out
        return r

    txt = open(rep, errors="replace").read()

    def grab(pat):
        m = re.search(pat, txt, re.M)
        return int(m.group(1)) if m else None
    r["epochs"] = grab(r'^Total number of epochs\s+(\d+)')
    r["sw"] = grab(r'^>> pure software \(SW\) epochs\s+(\d+)')
    r["hybrid"] = grab(r'^>> hybrid epochs[^\n]*?\s(\d+)\s*$')

    mem = {}
    m = re.search(r'^Memory usage information.*?\n(.*?)\n=====', txt, re.S | re.M)
    if m:
        for line in m.group(1).splitlines():
            mm = re.match(r'\s*(\w+)\s+\[0x[0-9a-fA-F]+ - 0x[0-9a-fA-F]+\]:\s+([\d\.]+)\s+(\S+)\s+/', line)
            if mm:
                mult = {"B": 1, "kB": 1000, "MB": 1000000,
                        "KiB": 1024, "MiB": 1048576}.get(mm.group(3), 1)
                mem[mm.group(1)] = int(float(mm.group(2)) * mult)
    r["cpuRAM2_B"] = mem.get("cpuRAM2")
    r["npuRAM6_B"] = mem.get("npuRAM6")
    r["octoFlash_B"] = mem.get("octoFlash")
    r["hyperRAM_B"] = mem.get("hyperRAM")

    src = open(nc, errors="replace").read()
    bodies = epoch_bodies(src)
    bad, unproven = [], {}
    for n, b in bodies.items():
        seen = set()
        for mm in LINK.finditer(b):
            du, _, dp, su, _, _ = mm.groups()
            seen.add((du, dp, su))
        if ("CONVACC", "0", "ACTIV") in seen:
            bad.append(n)
        for k in seen - PROVEN:
            unproven.setdefault(k, []).append(n)
    r["activ2conv_epochs"] = len(bad)
    r["activ2conv_list"] = sorted(bad)
    r["unproven_links"] = {"%s.%s <- %s" % k: len(v) for k, v in sorted(unproven.items())}

    m = re.search(r'll_atonn_rt_epoch_block_array\[\]\s*=\s*\{(.*?)\n  \};', src, re.S)
    arr = m.group(1) if m else ""
    r["array_entries"] = len(re.findall(r'\.epoch_num\s*=', arr))
    r["est_tot_cycles"] = sum(int(x) for x in re.findall(r'\.estimated_tot_cycles\s*=\s*(\d+)', arr))
    r["est_npu_cycles"] = sum(int(x) for x in re.findall(r'\.estimated_npu_cycles\s*=\s*(\d+)', arr))
    r["est_ms_at_600MHz"] = round(r["est_tot_cycles"] / 600000.0, 1)
    return r


if __name__ == "__main__":
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = [score(d) for d in dirs]
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
        sys.exit(0)
    hdr = ("%-22s %6s %4s %6s %9s %8s %8s %12s %8s" %
           ("tag", "epochs", "SW", "hybrid", "A->C", "cpuRAM2", "npuRAM6", "est_cycles", "est_ms"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print("%-22s %s" % (r["tag"], r["error"]))
            continue
        print("%-22s %6s %4s %6s %9s %8s %8s %12s %8s" % (
            r["tag"], r["epochs"], r["sw"], r["hybrid"], r["activ2conv_epochs"],
            r["cpuRAM2_B"], r["npuRAM6_B"], r["est_tot_cycles"], r["est_ms_at_600MHz"]))
        if r["unproven_links"]:
            print("%-22s   unproven stream-switch links: %s" % ("", r["unproven_links"]))
        if "--links" in sys.argv and r["activ2conv_list"]:
            print("%-22s   ACTIV->CONVACC at %s" % ("", r["activ2conv_list"]))
