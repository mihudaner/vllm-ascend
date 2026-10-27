# SPDX-License-Identifier: Apache-2.0
"""为每个已编译策略选择真实有效请求；PTO保留策略容量，Native只计算有效行。"""

import json
from pathlib import Path

from enumerate_policies import load_policy


def main():
    here = Path(__file__).resolve().parent
    cases = []
    for mode, package in (("spmd", "deepseek_v4_flash_csa_spmd_tnd"), ("mpmd", "deepseek_v4_flash_csa_tnd")):
        module = load_policy(here / "sources/vllm_ascend/ops/pypto" / package / "policy.py", mode + "_cases")
        rows = json.loads((here / f"{mode}_policies.json").read_text())["policies"]
        # 优先选择可由Native dummy请求承接padding的见证，不能按第一组代表误分类。
        reachable = {}
        for t in (1, 6, 16, 17, 23, 24, 31, 32, 33, 48, 49, 63, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193):
            for b in range(1, min(t, 49) + 1):
                for q in range(1, 7):
                    if b == 1 and t > q:
                        continue
                    for h in (8192, 15364, 32772, 196612, 262148):
                        for cols in (2048, 2049):
                            inputs = dict(
                                tokens=t, requests=b, max_query_tokens=q, max_seq_len=h, compressed_table_columns=cols
                            )
                            key = tuple(module.select_policy(**inputs).constexprs().values())
                            reachable.setdefault(key, inputs)
        for index, row in enumerate(rows):
            key = tuple(row["policy"].values())
            rep = dict(reachable.get(key, row["representative"]))
            t, b, q, h = (rep[k] for k in ("tokens", "requests", "max_query_tokens", "max_seq_len"))
            if t <= b * q:
                # 没有token padding时不能凭空追加失活请求槽；每个Native请求至少一行。
                active = b
                queries = [1] * b
                remaining = t - b
                for i in range(b):
                    extra = min(q - 1, remaining)
                    queries[i] += extra
                    remaining -= extra
                assert remaining == 0
            else:
                active = max(1, b - 1)
                queries = [q] * active
            histories = [h - n - (i % 4) for i, n in enumerate(queries)]
            valid = sum(queries)
            bounds = [0]
            for n in queries:
                bounds.append(bounds[-1] + n)
            bounds.extend([t] * (b - active))
            # 大历史真正需要超过2048页时，默认plan容量应走>2080的兜底区间。
            min_columns = (h + 127) // 128 + 1
            if rep["compressed_table_columns"] < min_columns:
                rep["compressed_table_columns"] = max(2081, min_columns)
            assert module.select_policy(**rep).constexprs() == row["policy"]
            assert len(bounds) == b + 1 and max(histories[i] + queries[i] for i in range(active)) == h
            cases.append(
                dict(
                    mode=mode,
                    index=index,
                    policy=row["policy"],
                    descriptor=rep,
                    native_entry_capacity=key in reachable,
                    scope="Native入口容量" if key in reachable else "非Native入口场景：PTO额外容量未分配给dummy请求",
                    queries=queries,
                    histories=histories,
                    valid_tokens=valid,
                    bounds=bounds,
                    seed=1024 + index,
                )
            )
        print(mode, len(rows), "非Native入口", sum(c["mode"] == mode and not c["native_entry_capacity"] for c in cases))
    (here / "function_cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
