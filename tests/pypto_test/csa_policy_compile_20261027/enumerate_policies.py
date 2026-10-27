# SPDX-License-Identifier: Apache-2.0
"""对冻结版本的policy谓词分区枚举；不加载PyPTO、不分配设备。"""

import argparse
import ast
import json
import sys
import types
from pathlib import Path


def load_policy(path, name):
    tree = ast.parse(path.read_text())
    # 唯一相对导入是异常格式化helper；合法枚举不会进入该分支。
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def enumerate_policy(module):
    rows = {}
    # T覆盖全部常量阈值与ceil(T/48)变化。T>193后只剩短档分组关系的真假，
    # 两个结果已经由T193/B24和T193/B49分别覆盖。
    tokens = (1, 6, 16, 17, 23, 24, 31, 32, 33, 48, 49, 63, 64, 65, 96, 97, 128, 129, 160, 161, 192, 193)
    # B>48时short_half_score_ub恒假；单请求、B>=24与其它T/B关系的真假
    # 均已在1..49内取到。query>=97时half-score谓词恒假，其余与97相同。
    for t in tokens:
        for b in range(1, 50):
            for q in range(1, 98):
                # cache_rows的五个等价区间；不按性能测试的8K/128K档位枚举。
                for history in (8192, 15364, 32772, 196612, 262148):
                    # None、<=2048及>2080走DEFAULT，2049..2080走EXTENDED。
                    for columns in (2048, 2049):
                        inputs = dict(
                            tokens=t,
                            requests=b,
                            max_query_tokens=q,
                            max_seq_len=history,
                            compressed_table_columns=columns,
                        )
                        policy = module.select_policy(**inputs).constexprs()
                        key = tuple(policy.values())
                        rows.setdefault(key, dict(policy=policy, representative=inputs))
    return [rows[key] for key in sorted(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    for mode, package in (("spmd", "deepseek_v4_flash_csa_spmd_tnd"), ("mpmd", "deepseek_v4_flash_csa_tnd")):
        source = args.repo / "vllm_ascend/ops/pypto" / package / "policy.py"
        rows = enumerate_policy(load_policy(source, mode + "_audit"))
        result = dict(method="frozen-selector predicate partitions; query 1..96 plus >=97 tail", policies=rows)
        (args.output / f"{mode}_policies.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(mode, len(rows), flush=True)


if __name__ == "__main__":
    main()
