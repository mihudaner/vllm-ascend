# SPDX-License-Identifier: Apache-2.0
"""总长度包含本轮query；覆盖整批短历史，而非借长请求抬高Host最大值。"""


def make_case(name, lengths, queries, *, tokens=None, requests=None, pages="original", reverse=False, lifecycle=False):
    assert len(lengths) == len(queries) and all(length >= q for length, q in zip(lengths, queries))
    actual = sum(queries)
    tokens = actual if tokens is None else tokens
    requests = len(queries) if requests is None else requests
    assert tokens >= actual and requests >= len(queries)
    assert tokens == actual or requests > len(queries)
    return dict(
        name=name,
        lengths=lengths,
        queries=queries,
        histories=[n - q for n, q in zip(lengths, queries)],
        tokens=tokens,
        requests=requests,
        pages=pages,
        reverse=reverse,
        lifecycle=lifecycle,
    )


def all_cases():
    cases = []
    # C4关闭、原页32、压缩页128、Top-512、Score768、排序1024、原8K门槛。
    for length in (
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        31,
        32,
        33,
        127,
        128,
        129,
        2047,
        2048,
        2049,
        3071,
        3072,
        3073,
        4095,
        4096,
        4097,
        8191,
        8192,
        8193,
    ):
        cases.append(make_case(f"b1_q1_l{length}", [length], [1], lifecycle=length in (1, 31, 127, 2047, 8191)))
    # 固定策略容量仍使用MAX_QUERY_TOKENS=6；真实每请求Q覆盖1..6。
    for query in range(1, 7):
        cases.append(make_case(f"b16_q{query}_l4096", [4096] * 16, [query] * 16, lifecycle=query == 6))
    for batch, length in (
        (4, 6),
        (16, 7),
        (16, 32),
        (16, 128),
        (16, 2048),
        (24, 6),
        (24, 4096),
        (24, 8191),
        (32, 2049),
        (40, 4097),
    ):
        cases.append(
            make_case(f"b{batch}_q6_l{length}", [length] * batch, [6] * batch, lifecycle=(batch, length) == (24, 4096))
        )
    cases += [
        make_case(
            "mixed_short_padded",
            [1, 3, 8, 128, 2048, 8191],
            [1, 2, 3, 4, 5, 6],
            tokens=32,
            requests=8,
            reverse=True,
            lifecycle=True,
        ),
        make_case(
            "mixed_short_long_control",
            [1, 3, 8, 128, 2048, 8193],
            [1, 2, 3, 4, 5, 6],
            tokens=32,
            requests=8,
            lifecycle=True,
        ),
        make_case(
            "b16_fragmented",
            [4096] * 16,
            [1, 2, 3, 4, 5, 6] * 2 + [1, 2, 3, 4],
            tokens=64,
            requests=17,
            pages="fragmented_history",
            lifecycle=True,
        ),
        make_case("b24_ascending", [8191] * 24, [6] * 24, pages="ascending_history"),
        make_case("one_active_large_padding", [4], [1], tokens=96, requests=2, lifecycle=True),
    ]
    assert len({c["name"] for c in cases}) == len(cases)
    for index, case in enumerate(cases):
        case["seed"] = 2026102900 + index
        # 固定页表容量允许后续跨C4/页/8K连续轨迹；不改变真实Q/L。
        case["table_history"] = 8256
    return cases
