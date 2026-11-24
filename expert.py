import numpy as np

from router import topk_router
from dispatch import dispatch


NUM_TOKENS = 4
NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 3
OUTPUT_SIZE = 2


def create_expert_weights():
    """
    每个 Expert 都有自己独立的 weight。

    shape:
        [num_experts, hidden_size, output_size]
    """

    expert_weights = np.array(
        [
            # Expert 0
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],

            # Expert 1
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [1.0, -1.0],
            ],

            # Expert 2
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 2.0],
            ],

            # Expert 3
            [
                [-1.0, 1.0],
                [2.0, 1.0],
                [1.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )

    return expert_weights


def run_experts(
    expert_batches,
    expert_weights,
):
    """
    对每个 Expert 的 token batch 执行 Linear：

        Y = X @ W_e

    return:
        expert_outputs

    expert_outputs[e] 中保存：
        token_id
        output
        weight
    """

    expert_outputs = [
        []
        for _ in range(len(expert_batches))
    ]


    # =================================
    # 遍历每个 Expert
    # =================================

    for expert_id in range(
        len(expert_batches)
    ):

        batch = expert_batches[
            expert_id
        ]

        W = expert_weights[
            expert_id
        ]


        # 如果这个 Expert 没有 token
        if len(batch) == 0:
            continue


        # =================================
        # 逐个 token 计算
        # =================================

        for item in batch:

            token = item["data"]

            token_id = item[
                "token_id"
            ]

            routing_weight = item[
                "weight"
            ]


            # -----------------------------
            # Expert computation
            #
            # [hidden_size]
            #      @
            # [hidden_size, output_size]
            #      =
            # [output_size]
            # -----------------------------

            output = token @ W


            expert_outputs[
                expert_id
            ].append(
                {
                    "token_id": token_id,
                    "output": output,
                    "weight": routing_weight,
                }
            )


    return expert_outputs


def main():

    # =================================
    # Step 1
    # Input Tokens
    # =================================

    X = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [2, 1, 1],
        ],
        dtype=np.float32,
    )


    print(
        "===== Input Tokens ====="
    )

    print(X)


    # =================================
    # Step 2
    # Router Scores
    # =================================

    scores = np.array(
        [
            [0.8, 0.1, 0.7, 0.2],
            [0.2, 0.9, 0.6, 0.1],
            [0.7, 0.2, 0.1, 0.8],
            [0.1, 0.6, 0.3, 0.9],
        ],
        dtype=np.float32,
    )


    # =================================
    # Step 3
    # Top-K Router
    # =================================

    topk_idx, topk_weights = (
        topk_router(
            scores,
            TOP_K,
        )
    )


    print(
        "\n===== Top-K Indices ====="
    )

    print(topk_idx)


    # =================================
    # Step 4
    # Dispatch
    # =================================

    expert_batches = dispatch(
        X,
        topk_idx,
        topk_weights,
        NUM_EXPERTS,
    )


    # =================================
    # Step 5
    # Expert Weights
    # =================================

    expert_weights = (
        create_expert_weights()
    )


    print(
        "\n===== Expert Weights ====="
    )

    for expert_id in range(
        NUM_EXPERTS
    ):

        print(
            f"\nExpert {expert_id} W:"
        )

        print(
            expert_weights[
                expert_id
            ]
        )


    # =================================
    # Step 6
    # Run Experts
    # =================================

    expert_outputs = (
        run_experts(
            expert_batches,
            expert_weights,
        )
    )


    # =================================
    # Step 7
    # Print Results
    # =================================

    print(
        "\n===== Expert Outputs ====="
    )


    for expert_id in range(
        NUM_EXPERTS
    ):

        print(
            f"\nExpert {expert_id}:"
        )

        outputs = (
            expert_outputs[
                expert_id
            ]
        )


        if len(outputs) == 0:

            print(
                "  No outputs"
            )

            continue


        for item in outputs:

            print(
                f"  Token {item['token_id']}"
                f" -> output={item['output']}, "
                f"weight={item['weight']:.4f}"
            )


if __name__ == "__main__":
    main()