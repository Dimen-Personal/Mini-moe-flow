import numpy as np

from router import topk_router
from dispatch import dispatch
from expert_mlp import (
    create_expert_weights,
    run_experts_mlp,
)


NUM_TOKENS = 4
NUM_EXPERTS = 4
TOP_K = 2

HIDDEN_SIZE = 3
INTERMEDIATE_SIZE = 2


def combine(
    expert_outputs,
    num_tokens,
    hidden_size
):
    Y = np.zeros(
        (
            num_tokens,
            hidden_size,
        ),
        dtype=np.float32,
    )


    for expert_id in range(
        len(expert_outputs)
    ):

        for item in expert_outputs[
            expert_id
        ]:

            token_id = item[
                "token_id"
            ]

            output = item[
                "output"
            ]

            weight = item[
                "weight"
            ]

            Y[token_id] += (
                weight * output
            )


    return Y





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
    # Top-K
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


    print(
        "\n===== Top-K Weights ====="
    )

    print(topk_weights)


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
    # Create Expert MLP Weights
    # =================================

    W1_all, W2_all = (
        create_expert_weights()
    )


    print(
        "\n===== Weight Shapes ====="
    )

    print(
        "W1:",
        W1_all.shape,
    )

    print(
        "W2:",
        W2_all.shape,
    )


    # =================================
    # Step 6
    # Expert MLP
    # =================================

    expert_outputs = (
        run_experts_mlp(
            expert_batches,
            W1_all,
            W2_all,
        )
    )


    # =================================
    # Step 7
    # Print Expert Internal Results
    # =================================

    print(
        "\n===== Expert MLP Outputs ====="
    )


    for expert_id in range(
        NUM_EXPERTS
    ):

        print(
            f"\nExpert {expert_id}:"
        )


        for item in expert_outputs[
            expert_id
        ]:

            print(
                f"  Token {item['token_id']}"
            )

            print(
                "    Linear1:",
                item[
                    "linear1_output"
                ],
            )

            print(
                "    SwiGLU:",
                item[
                    "swiglu_output"
                ],
            )

            print(
                "    Linear2:",
                item[
                    "output"
                ],
            )

            print(
                "    routing weight:",
                f"{item['weight']:.4f}",
            )


    # =================================
    # Step 8
    # Combine
    # =================================

    Y = combine(
        expert_outputs,
        NUM_TOKENS,
        HIDDEN_SIZE,
    )


    print(
        "\n===== Final MoE Output ====="
    )

    print(Y)


    print(
        "\n===== Output Shape ====="
    )

    print(Y.shape)
    
    
if __name__ == "__main__":
    main()
    
    