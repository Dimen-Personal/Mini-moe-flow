import numpy as np

from router import topk_router
from dispatch import dispatch
from combine import combine

from expert_mlp import (
    create_expert_weights,
    run_experts_mlp,
)

from expert_gemm import (
    run_experts_gemm,
)


NUM_TOKENS = 4
NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 3



def main():
    np.set_printoptions(
        precision = 6,     # 使得输出数组时的数字保留大约6位，但是实际数组不发生变化
        suppress = True    # 尽量不使用科学计数法进行显示
    )
    
    
    X = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [2, 1, 1],
        ],
        dtype=np.float32,
    )


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
    # Router + Top-K
    # =================================

    topk_idx, topk_weights = topk_router(
        scores,
        TOP_K,
    )


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


    print("===== Tokens Per Expert =====")

    for expert_id, batch in enumerate(
        expert_batches
    ):

        print(
            f"Expert {expert_id}: "
            f"{len(batch)} tokens"
        )


    # =================================
    # Step 5
    # Expert Weights
    # =================================

    W1_all, W2_all = (
        create_expert_weights()
    )


    # =================================
    # Step 6
    # Reference: Per-token GEMV
    # =================================

    reference_outputs = run_experts_mlp(
        expert_batches,
        W1_all,
        W2_all,
    )


    reference_Y = combine(
        reference_outputs,
        NUM_TOKENS,
        HIDDEN_SIZE,
    )


    # =================================
    # Step 7
    # Optimized: Per-expert GEMM
    # =================================

    gemm_outputs = run_experts_gemm(
        expert_batches,
        W1_all,
        W2_all,
    )


    gemm_Y = combine(
        gemm_outputs,
        NUM_TOKENS,
        HIDDEN_SIZE,
    )


    # =================================
    # Step 8
    # Print Results
    # =================================

    print("\n===== Per-token GEMV =====")

    print(reference_Y)


    print("\n===== Per-expert GEMM =====")

    print(gemm_Y)


    # =================================
    # Step 9
    # Correctness Check
    # =================================

    correct = np.allclose(
        reference_Y,
        gemm_Y,
        rtol=1e-5,
        atol=1e-6,
    )


    max_error = np.max(
        np.abs(reference_Y - gemm_Y)
    )


    print("\n===== Correctness =====")

    print("Correct:", correct)

    print("Max Error:", max_error)


    assert correct, "GEMM result mismatch"


if __name__ == "__main__":
    main()