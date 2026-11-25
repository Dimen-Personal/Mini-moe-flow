import numpy as np

from router import topk_router
from dispatch import dispatch
from expert import create_expert_weights, run_experts



NUM_TOKENS = 4
NUM_EXPERTS = 4
TOP_K = 2
OUTPUT_SIZE = 2



def combine(
    expert_outputs,
    num_tokens,
    output_size
):
    """
    expert_outputs:
        按 Expert 分组的计算结果

    return:
        Y shape = [num_tokens, output_size]
    """
    
    Y = np.zeros(
        (num_tokens, output_size),
        dtype = np.float32
    )
    
    
    
    for expert_id in range(len(expert_outputs)):
        outputs = expert_outputs[expert_id]
        
        
        for item in outputs:
            token_id = item["token_id"]
            output = item["output"]
            weight = item["weight"]
        
        
        
            Y[token_id] += weight * output
        
    
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

    print("===== Top-K Indices =====")
    print(topk_idx)

    print("\n===== Top-K Weights =====")
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
    # Expert Computation
    # =================================

    expert_weights = create_expert_weights()

    expert_outputs = run_experts(
        expert_batches,
        expert_weights,
    )


    # =================================
    # Step 6
    # Combine
    # =================================

    Y = combine(
        expert_outputs,
        NUM_TOKENS,
        OUTPUT_SIZE,
    )

    print("\n===== Final MoE Output =====")
    print(Y)

    print("\n===== Output Shape =====")
    print(Y.shape)


    # =================================
    # Step 7
    # Reference Implementation
    # =================================

    expected = np.zeros_like(Y)

    for token_id in range(NUM_TOKENS):

        for k in range(TOP_K):

            expert_id = topk_idx[token_id, k]

            weight = topk_weights[token_id, k]

            W = expert_weights[expert_id]

            expected[token_id] += (
                weight * (X[token_id] @ W)
            )


    print("\n===== Reference Output =====")
    print(expected)

    print("\n===== Correctness Check =====")
    print(np.allclose(Y, expected))


if __name__ == "__main__":
    main()