import numpy as np
from router import topk_router


NUM_TOKENS = 4
NUM_EXPERTS = 4
TOP_K = 2
HIDDEN_SIZE = 3



def dispatch(X, topk_idx, topk_weights, num_experts):
    
    """
    X:
        [num_tokens, hidden_size]

    topk_idx:
        [num_tokens, top_k]

    topk_weights:
        [num_tokens, top_k]

    return:
        expert_batches

    expert_batches[e] 里面保存：
        这个 expert 收到的所有 token
    """
    
    expert_batches = []
    
    for _ in range(num_experts):
        expert_batches.append([])




    # Go through every token
    num_tokens = X.shape[0]
    top_k = topk_idx.shape[1]
    
    
    for token_id in range(num_tokens):
        # 当前 Token 的 hidden vector
        token = X[token_id]
        
        
        # 一个 token 会去 top_k 个 expert
        for k in range(top_k):
            expert_id = topk_idx[token_id, k]
            weight = topk_weights[token_id, k]
            
            
            expert_batches[expert_id].append(
                {
                    "token_id": token_id,
                    "data": token.copy(),
                    "weight": weight
                }
            )
            
            
            
    return expert_batches




def main():
    # ==================================
    # Step 1
    # Input Tokens
    # ==================================

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
        "===== Input Tokens X ====="
    )

    print(X)


    # ==================================
    # Step 2
    # Router Scores
    # ==================================

    scores = np.array(
        [
            [0.8, 0.1, 0.7, 0.2],
            [0.2, 0.9, 0.6, 0.1],
            [0.7, 0.2, 0.1, 0.8],
            [0.1, 0.6, 0.3, 0.9],
        ],
        dtype=np.float32,
    )


    # ==================================
    # Step 3
    # Top-K Routing
    # ==================================

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


    # ==================================
    # Step 4
    # Dispatch
    # ==================================

    expert_batches = dispatch(
        X,
        topk_idx,
        topk_weights,
        NUM_EXPERTS,
    )


    # ==================================
    # Step 5
    # Print Dispatch Result
    # ==================================

    print(
        "\n===== Dispatch Result ====="
    )


    for expert_id in range(NUM_EXPERTS):

        print(
            f"\nExpert {expert_id}:"
        )

        batch = (
            expert_batches[
                expert_id
            ]
        )


        if len(batch) == 0:

            print(
                "  No tokens"
            )

            continue


        for item in batch:

            print(
                f"  Token {item['token_id']}"
                f" -> data={item['data']}, "
                f"weight={item['weight']:.4f}"
            )    
            
if __name__ == "__main__":
    main()
    