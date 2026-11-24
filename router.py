import numpy as np

NUM_TOKENS = 4
NUM_EXPERTS = 4
TOP_K = 2



def topk_router(scores, top_k):
    """
    scores shape:
        [num_tokens, num_experts]

    return:
        topk_idx:
            [num_tokens, top_k]

        topk_weights:
            [num_tokens, top_k]
    """
    
    topk_idx = np.argsort(scores, axis = 1)[:, -top_k:]  # 后面的表示数据切片，对象是所有行，列的话从倒数第top_k列开始到最后
    topk_idx = np.flip(topk_idx, axis = 1)    # 倒转顺序，最高的排在最前面
    
    
    
    topk_scores = np.take_along_axis(scores, topk_idx, axis = 1)   # 按照对应的下标在scores中去取数值
    
    
    
    # 计算routing weights, 这一版我们暂时先除以总和
    topk_weights = topk_scores / topk_scores.sum(axis = 1, keepdims = True)     # keepdims 保证结果的维度依旧是二维   
    
    
    return(topk_idx, topk_weights)




def main():
    
    scores = np.array(
        [
            [0.8, 0.1, 0.7, 0.2],
            [0.2, 0.9, 0.6, 0.1],
            [0.7, 0.2, 0.1, 0.8],
            [0.1, 0.6, 0.3, 0.9],
        ],
        dtype=np.float32,
    )
    
    
    
    print(
        "===== Router Scores ====="
    )

    print(scores)


    topk_idx, topk_weights = (
        topk_router(
            scores,
            TOP_K,
        )
    )


    print(
        "\n===== Top-K Expert Indices ====="
    )

    print(topk_idx)


    print(
        "\n===== Top-K Routing Weights ====="
    )

    print(topk_weights)


    print(
        "\n===== Routing Result ====="
    )
    
    
        
        
    for token_id in range(NUM_TOKENS):

        print(
            f"Token {token_id}:"
        )

        for k in range(TOP_K):

            expert_id = (
                topk_idx[token_id, k]
            )

            weight = (
                topk_weights[token_id, k]
            )

            print(
                f"  -> Expert {expert_id}, "
                f"weight = {weight:.4f}"
            )
    
    

if __name__ == "__main__":
    main()