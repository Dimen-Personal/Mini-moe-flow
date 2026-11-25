import numpy as np
from expert_mlp import silu


def swiglu_batch(linear1_output):
    """
    Input:
        [M, 2 * intermediate_size]

    Output:
        [M, intermediate_size]
    """
    
    intermediate_size = linear1_output.shape[-1] // 2
    
    gate = linear1_output[: , : intermediate_size]
    up = linear1_output[: , intermediate_size :]
    
    return silu(gate) * up





def run_experts_gemm(
    expert_batches,
    W1_all,
    W2_all
):
    """
    对每个 Expert 的全部 Token
    一次性执行矩阵计算。

    X:
        [M, hidden_size]

    W1:
        [hidden_size, 2 * intermediate_size]

    W2:
        [intermediate_size, hidden_size]

    Output:
        [M, hidden_size]
    """
    expert_outputs = [
        [] for _ in range(len(expert_batches))
    ]
    
    
    # =================================
    # Step 1
    # 遍历所有 Experts
    # =================================

    for expert_id in range(
        len(expert_batches)
    ):

        batch = expert_batches[expert_id]

        # 如果当前 Expert 没有收到 Token
        if len(batch) == 0:
            continue

        W1 = W1_all[expert_id]

        W2 = W2_all[expert_id]
        
        
        
        # 利用 stack将所有的数组叠成矩阵
        batch_matrix = np.stack(
            [item["data"] for item in batch],     # 这里是先从batch中取出每个item
            axis = 0
        )
        
        
        
        linear1_output = batch_matrix @ W1
        swiglu_output = swiglu_batch(linear1_output)
        linear2_output = swiglu_output @ W2
        
        
        
        # 这边enumerate 会提供索引加上batch里面的item 元素
        for row_id, item in enumerate(batch):

            expert_outputs[expert_id].append(
                {
                    "token_id":
                        item["token_id"],

                    "output":
                        linear2_output[row_id],

                    "weight":
                        item["weight"],

                    "linear1_output":
                        linear1_output[row_id],

                    "swiglu_output":
                        swiglu_output[row_id],
                }
            )
            
    return expert_outputs

        