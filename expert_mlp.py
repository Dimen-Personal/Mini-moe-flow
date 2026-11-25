import numpy as np



HIDDEN_SIZE = 3
INTERMEDIATE_SIZE = 2
NUM_EXPERTS = 4


def silu(x):
    """
    SiLU(x) = x * sigmoid(x)
    """
    
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    
    return x * sigmoid




def swiglu(linear1_output):
    """
    Linear1 output:

        shape = [2 * intermediate_size]

    Split:

        gate
        up

    SwiGLU:

        SiLU(gate) * up
    """


    intermediate_size = linear1_output.shape[-1] // 2
    
    gate = linear1_output[..., :intermediate_size]    # 后面处理多个token组成的batch的时候可以保证整个batch都被进行处理
    up = linear1_output[... ,intermediate_size:]
    
    output = silu(gate) * up
    
    
    return output




def create_expert_weights():
    # 每一个expert 都存在一套放大的weight和一套缩回原来hidden state尺寸的weight
    
    W1_all = np.array(
    [
        # =========================
        # Expert 0
        # =========================
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, -1.0],
        ],

        # =========================
        # Expert 1
        # =========================
        [
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, -1.0, 1.0, 1.0],
        ],

        # =========================
        # Expert 2
        # =========================
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 1.0],
        ],

        # =========================
        # Expert 3
        # =========================
        [
            [-1.0, 1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ],
    ],
        
    dtype=np.float32,
    )


    W2_all = np.array(
        [
            # Expert 0
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],

            # Expert 1
            [
                [2.0, 0.0, 1.0],
                [0.0, 1.0, 2.0],
            ],

            # Expert 2
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 2.0],
            ],

            # Expert 3
            [
                [-1.0, 1.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )


    return W1_all, W2_all




def run_experts_mlp(
    expert_batches, 
    W1_all,
    W2_all
):
    """
    每个 Expert：

        token
          ↓
        Linear 1
          ↓
        SwiGLU
          ↓
        Linear 2
          ↓
        output
    """
    
    expert_outputs = [
        [] for _ in range(len(expert_batches))
    ]
    
    
    
    for expert_id in range(len(expert_batches)):
        batch = expert_batches[expert_id]
        W1 = W1_all[expert_id]
        W2 = W2_all[expert_id]
        
        
        for item in batch:
            token_id = item["token_id"]
            token = item["data"]
            routing_weight = item["weight"]
            
            
            linear1_output = token @ W1
            swiglu_output = swiglu(linear1_output)
            linear2_output = swiglu_output @ W2
            
            
            expert_outputs[expert_id].append(
                {
                    "token_id": token_id,

                    "output": linear2_output,

                    "weight": routing_weight,

                    "linear1_output": linear1_output,

                    "swiglu_output": swiglu_output
                }
            )
            
    return expert_outputs







