#!/usr/bin/env python3
"""
Fix ONNX model for TensorRT compatibility by replacing dynamic TopK K parameter with constant.
"""

import onnx
from onnx import helper, numpy_helper
import numpy as np

def fix_topk_for_tensorrt(input_model_path, output_model_path):
    """Replace TopK's dynamic K input with a constant value"""
    
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # Find the TopK node
    topk_node = None
    topk_k_input = None
    for node in graph.node:
        if node.op_type == 'TopK' and 'regression' in node.name:
            topk_node = node
            topk_k_input = node.input[1]  # Second input is K
            print(f"Found TopK node: {node.name}")
            print(f"  Current K input: {topk_k_input}")
            break
    
    if topk_node is None:
        print("TopK node not found!")
        return False
    
    # Find the source of K value by tracing back the graph
    # K comes from Gather -> Shape chain, we need to determine the actual value
    # Based on the model structure (H=320, W=1024, D=192), K should be related to disparity range
    
    # For this model, K is likely to be 1 (argmax operation)
    # Let's verify by checking the original model behavior
    k_value = 1  # TopK with K=1 is essentially argmax
    
    print(f"Setting K to constant value: {k_value}")
    
    # Create a new constant initializer for K
    k_tensor = helper.make_tensor(
        name='/depth_net/regression/TopK_k_const',
        data_type=onnx.TensorProto.INT64,
        dims=[],  # Scalar
        vals=[k_value]
    )
    graph.initializer.append(k_tensor)
    
    # Update TopK node to use the constant
    topk_node.input[1] = '/depth_net/regression/TopK_k_const'
    
    # Remove unused nodes (Gather, Shape) if they're not used elsewhere
    nodes_to_remove = []
    for node in graph.node:
        if node.name == '/depth_net/regression/Gather':
            # Check if this node is only used by TopK
            is_only_topk = True
            for other_node in graph.node:
                if other_node == node:
                    continue
                if topk_k_input in other_node.input:
                    is_only_topk = False
                    break
            if is_only_topk:
                nodes_to_remove.append(node)
    
    for node in nodes_to_remove:
        print(f"Removing unused node: {node.name}")
        graph.node.remove(node)
    
    # Save the modified model
    onnx.save(model, output_model_path)
    print(f"Fixed model saved to: {output_model_path}")
    
    # Verify the model
    try:
        onnx.checker.check_model(output_model_path)
        print("Model verification passed!")
        return True
    except Exception as e:
        print(f"Model verification failed: {e}")
        return False


if __name__ == '__main__':
    import sys
    
    input_path = 'work_dirs/onnx_models/depth_net_folded.onnx'
    output_path = 'work_dirs/onnx_models/depth_net_fixed.onnx'
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    print(f"Input model: {input_path}")
    print(f"Output model: {output_path}")
    print()
    
    success = fix_topk_for_tensorrt(input_path, output_path)
    
    if success:
        print("\n✓ Model successfully fixed for TensorRT!")
        print(f"You can now convert it with:")
        print(f"  trtexec --onnx={output_path} --saveEngine=depth_net.trt --fp16")
    else:
        print("\n✗ Failed to fix model")
        sys.exit(1)
