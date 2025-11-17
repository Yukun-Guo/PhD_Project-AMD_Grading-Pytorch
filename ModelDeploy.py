from NetModule import NetModule
import torch
import onnxruntime as ort
import numpy as np
import os
import glob
from Utils.deploy_onnxmodel import load_onnxmodel

# Save ONNX model
def save_onnxmodel(ckpt_path, out_path: str='./model', opset: int = 18, input_shape: tuple = (1, 1, 480, 288), enable_quantize=False, saved_model_filename="model.onnx"):
    if ckpt_path is None:
        raise ValueError("Checkpoint path cannot be None")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    print(f"Loading model from checkpoint: {ckpt_path}")
    model = NetModule.load_from_checkpoint(ckpt_path)
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    input_sample = (torch.randn(input_shape, device=device), torch.randn(input_shape, device=device), torch.randn(input_shape, device=device), torch.randn(input_shape, device=device))

    input_names = ['mnv', 'fluid', 'ga', 'drusen']
    output_names = ['output']
    # allow dynamic batch, height and width
    dynamic_axes = {
        'mnv': {0: 'batch_size', 2: 'height', 3: 'width'},
        'fluid': {0: 'batch_size', 2: 'height', 3: 'width'},
        'ga': {0: 'batch_size', 2: 'height', 3: 'width'},
        'drusen': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size'},
    }
    # Ensure output directory exists (fixes FileNotFoundError when external data is written)
    os.makedirs(out_path, exist_ok=True)

    model_file = os.path.join(out_path, saved_model_filename)

    torch.onnx.export(
        model,
        input_sample,
        model_file,
        export_params=True,
        opset_version=opset,
        external_data=False,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=True,
    )

# load model from checkpoint
# search for checkpoint files in all subdirectories of logs folder
logs_folder = "./logs"
checkpoint_files = []

# Search for checkpoint files in all subdirectories
for root, dirs, files in os.walk(logs_folder):
    for file in files:
        if file.endswith('.ckpt'):
            checkpoint_files.append(os.path.join(root, file))

if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint files found in {logs_folder} directory or its subdirectories. "
                          f"Please ensure you have trained a model and saved checkpoints.")

# Sort by modification time (most recent first)
checkpoint_files.sort(key=os.path.getmtime, reverse=True)
checkpoint_path = checkpoint_files[0]

print(f"Found {len(checkpoint_files)} checkpoint files:")
for i, ckpt in enumerate(checkpoint_files[:3]):  # Show first 3
    print(f"  {i+1}. {ckpt}")
if len(checkpoint_files) > 3:
    print(f"  ... and {len(checkpoint_files) - 3} more")
    
print(f"Using checkpoint: {checkpoint_path}")

save_onnxmodel(checkpoint_path, out_path="./deployed_model",opset=18, input_shape=(1,1,304,304), enable_quantize=False, saved_model_filename="model.onnx")

# # decrypt model and test
# model_buffer = load_onnxmodel("./deployed_model/model.onnx")
# ort_sess = ort.InferenceSession(model_buffer)
# y = ort_sess.run(None, {"mnv": np.random.rand(1, 1, 304, 304).astype(np.float32), "fluid": np.random.rand(1, 1, 304, 304).astype(np.float32), "ga": np.random.rand(1, 1, 304, 304).astype(np.float32), "drusen": np.random.rand(1, 1, 304, 304).astype(np.float32)})
# try:
#     if isinstance(y, list) and len(y) > 0:
#         output = y[0]
#         print("Test: onnx output shape: ", output.shape)  # type: ignore
#         print("Test: onnx shape check: ", np.array_equal(output.shape, np.array([1,1])))  # type: ignore
#     else:
#         print("Test: onnx output: ", type(y), len(y) if hasattr(y, '__len__') else 'No length')
# except Exception as e:
#     print(f"Test: onnx output test failed: {e}")
#     print(f"Test: onnx output type: {type(y)}")

# # check inference correctness, compare with pytorch model
# print("Checking inference correctness...")
# model = NetModule.load_from_checkpoint(checkpoint_path)
# model.eval()
# model.to('cpu')
# inp = (torch.randn((1, 1, 304, 304), dtype=torch.float32), torch.randn((1, 1, 304, 304), dtype=torch.float32), torch.randn((1, 1, 304, 304), dtype=torch.float32), torch.randn((1, 1, 304, 304), dtype=torch.float32))
# with torch.no_grad():
#     pt_out = model(inp)
# pt_out_np = pt_out.cpu().numpy().astype(np.float32)
# onnx_outs = ort_sess.run(None, {"mnv": inp[0].numpy(), "fluid": inp[1].numpy(), "ga": inp[2].numpy(), "drusen": inp[3].numpy()})
# onnx_out = onnx_outs[0]
# print("Max absolute difference between PyTorch and ONNX Runtime outputs: ", np.max(np.abs(pt_out_np - onnx_out)))
# print("Inference correctness check passed:", np.allclose(pt_out_np, onnx_out, atol=1e-5))