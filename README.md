# sam3-onnx

## Setup

```sh
# clone this repository
git clone https://github.com/wkentaro/sam3-onnx.git
cd sam3-onnx

# clone submodule (sam3:onnx)
make build
```

## Inference with PyTorch

```sh
uv run infer_torch.py
# uv run infer_torch.py  --image <IMAGE_PATH> --prompt <PROMPT>
```

## Export to ONNX

```sh
uv run export_onnx.py
```

## Inference with ONNX Runtime

```sh
uv run infer_onnx.py
# uv run infer_onnx.py --image <IMAGE_PATH> --prompt <PROMPT>
```
