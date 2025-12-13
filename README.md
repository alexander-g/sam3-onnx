# sam3-onnx

## Setup

```sh
git clone https://github.com/wkentaro/sam3-onnx.git
cd sam3-onnx

make build  # git clone sam3:onnx and build uv
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
