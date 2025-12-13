# sam3-onnx

## Installation

```sh
git clone https://github.com/wkentaro/sam3-onnx.git
cd sam3-onnx

make build  # install deps with uv
```

## Usage

### Inference with pytorch

```sh
uv run infer_torch.py  # use official sam3 module
# uv run infer_torch.py --image <IMAGE_PATH> --prompt <PROMPT>
```

### Export to onnx

```sh
uv run export_onnx.py  # creates models/*.onnx
```

### Inference with onnx

```sh
uv run infer_onnx.py  # use models/*.onnx
# uv run infer_onnx.py --image <IMAGE_PATH> --prompt <PROMPT>
```
