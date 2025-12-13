# sam3-onnx

[ONNX](https://onnx.ai/) export and inference for [SAM3](https://github.com/facebookresearch/sam3).

## Quick start

```sh
git clone https://github.com/wkentaro/sam3-onnx.git && cd sam3-onnx

uvx hf download wkentaro/sam3-onnx-models --local-dir models  # download pre-exported models

uv run infer_onnx.py --image images/bus.jpg --prompt person
uv run infer_onnx.py --image images/sofa.jpg --prompt sofa
uv run infer_onnx.py --image images/dog.jpg --prompt dog
```

<img src="assets/example_bus_person.jpg" width="25%" /> <img src="assets/example_sofa_sofa.jpg" width="30%" /> <img src="assets/example_dog_dog.jpg" width="36%" />

## Installation

```sh
make build  # install dependencies with uv
```

## Usage

**Inference with pytorch**

```sh
uv run infer_torch.py  # use official sam3 module
# uv run infer_torch.py --image <IMAGE_PATH> --prompt <PROMPT>
```

**Export to onnx**

```sh
uv run export_onnx.py  # creates models/*.onnx

# uvx hf upload wkentaro/sam3-onnx-models models/ --include '*.onnx*'
```

**Inference with onnx**

```sh
uv run infer_onnx.py  # use models/*.onnx
# uv run infer_onnx.py --image <IMAGE_PATH> --prompt <PROMPT>
```

## Pre-exported ONNX models

If don't want to export yourself, download them from the [Hugging Face repo](https://huggingface.co/wkentaro/sam3-onnx-models):

```
models
├── sam3_decoder.onnx
├── sam3_decoder.onnx.data
├── sam3_image_encoder.onnx
├── sam3_image_encoder.onnx.data
├── sam3_language_encoder.onnx
└── sam3_language_encoder.onnx.data
```

## Examples

```sh
uv run infer_onnx.py --image images/bus.jpg --prompt person
uv run infer_onnx.py --image images/bus.jpg --prompt window
```

<img src="assets/example_bus_person.jpg" width="30%" /> <img src="assets/example_bus_window.jpg" width="30%" />

```sh
uv run infer_onnx.py --image images/sofa.jpg --prompt person
uv run infer_onnx.py --image images/sofa.jpg --prompt sofa
```

<img src="assets/example_sofa_person.jpg" width="40%" /> <img src="assets/example_sofa_sofa.jpg" width="40%" />

```sh
uv run infer_onnx.py --image images/dog.jpg --prompt dog
uv run infer_onnx.py --image images/dog.jpg --prompt ground
```

<img src="assets/example_dog_dog.jpg" width="40%" /> <img src="assets/example_dog_ground.jpg" width="40%" />
