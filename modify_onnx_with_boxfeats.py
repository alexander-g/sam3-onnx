import argparse
import os

import onnx
from   onnx import helper, TensorProto
import onnxruntime as ort
import numpy as np

from src.tokenization import tokenize




input_node_to_replace = 'backbone_fpn_2'
new_input_name = "box_feats"
node_names_to_clone = [
    '/Gather_2',
    '/Shape',
    '/Slice',
    '/Concat',
    '/Reshape',
    '/Transpose',
]
node_to_attach_to = '/geometry_encoder/img_pre_norm/LayerNormalization'

def main(args:argparse.Namespace):
    decoder = load_decoder(args.decoder)
    decoder = add_box_feats_to_decoder(decoder)
    decoder = replace_language_input_with_constant(decoder, args.language_encoder)

    outputfilename = os.path.join(
        os.path.dirname(args.decoder), 
        'sam3_decoder_with_box_feats.onnx'
    )
    onnx.save(decoder, outputfilename, save_as_external_data=False)
    test_modified_model(outputfilename)



def load_decoder(path:str):
    model = onnx.load(path)
    graph = model.graph

    nodenames = [n.name for n in graph.node]
    for node in node_names_to_clone:
        assert node in nodenames, node
    inputnames = [i.name for i in graph.input]
    assert input_node_to_replace in inputnames

    return model

def replace_language_input_with_constant(
    decoder: onnx.ModelProto, 
    language_encoder_path: str
):
    session_language = ort.InferenceSession(language_encoder_path)

    text_prompt = "visual"
    tokens = tokenize(texts=[text_prompt], context_length=32)
    language_mask, language_features, _ = \
        session_language.run(None, {"tokens": tokens})
    
    lfeats = onnx.numpy_helper.from_array(language_features, name='language_features')
    lmask  = onnx.numpy_helper.from_array(language_mask, name='language_mask')

    decoder.graph.initializer.append(lfeats)
    decoder.graph.initializer.append(lmask)
    for inputnode in list(decoder.graph.input):
        if inputnode.name in ['language_features', 'language_mask']:
            decoder.graph.input.remove(inputnode)

    return decoder




def add_box_feats_to_decoder(decoder:onnx.ModelProto):
    nodemap = {n.name:n for n in decoder.graph.node}
    input_rename_map = {input_node_to_replace: new_input_name}

    new_input = helper.make_tensor_value_info(new_input_name, TensorProto.FLOAT, [1,256,72,72])
    decoder.graph.input.extend([new_input])

    # duplicate the part from backbone_fpn_2 to where it gets combined with boxes
    # (not quite as far actually)
    cloned_nodes = []
    for nodename in node_names_to_clone:
        old_node = nodemap[nodename]
        new_node = clone_node(old_node, input_rename_map, suffix='_bxft')
        cloned_nodes.append(new_node)
    
    old_connection_node_name = nodemap[node_names_to_clone[-1]].output[0]
    new_connection_node_name = cloned_nodes[-1].output[0]

    for node in decoder.graph.node:
        if node.name == node_to_attach_to:
            node.input[:] = [
                new_connection_node_name 
                    if i==old_connection_node_name else i for i in node.input
            ]
    decoder.graph.node.extend(cloned_nodes)
    cleanup_graph(decoder.graph)

    new_decoder = helper.make_model(decoder.graph, ir_version = decoder.ir_version)
    new_decoder.opset_import.clear()
    new_decoder.opset_import.extend([helper.make_opsetid("", 21)])

    return new_decoder


def find_node_producing(
    graph:onnx.GraphProto, 
    output_name:str
) -> onnx.NodeProto|onnx.ValueInfoProto|str|None:
    for n in graph.node:
        if output_name in n.output:
            return n
    if output_name.startswith('onnx::'):
        return output_name
    return None


def clone_node(node, input_rename_map={}, output_rename_map={}, suffix:str='_clone'):
    new_inputs = [
        input_rename_map.get(i, f'{i}{suffix}') 
            if not ('constant' in i.lower() or 'onnx::' in i.lower())else i
                for i in node.input or []
    ]
    new_node = helper.make_node(
        node.op_type,
        inputs  = new_inputs,
        outputs = [output_rename_map.get(o, f"{o}{suffix}") for o in node.output],
        name    = (node.name + suffix) if node.name else None,
        domain  = node.domain,
        **{ a.name:helper.get_attribute_value(a) for a in node.attribute },
    )
    return new_node


def cleanup_graph(graph):
    used = set()
    for n in graph.node:
        used.update(n.input)
        used.update(n.output)

    # keep only referenced initializers
    used_inits = [init for init in graph.initializer if init.name in used]
    graph.initializer.clear()
    graph.initializer.extend(used_inits)

    # keep only referenced graph inputs (or those that are outputs)
    out_names = {o.name for o in graph.output}
    new_inputs = [i for i in graph.input if (i.name in used) or (i.name in out_names)]
    graph.input.clear()
    graph.input.extend(new_inputs)


def test_modified_model(path:str):
    session = ort.InferenceSession(path)
    session.run(None, {
        "original_height": np.array(1008, dtype=np.int64),
        "original_width":  np.array(1008, dtype=np.int64),
        "backbone_fpn_0":  np.random.random([1,256,288,288]).astype('float32'),
        "backbone_fpn_1":  np.random.random([1,256,144,144]).astype('float32'),
        "backbone_fpn_2":  np.random.random([1,256,72,72]).astype('float32'),
        "vision_pos_enc_2":  np.random.random([1,256,72,72]).astype('float32'),
        "box_coords": np.array([0.5, 0.2, 0.05, 0.05]).reshape(1,1,4).astype('float32'),
        "box_labels": np.array([[1]], dtype=np.int64),
        "box_masks":  np.array([[False]], dtype=np.bool_),
        'box_feats':  np.random.random([1,256,72,72]).astype('float32'),
    })



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', required=True, help='Path to sam3_decoder.onnx')
    parser.add_argument(
        '--language-encoder', 
        required = True, 
        help     = 'Path to sam3_language_encoder.onnx'
    )

    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)

    print('done')


