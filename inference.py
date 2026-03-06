import argparse
import time
import typing as tp

import numpy as np
import onnxruntime as ort
import PIL.Image, PIL.ImageDraw
PIL.Image.MAX_IMAGE_PIXELS = None

from src.tokenization import tokenize




#Helper functions for slicing images (for CHW dimension ordering)
def grid_for_patches(
    imageshape: tp.Tuple[int,int], 
    patchsize:  int, 
    slack:      int,
) -> np.ndarray:
    assert len(imageshape) == 2
    H,W       = imageshape[:2]
    stepsize  = patchsize - slack
    grid      = np.stack( 
        np.meshgrid( 
            np.minimum( np.arange(patchsize, H+stepsize, stepsize), H ), 
            np.minimum( np.arange(patchsize, W+stepsize, stepsize), W ),
            indexing = 'ij' 
        ), 
        axis = -1 
    )
    grid = np.concatenate([grid-patchsize, grid], axis=-1)
    grid = np.maximum(0, grid)
    return grid


def slice_into_patches_with_overlap(
    image:     np.ndarray, 
    patchsize: int=1024, 
    slack:     int=32
) -> tp.List[np.ndarray]:
    grid = grid_for_patches(image.shape[-2:], patchsize, slack)
    return slice_into_patches_from_grid(image, grid)


def slice_into_patches_from_grid(
    image: np.ndarray,
    grid:  np.ndarray,
) -> tp.List[np.ndarray]:
    assert grid.ndim in [2,3] and grid.shape[-1] == 4
    patches = [image[...,i0:i1, j0:j1] for i0,j0,i1,j1 in grid.reshape(-1, 4)]
    return patches


def stitch_overlapping_patches(
    patches:        tp.List[np.ndarray], 
    imageshape:     tp.Tuple[int,int], 
    slack:          int             = 32, 
    out:            np.ndarray|None = None,
) -> np.ndarray:
    patchsize = np.max(patches[0].shape[-2:])
    grid      = grid_for_patches(imageshape[-2:], patchsize, slack)
    halfslack = slack//2
    i0,i1     = (grid[grid.shape[0]-2,grid.shape[1]-2,(2,3)] - grid[-1,-1,(0,1)])//2
    d0 = np.stack( 
        np.meshgrid(
            [0]+[ halfslack]*(grid.shape[0]-2)+[i0]*(grid.shape[0]>1),
            [0]+[ halfslack]*(grid.shape[1]-2)+[i1]*(grid.shape[1]>1),
            indexing='ij' 
        ), 
        axis=-1
    )
    d1 = np.stack(
        np.meshgrid(     
            [-halfslack]*(grid.shape[0]-1)+[imageshape[-2]],      
            [-halfslack]*(grid.shape[1]-1)+[imageshape[-1]],
            indexing='ij'
        ), 
        axis=-1
    )
    d  = np.concatenate([d0,d1], axis=-1)
    if out is None:
        out = np.empty(patches[0].shape[:-2] + imageshape[-2:], dtype=patches[0].dtype)
    for patch,gi,di in zip(patches, d.reshape(-1,4), (grid+d).reshape(-1,4)):
        out[...,di[0]:di[2], di[1]:di[3]] = patch[...,gi[0]:gi[2], gi[1]:gi[3]]
    return out


# XYXY format
type Box = tp.Tuple[float, float, float, float]

def intersection(box0:Box, box1:Box) -> float:
    x0 = max(box0[0], box1[0])
    y0 = max(box0[1], box1[1])
    x1 = min(box0[2], box1[2])
    y1 = min(box0[3], box1[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)

def find_best_gridcell_for_box(grid:np.ndarray, box:Box) -> tp.Optional[int]:
    assert grid.ndim == 2 and grid.shape[1] == 4
    if len(grid) == 0:
        return None
    
    intersections = [intersection(cell, box) for cell in grid]
    best_index = int( np.argmax(intersections) )
    return best_index

def convert_box_to_relative_cxcywh(box:Box, relative_to:Box) -> Box:
    W = abs(relative_to[0] - relative_to[2])
    H = abs(relative_to[1] - relative_to[3])

    cx = (box[0] + box[2])/2 - relative_to[0]
    cy = (box[1] + box[3])/2 - relative_to[1]
    w  = abs(box[0] - box[2])
    h  = abs(box[1] - box[3])

    return (cx/W, cy/H, w/W, h/H) 


def main(args:argparse.Namespace):
    print('Loading onnx models...')
    session_image    = \
        ort.InferenceSession(f"{args.modeldir}/sam3_image_encoder.onnx")
    session_decode   = \
        ort.InferenceSession(f"{args.modeldir}/sam3_decoder_with_box_feats.onnx")

    print('Loading image...')
    image = PIL.Image.open(args.image).convert('RGB')
    print('OG image size:', image.size)
    og_size = image.size

    image = image.resize( [int(image.width*args.scale), int(image.height*args.scale)] )
    print('Scaled image size:', image.size)

    imagedata_chw = np.array(image).transpose(2,0,1)
    grid_yx = grid_for_patches(imagedata_chw.shape[-2:], patchsize=1008, slack=32)
    patches = slice_into_patches_from_grid(imagedata_chw, grid_yx)
    assert len(patches) > 0
    print(f'Number of patches: {len(patches)}')
    
    grid_xy = grid_yx.reshape(-1,4)[ :, (1,0,3,2) ]

    box = args.box
    box = scale_box(box, args.scale)
    best_grid_cell_index:int = find_best_gridcell_for_box(grid_xy, box) # type: ignore
    best_grid_cell:Box = grid_xy[best_grid_cell_index]
    relbox = convert_box_to_relative_cxcywh(box, best_grid_cell)

    encoder_outputs = []
    for i, patch in enumerate(patches):
        #print(patch.shape)
        output = session_image.run(None, {"image": patch})
        encoder_outputs.append(output)
    
    mask_patches = []
    for i, patch in enumerate(patches):
        print(f'Decoding {i} ...')

        box_feats = encoder_outputs[best_grid_cell_index][5]
        vision_pos_enc2, bb_fpn0, bb_fpn1, bb_fpn2 = encoder_outputs[i][2:6]

        output = session_decode.run(
            None,
            {
                "original_height": np.array(1008, dtype=np.int64),
                "original_width":  np.array(1008, dtype=np.int64),
                "backbone_fpn_0":  bb_fpn0,
                "backbone_fpn_1":  bb_fpn1,
                "backbone_fpn_2":  bb_fpn2,
                "vision_pos_enc_2":  vision_pos_enc2,
                "box_coords": np.array(relbox).reshape(1,1,4).astype('float32'),
                "box_labels": np.array([[1]], dtype=np.int64),
                "box_masks":  np.array([[False]], dtype=np.bool_),
                
                'box_feats': box_feats,
            },
        )

        boxes  = output[0]
        scores = output[1]
        masks  = output[2]
        mask_patches.append(masks.any(0)[0])
    
    full_mask = stitch_overlapping_patches(mask_patches, imagedata_chw.shape[-2:])
    outputimage = PIL.Image.fromarray(full_mask).resize(og_size, PIL.Image.Resampling.NEAREST)
    outputimage = draw_box(outputimage, args.box)
    outputimage.save('inference/full_mask.png')


def draw_box(image:PIL.Image.Image, box:Box):
    image = image.convert('RGB')
    draw = PIL.ImageDraw.Draw(image)
    draw.rectangle( list(box), outline=(255, 0, 0), width=5 )
    return image

def scale_box(box:Box, scale:float) -> Box:
    return tuple(i*scale for i in box)  # type: ignore [return-value]

def parse_box(x:str) -> Box:
    items = x.strip().split(',')
    assert len(items) == 4
    return tuple(float(i) for i in items) # type: ignore [return-value]



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to image')
    parser.add_argument('--modeldir', required=True, help='Path to directory with models')
    parser.add_argument(
        '--box', 
        required = True, 
        type = parse_box, 
        help = 'Box coordinates: x0,y0,x1,y1'
    )
    parser.add_argument('--scale', type=float, default=1.0, help='Scale factor to resize inputs')
    return parser



if __name__ == '__main__':
    #test()
    #test2()

    args = get_parser().parse_args()
    main(args)

    print('done')

