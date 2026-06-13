import warnings

import cv2
import numpy as np
import torch

from models.layers.canny import Canny2D


def _raft_flow(slices, device):
    """Optical flow via torchvision RAFT-Small; fallback when CUDA OpenCV is unavailable."""
    try:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    except ImportError:
        warnings.warn('torchvision RAFT not available; returning zero optical flow.')
        N, H, W = len(slices) - 1, slices.shape[-2], slices.shape[-1]
        return torch.zeros(N, 2, H, W, dtype=slices.dtype, device=device)

    if not hasattr(_raft_flow, 'model'):
        weights = Raft_Small_Weights.DEFAULT
        _raft_flow.transforms = weights.transforms()
        _raft_flow.model = raft_small(weights=weights, progress=False).to(device).eval()

    model = _raft_flow.model
    transforms = _raft_flow.transforms

    # slices: (N, H, W) uint8 in [0, 255] → (N, 3, H, W) float for RAFT
    frames = slices.float().unsqueeze(1).expand(-1, 3, -1, -1).contiguous()

    flows = []
    with torch.no_grad():
        for i in range(len(slices) - 1):
            img1_t, img2_t = transforms(frames[i:i+1], frames[i+1:i+2])
            flow = model(img1_t.to(device), img2_t.to(device))[-1]  # (1, 2, H, W)
            flows.append(flow)

    return torch.cat(flows, dim=0).to(dtype=slices.dtype, device=slices.device)  # (N-1, 2, H, W)


def get_optical_flow(slices, device):
    height, width = slices.shape[-2], slices.shape[-1]
    if not hasattr(get_optical_flow, 'of'):
        if hasattr(cv2, 'cuda_NvidiaOpticalFlow_2_0'):
            of = cv2.cuda_NvidiaOpticalFlow_2_0.create((width, height), 5, 1, 1, False, False, False, device.index)
        elif hasattr(cv2, 'cuda_NvidiaOpticalFlow_1_0'):
            of = cv2.cuda_NvidiaOpticalFlow_1_0.create(width, height, 5, False, False, False, device.index)
            warnings.warn('use cuda_NvidiaOpticalFlow_1_0!')
        else:
            warnings.warn('opencv-python does not support CUDA; using RAFT optical flow.')
            of = None  # None → RAFT path
        get_optical_flow.of = of

    if get_optical_flow.of is None:
        return _raft_flow(slices, device)

    of = get_optical_flow.of
    slices_np = slices.type(torch.uint8).cpu().numpy()
    flows = []
    for i in range(1, len(slices)):
        flow = of.calc(slices_np[i - 1, :, :], slices_np[i, :, :], None)
        if hasattr(cv2, 'cuda_NvidiaOpticalFlow_2_0'):
            flow = of.convertToFloat(flow[0], None)
        else:
            flow = of.upSampler(flow[0], width, height, of.getGridSize(), None)
        flows.append(flow)
    flows = np.stack(flows, axis=0).transpose((0, 3, 1, 2))
    flows = torch.from_numpy(flows).type(slices.dtype).to(slices.device)
    return flows


def get_edge(slices, device, threshold=0.1):
    if not hasattr(get_edge, 'canny'):
        get_edge.canny = Canny2D(threshold=threshold).to(device)
    with torch.no_grad():
        edge = get_edge.canny(slices.unsqueeze(1))
    return edge