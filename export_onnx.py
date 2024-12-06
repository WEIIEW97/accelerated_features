import torch
from modules.xfeat import XFeatModel
import onnxruntime

from .inference import XFEAT_CKPT

if __name__ == "__main__":
    net = XFeatModel().eval()
    net.load_state_dict(torch.load(XFEAT_CKPT, map_location=torch.device('cuda:0')))

    # dummy input
    x = torch.randn()