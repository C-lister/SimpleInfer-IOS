import torch
from model_zoo.conv_tasnet_local import ConvTasNet
from torch.utils.mobile_optimizer import optimize_for_mobile

model = ConvTasNet(partition_layer='input', end_layer='prediction')
model.eval()
example = torch.rand(1, 1, 16000)
traced_script_module = torch.jit.trace(model, example, strict=False)
torchscript_model_optimized = optimize_for_mobile(traced_script_module)
torchscript_model_optimized._save_for_lite_interpreter("convtasnet.pt")