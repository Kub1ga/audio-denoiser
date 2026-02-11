import torch
from denoiser import pretrained

model = pretrained.dns64(resample=1).cpu()
model.eval()

dummy = torch.randn(1, 1, 16000)

torch.onnx.export(
    model,
    dummy,
    "dns64_resample1.onnx",
    opset_version=18,
    input_names=["audio_in"],
    output_names=["audio_out"],
)

print("✅ ONNX berhasil (resample=1)")
