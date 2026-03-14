import base64
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def generate_gradcam_base64(model, image_tensor, target_class_idx):
    gradients = []
    activations = []

    target_layer = model.features[-1]

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor)
    score = output[:, target_class_idx]
    score.backward()

    fh.remove()
    bh.remove()

    grads = gradients[0]
    acts = activations[0]

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1).squeeze()
    cam = torch.relu(cam).detach().cpu().numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((224, 224))
    cam_arr = np.array(cam_img)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cam_arr, cmap="jet")
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
