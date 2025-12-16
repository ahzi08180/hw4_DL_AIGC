import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

DEVICE = "cpu"

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(checkpoint["classes"])
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint["classes"]


def predict_image(model, class_names, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_idx = probs.max(1)

    return class_names[top_idx.item()], top_prob.item()
