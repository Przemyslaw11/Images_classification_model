import torch
import os
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from .AlexNetModified import AlexNetModified
from torch.nn import Softmax
from torchvision.transforms import Resize
from torchvision import transforms
transform_resize = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])
app = FastAPI()

model = AlexNetModified()
dir_path = os.path.dirname(os.path.realpath(__file__))
model.load_state_dict(torch.load(os.path.join(dir_path, "modelWeights64x64.pth"), map_location="cpu"))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def predict_image(img, model):
    result = Softmax(dim=1)(model(img))
    print(result)
    probability = torch.max(result, dim=1)[0]
    _, preds  = torch.max(result, dim=1)
    return {"class" : classes[preds[0].item()], "prob": probability}


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_content = None
    try:
        file_content = file.file.read()
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    image = transform_resize((Image.open(io.BytesIO(file_content)).convert("RGB")))
    input = image.reshape(1,3,64,64)

    result = predict_image(input, model)
    cla55 = result["class"]
    prob = result["prob"]
    return {"prediction": cla55, "probability": f"{prob[0]*100:.2f}%"}
