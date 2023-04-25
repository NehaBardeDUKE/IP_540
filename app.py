import copy
import torch
import torchvision
from torchvision import transforms
import gradio as gr
import os

def load_model(model_path):
    trained_model = torch.load(model_path, map_location=torch.device("cpu"))
    best_model_wts = copy.deepcopy(trained_model)
    model = torchvision.models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(best_model_wts)
    return model

model_path = 'model7.pth'
model = load_model(model_path)
class_names = ['edited','real']
model_path_sw = 'model8.pth'
model_sw = load_model(model_path_sw)
class_names_sw = ['deepfake','facetune']
def model_pred(img):
    transform=transforms.Compose([   
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor=transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred_probs = torch.softmax(model(image_tensor), dim=1)
        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    return pred_labels_and_probs

def model_pred_sw(img):
    transform=transforms.Compose([   
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor=transform(img).unsqueeze(0)
    model_sw.eval()
    with torch.no_grad():
        pred_probs = torch.softmax(model_sw(image_tensor), dim=1)
        pred_labels_and_probs = {class_names_sw[i]: float(pred_probs[0][i]) for i in range(len(class_names_sw))}
    return pred_labels_and_probs
	
demo = gr.Blocks()

with demo:
    img = gr.Image(type="pil")
    pred_1 = gr.Label(num_top_classes=2, label="Predictions")
    b1 = gr.Button("Is this image REAL or is it EDITED")
    pred_2 = gr.Label(num_top_classes=2, label="Predictions")
    b2 = gr.Button("What was it created using", visible=True)
    
    def recognize_pred_1(inputs):
        pred_dict = model_pred(inputs)
        pred_label = max(pred_dict, key=lambda k: pred_dict[k])
        print (pred_label)
        #pred_1.value = pred_dict
        if pred_label == 'edited':
            pred=model_pred_sw(inputs)
            return pred
        else:
            
            return {'real':1,'edited':0}
    

    b1.click(model_pred, inputs=img, outputs=pred_1)
    b2.click(recognize_pred_1, inputs=img, outputs=pred_2)
    example_paths = [["examples/" + example] for example in os.listdir("examples")]
    examples = gr.Examples(
        examples=example_paths,
        inputs=img)
demo.launch(server_name="0.0.0.0",server_port=7000)