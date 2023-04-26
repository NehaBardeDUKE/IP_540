import copy
import torch
import torchvision
from torchvision import transforms
import gradio as gr
import os

# load the model on cpu
def load_model(model_path):
    #import the trained model weights		
    trained_model = torch.load(model_path, map_location=torch.device("cpu"))
    best_model_wts = copy.deepcopy(trained_model)
    # instantiate the pretrained model	
    model = torchvision.models.resnet152(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(best_model_wts)
    return model

# get prediction for real vs edited
def model_pred(img):
    # preprocessing the image per the resnet 152 specs	
    transform=transforms.Compose([   
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor=transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
	# get the predicted probabilities for both classes
        pred_probs = torch.softmax(model(image_tensor), dim=1)
        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    return pred_labels_and_probs
# get prediction for deepfake vs facetune
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
	#evaluate using the 2nd model 	
        pred_probs = torch.softmax(model_sw(image_tensor), dim=1)
        pred_labels_and_probs = {class_names_sw[i]: float(pred_probs[0][i]) for i in range(len(class_names_sw))}
    return pred_labels_and_probs
	

def launch():
	#load the first model weights and set class names
	model_path = 'model7.pth'
	model = load_model(model_path)
	class_names = ['edited','real']
	#load the second model weights and set classnames
	model_path_sw = 'model8.pth'
	model_sw = load_model(model_path_sw)
	class_names_sw = ['deepfake','facetune']
	#begin UI creation
	#define block
	demo = gr.Blocks()
	with demo:
		#Define the required input and output channels
		#input image box
		img = gr.Image(type="pil")
		#output label to print the predictions
		pred_1 = gr.Label(num_top_classes=2, label="Predictions")
		#button to trigger the prediction
		b1 = gr.Button("Is this image REAL or is it EDITED")
		#output label to prin the 2nd model prediction
		pred_2 = gr.Label(num_top_classes=2, label="Predictions")
		#button to trigger the 2nd model prediction
		b2 = gr.Button("What was it created using", visible=True)
		
		def recognize_pred_1(inputs):
			# repeat the first model prediction and get the max probability label for the same image
			pred_dict = model_pred(inputs)
			pred_label = max(pred_dict, key=lambda k: pred_dict[k])
			print (pred_label)
			#Perform the 2nd model prediction only if label is "edited"
			if pred_label == 'edited':
				pred=model_pred_sw(inputs)
				return pred
			else:
				#if the label is "real" then the 2nd model should not be sent any data and a default output should be displayed
				return {'real':1,'edited':0}
		
		#define the triggered functions on button clicks
		b1.click(model_pred, inputs=img, outputs=pred_1)
		b2.click(recognize_pred_1, inputs=img, outputs=pred_2)
		#provide a list of example images that the user can choose
		example_paths = [["examples/" + example] for example in os.listdir("examples")]
		examples = gr.Examples(
			examples=example_paths,
			inputs=img)
	#launch the demo on the required port	
	demo.launch(server_name="0.0.0.0",server_port=7000)
