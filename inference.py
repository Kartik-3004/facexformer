"""
Face Parsing - Task 0
Face Landmarks Detection - Task 1
Face Headpose Estimation - Task 2
Face Attributes Recognition - Task 3
Face Age/Gender/Race Estimation - Task 4
Face Landmarks Visibility Prediction - Task 5
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode
import argparse
from math import cos, sin
from PIL import Image
from network import FaceXFormer
from facenet_pytorch import MTCNN



def visualize_mask(image_tensor, mask):
    image = image_tensor.numpy().transpose(1, 2, 0) * 255 
    image = image.astype(np.uint8)
    
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mapping = np.array([
        [0, 0, 0],
        [0, 153, 255],
        [102, 255, 153],
        [0, 204, 153],
        [255, 255, 102],
        [255, 255, 204],
        [255, 153, 0],
        [255, 102, 255],
        [102, 0, 51],
        [255, 204, 255],
        [255, 0, 102]
    ])
    
    for index, color in enumerate(color_mapping):
        color_mask[mask == index] = color

    overlayed_image = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)

    return overlayed_image, image, color_mask

def visualize_landmarks(im, landmarks, color, thickness=3, eye_radius=0):
    im = im.permute(1, 2, 0).numpy()
    im = (im * 255).astype(np.uint8)
    im = np.ascontiguousarray(im)
    landmarks = landmarks.squeeze().numpy().astype(np.int32)
    for (x, y) in landmarks:
        cv2.circle(im, (x,y), eye_radius, color, thickness)
    return im

def visualize_head_pose(img, euler, tdx=None, tdy=None, size = 100):
    pitch, yaw, roll = euler[0], euler[1], euler[2]

    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,255,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(255,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,255),2)
    return img

def denorm_points(points, h, w, align_corners=False):
    if align_corners:
        denorm_points = (points + 1) / 2 * torch.tensor([w - 1, h - 1], dtype=torch.float32).to(points).view(1, 1, 2)
    else:
        denorm_points = ((points + 1) * torch.tensor([w, h], dtype=torch.float32).to(points).view(1, 1, 2) - 1) / 2

    return denorm_points



def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean 
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def adjust_bbox(x_min, y_min, x_max, y_max, image_width, image_height, margin_percentage=50):
    width = x_max - x_min
    height = y_max - y_min
    
    increase_width = width * (margin_percentage / 100.0) / 2
    increase_height = height * (margin_percentage / 100.0) / 2
    
    x_min_adjusted = max(0, x_min - increase_width)
    y_min_adjusted = max(0, y_min - increase_height)
    x_max_adjusted = min(image_width, x_max + increase_width)
    y_max_adjusted = min(image_height, y_max + increase_height)
    
    return x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted


def test(args):
    device = "cuda:" + str(args.gpu_num)
    model = FaceXFormer().to(device)
    weights_path = args.model_path
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict_backbone'])

    model.eval()
    transforms_image = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224,224), interpolation=InterpolationMode.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    mtcnn = MTCNN(keep_all=True)
    image = Image.open(args.image_path)
    width, height = image.size
    boxes, probs = mtcnn.detect(image)
    x_min, y_min, x_max, y_max = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
    x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, width, height)
    image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
    image = transforms_image(image)

    if args.task == "parsing":
        task = torch.tensor([0])
    elif args.task == "landmarks":
        task = torch.tensor([1])
    elif args.task == "headpose":
        task = torch.tensor([2])
    elif args.task == "attributes":
        task = torch.tensor([3])
    elif args.task == "age_gender_race":
        task = torch.tensor([4])
    elif args.task == "visibility":
        task = torch.tensor([5])
    data = {'image': image, 'label': {"segmentation":torch.zeros([224,224]), "lnm_seg": torch.zeros([5, 2]),"landmark": torch.zeros([68, 2]), "headpose": torch.zeros([3]), "attribute": torch.zeros([40]), "a_g_e": torch.zeros([3]), 'visibility': torch.zeros([29])}, 'task': task}
    images, labels, tasks = data["image"], data["label"], data["task"]
    images = images.unsqueeze(0).to(device=device)
    for k in labels.keys():
        labels[k] = labels[k].unsqueeze(0).to(device=device)
    tasks = tasks.to(device=device)

    landmark_output, headpose_output, attribute_output, visibility_output, age_output, gender_output, race_output, seg_output = model(images, labels, tasks)
    if tasks[0] == 0:
        preds = seg_output.softmax(dim=1)
        mask = torch.argmax(preds, dim=1)
        pred_mask = mask[0].detach().cpu().numpy()
        save_path = os.path.join(args.results_path, "parsing.png")
        cv2.imwrite(f"{save_path}", pred_mask)
        mask, face, color_mask = visualize_mask(unnormalize(images[0].detach().cpu()), pred_mask)
        save_path = os.path.join(args.results_path, "parsing_visualization.png")
        cv2.imwrite(f"{save_path}", mask[:, :, ::-1])
    if tasks[0] == 1:
        image = unnormalize(images[0].detach().cpu())
        denorm_landmarks = denorm_points(landmark_output.view(-1,68,2)[0],224,224)
        im = visualize_landmarks(image, denorm_landmarks.detach().cpu(), (255, 255, 0))
        save_path_viz = os.path.join(args.results_path, "landmarks.png")
        save_path = os.path.join(args.results_path, "landmarks.txt")
        cv2.imwrite(f"{save_path_viz}", im[:, :, ::-1])
        with open(f'{save_path}', 'w') as file:
            for landmark in denorm_landmarks[0]:
                x, y = landmark[0], landmark[1]
                file.write(f"{x.item()} {y.item()}\n")
        file.close()
    if tasks[0] == 2:
        image = unnormalize(images[0].detach().cpu())
        im = visualize_head_pose(image, headpose_output[0])
        save_path_viz = os.path.join(args.results_path, "headpose.png")
        save_path = os.path.join(args.results_path, "headpose.txt")
        cv2.imwrite(f"{save_path_viz}", im[:, :, ::-1])
        with open(f'{save_path}', 'w') as file:
            file.write(f"Pitch: {headpose_output[0][0].item()*180/np.pi} \n")
            file.write(f"Yaw: {headpose_output[0][1].item()*180/np.pi} \n")
            file.write(f"Roll: {headpose_output[0][2].item()*180/np.pi}")
        file.close()
    if tasks[0] == 3:
        probs = torch.sigmoid(attribute_output[0])
        preds = (probs >= 0.5).float()
        pred = preds.tolist()
        pred_str = [str(int(b)) for b in pred]
        joined_pred = " ".join(pred_str)
        save_path = os.path.join(args.results_path, "attribute.txt")
        with open(f'{save_path}', 'w') as file:
            file.write(joined_pred)
        file.close()
    if tasks[0] == 4:
        age_preds = torch.argmax(age_output, dim=1)[0]
        gender_preds = torch.argmax(gender_output, dim=1)[0]
        race_preds = torch.argmax(race_output, dim=1)[0]
        save_path = os.path.join(args.results_path, "age_gender_race.txt")
        with open(f'{save_path}', 'w') as file:
            file.write(f"Age: {age_preds.item()} \n")
            file.write(f"Gender: {gender_preds.item()} \n")
            file.write(f"Race: {race_preds.item()}")
        file.close()
    if tasks[0] == 5:
        probs = torch.sigmoid(visibility_output[0])
        preds = (probs >= 0.5).float()
        pred = preds.tolist()
        pred_str = [str(int(b)) for b in pred]
        joined_pred =  " ".join(pred_str)
        save_path = os.path.join(args.results_path, "visibility.txt")
        with open(f'{save_path}', 'w') as file:
            file.write(joined_pred)
        file.close()
    image = unnormalize(images[0].detach().cpu())
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    save_path = os.path.join(args.results_path, "face.png")
    cv2.imwrite(f"{save_path}", image[:, :, ::-1])


            



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Provide absolute path to your weights file")
    parser.add_argument("--image_path", type=str, help="Provide absolute path to the image you want to perform inference on")
    parser.add_argument("--results_path", type=str, help="Provide path to the folder where results need to be saved")
    parser.add_argument("--task", type=str, help="parsing" or "landmarks" or "headpose" or "attributes" or "age_gender_race" or "visibility")
    parser.add_argument("--gpu_num", type=str, help="Provide the gpu number")
    args = parser.parse_args()  
    test(args)