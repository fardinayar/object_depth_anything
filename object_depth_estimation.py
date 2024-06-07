from depth.depth_anything.dpt import DepthAnything_encoder, DepthAnything_decoder
from depth.options import MonodepthOptions
import torch
import os
from ultralytics import YOLO
import PIL.Image as pil
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np
import imageio
import tqdm

def get_depth_model(opt):
    encoder = DepthAnything_encoder('vits')
    dim = encoder.pretrained.blocks[0].attn.qkv.in_features
    depth_decoder = DepthAnything_decoder(dim).from_pretrained('LiheYoung/depth_anything_vits14', dim=dim)
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    scale_path = os.path.join(opt.load_weights_folder, "scale.pth")
    
    
    scale = torch.nn.Module()
    scale.p = torch.nn.Parameter(torch.tensor([1.0]), True)
    encoder_dict = torch.load(encoder_path)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))
    scale.load_state_dict(torch.load(scale_path))
    
    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()
    scale.cuda()
    scale.eval()
    
    def inference_depth(input_image):
        feed_height = encoder_dict['height']
        feed_width = encoder_dict['width']
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0).cuda()
        features, h, w = encoder(input_image)
        output = depth_decoder(features, h, w)
        depth = output[("disp", 0)] * scale.p
        depth = torch.nn.functional.interpolate(
                depth, (original_height, original_width), mode="bilinear", align_corners=False)
        depth = depth.squeeze().detach().cpu().numpy()
        return depth
    
    return inference_depth


def get_yolo_model():
    
    # Load a model
    model = YOLO("yolov8n.pt",verbose=False)  # load a pretrained model (recommended for training)

    model.cuda()
    
    return model


class ObjectDepthEstimation:
    def __init__(self, options):
        self.depth_model = get_depth_model(options)
        self.object_detection_model = get_yolo_model()
        self.opts = options
        self.root_path = self.opts.image_folder
        self.imagesList = sorted(os.listdir(self.root_path))

    
    def add_depth_to_results(self, results, depth, image_name=None):
        image = results.orig_img
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classes = results.names
        depths = []
        for i, box in enumerate(results.boxes.xyxy):     
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])
                label = results.boxes.cls[i].item()
                label = classes[label] + " | " + "{:.2f}".format(depth[yA:yB, xA:xB].min())
                '''if classes[results.boxes.cls[i].item()] == 'person':
                    continue'''
                depths.append(depth[yA:yB, xA:xB].min())
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                cv2.putText(image, label, (xA, yA-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
        setattr(results.boxes, 'depth', np.array(depths))
        if image_name:
            setattr(results.boxes, 'image_name', image_name)
        return image, results
    
    def inference(self):
        loader = self.image_loader()
        output_images = []
        all_results = []
        for image, image_name in tqdm.tqdm(loader, total=len(self.imagesList)):
            results =  self.object_detection_model(image,verbose=False)
            depth = self.depth_model(image)
            output_image, results = self.add_depth_to_results(results[0].to('cpu').numpy(), depth, image_name)
            normalized_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            output_images.append(np.vstack([output_image, np.concatenate([normalized_depth[:,:,None],]*3, axis=-1)]).astype('uint8'))
            all_results.append(results)
        if self.opts.output_type == 'video':
            imageio.mimsave('demo.gif', output_images[:100])
        if self.opts.output_type == 'csv':
            self.save_csv(all_results)
            
    def image_loader(self):
        
        for image_name in self.imagesList:
            image = pil.open(self.root_path + image_name)
            yield image, image_name
        
    def save_csv(self, all_results):
        df = []
        classes = all_results[0].names
        for results in all_results:
            for i in range(len(results.boxes)):
                  df.append([getattr(results.boxes,'image_name'), classes[results.boxes.cls[i]], getattr(results.boxes, 'depth')[i]])
        df = pd.DataFrame(df, columns=["image name", "object class", "depth"])
        df.to_csv('results.csv', index=False)
        
            
    
        
        

if __name__ == '__main__':
    options = MonodepthOptions()
    options = options.parse()
    options.load_weights_folder = 'mono_weights'
    print(options)
    ode = ObjectDepthEstimation(options)
    ode.inference()

    
    
    