from __future__ import print_function
import os
import argparse
import numpy as np
import cv2
from Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def main():
    rf = Pytorch_RetinaFace()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "./Pytorch_Retinaface/outputs")

    image_path = os.path.join(current_dir, "./Pytorch_Retinaface/images/test.webp")
    if not os.path.exists(image_path):
        raise FileNotFoundError
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    img = np.float32(img_raw)

    dets = rf.detect_faces(img)

    # Crop and save each detected face
    cropped_imgs = rf.center_and_crop_rescale(img_raw, dets)
    for index, cropped_img in enumerate(cropped_imgs):
        # Save the final image
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"cropped_face_{index}.jpg"), cropped_img)
        print(f"Saved: cropped_face_{index}.jpg")

if __name__ == '__main__':
    main()