import os
import time
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

from .data import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm

class Pytorch_RetinaFace:
    def __init__(self, cfg="mobile0.25", pretrained_path="./weights/mobilenet0.25_Final.pth", weights_path="./weights/mobilenetV1X0.25_pretrain.tar", device="cuda", vis_thres=0.6, top_k=5000, keep_top_k=750, nms_threshold=0.4, confidence_threshold=0.02):
        self.vis_thres = vis_thres
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.cfg = cfg_mnet if cfg=="mobile0.25" else cfg_re50
        self.device = torch.device(device)
        self.net = RetinaFace(cfg=self.cfg, weights_path=weights_path, phase='test', device=device).to(self.device)
        self.load_model_weights(pretrained_path)
        self.net.eval()

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model_weights(self, pretrained_path):
        pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_path)
        print('Loading pretrained model from {}'.format(pretrained_path))
        pretrained_dict = torch.load(pretrained_path, map_location=self.device)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(self.net, pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.to(self.device)
        return self.net

    def center_and_crop_rescale(self, image, dets, scale_factor=4, shift_factor=0.35, aspect_ratio=1.0):
        cropped_imgs = []
        for index, bbox in enumerate(dets):
            if bbox[4] < self.vis_thres:
                continue

            x1, y1, x2, y2 = map(int, bbox[:4])
            face_width = x2 - x1
            face_height = y2 - y1

            # New height and width based on scale factor
            new_face_height = int(face_height * scale_factor)
            # new_face_width = int(new_face_height * (face_width / face_height))
            new_face_width = int(new_face_height / aspect_ratio)

            # Center coordinates of the detected face
            center_x = x1 + face_width // 2
            center_y = y1 + face_height // 2 + int(new_face_height * (0.5 - shift_factor))

            # Crop coordinates, adjusted to the image boundaries
            crop_x1 = max(0, center_x - new_face_width // 2)
            crop_x2 = min(image.shape[1], center_x + new_face_width // 2)
            crop_y1 = max(0, center_y - new_face_height // 2)
            crop_y2 = min(image.shape[0], center_y + new_face_height // 2)

            # Crop the region and add padding to form a square
            cropped_imgs.append(image[crop_y1:crop_y2, crop_x1:crop_x2])
        return cropped_imgs

    def detect_faces(self, img):
        resize = 1

        if len(img.shape) == 4 and img.shape[0] == 1:
            img = torch.squeeze(img, 0)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(self.device)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        mean_values = torch.tensor([104, 117, 123], dtype=torch.float32).to(self.device)
        mean_values = mean_values.view(1, 1, 3)
        img -= mean_values
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        scale = scale.to(self.device)

        tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # Ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # Keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # Perform NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # Keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets
