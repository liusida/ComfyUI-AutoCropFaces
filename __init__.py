import torch
from .Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace

class AutoCropFaces:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_number_of_faces": ("INT", {
                    "default": 5, 
                    "min": 1,
                    "max": 50,
                    "step": 1,
                }),
                "index_of_face": ("INT", {
                    "default": 1,
                    "min": 1,
                    "step": 1,
                    "display": "number"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 4,
                    "min": 0.5,
                    "max": 10,
                    "step": 0.5,
                    "display": "slider"
                }),
                "shift_factor": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1,
                    "step": 0.1,
                    "display": "slider"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("face",)

    FUNCTION = "auto_crop_faces"

    CATEGORY = "Faces"

    def auto_crop_faces(self, image, max_number_of_faces, index_of_face, scale_factor, shift_factor):
        #TODO: currently only support one single image. No batch.
        image_without_batch = image[0]
        image_255 = image_without_batch * 255
        rf = Pytorch_RetinaFace(top_k=50, keep_top_k=max_number_of_faces)
        dets = rf.detect_faces(image_255)
        cropped_images = rf.center_and_crop_rescale(image_without_batch, dets, scale_factor=scale_factor, shift_factor=shift_factor)
        if len(cropped_images)>=1:
            clamped_index = max(1, min(index_of_face, len(cropped_images)))
            cropped_image = torch.unsqueeze(cropped_images[clamped_index-1], 0)
            return (cropped_image,)
        return (image,)

NODE_CLASS_MAPPINGS = {
    "AutoCropFaces": AutoCropFaces
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropFaces": "Auto Crop Faces"
}
