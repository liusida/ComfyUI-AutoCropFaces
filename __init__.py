import torch
import comfy.utils
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
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
                "selected_number_of_faces": ("INT", {
                    "default": -1,
                    "min": -1,
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
                    "step": 0.01,
                    "display": "slider"
                }),
                "aspect_ratio": ("FLOAT", {
                    "default": 1, 
                    "min": 0.2,
                    "max": 5,
                    "step": 0.1,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("face",)

    FUNCTION = "auto_crop_faces"

    CATEGORY = "Faces"

    def auto_crop_faces_in_image (self, image, max_number_of_faces, scale_factor, shift_factor, aspect_ratio, method='lanczos'): 
        image_255 = image * 255
        rf = Pytorch_RetinaFace(top_k=50, keep_top_k=max_number_of_faces)
        dets = rf.detect_faces(image_255)
        cropped_faces, bbox_info = rf.center_and_crop_rescale(image, dets, scale_factor=scale_factor, shift_factor=shift_factor, aspect_ratio=aspect_ratio)

        # Add a batch dimension to each cropped face
        cropped_faces_with_batch = [face.unsqueeze(0) for face in cropped_faces]
        return cropped_faces_with_batch, bbox_info

    def auto_crop_faces(self, image, max_number_of_faces, index_of_face, selected_number_of_faces, scale_factor, shift_factor, aspect_ratio, method='lanczos'):

        selected_faces, detected_cropped_faces = [], []
        selected_crop_data, detected_crop_data = [], []
        original_images = []

        remaining_face_count = max_number_of_faces
        for i in range(image.shape[0]):  # Loop through each image in the batch

            original_images.append(image[i].unsqueeze(0))

            cropped_images, infos = self.auto_crop_faces_in_image(
                image[i],
                max_number_of_faces,
                scale_factor,
                shift_factor,
                aspect_ratio,
                method)

            detected_cropped_faces.extend(cropped_images)
            detected_crop_data.extend(infos)

            remaining_face_count = remaining_face_count - len(detected_cropped_faces)
            if remaining_face_count <= 0:
                break

        if not detected_cropped_faces or len(detected_cropped_faces) == 0:
            selected_faces = original_images
            selected_crop_data = [(0, 0, original_images.shape[3], original_images.shape[2])] * original_images.shape[0]

        index_of_face = 0 if index_of_face <= -1 else index_of_face
        start = max(0, min(index_of_face, len(detected_cropped_faces) - 1))
        end = start + selected_number_of_faces if selected_number_of_faces > 0 else len(detected_cropped_faces)
        selected_faces = detected_cropped_faces[start:end]
        selected_crop_data = detected_crop_data[start:end]

        out = selected_faces[0]
        if len(selected_faces) == 0: 
            return (image, ((1, 1), (0, 0, 1, 1)))
        elif len(selected_faces) <= 1:
            return (out, selected_crop_data[0])

        shape = out.shape
        for i in range(1, len(selected_faces)):
            resized_image = selected_faces[i]
            if shape != selected_faces[i].shape:
                resized_image = comfy.utils.common_upscale(
                    selected_faces[i].movedim(-1, 1),
                    shape[2],
                    shape[1],
                    method,
                    "" # Only "center" is implemented right now.
                ).movedim(1, -1)
            out = torch.cat((out, resized_image), dim=0)

        return (out, selected_crop_data)

NODE_CLASS_MAPPINGS = {
    "AutoCropFaces": AutoCropFaces
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropFaces": "Auto Crop Faces"
}
