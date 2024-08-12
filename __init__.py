import torch
import comfy.utils
from .Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace
from comfy.model_management import get_torch_device

class AutoCropFaces:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "number_of_faces": ("INT", {
                    "default": 5, 
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10,
                    "step": 0.5,
                    "display": "slider"
                }),
                "shift_factor": ("FLOAT", {
                    "default": 0.45,
                    "min": 0,
                    "max": 1,
                    "step": 0.01,
                    "display": "slider"
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "step": 1,
                    "display": "number"
                }),
                "max_faces_per_image": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                }),
                # "aspect_ratio": ("FLOAT", {
                #     "default": 1, 
                #     "min": 0.2,
                #     "max": 5,
                #     "step": 0.1,
                # }),
                "aspect_ratio": (["9:16", "2:3", "3:4", "4:5", "1:1", "5:4", "4:3", "3:2", "16:9"], {
                    "default": "1:1",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("face",)

    FUNCTION = "auto_crop_faces"

    CATEGORY = "Faces"

    def aspect_ratio_string_to_float(self, str_aspect_ratio="1:1"):
        a, b = map(float, str_aspect_ratio.split(':'))
        return a / b

    def auto_crop_faces_in_image (self, image, max_number_of_faces, scale_factor, shift_factor, aspect_ratio, method='lanczos'): 
        image_255 = image * 255
        rf = Pytorch_RetinaFace(top_k=50, keep_top_k=max_number_of_faces, device=get_torch_device())
        dets = rf.detect_faces(image_255)
        cropped_faces, bbox_info = rf.center_and_crop_rescale(image, dets, scale_factor=scale_factor, shift_factor=shift_factor, aspect_ratio=aspect_ratio)

        # Add a batch dimension to each cropped face
        cropped_faces_with_batch = [face.unsqueeze(0) for face in cropped_faces]
        return cropped_faces_with_batch, bbox_info

    def auto_crop_faces(self, image, number_of_faces, start_index, max_faces_per_image, scale_factor, shift_factor, aspect_ratio, method='lanczos'):
        """ 
        "image" - Input can be one image or a batch of images with shape (batch, width, height, channel count)
        "number_of_faces" - This is passed into PyTorch_RetinaFace which allows you to define a maximum number of faces to look for.
        "start_index" - The starting index of which face you select out of the set of detected faces.
        "scale_factor" - How much crop factor or padding do you want around each detected face.
        "shift_factor" - Pan up or down relative to the face, 0.5 should be right in the center.
        "aspect_ratio" - When we crop, you can have it crop down at a particular aspect ratio.
        "method" - Scaling pixel sampling interpolation method.
        """
        
        # Turn aspect ratio to float value
        aspect_ratio = self.aspect_ratio_string_to_float(aspect_ratio)

        selected_faces, detected_cropped_faces = [], []
        selected_crop_data, detected_crop_data = [], []
        original_images = []

        # Loop through the input batches. Even if there is only one input image, it's still considered a batch.
        for i in range(image.shape[0]):

            original_images.append(image[i].unsqueeze(0)) # Temporarily the image, but insure it still has the batch dimension.
            # Detect the faces in the image, this will return multiple images and crop data for it.
            cropped_images, infos = self.auto_crop_faces_in_image(
                image[i],
                max_faces_per_image,
                scale_factor,
                shift_factor,
                aspect_ratio,
                method)

            detected_cropped_faces.extend(cropped_images)
            detected_crop_data.extend(infos)

        # If we haven't detected anything, just return the original images, and default crop data.
        if not detected_cropped_faces or len(detected_cropped_faces) == 0:
            selected_crop_data = [(0, 0, img.shape[3], img.shape[2]) for img in original_images]
            return (image, selected_crop_data)

         # Circular index calculation
        start_index = start_index % len(detected_cropped_faces)

        if number_of_faces >= len(detected_cropped_faces):
            selected_faces = detected_cropped_faces[start_index:] + detected_cropped_faces[:start_index]
            selected_crop_data = detected_crop_data[start_index:] + detected_crop_data[:start_index]
        else:
            end_index = (start_index + number_of_faces) % len(detected_cropped_faces)
            if start_index < end_index:
                selected_faces = detected_cropped_faces[start_index:end_index]
                selected_crop_data = detected_crop_data[start_index:end_index]
            else:
                selected_faces = detected_cropped_faces[start_index:] + detected_cropped_faces[:end_index]
                selected_crop_data = detected_crop_data[start_index:] + detected_crop_data[:end_index]

        # If we haven't selected anything, then return original images.
        if len(selected_faces) == 0: 
            # selected_crop_data = [(0, 0, img.shape[3], img.shape[2]) for img in original_images]
            return (image, None)

        # If there is only one detected face in batch of images, just return that one.
        elif len(selected_faces) <= 1:
            out = selected_faces[0]
            crop_data = selected_crop_data[0] # to be compatible with WAS
            return (out, crop_data)

        # Determine the index of the face with the maximum width
        max_width_index = max(range(len(selected_faces)), key=lambda i: selected_faces[i].shape[1])

        # Determine the maximum width
        max_width = selected_faces[max_width_index].shape[1]
        max_height = selected_faces[max_width_index].shape[2]
        shape = (max_height, max_width)

        out = None
        # All images need to have the same width/height to fit into the tensor such that we can output as image batches.
        for face_image in selected_faces:
            if shape != face_image.shape[1:3]: # Determine whether cropped face image size matches largest cropped face image. 
                face_image = comfy.utils.common_upscale( # This method expects (batch, channel, height, width)
                    face_image.movedim(-1, 1), # Move channel dimension to width dimension
                    max_height, # Height
                    max_width, # Width
                    method, # Pixel sampling method.
                    "" # Only "center" is implemented right now, and we don't want to use that.
                ).movedim(1, -1)
            # Append the fitted image into the tensor.
            if out is None:
                out = face_image
            else:
                out = torch.cat((out, face_image), dim=0)

        #TODO: WAS doesn't not support multiple faces, so this won't work with WAS.
        return (out, selected_crop_data)

NODE_CLASS_MAPPINGS = {
    "AutoCropFaces": AutoCropFaces
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropFaces": "Auto Crop Faces"
}
