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
                    "max": 100,
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
        """ 
        "image" - Input can be one image or a batch of images with shape (batch, width, height, channel count)
        "max_number_of_faces" - This is passed into PyTorch_RetinaFace which allows you to define a maximum number of faces to look for.
        "index_of_face" - The starting index of which face you select out of the set of detected faces.
        "selected_number_of_faces" - The number of faces you want to select from the set of detected faces starting from "index_of_face", if 
            this is -1, then it will be either "max_number_of_faces" or the number of detected faces, whichever is less.
        "scale_factor" - How much crop factor or padding do you want around each detected face.
        "shift_factor" - Pan up or down relative to the face, 0.5 should be right in the center.
        "aspect_ratio" - When we crop, you can have it crop down at a particular aspect ratio.
        "method" - Scaling pixel sampling interpolation method.
        """

        selected_faces, detected_cropped_faces = [], []
        selected_crop_data, detected_crop_data = [], []
        original_images = []

        # Foreach detected face, we substract that, counting down until 0, then stop detecting anymore faces.
        remaining_face_count = max_number_of_faces

        # Loop through the input batches. Even if there is only one input image, it's still considered a batch.
        for i in range(image.shape[0]):

            original_images.append(image[i].unsqueeze(0)) # Temporarily the image, but insure it still has the batch dimension.
            # Detect the faces in the image, this will return multiple images and crop data for it.
            cropped_images, infos = self.auto_crop_faces_in_image(
                image[i],
                max_number_of_faces,
                scale_factor,
                shift_factor,
                aspect_ratio,
                method)

            detected_cropped_faces.extend(cropped_images)
            detected_crop_data.extend(infos)

            # Count down until we've reached our "max_number_of_faces"
            remaining_face_count = remaining_face_count - len(detected_cropped_faces)
            if remaining_face_count <= 0: # We've reached the limit, break.
                break

        # If we haven't detected anything, just return the original images, and default crop data.
        if not detected_cropped_faces or len(detected_cropped_faces) == 0:
            selected_crop_data = [(0, 0, img.shape[3], img.shape[2]) for img in original_images]
            return (image, selected_crop_data)

        index_of_face = 0 if index_of_face <= -1 else index_of_face

        # Get the range at which we want to select the faces.
        start = max(0, min(index_of_face, len(detected_cropped_faces) - 1))
        end = start + min(max_number_of_faces, selected_number_of_faces) if selected_number_of_faces > 0 else min(max_number_of_faces, len(detected_cropped_faces))

        selected_faces = detected_cropped_faces[start:end]
        selected_crop_data = detected_crop_data[start:end]

        out = selected_faces[0]

        # If we haven't selected anything, then return original images.
        if len(selected_faces) == 0: 
            selected_crop_data = [(0, 0, img.shape[3], img.shape[2]) for img in original_images]
            return (image, selected_crop_data)

        # If there is only one detected face in batch of images, just return that one.
        elif len(selected_faces) <= 1:
            return (out, selected_crop_data)

        shape = out.shape

        # All images need to have the same width/height to fit into the tensor such that we can output as image batches.
        for i in range(1, len(selected_faces)):
            resized_image = selected_faces[i]
            if shape != selected_faces[i].shape: # Check all images against the first image and scale it to that size.
                resized_image = comfy.utils.common_upscale( # This method expects (batch, channel, height, width)
                    selected_faces[i].movedim(-1, 1), # Move channel dimension to width dimension
                    shape[2], # Height
                    shape[1], # Width
                    method, # Pixel sampling method.
                    "" # Only "center" is implemented right now, and we don't want to use that.
                ).movedim(1, -1)
            # Append the fitted image into the tensor.
            out = torch.cat((out, resized_image), dim=0)

        return (out, selected_crop_data)

NODE_CLASS_MAPPINGS = {
    "AutoCropFaces": AutoCropFaces
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropFaces": "Auto Crop Faces"
}
