import torch
import comfy.utils
from .Pytorch_Retinaface.pytorch_retinaface import Pytorch_RetinaFace
from comfy.model_management import get_torch_device


class AutoCropFaces:
    # ───────────────────────────────────────────────────────────── INPUT TYPES ──
    @classmethod
    def INPUT_TYPES(cls):
        """Add dropdown 'face_filter' to choose sort order."""
        return {
            "required": {
                "image": ("IMAGE",),

                # number of faces to output
                "number_of_faces": (
                    "INT",
                    {"default": 5, "min": 1, "max": 100, "step": 1},
                ),

                # detection / crop parameters
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.5,
                        "max": 20,
                        "step": 0.5,
                        "display": "slider",
                    },
                ),
                "shift_factor": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0,
                        "max": 1,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),

                # start position in ordered list (circular)
                "start_index": ("INT", {"default": 0, "step": 1, "display": "number"}),

                # max faces to *detect* per image
                "max_faces_per_image": (
                    "INT",
                    {"default": 50, "min": 1, "max": 1000, "step": 1},
                ),

                # output aspect-ratio
                "aspect_ratio": (
                    [
                        "9:16",
                        "2:3",
                        "3:4",
                        "4:5",
                        "1:1",
                        "5:4",
                        "4:3",
                        "3:2",
                        "16:9",
                    ],
                    {"default": "1:1"},
                ),

                # ── NEW ── how to sort faces before selection
                "face_filter": (
                    ["largest_first", "smallest_first", "left_to_right"],
                    {"default": "largest_first"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROP_DATA")
    RETURN_NAMES = ("face",)
    FUNCTION = "auto_crop_faces"
    CATEGORY = "Faces"

    # ───────────────────────────────────────────────────────────── HELPERS ──
    @staticmethod
    def _aspect_ratio_to_float(ratio_str: str) -> float:
        a, b = map(float, ratio_str.split(":"))
        return a / b

    @staticmethod
    def _calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        Boxes are expected to be in [x1, y1, x2, y2] format.
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou

    def _filter_duplicate_detections(self, detected_faces, detected_infos, raw_boxes, iou_threshold=0.85):
        if not detected_faces:
            return [], [], []

        unique_faces = []
        unique_infos = []
        unique_raw_boxes = []

        for i in range(len(raw_boxes)):
            is_duplicate = False
            for unique_box in unique_raw_boxes:
                if self._calculate_iou(raw_boxes[i], unique_box) > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_faces.append(detected_faces[i])
                unique_infos.append(detected_infos[i])
                unique_raw_boxes.append(raw_boxes[i])

        return unique_faces, unique_infos, unique_raw_boxes

    def _detect_and_crop(
        self,
        img_tensor,
        max_faces,
        scale,
        shift,
        aspect_ratio_float,
        method="lanczos",
    ):
        """
        Run RetinaFace on a single image tensor and return:
        • list[cropped_face_tensor]
        • list[tuple]  (cx, cy, scale) — centre-X, centre-Y, size factor
        """
        Run RetinaFace on a single image tensor and return:
        • list[cropped_face_tensor]
        • list[tuple]  (cx, cy, scale) — centre-X, centre-Y, size factor
        • list[list] raw_detections [x1, y1, x2, y2, (confidence)]
        """
        img_255 = img_tensor * 255
        rf = Pytorch_RetinaFace(
            top_k=50, keep_top_k=max_faces, device=get_torch_device()
        )
        # detections_from_rf is a list of lists, e.g., [[x1,y1,x2,y2,conf], ...] or None
        detections_from_rf = rf.detect_faces(img_255)

        valid_detections_for_crop = []
        raw_bounding_boxes = [] # To store just the bbox for duplicate checking

        if detections_from_rf is not None:
            for det in detections_from_rf:
                if isinstance(det, (list, tuple)) and len(det) >= 4:
                    # Pytorch_RetinaFace.detect_faces returns [x1,y1,x2,y2,confidence]
                    # center_and_crop_rescale expects a list of such detections
                    valid_detections_for_crop.append(det)
                    raw_bounding_boxes.append(list(det[:4])) # Store [x1,y1,x2,y2] for duplicate check
                # else:
                    # print(f"Debug: Skipping invalid detection format: {det}")

        if not valid_detections_for_crop:
            # print("Debug: No valid detections found by _detect_and_crop.")
            return [], [], []

        # Pass all valid detections (including confidence) to center_and_crop_rescale
        crops, infos = rf.center_and_crop_rescale(
            img_tensor,
            valid_detections_for_crop,
            scale_factor=scale,
            shift_factor=shift,
            aspect_ratio=aspect_ratio_float,
        )

        # ensure batch-dim for crops
        processed_crops = [c.unsqueeze(0) for c in crops]

        # Return processed_crops, their corresponding infos, and the raw_bounding_boxes that led to these crops
        return processed_crops, infos, raw_bounding_boxes

    # ────────────────────────────────────────────────────────────── MAIN ──
    def auto_crop_faces(
        self,
        image,
        number_of_faces,
        start_index,
        max_faces_per_image,
        scale_factor,
        shift_factor,
        aspect_ratio,
        face_filter,  # new parameter
        method="lanczos",
    ):
        """
        Detect faces, order them according to *face_filter*,
        then return `number_of_faces` starting at `start_index`.
        """
        if image is None:
            print("ERROR: AutoCropFaces node received no image input. Please check the workflow connections.")
            # Return dummy outputs to prevent crash
            # Expected return: IMAGE, CROP_DATA
            # Create a small dummy tensor for the image output
            dummy_image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device=get_torch_device())
            return dummy_image_tensor, None

        aspect_ratio_f = self._aspect_ratio_to_float(aspect_ratio)

        detected_faces, detected_infos, all_raw_boxes = [], [], []

        # iterate over batch
        for i in range(image.shape[0]):
            # _detect_and_crop now returns raw_bounding_boxes as the third item
            crops, infos, raw_boxes = self._detect_and_crop(
                image[i],
                max_faces_per_image,
                scale_factor,
                shift_factor,
                aspect_ratio_f,
                method,
            )
            if crops: # Only extend if detections were made
                detected_faces.extend(crops)
                detected_infos.extend(infos)
                all_raw_boxes.extend(raw_boxes) # Collect raw bounding boxes

        # nothing detected → pass-through
        if not detected_faces:
            # print("Debug: No faces detected in any image in the batch.")
            fallback = [(0, 0, image.shape[3], image.shape[2])]  # dummy crop
            return image, fallback

        # Filter out duplicate detections
        # print(f"Debug: Before filtering: {len(detected_faces)} faces, {len(all_raw_boxes)} boxes.")
        detected_faces, detected_infos, all_raw_boxes = self._filter_duplicate_detections(
            detected_faces, detected_infos, all_raw_boxes
        )
        # print(f"Debug: After filtering: {len(detected_faces)} faces.")

        # if after filtering, no faces are left
        if not detected_faces:
            # print("Debug: No faces left after filtering duplicates.")
            fallback = [(0, 0, image.shape[3], image.shape[2])]
            return image, fallback

        # ── ORDER / FILTER ───────────────────────────────────────────────
        # Ensure face_filter is applied to unique faces
        if face_filter == "largest_first":
            order = sorted(
                range(len(detected_faces)),
                key=lambda i: detected_faces[i].shape[1]  # width
                * detected_faces[i].shape[2],  # height
                reverse=True,
            )
        elif face_filter == "smallest_first":
            order = sorted(
                range(len(detected_faces)),
                key=lambda i: detected_faces[i].shape[1]
                * detected_faces[i].shape[2],
            )
        else:  # "left_to_right"
            # infos[0] is centre-x for every face tuple
            order = sorted(range(len(detected_faces)), key=lambda i: detected_infos[i][0])

        detected_faces = [detected_faces[i] for i in order]
        detected_infos = [detected_infos[i] for i in order]

        # ── SELECT SUBSET (circular slice) ───────────────────────────────
        start_index %= len(detected_faces)

        if number_of_faces >= len(detected_faces):
            faces_sel = detected_faces[start_index:] + detected_faces[:start_index]
            infos_sel = detected_infos[start_index:] + detected_infos[:start_index]
        else:
            end = (start_index + number_of_faces) % len(detected_faces)
            if start_index < end:
                faces_sel = detected_faces[start_index:end]
                infos_sel = detected_infos[start_index:end]
            else:
                faces_sel = detected_faces[start_index:] + detected_faces[:end]
                infos_sel = detected_infos[start_index:] + detected_infos[:end]

        # ── PREPARE OUTPUT ───────────────────────────────────────────────
        if not faces_sel:
            return image, None
        if len(faces_sel) == 1:
            return faces_sel[0], infos_sel[0]

        # pad / upscale to common size
        max_w = max(f.shape[1] for f in faces_sel)
        max_h = max(f.shape[2] for f in faces_sel)

        out = None
        for f in faces_sel:
            if (f.shape[1], f.shape[2]) != (max_w, max_h):
                f = comfy.utils.common_upscale(
                    f.movedim(-1, 1),  # (B,C,H,W)
                    max_w, # target width
                    max_h, # target height
                    method,
                    "disabled", # crop parameter for comfy.utils.common_upscale
                ).movedim(1, -1)
            out = f if out is None else torch.cat((out, f), dim=0)

        return out, infos_sel


# ────────────────────────────────────────────────────────── NODE REGISTRY ──
NODE_CLASS_MAPPINGS = {"AutoCropFaces": AutoCropFaces}
NODE_DISPLAY_NAME_MAPPINGS = {"AutoCropFaces": "Auto Crop Faces"}
