import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, UnidentifiedImageError
import io
import os
import math

#if rembg available
try:
    from rembg import remove as remove_background_ai
    REMBG_AVAILABLE = True
    print("INFO: rembg (AI background removal) is available.")
except ImportError:
    REMBG_AVAILABLE = False
    print("WARNING: rembg library not found. AI background removal will not be available.")
    print("         Please install it: pip install rembg")
    def remove_background_ai(data, *args, **kwargs):
        print("ERROR: rembg not available, cannot remove background with AI.")
        return data

# --- Helper: Image Format Conversions ---
def pillow_to_cv2_bgr(pil_image):
    if pil_image.mode == 'RGBA': # rembg might output RGBA
        pil_image = pil_image.convert('RGB') # Convert to RGB first
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_bgr_to_pillow_rgb(cv2_image_bgr):
    return Image.fromarray(cv2.cvtColor(cv2_image_bgr, cv2.COLOR_BGR2RGB))

# initial photo validation
def initial_photo_validation(pil_image_original):
    """Basic checks on the uploaded photo."""
    feedback = []
    width, height = pil_image_original.size
    if width < 600 or height < 800: # Example minimums
        feedback.append("Photo resolution is too low. Please use a higher quality image.")
    
    # todo - here we can apply more checks for blur and tilted face etc.    
    if not feedback:
        return True, "Initial validation passed."
    else:
        return False, "Issues found: " + "; ".join(feedback)

# here we can apply better face and area detection with or without ai
# Using OpenCV's DNN module with a pre-trained model is a good step up from Haar.
# or using MediaPipe (from Google) is also excellent for this
# a DNN/MediaPipe detector would be better.

def detect_main_face_advanced(cv2_image_bgr):
    """
    Placeholder for advanced face detection (e.g., OpenCV DNN, MediaPipe).
    For this example, we'll fall back to Haar if advanced isn't implemented.
    Returns (x, y, w, h) or None.
    """
    # TODO: implement DNN/MediaPipe face detector for better accuracy and robustness
    print("INFO: Using Haar Cascade for face detection (for a 'best' product, consider DNN/MediaPipe).")
    gray_image = cv2.cvtColor(cv2_image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise IOError(f"Failed to load Haar Cascade from {cascade_path}")
    except Exception as e:
        print(f"ERROR: Could not load Haar Cascade: {e}")
        return None
    
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if not len(faces):
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True) # Largest face
    return faces[0]


def auto_adjust_lighting_color(pil_image, method="pillow_basic"):
    """
    Applies subtle auto-adjustments.
    Method can be 'pillow_basic', 'cv2_clahe', or future 'ai_magic'.
    """
    print(f"INFO: Applying lighting/color adjustment using: {method}")
    if method == "pillow_basic":
        # simple global brightness/contrast increaser
        enhancer = ImageEnhance.Contrast(pil_image)
        img_contrast = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Brightness(img_contrast)
        img_final = enhancer.enhance(1.05)
        return img_final
    elif method == "cv2_clahe": # better one, but sophisticated
        cv_img = pillow_to_cv2_bgr(pil_image)
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final_cv_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return cv2_bgr_to_pillow_rgb(final_cv_img)
        # we can make better photo enchancement using ai or anything else
        # (e.g., correcting color cast, improving skin tones without over-processing)
    else:
        return pil_image

def crop_and_resize_to_spec(pil_image_subject_on_bg, face_roi_in_current_image, spec):
    """
    Precisely crops and resizes the image based on detected face and document specifications.
    pil_image_subject_on_bg: Pillow image (subject already on target background).
    face_roi_in_current_image: (x,y,w,h) of the face in pil_image_subject_on_bg.
    spec: Dictionary with document specifications.
    """
    cv_image = pillow_to_cv2_bgr(pil_image_subject_on_bg)
    fx, fy, fw, fh = face_roi_in_current_image
    orig_h, orig_w = cv_image.shape[:2]

    target_w_px = spec['output_pixel_dims']['width']
    target_h_px = spec['output_pixel_dims']['height']
    head_h_ratio = spec['head_position']['head_height_ratio']
    head_top_margin_ratio = spec['head_position']['top_of_head_to_photo_top_ratio']

    if fh == 0: return None
    desired_head_h_final_px = target_h_px * head_h_ratio
    scale_factor = desired_head_h_final_px / fh

    crop_w_orig = target_w_px / scale_factor
    crop_h_orig = target_h_px / scale_factor

    face_center_x = fx + fw / 2
    crop_x1 = face_center_x - (crop_w_orig / 2)

    margin_above_head_orig = (target_h_px * head_top_margin_ratio) / scale_factor
    crop_y1 = fy - margin_above_head_orig

    crop_x1 = int(max(0, crop_x1))
    crop_y1 = int(max(0, crop_y1))
    crop_x2 = int(min(orig_w, crop_x1 + crop_w_orig))
    crop_y2 = int(min(orig_h, crop_y1 + crop_h_orig))

    actual_crop_w = crop_x2 - crop_x1
    actual_crop_h = crop_y2 - crop_y1

    if actual_crop_w <= 0 or actual_crop_h <= 0:
        print("ERROR: Invalid crop dimensions after calculation.")
        return None

    # crop using Pillow to maintain PIL object
    cropped_pil = pil_image_subject_on_bg.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    final_pil_image = cropped_pil.resize((target_w_px, target_h_px), Image.Resampling.LANCZOS)
    return final_pil_image

def final_quality_checks(pil_image_final, spec):
    feedback = []
    cv_final_gray = cv2.cvtColor(pillow_to_cv2_bgr(pil_image_final), cv2.COLOR_BGR2GRAY)
    
    # Blur check
    laplacian_var = cv2.Laplacian(cv_final_gray, cv2.CV_64F).var()
    blur_threshold = spec.get('quality_checks', {}).get('min_laplacian_variance', 100)
    if laplacian_var < blur_threshold:
        feedback.append(f"Image may be too blurry (Laplacian Var: {laplacian_var:.2f} < {blur_threshold}).")
    else:
        print(f"INFO: Sharpness check passed (Laplacian Var: {laplacian_var:.2f}).")

    #we can apply other checks also

    if not feedback:
        return True, "Final quality checks passed."
    else:
        return False, "Final quality issues: " + "; ".join(feedback)

DEFAULT_SPECIFICATIONS = {
    "document_name": "Generic Visa Photo",
    "output_pixel_dims": { # Pixel dimensions for the final photo
        "width": 413,  # Example: 35mm at 300 DPI
        "height": 531  # Example: 45mm at 300 DPI
    },
    "background": {
        "type": "color", # 'color' or 'transparent' (if AI can make it transparent first)
        "color_rgb": (240, 240, 240), # Light gray. For pure white: (255,255,255)
        "allow_ai_background_removal": True
    },
    "head_position": { # Rules for final cropped image
        "head_height_ratio": 0.70, # Head (chin to crown) should be ~70% of photo height
        "top_of_head_to_photo_top_ratio": 0.12, # Top of head ~12% from photo top
    },
    "auto_adjustments": {
        "lighting_method": "cv2_clahe" # "pillow_basic", "cv2_clahe", or "none"
    },
    "quality_checks": {
        "min_laplacian_variance": 80.0, # Threshold for blur detection
        "ensure_single_face": True,
    },
    "output_format": {
        "type": "JPEG",
        "quality": 95,
        "dpi": (300, 300)
    }
}

def create_compliant_photo(input_image_path, output_image_path, specifications):
    """
    The main processing pipeline for creating a compliant photo.
    """
    print(f"--- Starting photo processing for: {input_image_path} ---")
    try:
        pil_original = Image.open(input_image_path)
        # Ensure it's RGB for consistency
        if pil_original.mode == 'RGBA' or pil_original.mode == 'P':
            pil_original = pil_original.convert('RGB')
        print(f"INFO: Original image loaded: {pil_original.size} mode {pil_original.mode}")
    except FileNotFoundError:
        print(f"ERROR: Input image not found at {input_image_path}")
        return False, "Input image not found."
    except UnidentifiedImageError:
        print(f"ERROR: Cannot identify image file. It might be corrupted or unsupported: {input_image_path}")
        return False, "Corrupted or unsupported image file."
    except Exception as e:
        print(f"ERROR: Could not load image: {e}")
        return False, f"Could not load image: {e}"

    valid, msg = initial_photo_validation(pil_original)
    if not valid:
        print(f"ERROR: Initial validation failed: {msg}")
        return False, msg
    print(f"INFO: {msg}")

    pil_subject_on_target_bg = pil_original
    if specifications['background']['allow_ai_background_removal'] and REMBG_AVAILABLE:
        print("INFO: Attempting AI background removal...")
        try:
            img_byte_arr = io.BytesIO()
            pil_original.save(img_byte_arr, format='PNG')
            input_bytes = img_byte_arr.getvalue()

            output_bytes_with_alpha = remove_background_ai(input_bytes, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10)
            
            pil_subject_transparent = Image.open(io.BytesIO(output_bytes_with_alpha)).convert("RGBA")
            print("INFO: AI background removal successful. Subject isolated.")

            bg_color = specifications['background']['color_rgb']
            target_bg = Image.new("RGB", pil_subject_transparent.size, bg_color)
            
            target_bg.paste(pil_subject_transparent, (0,0), pil_subject_transparent)
            pil_subject_on_target_bg = target_bg.convert("RGB")
            print(f"INFO: Subject placed on new background color {bg_color}.")

        except Exception as e:
            print(f"WARNING: AI background removal failed: {e}. Proceeding with original image for background.")
            bg_color = specifications['background']['color_rgb']
            pil_subject_on_target_bg = Image.new("RGB", pil_original.size, bg_color)
            print("WARNING: Using original image due to rembg failure. Background may not be compliant.")
    else:
        print("INFO: AI background removal skipped (not enabled or rembg not available).")
        print("INFO: Current image background will be subject to cropping rules.")

    adjustment_method = specifications['auto_adjustments']['lighting_method']
    if adjustment_method != "none":
        pil_adjusted = auto_adjust_lighting_color(pil_subject_on_target_bg, method=adjustment_method)
    else:
        pil_adjusted = pil_subject_on_target_bg
    print("INFO: Lighting/color adjustments applied.")

    cv_for_detection = pillow_to_cv2_bgr(pil_adjusted)
    print("INFO: Detecting face for precise cropping...")
    face_roi = detect_main_face_advanced(cv_for_detection)
    
    if face_roi is None:
        print("ERROR: No face detected in the adjusted image. Cannot proceed with cropping.")
        print("INFO: Trying face detection on the original image (before adjustments)...")
        face_roi_orig = detect_main_face_advanced(pillow_to_cv2_bgr(pil_original))
        if face_roi_orig:
            print("WARNING: Face found on original but not adjusted. Proceeding with original for cropping basis, but final image will be the adjusted one.")
            return False, "Face detection failed on processed image."
        else:
            return False, "Face detection failed on processed image (and original)."
            
    print(f"INFO: Face detected at (x,y,w,h): {face_roi} in adjusted image.")

    print("INFO: Applying precise cropping and resizing...")
    pil_final_candidate = crop_and_resize_to_spec(pil_adjusted, face_roi, specifications)

    if pil_final_candidate is None:
        print("ERROR: Precise cropping and resizing failed.")
        return False, "Cropping and resizing to specifications failed."
    print(f"INFO: Image cropped and resized to {pil_final_candidate.size}.")

    valid_final, msg_final = final_quality_checks(pil_final_candidate, specifications)
    if not valid_final:
        print(f"WARNING: Final quality checks raised issues: {msg_final}")
    else:
        print(f"INFO: {msg_final}")

    try:
        output_format = specifications['output_format']['type'].upper()
        if output_format == "JPEG":
            pil_final_candidate.save(
                output_image_path,
                "JPEG",
                quality=specifications['output_format']['quality'],
                dpi=specifications['output_format']['dpi']
            )
        elif output_format == "PNG":
             pil_final_candidate.save(
                output_image_path,
                "PNG",
                dpi=specifications['output_format']['dpi']
            )
        else:
            print(f"ERROR: Unsupported output format: {output_format}")
            return False, f"Unsupported output format: {output_format}"
        
        print(f"--- Successfully processed photo saved to: {output_image_path} ---")
        return True, f"Photo processed and saved to {output_image_path}. Issues: {msg_final if not valid_final else 'None'}"

    except Exception as e:
        print(f"ERROR: Failed to save the final image: {e}")
        return False, f"Failed to save final image: {e}"


if __name__ == "__main__":
    sample_input = "1.webp"
    processed_output = "output.jpg"

    if not os.path.exists(sample_input):
        print(f"ERROR: Test input image '{sample_input}' not found.")
        print("Please create this file with a photo of a person (frontal face, clear).")
        exit()

    current_specifications = DEFAULT_SPECIFICATIONS
    current_specifications["document_name"] = "Canada Visa Photo Example"

    success, message = create_compliant_photo(sample_input, processed_output, current_specifications)

    print("\n--- Processing Summary ---")
    print(f"Success: {success}")
    print(f"Message: {message}")

    if success:
        print(f"\nTo view the result, open: {processed_output}")