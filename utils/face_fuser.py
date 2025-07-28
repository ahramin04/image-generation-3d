import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import torch
import os

# Initialize face analysis with higher detection size for 4K
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(1024, 1024))  # Increased from 640x640 for better 4K detection
swapper = get_model("models/inswapper_128.onnx", download=False)


def detect_style_from_prompt(prompt):
    """
    Detect requested style from the prompt
    Returns: style_name, style_strength
    """
    prompt = prompt.lower()
    
    # Enhanced style mappings with strong pixar detection
    style_mappings = {
        "pixar": {
            "keywords": ["pixar", "3d cartoon", "3d animated", "pixar-style", "pixar character", "animation studio"],
            "strength": 0.95,
        },
        "anime": {
            "keywords": ["anime", "manga", "japanese animation"],
            "strength": 0.85,
        },
        "cartoon": {
            "keywords": ["cartoon", "toon", "animated", "disney"],
            "strength": 0.8,
        },
        "painting": {
            "keywords": ["painting", "oil painting", "acrylic", "watercolor"],
            "strength": 0.75,
        },
    }
    
    # Check for style matches
    for style, config in style_mappings.items():
        if any(keyword in prompt for keyword in config["keywords"]):
            return style, config["strength"]
    
    # Check for expressive eyes which is a Pixar characteristic
    if "expressive eyes" in prompt and any(word in prompt for word in ["animated", "character"]):
        return "pixar", 0.9
        
    # Check for general styling terms
    general_style_keywords = ["style", "styled", "stylized", "artistic"]
    if any(keyword in prompt for keyword in general_style_keywords):
        return "general", 0.7
        
    return None, 0.0


def match_clarity(source_face, target_face):
    """
    Match the clarity of source face to target face - enhanced for 4K
    """
    # Convert to LAB color space to manipulate clarity without affecting colors
    source_lab = cv2.cvtColor(source_face, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB)
    
    # Extract L channel (luminance/clarity)
    source_l = source_lab[:,:,0]
    target_l = target_lab[:,:,0]
    
    # Calculate average clarity metrics
    source_clarity = np.mean(cv2.Laplacian(source_l, cv2.CV_64F).var())
    target_clarity = np.mean(cv2.Laplacian(target_l, cv2.CV_64F).var())
    
    # Calculate adjustment factor - enhanced for better clarity
    adjustment_factor = (target_clarity / max(source_clarity, 1e-5)) * 1.2  # Increased for better match
    adjustment_factor = min(max(adjustment_factor, 0.6), 1.8)  # Extended adjustment range
    
    # Apply clarity adjustment - optimized for 4K
    adjusted_source = source_face.copy()
    if adjustment_factor > 1:
        # Increase sharpness - improved for 4K detail with multiple passes
        adjusted_source = cv2.detailEnhance(adjusted_source, 
                                          sigma_s=1.5,  # Reduced for finer detail 
                                          sigma_r=0.15 * (adjustment_factor - 0.8))
        # Second pass for even finer details
        adjusted_source = cv2.detailEnhance(adjusted_source,
                                          sigma_s=1.0,
                                          sigma_r=0.10)
    else:
        # Decrease sharpness - adaptive to 4K
        kernel_size = int(3 * (1 - adjustment_factor) + 1)  # Smaller kernel for better detail
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        adjusted_source = cv2.GaussianBlur(adjusted_source, (kernel_size, kernel_size), 0)
    
    return adjusted_source


def apply_style(img, style_type, strength):
    """Apply style based on detected style type - enhanced for 4K detail"""
    if strength <= 0:
        return img
        
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if style_type == "pixar":
        # Enhanced Pixar style for 4K - stronger effect with better detail preservation
        pil_img = ImageEnhance.Color(pil_img).enhance(1.3)  # More vibrant for 4K
        pil_img = ImageEnhance.Brightness(pil_img).enhance(1.15)  # Brighter
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)  # Higher contrast for 3D look
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=150))  # Better detail for 4K
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)  # More defined features
    
    elif style_type == "anime":
        # Enhanced anime style for 4K
        pil_img = ImageEnhance.Color(pil_img).enhance(1.4)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3) 
        pil_img = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.6)  # Sharper lines for 4K
        
    elif style_type == "cartoon":
        # Enhanced cartoon style for 4K
        pil_img = ImageEnhance.Color(pil_img).enhance(1.25)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.25)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5 * strength))
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)  # Higher sharpness for 4K
        
    else:
        # General style enhancement for 4K
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.0 + 0.2 * strength)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.6 * strength))
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.0 + 0.3 * strength)  # Better sharpness for 4K
        pil_img = ImageEnhance.Color(pil_img).enhance(1.0 + 0.15 * strength)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def seamless_blend(face_area, target_img, face_mask):
    """
    Create a seamless blend between face and target image with multiple techniques
    """
    # Create a refined mask with feathered edges
    refined_mask = cv2.GaussianBlur(face_mask, (15, 15), 5)
    
    # Try multiple blending techniques and pick the best one
    try:
        # First try Poisson blending (best quality but can fail)
        center = (face_area.shape[1] // 2, face_area.shape[0] // 2)
        poisson_blend = cv2.seamlessClone(
            face_area, 
            target_img, 
            refined_mask, 
            center, 
            cv2.NORMAL_CLONE
        )
        
        # Save debug outputs
        cv2.imwrite("debug/poisson_blend.jpg", poisson_blend)
        return poisson_blend
        
    except Exception as e:
        print(f"Poisson blending failed: {e}, trying alpha blending")
        
        # Fallback to alpha blending
        mask_3ch = np.stack([refined_mask, refined_mask, refined_mask], axis=2) / 255.0
        alpha_blend = (face_area * mask_3ch + target_img * (1 - mask_3ch)).astype(np.uint8)
        
        cv2.imwrite("debug/alpha_blend.jpg", alpha_blend)
        return alpha_blend


def process_in_tiles(func, img, *args, **kwargs):
    """
    Process large 4K images in tiles to avoid memory issues
    """
    # For smaller images, process normally
    if img.shape[0] * img.shape[1] < 1920*1080:
        return func(img, *args, **kwargs)
        
    # For large images, we'll use a memory-efficient approach
    # This is a placeholder - actual implementation would involve splitting the image
    # into tiles, processing each tile, and then recombining them
    
    # For now, we'll just resize if too large, process, then resize back
    h, w = img.shape[:2]
    if h > 2160 or w > 3840:
        scale_factor = min(2160/h, 3840/w)
        resized_img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
        processed = func(resized_img, *args, **kwargs)
        return cv2.resize(processed, (w, h))
    else:
        return func(img, *args, **kwargs)


def face_swap(source_img, target_img, style_type=None, style_strength=0.0):
    """
    Swap face from source image to target image with enhanced processing
    
    Parameters:
    - source_img: PIL Image with the face to use
    - target_img: PIL Image where the face will be inserted
    - style_type: Type of style to apply (pixar, anime, etc.)
    - style_strength: 0.0 = exact input face, 1.0 = fully styled face
    
    Returns:
    - PIL Image with the swapped face
    """
    # Convert PIL to OpenCV format
    source_cv = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
    target_cv = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    # Create debug directory
    os.makedirs("debug", exist_ok=True)
    
    # Better face detection with debug output
    print("Detecting faces in source image...")
    source_faces = app.get(source_cv)
    cv2.imwrite("debug/source_image.jpg", source_cv)
    print(f"Found {len(source_faces)} faces in source image")
    
    print("Detecting faces in target image...")
    target_faces = app.get(target_cv)
    cv2.imwrite("debug/target_image.jpg", target_cv)
    print(f"Found {len(target_faces)} faces in target image")
    
    # Debug visualization 
    if len(source_faces) > 0:
        debug_source = source_cv.copy()
        face = source_faces[0]
        bbox = face.bbox.astype(np.int32)
        cv2.rectangle(debug_source, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imwrite("debug/source_face_detected.jpg", debug_source)
    
    if len(target_faces) > 0:
        debug_target = target_cv.copy()
        for i, face in enumerate(target_faces):
            bbox = face.bbox.astype(np.int32)
            cv2.rectangle(debug_target, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(debug_target, f"Face #{i+1}", (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite("debug/target_faces_detected.jpg", debug_target)
    
    # Select the largest (main) face if multiple faces are detected
    if len(target_faces) > 1:
        print("Multiple faces detected in target, using the largest face")
        largest_idx = 0
        largest_size = 0
        for i, face in enumerate(target_faces):
            bbox = face.bbox.astype(np.int32)
            size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if size > largest_size:
                largest_size = size
                largest_idx = i
        target_faces = [target_faces[largest_idx]]
        print(f"Selected face #{largest_idx+1} as the main face")
    
    if len(source_faces) == 0:
        print("ERROR: No face detected in source image. Try a clearer face photo.")
        return target_img
        
    if len(target_faces) == 0:
        print("ERROR: No face detected in target image. Regenerating image might help.")
        return target_img
    
    try:
        # Apply style to source face if requested (memory-optimized)
        if style_strength > 0 and style_type:
            source_cv = apply_style(source_cv, style_type, style_strength)
        
        # Extract face areas for clarity matching
        source_face = source_faces[0]
        target_face = target_faces[0]
        
        # Match clarity between faces
        src_bbox = source_face.bbox.astype(np.int32)
        tgt_bbox = target_face.bbox.astype(np.int32)
        
        # Ensure bounding box coordinates are valid
        src_bbox = [max(0, int(coord)) for coord in src_bbox]
        tgt_bbox = [max(0, int(coord)) for coord in tgt_bbox]
        
        # Extract face regions
        source_face_region = source_cv[
            src_bbox[1]:src_bbox[3], 
            src_bbox[0]:src_bbox[2]
        ]
        target_face_region = target_cv[
            tgt_bbox[1]:tgt_bbox[3], 
            tgt_bbox[0]:tgt_bbox[2]
        ]
        
        # Skip if any region is empty
        if source_face_region.size > 0 and target_face_region.size > 0:
            # Match clarity - optimized for 4K
            source_cv = match_clarity(source_cv, target_cv)
        
        # For very large images, use memory-efficient batched processing
        if torch.cuda.is_available() and target_cv.shape[0] * target_cv.shape[1] > 2048*2048:
            # Free up CUDA memory before the intensive operation
            torch.cuda.empty_cache()
        
        # Perform the face swap with clarity-matched face
        swapped = swapper.get(target_cv, target_faces[0], source_faces[0], paste_back=True)
        
        # Create a mask for the swapped face area - improved for 4K detail
        mask = np.zeros_like(swapped[:,:,0])
        face_landmarks = target_face.landmark_2d_106
        hull = cv2.convexHull(face_landmarks.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Dilate mask to cover face edges - enhanced for 4K
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)  # Reduced dilation
        
        # Blur the edges of the mask - smoother for 4K
        mask = cv2.GaussianBlur(mask, (13, 13), 5)  # Improved mask edge
        
        # Extract face area and create 3-channel mask
        mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
        
        # Apply additional face enhancement to ensure high resolution
        face_area = swapped * (mask_3ch > 0.05)
        enhanced_face = cv2.detailEnhance(face_area, sigma_s=1.0, sigma_r=0.15)
        swapped = swapped * (1.0 - mask_3ch) + enhanced_face * mask_3ch
        
        # Blend the swapped face with the target image
        blended = swapped * mask_3ch + target_cv * (1 - mask_3ch)
        
        # Apply final 4K enhancement specifically to face region
        face_region_mask = (mask_3ch > 0.05).astype(np.float32)
        blended_face = blended * face_region_mask
        enhanced_face_region = cv2.detailEnhance(blended_face, sigma_s=0.8, sigma_r=0.2)
        blended = blended * (1.0 - face_region_mask) + enhanced_face_region
        
        final = Image.fromarray(cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2RGB))
        return final
        
    except Exception as e:
        print(f"Face swap failed: {e}")
        return target_img
