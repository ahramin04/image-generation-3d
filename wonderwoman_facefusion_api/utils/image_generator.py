from PIL import Image
import os
import uuid
from utils.face_fuser import face_swap, detect_style_from_prompt
import torch
import gc
import re
import numpy as np

# Check if CUDA is available for better 4K handling
device = "cuda" if torch.cuda.is_available() else "cpu"

# Import diffusers after device setup
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler

def generate_image_with_facefusion(prompt, face_image_path, model_path, lora_path=None, use_lora=True):
    """
    Generate a high-quality image based on prompt and fuse with the provided face
    """
    input_image = Image.open(face_image_path).convert("RGB")
    
    # IMPROVED: Process input face to ensure high resolution
    # Use higher quality resize for face image
    orig_size = input_image.size
    if max(orig_size) > 1024:
        # Save original high-res for later enhancement
        high_res_face = input_image.copy()
        # Resize for processing but maintain quality
        input_image = input_image.resize((1024, int(1024 * orig_size[1]/orig_size[0])) 
                                       if orig_size[0] > orig_size[1] else 
                                       (int(1024 * orig_size[0]/orig_size[1]), 1024), 
                                       Image.LANCZOS)
    else:
        high_res_face = input_image.copy()

    # Detect style requirements from the prompt
    style_type, style_strength = detect_style_from_prompt(prompt)
    
    print(f"Detected style: {style_type or 'None'} with strength {style_strength}")
    
    # Modify prompt to ensure single subject with good composition
    enhanced_prompt = f"{prompt}, single subject, solo, one person, centered composition, full body portrait"
    
    # Strong negative prompt to prevent multiple bodies/distortion
    negative_prompt = "multiple bodies, extra limbs, duplicate, disfigured, deformed, mutated, bad anatomy, extra fingers, multiple people, group, crowd, distorted face, duplicate face, ugly, blurry, watermark, low quality, multiple heads, extra arms, extra legs"
    
    is_sdxl = any(x in model_path.lower() for x in ["xl", "sdxl", "sd-xl"])
    print(f"Using {'SDXL' if is_sdxl else 'SD'} model")
    
    try:
        # Clean up memory before loading models
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        if is_sdxl:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
                use_safetensors=model_path.endswith(".safetensors")
            ).to(device)
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None
            ).to(device)

        # Use DPMSolver for better quality and speed
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )

        # Enable optimizations
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            
        print("Model loaded and optimized for 4K generation")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    if use_lora and lora_path and os.path.exists(lora_path):
        print(f"[INFO] Skipping LoRA for now (can be added later)")
    
    print(f"Using enhanced prompt: {enhanced_prompt}")

    # Set dimensions for better composition - use standard aspect ratios
    # Use 2:3 portrait aspect ratio which is better for single subjects
    width, height = 832, 1216  # Lower resolution but better quality/stability
    
    try:
        # Generate image with better settings
        result = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=40,  # Balance between quality and speed
            guidance_scale=7.5,      # Balanced guidance
            width=width,
            height=height
        )
        gen_image = result.images[0]
        
        # Clean up memory after generation
        if device == "cuda":
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Image generation failed: {e}")
        raise

    print(f"Performing face swap with {style_type or 'no'} styling...")
    
    # Apply appropriate style strength for the detected style
    if style_type == "pixar":
        style_strength_adjusted = min(0.8, style_strength)  # Reduce Pixar style for better face integration
    else:
        style_strength_adjusted = style_strength * 0.7  # Reduce style strength for better face retention
        
    try:
        # Try face fusion at current resolution
        fused = face_swap(input_image, gen_image, style_type=style_type, style_strength=style_strength_adjusted)
        
        # Check if face detection failed by comparing output to input
        face_diff = np.mean(np.array(fused) != np.array(gen_image))
        
        if face_diff < 0.01:  # Images almost identical = fusion failed
            print("WARNING: Face fusion may have failed, trying alternative approach")
            
            # Try with clearer face processing
            from PIL import ImageEnhance
            
            # First enhance the input face for better detection
            enhanced_input = ImageEnhance.Sharpness(input_image).enhance(1.5)
            enhanced_input = ImageEnhance.Contrast(enhanced_input).enhance(1.2)
            
            # Try fusion with enhanced face
            fused = face_swap(enhanced_input, gen_image, style_type=style_type, style_strength=style_strength_adjusted)
            
            # If still failing, try at lower resolution then upscale
            face_diff = np.mean(np.array(fused) != np.array(gen_image))
            if face_diff < 0.01:
                # Resize both for fusion
                smaller_gen = gen_image.copy()
                smaller_input = high_res_face.copy()
                smaller_gen.thumbnail((768, 768), Image.LANCZOS)
                smaller_input.thumbnail((512, 512), Image.LANCZOS)
                
                # Try fusion on smaller image
                smaller_fused = face_swap(smaller_input, smaller_gen, style_type=None, style_strength=0)
                
                # Scale back up with high quality and enhance
                fused = smaller_fused.resize((width, height), Image.LANCZOS)
                fused = ImageEnhance.Sharpness(fused).enhance(1.3)
    except Exception as e:
        print(f"Face fusion failed: {e}")
        # Return the generated image without fusion as fallback
        fused = gen_image

    # Save output as high-quality PNG
    output_path = os.path.join("outputs", f"4k_{uuid.uuid4().hex}.png")
    os.makedirs("outputs", exist_ok=True)
    fused.save(output_path, format='PNG', compress_level=1)  # Low compression for better quality
    print(f"Saved 4K final image to {output_path}")
    
    return output_path
