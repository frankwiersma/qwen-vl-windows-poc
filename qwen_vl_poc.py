from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os
import shutil

# Print if using GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model with lower precision to reduce memory usage
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,  # Use float16 instead of auto
    device_map="auto",
    low_cpu_mem_usage=True
)

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Get list of jpg files from pics folder
pics_folder = "./pics"
jpg_files = [f for f in os.listdir(pics_folder) if f.endswith('.jpg')]

for jpg_file in jpg_files:
    try:
        # Load and resize image to reduce memory usage
        image_path = os.path.join(pics_folder, jpg_file)
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))  # Resize to smaller dimensions

        # Define input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe what you see in this image in exactly 5 words"}
                ]
            }
        ]

        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(device)

        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate description with reduced tokens
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,  # Reduced from 128
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract just the model's response
        response_start = description.rfind("assistant")  # Use rfind to get last occurrence
        if response_start != -1:
            description = description[response_start + 9:].strip()
            
            # Find the actual response after any potential system/user text
            # Look for common endings of the prompt
            prompt_ends = [
                "in exactly 5 words",
                "Describe what you see",
                "in this image"
            ]
            
            for prompt_end in prompt_ends:
                idx = description.find(prompt_end)
                if idx != -1:
                    description = description[idx + len(prompt_end):].strip()
            
            # Split into words and take first 5 only
            words = [word for word in description.split() if len(word) > 1][:5]  # Filter out single chars
            if words:  # Only proceed if we have actual words
                description = ' '.join(words)
            else:
                raise Exception("No valid description generated")
        
        # Create new filename from description (clean up description for filename)
        new_filename = description.strip().lower()
        new_filename = ''.join(c if c.isalnum() else '_' for c in new_filename)
        new_filename = new_filename[:50]  # Limit length
        
        # Add counter if filename already exists
        counter = 1
        base_filename = new_filename
        while os.path.exists(os.path.join(pics_folder, new_filename + '.jpg')):
            new_filename = f"{base_filename}_{counter}"
            counter += 1
        new_filename += '.jpg'
        
        # Rename file
        new_filepath = os.path.join(pics_folder, new_filename)
        shutil.move(image_path, new_filepath)
        print(f"Renamed {jpg_file} to {new_filename}")

        # Clear CUDA cache after processing each image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error processing {jpg_file}: {str(e)}")
        continue
