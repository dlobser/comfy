import numpy as np
from PIL import Image
import os
import time
import json
import requests
import uuid
import random

# Allow large images
Image.MAX_IMAGE_PIXELS = None

class MultiScaleGridInpainterResume:
    def __init__(self, input_image_path, mask_path="mask.png", temp_file="q.png", 
                 comfy_server="127.0.0.1:8188", workflow_path="DifferentialDiffusionForGridAPI.json",
                 progress_save_interval=300, force_rebuild=False):
        """
        Initialize the multi-scale grid inpainter with resume capability and progress saves
        
        Args:
            input_image_path: Path to the 512x512 input image
            mask_path: Path to the 512x512 alpha mask (black and white PNG)
            temp_file: Temporary file name for ComfyUI processing
            comfy_server: ComfyUI server address
            workflow_path: Path to your ComfyUI workflow JSON file
            progress_save_interval: Save work-in-progress every N tiles (default 300)
            force_rebuild: If True, rebuild all steps even if output files exist
        """
        self.input_image_path = input_image_path
        self.mask_path = mask_path
        self.temp_file = temp_file
        self.comfy_server = comfy_server
        self.workflow_path = workflow_path
        self.progress_save_interval = progress_save_interval
        self.force_rebuild = force_rebuild
        
        # Load workflow template
        with open(workflow_path, 'r') as f:
            self.workflow_template = json.load(f)
        
        # Load and prepare the alpha mask
        self.load_alpha_mask()
        
        # Define the scaling pipeline
        self.scale_steps = [
            {
                "name": "1K",
                "target_size": 1024,
                "tile_size": 512,
                "stride": 256,
                "brightness": -8.0,
                "prompt": "extremely detailed, intricate textures, high resolution, sharp focus, photorealistic",
                "output_file": "result_1k.png"
            },
            {
                "name": "2K", 
                "target_size": 2048,
                "tile_size": 512,
                "stride": 256,
                "brightness": -6.5,
                "prompt": "ultra high detail, fine textures, crisp edges, professional photography, masterpiece quality",
                "output_file": "result_2k.png"
            },
            {
                "name": "4K",
                "target_size": 4096,
                "tile_size": 512,
                "stride": 256, 
                "brightness": -5.0,
                "prompt": "hyper detailed, microscopic textures, perfect clarity, award winning photography, 8k quality",
                "output_file": "result_4k.png"
            },
            {
                "name": "8K",
                "target_size": 8192,
                "tile_size": 512,
                "stride": 256,
                "brightness": -3.75,
                "prompt": "incredibly intricate details, fiber-level textures, museum quality, ultra sharp, pristine clarity",
                "output_file": "result_8k.png"
            },
            {
                "name": "16K",
                "target_size": 16384,
                "tile_size": 512,
                "stride": 256,
                "brightness": -2.5,
                "prompt": "maximum detail, every surface perfectly rendered, gallery quality, flawless execution, legendary sharpness",
                "output_file": "result_16k.png"
            }
        ]
        
        print("Multi-Scale Grid Inpainting Pipeline (with Resume & Progress Saves)")
        print("=" * 70)
        print(f"Input image: {input_image_path}")
        print(f"Alpha mask: {mask_path}")
        print(f"Progress saves: Every {progress_save_interval} tiles")
        print(f"Force rebuild: {force_rebuild}")
        print(f"Pipeline steps: {len(self.scale_steps)}")
        
        # Check which steps can be resumed
        self.check_existing_outputs()
    
    def check_existing_outputs(self):
        """Check which output files already exist and can be resumed from"""
        print("\nChecking for existing output files...")
        self.existing_files = {}
        
        for i, step in enumerate(self.scale_steps):
            output_file = step["output_file"]
            if os.path.exists(output_file) and not self.force_rebuild:
                file_size = os.path.getsize(output_file) / (1024*1024)
                print(f"  ✓ Found {step['name']}: {output_file} ({file_size:.1f} MB)")
                self.existing_files[i] = output_file
            else:
                print(f"  ⏸ Missing {step['name']}: {output_file}")
        
        if self.existing_files and not self.force_rebuild:
            last_completed = max(self.existing_files.keys())
            print(f"\n🔄 Can resume from step {last_completed + 2} ({self.scale_steps[last_completed]['name']} → {self.scale_steps[last_completed + 1]['name'] if last_completed + 1 < len(self.scale_steps) else 'Complete'})")
        elif self.force_rebuild:
            print("\n🔥 Force rebuild enabled - will regenerate all steps")
        else:
            print("\n🚀 Starting from the beginning")
    
    def load_alpha_mask(self):
        """Load and prepare the alpha mask"""
        if not os.path.exists(self.mask_path):
            print(f"Warning: Alpha mask '{self.mask_path}' not found!")
            print("Creating a default circular mask...")
            self.create_default_mask()
        
        mask_img = Image.open(self.mask_path)
        if mask_img.mode != 'L':  # Convert to grayscale if needed
            mask_img = mask_img.convert('L')
        
        # Convert to numpy array and normalize to 0-1 range
        self.alpha_mask = np.array(mask_img).astype(np.float32) / 255.0
        print(f"  Loaded alpha mask: {self.alpha_mask.shape}")
    
    def create_default_mask(self):
        """Create a default circular feathered mask if none exists"""
        size = 512
        center = size // 2
        y, x = np.ogrid[:size, :size]
        
        # Create circular mask with feathered edges
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        inner_radius = center * 0.6  # Fully opaque in center
        outer_radius = center * 0.9  # Fully transparent at edge
        
        mask = np.ones((size, size), dtype=np.float32)
        
        # Create feathered edge
        fade_region = (distance > inner_radius) & (distance < outer_radius)
        fade_factor = (outer_radius - distance[fade_region]) / (outer_radius - inner_radius)
        mask[fade_region] = fade_factor
        
        # Fully transparent outside outer radius
        mask[distance >= outer_radius] = 0
        
        # Save as PNG
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_img.save(self.mask_path)
        print(f"  Created default circular mask: {self.mask_path}")
    
    def simple_upscale(self, input_path, target_size, output_path):
        """Simple upscale using PIL"""
        print(f"  Upscaling {input_path} to {target_size}x{target_size}")
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        upscaled = img.resize((target_size, target_size), Image.LANCZOS)
        upscaled.save(output_path)
        print(f"  Saved upscaled image: {output_path}")
        return output_path
    
    def calculate_grid_positions(self, width, height, tile_size, stride):
        """Calculate all the (x, y) positions for the grid"""
        positions = []
        
        y = 0
        while y + tile_size <= height:
            x = 0
            while x + tile_size <= width:
                positions.append((x, y))
                x += stride
            y += stride
        
        return positions
    
    def extract_tile(self, canvas, x, y, tile_size):
        """Extract a tile from the canvas"""
        tile = canvas[y:y+tile_size, x:x+tile_size]
        return Image.fromarray(tile)
    
    def insert_tile_with_alpha(self, canvas, x, y, tile_size):
        """Insert a processed tile back into the canvas with alpha blending"""
        output_dir = r"C:\_Main\ai\ComfyOutput"
        
        import glob
        pattern = os.path.join(output_dir, "SavedGrid_*.png")
        files = glob.glob(pattern)
        
        if not files:
            print(f"    No SavedGrid output files found in {output_dir}")
            return False
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getctime)
        
        processed_image = Image.open(latest_file)
        if processed_image.mode != 'RGB':
            processed_image = processed_image.convert('RGB')
        processed_array = np.array(processed_image).astype(np.float32)
        
        # Get the region from the current canvas
        canvas_region = canvas[y:y+tile_size, x:x+tile_size].astype(np.float32)
        
        # Expand alpha mask to 3 channels (RGB)
        alpha_3d = np.stack([self.alpha_mask] * 3, axis=2)
        
        # Alpha blend: result = old * (1 - alpha) + new * alpha
        blended = canvas_region * (1.0 - alpha_3d) + processed_array * alpha_3d
        
        # Convert back to uint8 and insert into canvas
        canvas[y:y+tile_size, x:x+tile_size] = blended.astype(np.uint8)
        
        return True
    
    def save_progress(self, canvas, step_name, tile_index):
        """Save work-in-progress image"""
        progress_filename = f"progress_{step_name.lower()}_{tile_index:04d}.png"
        progress_image = Image.fromarray(canvas)
        progress_image.save(progress_filename)
        print(f"    💾 Progress saved: {progress_filename}")
    
    def run_comfy_api(self, prompt_text, brightness_value):
        """Run ComfyUI API with specified prompt and brightness"""
        try:
            # Create a copy of the workflow template
            workflow = self.workflow_template.copy()
            
            # Update the image path in node 14
            abs_temp_path = os.path.abspath(self.temp_file)
            workflow["14"]["inputs"]["image"] = f'"{abs_temp_path}"'
            
            # Update the prompt in node 7
            workflow["7"]["inputs"]["text"] = prompt_text
            
            # Update brightness in node 11
            workflow["11"]["inputs"]["brightness"] = brightness_value
            
            # Randomize the seed
            random_seed = random.randint(1, 999999999)
            workflow["6"]["inputs"]["seed"] = random_seed
            
            # Generate unique client ID
            client_id = str(uuid.uuid4())
            
            # Queue the prompt
            prompt_data = {
                "prompt": workflow,
                "client_id": client_id
            }
            
            response = requests.post(f"http://{self.comfy_server}/prompt", json=prompt_data)
            
            if response.status_code != 200:
                print(f"    Error queuing prompt: {response.status_code}")
                return False
            
            result = response.json()
            prompt_id = result.get("prompt_id")
            
            if not prompt_id:
                print("    No prompt_id returned")
                return False
            
            # Wait for completion
            return self._wait_for_completion(prompt_id)
            
        except Exception as e:
            print(f"    Error running ComfyUI API: {e}")
            return False
    
    def _wait_for_completion(self, prompt_id, timeout=300):
        """Wait for ComfyUI to complete processing"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{self.comfy_server}/history/{prompt_id}")
                
                if response.status_code == 200:
                    history = response.json()
                    
                    if prompt_id in history:
                        prompt_history = history[prompt_id]
                        
                        if "outputs" in prompt_history:
                            return True
                
                time.sleep(1)
                
            except Exception as e:
                print(f"    Error checking completion: {e}")
                time.sleep(1)
        
        print(f"    Timeout waiting for prompt {prompt_id}")
        return False
    
    def process_scale_step(self, input_image_path, step_config):
        """Process a single scale step"""
        step_name = step_config["name"]
        target_size = step_config["target_size"]
        tile_size = step_config["tile_size"]
        stride = step_config["stride"]
        brightness = step_config["brightness"]
        prompt = step_config["prompt"]
        output_file = step_config["output_file"]
        
        print(f"\nProcessing {step_name} ({target_size}x{target_size}) with Alpha Blending & Progress Saves")
        print("-" * 70)
        print(f"Brightness: {brightness}")
        print(f"Prompt: {prompt[:60]}...")
        print(f"Progress saves: Every {self.progress_save_interval} tiles")
        
        # Step 1: Simple upscale to target size
        upscaled_path = f"temp_upscaled_{step_name.lower()}.png"
        self.simple_upscale(input_image_path, target_size, upscaled_path)
        
        # Step 2: Load upscaled image for grid processing
        image = Image.open(upscaled_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        canvas = np.array(image)
        height, width = canvas.shape[:2]
        
        # Step 3: Calculate grid positions
        grid_positions = self.calculate_grid_positions(width, height, tile_size, stride)
        total_tiles = len(grid_positions)
        print(f"Grid processing: {total_tiles} tiles ({tile_size}x{tile_size}, stride {stride}) with alpha blending")
        
        # Step 4: Process each tile
        for i, (x, y) in enumerate(grid_positions):
            print(f"  Tile {i+1}/{total_tiles} at ({x}, {y})", end=" ")
            
            # Extract tile
            tile = self.extract_tile(canvas, x, y, tile_size)
            tile.save(self.temp_file)
            
            # Process with ComfyUI
            success = self.run_comfy_api(prompt, brightness)
            if not success:
                print("FAILED")
                continue
            
            # Insert processed tile back with alpha blending
            success = self.insert_tile_with_alpha(canvas, x, y, tile_size)
            if success:
                print("✓", end="")
            else:
                print("FAILED BLEND")
                continue
            
            # Save progress every N tiles
            if (i + 1) % self.progress_save_interval == 0:
                self.save_progress(canvas, step_name, i + 1)
                print(" [PROGRESS SAVED]")
            else:
                print()
        
        # Step 5: Save final result for this scale
        final_image = Image.fromarray(canvas)
        final_image.save(output_file)
        print(f"✅ {step_name} completed with alpha blending: {output_file}")
        
        # Clean up temp upscaled file
        if os.path.exists(upscaled_path):
            os.remove(upscaled_path)
        
        return output_file
    
    def run_full_pipeline(self):
        """Run the complete multi-scale pipeline with resume capability"""
        print("\nStarting Multi-Scale Pipeline with Resume & Progress Saves...")
        print("=" * 70)
        
        # Determine starting point
        if self.existing_files and not self.force_rebuild:
            # Find the last completed step
            last_completed_index = max(self.existing_files.keys())
            start_index = last_completed_index + 1
            current_input = self.existing_files[last_completed_index]
            
            if start_index >= len(self.scale_steps):
                print("🎉 All steps already completed!")
                return
            
            print(f"🔄 Resuming from step {start_index + 1} ({self.scale_steps[start_index]['name']})")
            print(f"📁 Using existing output as input: {current_input}")
        else:
            # Start from the beginning
            start_index = 0
            current_input = self.input_image_path
            print(f"🚀 Starting from the beginning with: {current_input}")
        
        # Process remaining steps
        for i in range(start_index, len(self.scale_steps)):
            step_config = self.scale_steps[i]
            
            # Skip if exists and not forcing rebuild
            if i in self.existing_files and not self.force_rebuild:
                print(f"\n⏭️  Skipping {step_config['name']} - already exists: {step_config['output_file']}")
                current_input = step_config["output_file"]
                continue
            
            step_start_time = time.time()
            
            # Process this scale step
            result_file = self.process_scale_step(current_input, step_config)
            
            # The output becomes input for the next step
            current_input = result_file
            
            step_duration = time.time() - step_start_time
            print(f"Step {i+1} completed in {step_duration/60:.1f} minutes")
            
            # Clean up temp file
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
        
        print("\n" + "=" * 70)
        print("🎉 FULL PIPELINE COMPLETED!")
        print("Results:")
        for step in self.scale_steps:
            if os.path.exists(step["output_file"]):
                file_size = os.path.getsize(step["output_file"]) / (1024*1024)
                print(f"  {step['name']}: {step['output_file']} ({file_size:.1f} MB)")


def main():
    # Configuration
    input_image = "input_512.png"  # Your 512x512 starting image
    mask_image = "mask.png"        # Your 512x512 alpha mask
    workflow_json = "DifferentialDiffusionForGridAPI.json"
    
    # Advanced options
    progress_save_interval = 300   # Save progress every N tiles
    force_rebuild = False          # Set to True to regenerate all steps
    
    # Verify input file exists
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found!")
        print("Please place your 512x512 starting image in the current directory.")
        return
    
    # Create the multi-scale inpainter with resume capability
    pipeline = MultiScaleGridInpainterResume(
        input_image_path=input_image,
        mask_path=mask_image,
        temp_file="q.png",
        comfy_server="127.0.0.1:8188",
        workflow_path=workflow_json,
        progress_save_interval=progress_save_interval,
        force_rebuild=force_rebuild
    )
    
    # Run the complete pipeline
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()