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

class MultiScaleGridInpainter:
    def __init__(self, input_image_path, temp_file="q.png", 
                 comfy_server="127.0.0.1:8188", workflow_path="DifferentialDiffusionForGridAPI.json"):
        """
        Initialize the multi-scale grid inpainter
        
        Args:
            input_image_path: Path to the 512x512 input image
            temp_file: Temporary file name for ComfyUI processing
            comfy_server: ComfyUI server address
            workflow_path: Path to your ComfyUI workflow JSON file
        """
        self.input_image_path = input_image_path
        self.temp_file = temp_file
        self.comfy_server = comfy_server
        self.workflow_path = workflow_path
        
        # Load workflow template
        with open(workflow_path, 'r') as f:
            self.workflow_template = json.load(f)
        
        # Define the scaling pipeline
        self.scale_steps = [
            {
                "name": "1K",
                "target_size": 1024,
                "tile_size": 512,
                "stride": 256,
                "brightness": -6.5,
                "prompt": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with (leaves:1.2) made of (birds) and (beetles), outsider art in the style of Hieronymus Bosch",
                "output_file": "result_1k.png"
            },
            {
                "name": "2K", 
                "target_size": 2048,
                "tile_size": 512,
                "stride": 256,
                "brightness": -6,
                "prompt": "vast abstract landscape, a photograph, (leaves:1.2) made of (birds) and (beetles), outsider art in the style of Hieronymus Bosch",
                "output_file": "result_2k.png"
            },
            {
                "name": "4K",
                "target_size": 4096,
                "tile_size": 512,
                "stride": 256, 
                "brightness": -5.5,
                "prompt": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with (leaves:1.2) made of (birds) and (beetles), outsider art in the style of Hieronymus Bosch, extremely intricate and detailed, tree with gnarled bark made of flock of hundreds of birds, starlings, crows, mandelbulb fractal bird landscape made of feathers, wax, paper cutout, and bones by Ernst Haeckel, beautiful oil painting by hieronymous bosch",
                "output_file": "result_4k.png"
            },
            {
                "name": "8K",
                "target_size": 8192,
                "tile_size": 512,
                "stride": 256,
                "brightness": -4.5,
                "prompt": "extremely intricate and detailed, a giant tree with gnarled bark with leaves made of peacock feathers in a forest made of flock of hundreds of birds, with a giant (rainbow:1.2), made of wax, paper cutout,  beautiful oil painting by hieronymous bosch",
                "output_file": "result_8k.png"
            },
            {
                "name": "16K",
                "target_size": 16384,
                "tile_size": 512,
                "stride": 256,
                "brightness": -3.5,
                "prompt": "extremely intricate and detailed, tree with gnarled bark made of flock of hundreds of birds, starlings, crows, mandelbulb fractal bird landscape made of feathers, wax, paper cutout, and bones by Ernst Haeckel, beautiful oil painting by hieronymous bosch",
                "output_file": "result_16k.png"
            }
        ]
        
        print("Multi-Scale Grid Inpainting Pipeline")
        print("=" * 50)
        print(f"Input image: {input_image_path}")
        print(f"Pipeline steps: {len(self.scale_steps)}")
        for i, step in enumerate(self.scale_steps):
            print(f"  {i+1}. {step['name']} ({step['target_size']}x{step['target_size']}) - Brightness: {step['brightness']}")
    
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
    
    def insert_tile(self, canvas, x, y, tile_size):
        """Insert a processed tile back into the canvas"""
        output_dir = r"C:\_Main\ai\ComfyOutput"
        
        import glob
        pattern = os.path.join(output_dir, "SavedGrid_*.png")
        files = glob.glob(pattern)
        
        if not files:
            print(f"    No SavedGrid output files found in {output_dir}")
            return False
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getctime)
        print(f"    Loading processed result: {os.path.basename(latest_file)}")
        
        processed_image = Image.open(latest_file)
        if processed_image.mode != 'RGB':
            processed_image = processed_image.convert('RGB')
        processed_array = np.array(processed_image)
        
        # Insert the processed tile back into the canvas
        canvas[y:y+tile_size, x:x+tile_size] = processed_array
        return True
    
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
        
        print(f"\nProcessing {step_name} ({target_size}x{target_size})")
        print("-" * 40)
        print(f"Brightness: {brightness}")
        print(f"Prompt: {prompt[:60]}...")
        
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
        print(f"Grid processing: {total_tiles} tiles ({tile_size}x{tile_size}, stride {stride})")
        
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
            
            # Insert processed tile back
            success = self.insert_tile(canvas, x, y, tile_size)
            if success:
                print("✓")
            else:
                print("FAILED INSERT")
                continue
        
        # Step 5: Save final result for this scale
        final_image = Image.fromarray(canvas)
        final_image.save(output_file)
        print(f"✓ {step_name} completed: {output_file}")
        
        # Clean up temp upscaled file
        if os.path.exists(upscaled_path):
            os.remove(upscaled_path)
        
        return output_file
    
    def run_full_pipeline(self):
        """Run the complete multi-scale pipeline"""
        print("\nStarting Multi-Scale Pipeline...")
        print("=" * 50)
        
        current_input = self.input_image_path
        
        for i, step_config in enumerate(self.scale_steps):
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
        
        print("\n" + "=" * 50)
        print("🎉 FULL PIPELINE COMPLETED!")
        print("Results:")
        for step in self.scale_steps:
            if os.path.exists(step["output_file"]):
                file_size = os.path.getsize(step["output_file"]) / (1024*1024)
                print(f"  {step['name']}: {step['output_file']} ({file_size:.1f} MB)")


def main():
    # Configuration
    input_image = "input_512.png"  # Your 512x512 starting image
    workflow_json = "DifferentialDiffusionForGridAPI.json"
    
    # Verify input file exists
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found!")
        print("Please place your 512x512 starting image in the current directory.")
        return
    
    # Create the multi-scale inpainter
    pipeline = MultiScaleGridInpainter(
        input_image_path=input_image,
        temp_file="q.png",
        comfy_server="127.0.0.1:8188",
        workflow_path=workflow_json
    )
    
    # Run the complete pipeline
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()