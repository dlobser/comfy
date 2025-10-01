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

class MultiScaleGridInpainterResumeColorGuided:
    def __init__(self, input_image_path, color_map_path, mask_path="mask.png", temp_file="q.png", 
                 comfy_server="127.0.0.1:8188", workflow_path="DifferentialDiffusionAPIWCtrlNet.json",
                 progress_save_interval=300, force_rebuild=False, output_dir="output"):
        """
        Initialize the multi-scale grid inpainter with color guidance, resume capability and progress saves
        
        Args:
            input_image_path: Path to the 512x512 input image
            color_map_path: Path to the 512x512 color guide image
            mask_path: Path to the 512x512 alpha mask (black and white PNG)
            temp_file: Temporary file name for ComfyUI processing
            comfy_server: ComfyUI server address
            workflow_path: Path to your ComfyUI workflow JSON file
            progress_save_interval: Save work-in-progress every N tiles (default 300)
            force_rebuild: If True, rebuild all steps even if output files exist
            output_dir: Subdirectory where all output images will be saved
        """
        self.input_image_path = input_image_path
        self.color_map_path = color_map_path
        self.mask_path = mask_path
        self.temp_file = temp_file
        self.comfy_server = comfy_server
        self.workflow_path = workflow_path
        self.progress_save_interval = progress_save_interval
        self.force_rebuild = force_rebuild
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        
        # Load workflow template
        with open(workflow_path, 'r') as f:
            self.workflow_template = json.load(f)
        
        # Load and prepare the alpha mask
        self.load_alpha_mask()
        
        # Load the color map (512x512)
        self.load_color_map()
        
        # Define color-to-variable mapping
        # Each color map can define multiple variables that will be replaced in prompts
        # Format: (R, G, B): {"variable_name": "replacement_value"}
        self.color_variables = {
            (255, 0, 0): {"thing": "birds", "texture": "feathers"},
            (255, 255,  0): {"thing": "tree trunks", "texture": "gnarled spiral swirls of mandelbulb fractals"},
            (0, 255,  0): {"thing": "leaves", "texture": "scraps of paper"},
            (0, 255, 255): {"thing": "grass and plants", "texture": "tiny little beads"},
            (0, 0,255): {"thing": "giant flowers", "texture": "wax, wire, and felt"},
        }
        
        # Default variable values if no color match is found
        self.default_variables = {"material": "natural materials", "texture": "detailed"}
        
        # Define the scaling pipeline with prompts containing {variables}
        # Output files now include the output_dir path
        self.scale_steps = [
            {
                "name": "1K",
                "target_size": 1024,
                "tile_size": 512,
                "stride": 256,
                "brightness": -8.5,
                "prompt": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with ({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "output_file": os.path.join(output_dir, "result_1k.png")
            },
            {
                "name": "2K", 
                "target_size": 2048,
                "tile_size": 512,
                "stride": 256,
                "brightness": -7,
                "prompt": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with ({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "output_file": os.path.join(output_dir, "result_2k.png")
            },
            {
                "name": "4K",
                "target_size": 4096,
                "tile_size": 512,
                "stride": 256, 
                "brightness": -5,
                "prompt": "({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "output_file": os.path.join(output_dir, "result_4k.png")
            },
            {
                "name": "8K",
                "target_size": 8192,
                "tile_size": 512,
                "stride": 256,
                "brightness": -4.5,
                "prompt": "({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "output_file": os.path.join(output_dir, "result_8k.png")
            },
            {
                "name": "16K",
                "target_size": 16384,
                "tile_size": 512,
                "stride": 256,
                "brightness": -3,
                "prompt": "a macro photograph of ({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "output_file": os.path.join(output_dir, "result_16k.png")
            }
        ]
        
        print("Multi-Scale Color-Guided Grid Inpainting Pipeline")
        print("=" * 70)
        print(f"Input image: {input_image_path}")
        print(f"Color map: {color_map_path}")
        print(f"Alpha mask: {mask_path}")
        print(f"Progress saves: Every {progress_save_interval} tiles")
        print(f"Force rebuild: {force_rebuild}")
        print(f"Pipeline steps: {len(self.scale_steps)}")
        
        # Print available color mappings
        print("\nColor-to-variable mappings:")
        for color, variables in self.color_variables.items():
            var_str = ", ".join([f"{k}={v}" for k, v in variables.items()])
            print(f"  RGB{color}: {var_str}")
        
        # Check which steps can be resumed
        self.check_existing_outputs()
    
    def load_color_map(self):
        """Load the 512x512 color map"""
        if not os.path.exists(self.color_map_path):
            raise FileNotFoundError(f"Color map not found: {self.color_map_path}")
        
        self.color_map = Image.open(self.color_map_path)
        if self.color_map.mode != 'RGB':
            self.color_map = self.color_map.convert('RGB')
        self.color_map_array = np.array(self.color_map)
        
        # Verify it's 512x512
        if self.color_map_array.shape[:2] != (512, 512):
            print(f"Warning: Color map is {self.color_map_array.shape[:2]}, expected (512, 512)")
            print("Resizing color map to 512x512...")
            self.color_map = self.color_map.resize((512, 512), Image.NEAREST)
            self.color_map_array = np.array(self.color_map)
        
        print(f"Loaded color map: {self.color_map_array.shape}")
    
    def get_variables_for_tile(self, x, y, target_size, tile_size):
        """
        Determine which variables to use based on the color map at tile position
        Scales the coordinates from target_size back to 512x512 color map
        Returns a dictionary of variable replacements
        """
        # Calculate the center of the tile in the target resolution
        center_x = x + tile_size // 2
        center_y = y + tile_size // 2
        
        # Scale back to 512x512 color map coordinates
        scale_factor = 512.0 / target_size
        color_map_x = int(center_x * scale_factor)
        color_map_y = int(center_y * scale_factor)
        
        # Clamp to valid range
        color_map_x = min(max(0, color_map_x), 511)
        color_map_y = min(max(0, color_map_y), 511)
        
        # Get the color at this position
        sample_color = tuple(self.color_map_array[color_map_y, color_map_x])
        
        # Find closest matching color
        closest_color = self._find_closest_color(sample_color)
        variables = self.color_variables.get(closest_color, self.default_variables)
        
        return variables, sample_color, closest_color
    
    def replace_variables_in_prompt(self, prompt_template, variables):
        """
        Replace all {variable} placeholders in the prompt with their values
        """
        result = prompt_template
        for var_name, var_value in variables.items():
            placeholder = "{" + var_name + "}"
            result = result.replace(placeholder, var_value)
        return result
    
    def _color_distance(self, color1, color2):
        """Calculate Euclidean distance between two RGB colors"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    
    def _find_closest_color(self, sample_color):
        """Find the closest color in the color_variables dictionary"""
        min_distance = float('inf')
        closest_color = None
        
        for defined_color in self.color_variables.keys():
            distance = self._color_distance(sample_color, defined_color)
            if distance < min_distance:
                min_distance = distance
                closest_color = defined_color
        
        return closest_color
    
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
                print(f"  ⊗ Missing {step['name']}: {output_file}")
        
        if self.existing_files and not self.force_rebuild:
            last_completed = max(self.existing_files.keys())
            print(f"\n📄 Can resume from step {last_completed + 2}")
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
        if mask_img.mode != 'L':
            mask_img = mask_img.convert('L')
        
        self.alpha_mask = np.array(mask_img).astype(np.float32) / 255.0
        print(f"  Loaded alpha mask: {self.alpha_mask.shape}")
    
    def create_default_mask(self):
        """Create a default circular feathered mask if none exists"""
        size = 512
        center = size // 2
        y, x = np.ogrid[:size, :size]
        
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        inner_radius = center * 0.6
        outer_radius = center * 0.9
        
        mask = np.ones((size, size), dtype=np.float32)
        
        fade_region = (distance > inner_radius) & (distance < outer_radius)
        fade_factor = (outer_radius - distance[fade_region]) / (outer_radius - inner_radius)
        mask[fade_region] = fade_factor
        mask[distance >= outer_radius] = 0
        
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
        
        latest_file = max(files, key=os.path.getctime)
        
        processed_image = Image.open(latest_file)
        if processed_image.mode != 'RGB':
            processed_image = processed_image.convert('RGB')
        processed_array = np.array(processed_image).astype(np.float32)
        
        canvas_region = canvas[y:y+tile_size, x:x+tile_size].astype(np.float32)
        
        alpha_3d = np.stack([self.alpha_mask] * 3, axis=2)
        
        blended = canvas_region * (1.0 - alpha_3d) + processed_array * alpha_3d
        
        canvas[y:y+tile_size, x:x+tile_size] = blended.astype(np.uint8)
        
        return True
    
    def save_progress(self, canvas, step_name, tile_index):
        """Save work-in-progress image"""
        progress_filename = os.path.join(self.output_dir, f"progress_{step_name.lower()}_{tile_index:04d}.png")
        progress_image = Image.fromarray(canvas)
        progress_image.save(progress_filename)
        print(f"    💾 Progress saved: {progress_filename}")
    
    def run_comfy_api(self, prompt_text, brightness_value):
        """Run ComfyUI API with specified prompt and brightness"""
        try:
            workflow = self.workflow_template.copy()
            
            abs_temp_path = os.path.abspath(self.temp_file)
            workflow["14"]["inputs"]["image"] = f'"{abs_temp_path}"'
            
            workflow["7"]["inputs"]["text"] = prompt_text
            
            workflow["11"]["inputs"]["brightness"] = brightness_value
            
            random_seed = random.randint(1, 999999999)
            workflow["6"]["inputs"]["seed"] = random_seed
            
            client_id = str(uuid.uuid4())
            
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
                print(f"    No prompt_id returned")
                return False
            
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
        """Process a single scale step with color guidance"""
        step_name = step_config["name"]
        target_size = step_config["target_size"]
        tile_size = step_config["tile_size"]
        stride = step_config["stride"]
        brightness = step_config["brightness"]
        prompt_template = step_config["prompt"]
        output_file = step_config["output_file"]
        
        print(f"\nProcessing {step_name} ({target_size}x{target_size}) with Color-Guided Variables")
        print("-" * 70)
        print(f"Brightness: {brightness}")
        print(f"Prompt template: {prompt_template[:80]}...")
        print(f"Progress saves: Every {self.progress_save_interval} tiles")
        
        # Step 1: Simple upscale to target size (also save in output_dir)
        upscaled_path = os.path.join(self.output_dir, f"temp_upscaled_{step_name.lower()}.png")
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
        
        # Step 4: Process each tile with color-guided variables
        for i, (x, y) in enumerate(grid_positions):
            # Get color-guided variables for this tile
            variables, sample_color, closest_color = self.get_variables_for_tile(x, y, target_size, tile_size)
            
            # Replace variables in the prompt template
            final_prompt = self.replace_variables_in_prompt(prompt_template, variables)
            
            print(f"  Tile {i+1}/{total_tiles} at ({x}, {y})")
            print(f"    Color: RGB{sample_color} → RGB{closest_color}")
            var_str = ", ".join([f"{k}={v}" for k, v in variables.items()])
            print(f"    Variables: {var_str}")
            print(f"    Prompt: {final_prompt[:60]}...", end=" ")
            
            # Extract tile
            tile = self.extract_tile(canvas, x, y, tile_size)
            tile.save(self.temp_file)
            
            # Process with ComfyUI using final prompt
            success = self.run_comfy_api(final_prompt, brightness)
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
        print(f"✅ {step_name} completed: {output_file}")
        
        # Clean up temp upscaled file
        if os.path.exists(upscaled_path):
            os.remove(upscaled_path)
        
        return output_file
    
    def run_full_pipeline(self):
        """Run the complete multi-scale pipeline with color guidance and resume capability"""
        print("\nStarting Multi-Scale Color-Guided Pipeline...")
        print("=" * 70)
        
        # Determine starting point
        if self.existing_files and not self.force_rebuild:
            last_completed_index = max(self.existing_files.keys())
            start_index = last_completed_index + 1
            current_input = self.existing_files[last_completed_index]
            
            if start_index >= len(self.scale_steps):
                print("🎉 All steps already completed!")
                return
            
            print(f"📄 Resuming from step {start_index + 1}")
            print(f"📁 Using existing output as input: {current_input}")
        else:
            start_index = 0
            current_input = self.input_image_path
            print(f"🚀 Starting from the beginning with: {current_input}")
        
        # Process remaining steps
        for i in range(start_index, len(self.scale_steps)):
            step_config = self.scale_steps[i]
            
            # Skip if exists and not forcing rebuild
            if i in self.existing_files and not self.force_rebuild:
                print(f"\n⏭️ Skipping {step_config['name']} - already exists")
                current_input = step_config["output_file"]
                continue
            
            step_start_time = time.time()
            
            # Process this scale step with color guidance
            result_file = self.process_scale_step(current_input, step_config)
            
            # The output becomes input for the next step
            current_input = result_file
            
            step_duration = time.time() - step_start_time
            print(f"Step {i+1} completed in {step_duration/60:.1f} minutes")
            
            # Clean up temp file
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
        
        print("\n" + "=" * 70)
        print("🎉 FULL COLOR-GUIDED PIPELINE COMPLETED!")
        print("Results:")
        for step in self.scale_steps:
            if os.path.exists(step["output_file"]):
                file_size = os.path.getsize(step["output_file"]) / (1024*1024)
                print(f"  {step['name']}: {step['output_file']} ({file_size:.1f} MB)")


def main():
    # Configuration
    input_image = "input_512.png"    # Your 512x512 starting image
    color_map = "color_map_512.png"  # Your 512x512 color guide
    mask_image = "mask.png"          # Your 512x512 alpha mask
    workflow_json = "DifferentialDiffusionAPIWCtrlNet.json"
    
    # Output directory - all results will be saved here
    output_directory = "output"      # Change this to your preferred subdirectory name
    
    # Advanced options
    progress_save_interval = 300     # Save progress every N tiles
    force_rebuild = False            # Set to True to regenerate all steps
    
    # Verify input files exist
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found!")
        return
    
    if not os.path.exists(color_map):
        print(f"Error: Color map '{color_map}' not found!")
        return
    
    # Create the multi-scale color-guided inpainter
    pipeline = MultiScaleGridInpainterResumeColorGuided(
        input_image_path=input_image,
        color_map_path=color_map,
        mask_path=mask_image,
        temp_file="q.png",
        comfy_server="127.0.0.1:8188",
        workflow_path=workflow_json,
        progress_save_interval=progress_save_interval,
        force_rebuild=force_rebuild,
        output_dir=output_directory
    )
    
    # Run the complete pipeline
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()