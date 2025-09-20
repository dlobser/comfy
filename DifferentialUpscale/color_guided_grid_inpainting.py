import numpy as np
from PIL import Image
import os
import time
import json
import requests
import uuid

class ColorGuidedGridInpainter:
    def __init__(self, input_image_path, color_map_path, tile_size=512, stride=256, 
                 temp_file="q.png", comfy_server="127.0.0.1:8188", 
                 workflow_path="DifferentialDiffusionForGridAPI.json"):
        """
        Initialize the color-guided grid inpainter
        
        Args:
            input_image_path: Path to the image to process
            color_map_path: Path to the color map image (same size as input)
            tile_size: Size of each tile to process (default 512)
            stride: How much to move between tiles (default 256 for 50% overlap)
            temp_file: Temporary file name for ComfyUI processing
            comfy_server: ComfyUI server address
            workflow_path: Path to your ComfyUI workflow JSON file
        """
        self.input_image_path = input_image_path
        self.color_map_path = color_map_path
        self.tile_size = tile_size
        self.stride = stride
        self.temp_file = temp_file
        self.comfy_server = comfy_server
        self.workflow_path = workflow_path
        
        # Define color-to-prompt mapping
        # Format: (R, G, B): "prompt text"
        self.color_prompts = {
            (255, 0, 0): "extremely intricate and detailed, hundreds of leaves made of feathers hanging by strings with beads, a giant tree with gnarled bark with leaves made of peacock feathers in a forest made of flock of hundreds of birds, made of wax, paper cutout, beautiful oil painting by hieronymous bosch",
            (0, 0, 255): "photorealistic, highly detailed landscape, natural lighting, beautiful scenery",
            (0, 255, 0): "abstract art, vibrant colors, modern artistic style",
            (255, 255, 0): "steampunk machinery, brass gears, industrial design",
            (255, 0, 255): "fantasy magical forest, ethereal lighting, mystical atmosphere",
            (0, 255, 255): "cyberpunk cityscape, neon lights, futuristic architecture",
            (128, 128, 128): "black and white photography, high contrast, dramatic lighting"
        }
        
        # Default prompt if no color match is found
        self.default_prompt = "high quality, detailed, photorealistic"
        
        # Load workflow template
        with open(workflow_path, 'r') as f:
            self.workflow_template = json.load(f)
        
        # Load the main image into memory
        self.image = Image.open(input_image_path)
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
        self.canvas = np.array(self.image)
        self.height, self.width = self.canvas.shape[:2]
        
        # Load the color map
        self.color_map = Image.open(color_map_path)
        if self.color_map.mode != 'RGB':
            self.color_map = self.color_map.convert('RGB')
        self.color_map_array = np.array(self.color_map)
        
        # Verify both images are the same size
        if self.color_map_array.shape[:2] != (self.height, self.width):
            raise ValueError(f"Color map size {self.color_map_array.shape[:2]} doesn't match input image size {(self.height, self.width)}")
        
        print(f"Loaded image: {self.width}x{self.height}")
        print(f"Loaded color map: {self.color_map_array.shape}")
        print(f"Tile size: {self.tile_size}, Stride: {self.stride}")
        
        # Calculate grid positions
        self.grid_positions = self._calculate_grid_positions()
        print(f"Total tiles to process: {len(self.grid_positions)}")
        
        # Print available color mappings
        print("\nColor-to-prompt mappings:")
        for color, prompt in self.color_prompts.items():
            print(f"  RGB{color}: {prompt[:60]}...")
    
    def _calculate_grid_positions(self):
        """Calculate all the (x, y) positions for the grid"""
        positions = []
        
        y = 0
        while y + self.tile_size <= self.height:
            x = 0
            while x + self.tile_size <= self.width:
                positions.append((x, y))
                x += self.stride
            y += self.stride
        
        return positions
    
    def get_prompt_for_tile(self, x, y):
        """
        Determine which prompt to use based on the color map at tile position
        Samples the center pixel of the tile area in the color map
        Uses closest color matching with Euclidean distance
        """
        # Get center point of the tile
        center_x = x + self.tile_size // 2
        center_y = y + self.tile_size // 2
        
        # Make sure we're within bounds
        center_x = min(center_x, self.width - 1)
        center_y = min(center_y, self.height - 1)
        
        # Get the color at this position
        sample_color = tuple(self.color_map_array[center_y, center_x])
        
        # Find closest matching color using Euclidean distance
        closest_color = self._find_closest_color(sample_color)
        prompt = self.color_prompts[closest_color]
        
        # Calculate distance for debugging
        distance = self._color_distance(sample_color, closest_color)
        
        print(f"  Tile center ({center_x}, {center_y}) has color RGB{sample_color}")
        print(f"  Closest match: RGB{closest_color} (distance: {distance:.1f})")
        print(f"  Using prompt: {prompt[:60]}...")
        
        return prompt
    
    def _color_distance(self, color1, color2):
        """Calculate Euclidean distance between two RGB colors"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return ((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2) ** 0.5
    
    def _find_closest_color(self, sample_color):
        """Find the closest color in the color_prompts dictionary"""
        min_distance = float('inf')
        closest_color = None
        
        for defined_color in self.color_prompts.keys():
            distance = self._color_distance(sample_color, defined_color)
            if distance < min_distance:
                min_distance = distance
                closest_color = defined_color
        
        return closest_color
    
    def extract_tile(self, x, y):
        """Extract a tile from the current canvas"""
        tile = self.canvas[y:y+self.tile_size, x:x+self.tile_size]
        return Image.fromarray(tile)
    
    def run_comfy_api(self, prompt_text):
        """
        Run ComfyUI API using your workflow with the specified prompt
        """
        print("Running ComfyUI processing...")
        
        try:
            # Create a copy of the workflow template
            workflow = self.workflow_template.copy()
            
            # Update the image path in node 14
            abs_temp_path = os.path.abspath(self.temp_file)
            workflow["14"]["inputs"]["image"] = f'"{abs_temp_path}"'
            
            # Update the prompt in node 7
            workflow["7"]["inputs"]["text"] = prompt_text
            
            # Generate unique client ID
            client_id = str(uuid.uuid4())
            
            # Queue the prompt
            prompt_data = {
                "prompt": workflow,
                "client_id": client_id
            }
            
            response = requests.post(f"http://{self.comfy_server}/prompt", json=prompt_data)
            
            if response.status_code != 200:
                print(f"Error queuing prompt: {response.status_code}")
                return False
            
            result = response.json()
            prompt_id = result.get("prompt_id")
            
            if not prompt_id:
                print("No prompt_id returned")
                return False
            
            print(f"Queued prompt with ID: {prompt_id}")
            
            # Wait for completion
            return self._wait_for_completion(prompt_id)
            
        except Exception as e:
            print(f"Error running ComfyUI API: {e}")
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
                            print("ComfyUI processing completed!")
                            return True
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error checking completion: {e}")
                time.sleep(1)
        
        print(f"Timeout waiting for prompt {prompt_id}")
        return False
    
    def insert_tile(self, x, y):
        """Insert a processed tile back into the canvas"""
        output_dir = r"C:\_Main\ai\ComfyOutput"
        
        import glob
        pattern = os.path.join(output_dir, "SavedGrid_*.png")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No SavedGrid output files found in {output_dir}")
            return False
        
        # Get the most recent file
        latest_file = max(files, key=os.path.getctime)
        print(f"  Loading processed result: {latest_file}")
        
        processed_image = Image.open(latest_file)
        if processed_image.mode != 'RGB':
            processed_image = processed_image.convert('RGB')
        processed_array = np.array(processed_image)
        
        # Insert the processed tile back into the canvas
        self.canvas[y:y+self.tile_size, x:x+self.tile_size] = processed_array
        return True
    
    def process_grid(self, save_intermediate=False, output_path="result.png"):
        """Process the entire grid with color-guided prompts"""
        total_tiles = len(self.grid_positions)
        
        for i, (x, y) in enumerate(self.grid_positions):
            print(f"\nProcessing tile {i+1}/{total_tiles} at position ({x}, {y})")
            
            # 1. Determine prompt based on color map
            prompt = self.get_prompt_for_tile(x, y)
            
            # 2. Extract tile from current canvas
            tile = self.extract_tile(x, y)
            
            # 3. Save tile to temp file
            tile.save(self.temp_file)
            print(f"  Saved tile to {self.temp_file}")
            
            # 4. Run ComfyUI processing with the specific prompt
            success = self.run_comfy_api(prompt)
            if not success:
                print(f"  Error processing tile at ({x}, {y})")
                continue
            
            # 5. Insert processed tile back into canvas
            success = self.insert_tile(x, y)
            if success:
                print(f"  Inserted processed tile back into canvas")
            else:
                print(f"  Failed to insert processed tile")
                continue
            
            # 6. Optionally save intermediate result
            if save_intermediate:
                intermediate_path = f"intermediate_{i:03d}.png"
                Image.fromarray(self.canvas).save(intermediate_path)
                print(f"  Saved intermediate result: {intermediate_path}")
        
        # Save final result
        final_image = Image.fromarray(self.canvas)
        final_image.save(output_path)
        print(f"\nFinal result saved to: {output_path}")
        
        # Clean up temp file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)


def main():
    # Example usage
    input_image = "input_image.png"      # Your main image to process
    color_map = "color_map.png"          # Color guide image (same size as input)
    workflow_json = "DifferentialDiffusionForGridAPI.json"
    
    # Create the inpainter
    inpainter = ColorGuidedGridInpainter(
        input_image_path=input_image,
        color_map_path=color_map,
        tile_size=512,
        stride=256,
        temp_file="q.png",
        comfy_server="127.0.0.1:8188",
        workflow_path=workflow_json
    )
    
    # Process the grid
    inpainter.process_grid(
        save_intermediate=False,  # Set to True if you want progress saves
        output_path="final_color_guided_result.png"
    )


if __name__ == "__main__":
    main()