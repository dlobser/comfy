import numpy as np
from PIL import Image, ImageFile
import os
import time
import subprocess
import json
import requests
import uuid
import websocket
import threading
import random


# Increase PIL's image size limit for large images
Image.MAX_IMAGE_PIXELS = None  # Removes the limit entirely

class GridInpainter:
    def __init__(self, input_image_path, tile_size=512, stride=256, temp_file="q.png", 
                 comfy_server="127.0.0.1:8188", workflow_path="DifferentialDiffusionForGridAPI.json"):
        """
        Initialize the grid inpainter
        
        Args:
            input_image_path: Path to the 2K input image
            tile_size: Size of each tile to process (default 512)
            stride: How much to move between tiles (default 256 for 50% overlap)
            temp_file: Temporary file name for ComfyUI processing (default q.png to match your workflow)
            comfy_server: ComfyUI server address (default 127.0.0.1:8188)
            workflow_path: Path to your ComfyUI workflow JSON file
        """
        self.input_image_path = input_image_path
        self.tile_size = tile_size
        self.stride = stride
        self.temp_file = temp_file
        self.comfy_server = comfy_server
        self.workflow_path = workflow_path
        
        # Load workflow template
        with open(workflow_path, 'r') as f:
            self.workflow_template = json.load(f)
        
        # Load the main image into memory
        self.image = Image.open(input_image_path)
        # Ensure image is RGB
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')
        self.canvas = np.array(self.image)
        self.height, self.width = self.canvas.shape[:2]
        
        print(f"Loaded image: {self.width}x{self.height}")
        print(f"Tile size: {self.tile_size}, Stride: {self.stride}")
        
        # Calculate grid positions
        self.grid_positions = self._calculate_grid_positions()
        print(f"Total tiles to process: {len(self.grid_positions)}")
    
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
    
    def extract_tile(self, x, y):
        """Extract a tile from the current canvas"""
        tile = self.canvas[y:y+self.tile_size, x:x+self.tile_size]
        return Image.fromarray(tile)
    
    def insert_tile(self, x, y):
        """Insert a processed tile back into the canvas"""
        # ComfyUI saves with "SavedGrid" prefix to your output directory
        output_dir = r"C:\_Main\ai\ComfyOutput"
        
        # Look for files with SavedGrid prefix
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
        # Ensure processed image matches canvas format
        if processed_image.mode != 'RGB':
            processed_image = processed_image.convert('RGB')
        processed_array = np.array(processed_image)
        
        # Debug: Print shapes to understand the mismatch
        print(f"  Canvas shape: {self.canvas.shape}")
        print(f"  Processed array shape: {processed_array.shape}")
        
        # Ensure shapes match
        if len(self.canvas.shape) == 3 and len(processed_array.shape) == 3:
            # Both are color images
            self.canvas[y:y+self.tile_size, x:x+self.tile_size] = processed_array
        elif len(self.canvas.shape) == 2 and len(processed_array.shape) == 3:
            # Canvas is grayscale, processed is color - convert processed to grayscale
            processed_gray = np.mean(processed_array, axis=2).astype(np.uint8)
            self.canvas[y:y+self.tile_size, x:x+self.tile_size] = processed_gray
        elif len(self.canvas.shape) == 3 and len(processed_array.shape) == 2:
            # Canvas is color, processed is grayscale - convert grayscale to color
            processed_color = np.stack([processed_array] * 3, axis=2)
            self.canvas[y:y+self.tile_size, x:x+self.tile_size] = processed_color
        else:
            # Both grayscale
            self.canvas[y:y+self.tile_size, x:x+self.tile_size] = processed_array
        
        return True
    
    def run_comfy_api(self):
        """
        Run ComfyUI API using your workflow
        """
        print("Running ComfyUI processing...")
        
        try:
            # Create a copy of the workflow template
            workflow = self.workflow_template.copy()
            
            # Update the image path in node 14 to point to our temp file
            # Get absolute path to ensure ComfyUI can find it
            abs_temp_path = os.path.abspath(self.temp_file)
            # Convert to Windows path format with quotes to match your workflow
            workflow["14"]["inputs"]["image"] = f'"{abs_temp_path}"'
            
            # Randomize the seed for each tile
            random_seed = random.randint(1, 999999999)
            workflow["6"]["inputs"]["seed"] = random_seed
            print(f"  Using random seed: {random_seed}")
            
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
            
            # Wait for completion by polling the history
            return self._wait_for_completion(prompt_id)
            
        except Exception as e:
            print(f"Error running ComfyUI API: {e}")
            return False
    
    def _wait_for_completion(self, prompt_id, timeout=300):
        """
        Wait for ComfyUI to complete processing by polling the history
        """
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
                
                # Check every second
                time.sleep(1)
                
            except Exception as e:
                print(f"Error checking completion: {e}")
                time.sleep(1)
        
        print(f"Timeout waiting for prompt {prompt_id}")
        return False
    
    def process_grid(self, save_intermediate=False, output_path="result.png"):
        """
        Process the entire grid
        
        Args:
            save_intermediate: Whether to save intermediate results
            output_path: Final output image path
        """
        total_tiles = len(self.grid_positions)
        
        for i, (x, y) in enumerate(self.grid_positions):
            print(f"Processing tile {i+1}/{total_tiles} at position ({x}, {y})")
            
            # 1. Extract tile from current canvas
            tile = self.extract_tile(x, y)
            
            # 2. Save tile to temp file
            tile.save(self.temp_file)
            print(f"  Saved tile to {self.temp_file}")
            
            # 3. Run ComfyUI processing
            success = self.run_comfy_api()
            if not success:
                print(f"  Error processing tile at ({x}, {y})")
                continue
            
            # 4. Insert processed tile back into canvas
            success = self.insert_tile(x, y)
            if success:
                print(f"  Inserted processed tile back into canvas")
            else:
                print(f"  Failed to insert processed tile")
                continue
            
            # 5. Optionally save intermediate result
            if save_intermediate:
                intermediate_path = f"intermediate_{i:03d}.png"
                Image.fromarray(self.canvas).save(intermediate_path)
                print(f"  Saved intermediate result: {intermediate_path}")
        
        # Save final result
        final_image = Image.fromarray(self.canvas)
        final_image.save(output_path)
        print(f"Final result saved to: {output_path}")
        
        # Clean up temp file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)


def main():
    # Example usage
    input_image = "input_2k.png"  # Replace with your input image path
    workflow_json = "DifferentialDiffusionForGridAPI.json"  # Your workflow file
    
    # Create the inpainter
    inpainter = GridInpainter(
        input_image_path=input_image,
        tile_size=512,
        stride=256,
        temp_file="q.png",  # This matches your workflow
        comfy_server="127.0.0.1:8188",  # Adjust if your ComfyUI runs elsewhere
        workflow_path=workflow_json
    )
    
    # Process the grid
    inpainter.process_grid(
        save_intermediate=False,  # Set to False if you don't want intermediate saves
        output_path="final_result.png"
    )


if __name__ == "__main__":
    main()