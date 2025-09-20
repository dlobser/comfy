# Grid Inpainting Script

This script implements a grid-based image processing pipeline that:

1. Loads a 2K image into memory
2. Processes it in overlapping tiles (512x512 with 256px stride)
3. Saves each tile, runs ComfyUI processing, and composites the result back
4. Keeps the full image in memory to avoid I/O bottlenecks

## Key Features

- **Memory-efficient**: Only writes small tiles to disk, keeps main canvas in RAM
- **Configurable**: Tile size and stride can be adjusted
- **Progress tracking**: Shows current tile being processed
- **Intermediate saves**: Option to save progress at each step
- **Clean interface**: Easy to integrate with your ComfyUI API

## Usage

1. **Install dependencies**:
   ```bash
   pip install pillow numpy
   ```

2. **Update the ComfyUI integration**:
   - Modify the `run_comfy_api()` method to call your actual ComfyUI setup
   - This could be an HTTP API call, subprocess, or other integration method

3. **Run the script**:
   ```python
   python grid_inpainting.py
   ```

## Customization Points

### ComfyUI Integration
The `run_comfy_api()` method is where you'll integrate with your ComfyUI workflow:

```python
def run_comfy_api(self):
    # Option 1: HTTP API call
    response = requests.post("http://localhost:8188/api/prompt", json=your_payload)
    
    # Option 2: Subprocess call
    subprocess.run(["python", "comfy_script.py", "--input", self.temp_file])
    
    # Option 3: Direct Python API if available
    comfy_result = your_comfy_function(self.temp_file)
```

### Grid Configuration
- `tile_size`: Size of each processing tile (default 512)
- `stride`: Overlap amount (default 256 for 50% overlap)
- Smaller stride = more overlap = smoother blending

## File Flow

1. `input_2k.png` → Load into memory as numpy array
2. For each grid position:
   - Extract tile → `comfy_input.png`
   - ComfyUI processes `comfy_input.png` → overwrites with result
   - Load result and composite back into memory canvas
3. Save final result → `final_result.png`

## Memory Usage

For a 2048x2048 RGB image:
- Canvas in memory: ~12MB
- Temp tiles: ~768KB each
- Total memory footprint: Very manageable

This approach eliminates the bottleneck of writing/reading the full 2K image at each step.
