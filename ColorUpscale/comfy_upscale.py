import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import time
import json
import requests
import uuid
import random
from typing import Dict, Tuple

# Allow large images
Image.MAX_IMAGE_PIXELS = None


def _json_literal(value):
    """
    Return a JSON-safe literal string for replacement into a JSON template.
    Example: _json_literal("C:/path.png") -> "\"C:/path.png\""
             _json_literal(0.42) -> "0.42"
    """
    return json.dumps(value, ensure_ascii=False)


class MultiScaleGridInpainterResumeColorGuided:
    def __init__(
        self,
        input_image_path="input.png",
        color_map_path="color.png",
        control_image_path="ctrl.png",
        mask_path="mask.png",
        temp_file="q.png",
        comfy_server="127.0.0.1:8188",
        workflow_path="mapWorkflow.json",
        progress_save_interval=300,
        force_rebuild=False,
        output_dir="output",
        test_run=False,
        test_pos: Tuple[float, float] = (0.5, 0.5),  # (x_percent, y_percent)
    ):
        """
        Initialize the multi-scale grid inpainter with color guidance, resume capability and progress saves

        New in this version:
        - Variable-driven workflow templating (use {placeholders} in workflow JSON)
        - ControlNet strength per scale step via self.scale_steps[*]['controlnet_strength']
        - ControlNet source image crop matched to the current color image tile
        - test_run mode: render a single tile per level at a given normalized position

        Args:
            input_image_path: Path to the 512x512 input image
            color_map_path: Path to the 512x512 color guide image
            control_image_path: Optional path to a (any resolution) ControlNet guidance image
            mask_path: Path to the 512x512 alpha mask (black and white PNG)
            temp_file: Temporary file name for ComfyUI color tile processing
            comfy_server: ComfyUI server address
            workflow_path: Path to your ComfyUI workflow JSON file (with {placeholders})
            progress_save_interval: Save work-in-progress every N tiles (default 300)
            force_rebuild: If True, rebuild all steps even if output files exist
            output_dir: Subdirectory where all output images will be saved
            test_run: If True, only one tile per level is rendered at test_pos
            test_pos: Normalized position (0..1, 0..1) to pick the tile within the level grid
        """
        self.input_image_path = input_image_path
        self.color_map_path = color_map_path
        self.control_image_path = control_image_path
        self.mask_path = mask_path
        self.temp_file = temp_file
        self.temp_control_file = "q_control.png"
        self.comfy_server = comfy_server
        self.workflow_path = workflow_path
        self.progress_save_interval = progress_save_interval
        self.force_rebuild = force_rebuild
        self.output_dir = output_dir
        self.test_run = test_run
        self.test_pos = test_pos

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(self.output_dir)}")

        # Load workflow template (as Python object)
        with open(workflow_path, 'r', encoding='utf-8') as f:
            self.workflow_template = json.load(f)

        # Load and prepare the alpha mask
        self.load_alpha_mask()

        # Load the color map (512x512)
        self.load_color_map()

        # Default ControlNet image existence check (optional)
        if self.control_image_path and not os.path.exists(self.control_image_path):
            print(f"Warning: ControlNet image not found: {self.control_image_path}. ControlNet cropping will be skipped.")
            self.control_image_path = None

        # Define color-to-variable mapping (example mapping)
        self.color_variables = {
            (255, 0, 0): {"thing": "mountains, extremely tall and jagged", "texture": "pebbles and rocks"},
            (255, 255, 0): {"thing": "a desert filled with camels, sand dunes, rocky outcroppings", "texture": "gnarled spiral swirls of mandelbulb fractals"},
            (0, 255, 0): {"thing": "Grassy forests full of leaves trees, birds, deer, squirrels, dogs", "texture": "scraps of paper"},
            (0, 255, 255): {"thing": "Icy glacier and snow capped mountains filled with polar bears, penguins, and snow", "texture": "tiny little beads"},
            (0, 0, 255): {"thing": "An ocean full of strange undersea creatures and fish, whales, squid", "texture": "wax, wire, and felt"},
        }

        # Default variable values if no color match is found
        self.default_variables = {"material": "natural materials", "texture": "detailed"}

        # Scaling pipeline with prompts and per-level ControlNet strength
        self.scale_steps = [
            {
                "name": "1K",
                "target_size": 1024,
                "tile_size": 512,
                "stride": 256,
                "brightness": -6,
                "controlnet_strength": 1,
                # These two feed {positivePrompt} / {negativePrompt} in the workflow
                "positive": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with ({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "negative": "(blurry:1.2), compression artifact, jpeg, low quality, deformed, frame, picture frame, blurry, compression, artifact",
                "output_file": os.path.join(output_dir, "result_1k.png"),
            },
            {
                "name": "2K",
                "target_size": 2048,
                "tile_size": 512,
                "stride": 256,
                "brightness": -5,
                "controlnet_strength": 1,
                "positive": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with ({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "negative": "(blurry:1.2), compression artifact, jpeg, low quality, deformed, frame, picture frame, blurry, compression, artifact",
                "output_file": os.path.join(output_dir, "result_2k.png"),
            },
            {
                "name": "4K",
                "target_size": 4096,
                "tile_size": 512,
                "stride": 256,
                "brightness": -4,
                "controlnet_strength": 1,
                "positive": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with ({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "negative": "(blurry:1.2), compression artifact, jpeg, low quality, deformed, frame, picture frame, blurry, compression, artifact",
                "output_file": os.path.join(output_dir, "result_4k.png"),
            },
            {
                "name": "8K",
                "target_size": 8192,
                "tile_size": 512,
                "stride": 256,
                "brightness": -3,
                "controlnet_strength": .6,
                "positive": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with ({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "negative": "(blurry:1.2), compression artifact, jpeg, low quality, deformed, frame, picture frame, blurry, compression, artifact",
                "output_file": os.path.join(output_dir, "result_8k.png"),
            },
            {
                "name": "16K",
                "target_size": 16384,
                "tile_size": 512,
                "stride": 256,
                "brightness": -2.5,
                "controlnet_strength": 0.3,
                "positive": "vast abstract landscape, a photograph, a beautiful (dense forest:1.2) with ({thing}:1.2) made of ({texture}), outsider art in the style of Hieronymus Bosch",
                "negative": "(blurry:1.2), compression artifact, jpeg, low quality, deformed, frame, picture frame, blurry, compression, artifact",
                "output_file": os.path.join(output_dir, "result_16k.png"),
            },
        ]

        print("Multi-Scale Color-Guided Grid Inpainting Pipeline (Variable-Driven)")
        print("=" * 70)
        print(f"Input image: {input_image_path}")
        print(f"Color map: {color_map_path}")
        print(f"Control image: {self.control_image_path}")
        print(f"Alpha mask: {mask_path}")
        print(f"Progress saves: Every {progress_save_interval} tiles")
        print(f"Force rebuild: {force_rebuild}")
        print(f"Pipeline steps: {len(self.scale_steps)}")
        print(f"test_run: {self.test_run}, test_pos: {self.test_pos}")

        # Print available color mappings
        print("\nColor-to-variable mappings:")
        for color, variables in self.color_variables.items():
            var_str = ", ".join([f"{k}={v}" for k, v in variables.items()])
            print(f"  RGB{color}: {var_str}")

        # Check which steps can be resumed
        self.check_existing_outputs()

    # ----------------------------
    # Asset loading helpers
    # ----------------------------
    def load_color_map(self):
        if not os.path.exists(self.color_map_path):
            raise FileNotFoundError(f"Color map not found: {self.color_map_path}")
        self.color_map = Image.open(self.color_map_path)
        if self.color_map.mode != 'RGB':
            self.color_map = self.color_map.convert('RGB')
        self.color_map_array = np.array(self.color_map)
        if self.color_map_array.shape[:2] != (512, 512):
            print(f"Warning: Color map is {self.color_map_array.shape[:2]}, expected (512, 512)")
            print("Resizing color map to 512x512...")
            self.color_map = self.color_map.resize((512, 512), Image.NEAREST)
            self.color_map_array = np.array(self.color_map)
        print(f"Loaded color map: {self.color_map_array.shape}")

    def load_alpha_mask(self):
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
        size = 512
        center = size // 2
        y, x = np.ogrid[:size, :size]
        distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
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

    # ----------------------------
    # Utility
    # ----------------------------
    def check_existing_outputs(self):
        print("\nChecking for existing output files...")
        self.existing_files = {}
        for i, step in enumerate(self.scale_steps):
            output_file = step["output_file"]
            if os.path.exists(output_file) and not self.force_rebuild:
                file_size = os.path.getsize(output_file) / (1024 * 1024)
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

    def simple_upscale(self, input_path, target_size, output_path):
        print(f"  Upscaling {input_path} to {target_size}x{target_size}")
        img = Image.open(input_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        upscaled = img.resize((target_size, target_size), Image.LANCZOS)
        upscaled.save(output_path)
        print(f"  Saved upscaled image: {output_path}")
        return output_path

    def calculate_grid_positions(self, width, height, tile_size, stride):
        positions = []
        y = 0
        while y + tile_size <= height:
            x = 0
            while x + tile_size <= width:
                positions.append((x, y))
                x += stride
            y += stride
        return positions

    def pick_test_position(self, width, height, tile_size, stride):
        tx = int(round((width - tile_size) * min(max(self.test_pos[0], 0.0), 1.0)))
        ty = int(round((height - tile_size) * min(max(self.test_pos[1], 0.0), 1.0)))
        # snap to grid stride so it aligns with normal tile layout
        if stride > 0:
            tx = (tx // stride) * stride
            ty = (ty // stride) * stride
        tx = max(0, min(tx, width - tile_size))
        ty = max(0, min(ty, height - tile_size))
        return [(tx, ty)]

    def extract_tile(self, canvas, x, y, tile_size):
        tile = canvas[y : y + tile_size, x : x + tile_size]
        return Image.fromarray(tile)

    def insert_tile_with_alpha(self, canvas, x, y, tile_size):
        output_dir = r"C:\\_Main\\ai\\ComfyOutput"
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
        canvas_region = canvas[y : y + tile_size, x : x + tile_size].astype(np.float32)
        alpha_3d = np.stack([self.alpha_mask] * 3, axis=2)
        blended = canvas_region * (1.0 - alpha_3d) + processed_array * alpha_3d
        canvas[y : y + tile_size, x : x + tile_size] = blended.astype(np.uint8)
        return True

    # ----------------------------
    # ControlNet crop helper
    # ----------------------------
    def crop_control_image_like_tile(self, target_size, tile_size, x, y):
        """
        Crop the ControlNet source image so that it matches the same field-of-view as the
        (x,y,tile_size) tile on a target_size x target_size canvas.

        If control image = C x C (or generic W x H), the crop width/height = (tile_size/target_size) * (W/H).
        Example: target_size=1024, tile_size=512, control=8192 -> crop 4096.
        """
        if not self.control_image_path:
            return None
        img = Image.open(self.control_image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        W, H = img.size
        # compute crop dimensions
        crop_w = max(1, int(round(W * (tile_size / float(target_size)))))
        crop_h = max(1, int(round(H * (tile_size / float(target_size)))))
        # compute top-left in control image space using proportional mapping
        left = int(round((x / float(target_size)) * W))
        top = int(round((y / float(target_size)) * H))
        # clamp to image
        left = max(0, min(left, W - crop_w))
        top = max(0, min(top, H - crop_h))
        box = (left, top, left + crop_w, top + crop_h)
        crop = img.crop(box)
        # Resize crop to the model tile size expected by the workflow (here: 512x512)
        crop = crop.resize((tile_size, tile_size), Image.LANCZOS)
        crop.save(self.temp_control_file)
        return os.path.abspath(self.temp_control_file)

    # ----------------------------
    # Workflow templating + API
    # ----------------------------
    def render_workflow(self, replacements: Dict[str, object]):
        """
        Replace quoted placeholders like "{positivePrompt}" with proper JSON literals.
        Keep all placeholders quoted in the JSON file so it's valid JSON on disk.
        """
        template_str = json.dumps(self.workflow_template, ensure_ascii=False)
        for key, val in replacements.items():
            # Look for the quoted token exactly as it appears in valid JSON:
            # e.g.  "{positivePrompt}"
            quoted_token = '"' + '{' + key + '}' + '"'
            template_str = template_str.replace(quoted_token, _json_literal(val))
        return json.loads(template_str)

    def run_comfy_api(self, rendered_workflow):
        try:
            client_id = str(uuid.uuid4())
            prompt_data = {"prompt": rendered_workflow, "client_id": client_id}
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

    # ----------------------------
    # Main per-level processing
    # ----------------------------
    def get_variables_for_tile(self, x, y, target_size, tile_size):
        center_x = x + tile_size // 2
        center_y = y + tile_size // 2
        scale_factor = 512.0 / target_size
        color_map_x = int(center_x * scale_factor)
        color_map_y = int(center_y * scale_factor)
        color_map_x = min(max(0, color_map_x), 511)
        color_map_y = min(max(0, color_map_y), 511)
        sample_color = tuple(self.color_map_array[color_map_y, color_map_x])
        closest_color = self._find_closest_color(sample_color)
        variables = self.color_variables.get(closest_color, self.default_variables)
        return variables, sample_color, closest_color

    def replace_variables_in_prompt(self, prompt_template, variables):
        result = prompt_template
        for var_name, var_value in variables.items():
            result = result.replace("{" + var_name + "}", var_value)
        return result

    def _color_distance(self, c1, c2):
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2) ** 0.5

    def _find_closest_color(self, sample_color):
        min_d = float('inf')
        closest = None
        for defined in self.color_variables.keys():
            d = self._color_distance(sample_color, defined)
            if d < min_d:
                min_d = d
                closest = defined
        return closest

    def process_scale_step(self, input_image_path, step_config):
        step_name = step_config["name"]
        target_size = step_config["target_size"]
        tile_size = step_config["tile_size"]
        stride = step_config["stride"]
        brightness = step_config["brightness"]
        cn_strength = step_config.get("controlnet_strength", 0.25)
        pos_template = step_config["positive"]
        neg_template = step_config["negative"]
        output_file = step_config["output_file"]

        print(f"\nProcessing {step_name} ({target_size}x{target_size}) with Color-Guided Variables")
        print("-" * 70)
        print(f"Brightness: {brightness} | ControlNet strength: {cn_strength}")
        print(f"Progress saves: Every {self.progress_save_interval} tiles")

        # Step 1: Simple upscale to target size
        upscaled_path = os.path.join(self.output_dir, f"temp_upscaled_{step_name.lower()}.png")
        self.simple_upscale(input_image_path, target_size, upscaled_path)

        # Step 2: Load upscaled image for grid processing
        image = Image.open(upscaled_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        canvas = np.array(image)
        height, width = canvas.shape[:2]

        # Step 3: Choose positions
        if self.test_run:
            grid_positions = self.pick_test_position(width, height, tile_size, stride)
        else:
            grid_positions = self.calculate_grid_positions(width, height, tile_size, stride)
        total_tiles = len(grid_positions)
        print(f"Grid processing: {total_tiles} tile(s) ({tile_size}x{tile_size}, stride {stride})")

        for i, (x, y) in enumerate(grid_positions):
            variables, sample_color, closest_color = self.get_variables_for_tile(x, y, target_size, tile_size)
            pos_prompt = self.replace_variables_in_prompt(pos_template, variables)
            neg_prompt = self.replace_variables_in_prompt(neg_template, variables)

            print(f"  Tile {i+1}/{total_tiles} at ({x}, {y})  Color: RGB{sample_color} → RGB{closest_color}")

            # Prepare the color tile for Comfy
            tile_img = self.extract_tile(canvas, x, y, tile_size)

            def _norm(p: str) -> str:
                return p.replace("\\", "/")

            tile_img.save(self.temp_file)
            abs_tile_path = _norm(os.path.abspath(self.temp_file))

            control_path_for_json = None
            if self.control_image_path:
                control_crop_path = self.crop_control_image_like_tile(target_size, tile_size, x, y)
                if not control_crop_path or not os.path.exists(control_crop_path):
                    print("    ERROR: control crop missing; aborting this tile")
                    return False
                control_path_for_json = _norm(os.path.abspath(control_crop_path))
            else:
                # If no ctrl.png given, just use the color tile
                control_path_for_json = abs_tile_path

            replacements = {
                "positivePrompt": pos_prompt,
                "negativePrompt": neg_prompt,
                "seed": random.randint(1, 999999999),
                "brightness": float(brightness),
                "colorTilePath": abs_tile_path,
                "controlNetStrength": float(cn_strength),
                "controlNetImagePath": control_path_for_json
            }

            rendered = self.render_workflow(replacements)
            with open(os.path.join(self.output_dir, "debug_rendered_prompt.json"), "w", encoding="utf-8") as f:
                json.dump(rendered, f, ensure_ascii=False, indent=2)

            success = self.run_comfy_api(rendered)

            # tile_img.save(self.temp_file)
            # abs_tile_path = os.path.abspath(self.temp_file)

            # # Prepare the (optional) ControlNet crop
            # control_crop_path = None
            # if self.control_image_path:
            #     control_crop_path = self.crop_control_image_like_tile(target_size, tile_size, x, y)

            # # Build replacements for the workflow template
            # replacements = {
            #     # prompts
            #     "positivePrompt": pos_prompt,
            #     "negativePrompt": neg_prompt,
            #     # seed is per tile for variety
            #     "seed": random.randint(1, 999999999),
            #     # brightness node input
            #     "brightness": float(brightness),
            #     # color (inpainting) input image path for VHS_LoadImagePath (or equivalent)
            #     "colorTilePath": abs_tile_path,
            #     # ControlNet params
            #     "controlNetStrength": float(cn_strength),
            #     # If None -> put a known dummy so JSON stays valid; workflow should ignore if not wired
            #     "controlNetImagePath": control_crop_path if control_crop_path else abs_tile_path,
            # }

            # Render the workflow and run
            # rendered = self.render_workflow(replacements)
            # success = self.run_comfy_api(rendered)
            # if not success:
            #     print("    FAILED render or run")
            #     continue

            # Insert processed tile back with alpha blending
            success = self.insert_tile_with_alpha(canvas, x, y, tile_size)
            if not success:
                print("    FAILED BLEND")
                continue

            # Save progress periodically (ignored in test_run if only 1 tile)
            if not self.test_run and (i + 1) % self.progress_save_interval == 0:
                self.save_progress(canvas, step_name, i + 1)
                print("    [PROGRESS SAVED]")

        # Save final result for this scale
        final_image = Image.fromarray(canvas)
        final_image.save(output_file, format="PNG")
        final_image.close()
        print(f"✅ {step_name} completed: {output_file}")
        if os.path.exists(upscaled_path):
            os.remove(upscaled_path)
        return output_file

    def save_progress(self, canvas, step_name, tile_index):
        progress_filename = os.path.join(self.output_dir, f"progress_{step_name.lower()}_{tile_index:04d}.png")
        Image.fromarray(canvas).save(progress_filename)
        print(f"    💾 Progress saved: {progress_filename}")

    # ----------------------------
    # Pipeline driver
    # ----------------------------
    def run_full_pipeline(self):
        print("\nStarting Multi-Scale Color-Guided Pipeline...")
        print("=" * 70)
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

        for i in range(start_index, len(self.scale_steps)):
            step_config = self.scale_steps[i]
            if i in self.existing_files and not self.force_rebuild:
                print(f"\n⏭️ Skipping {step_config['name']} - already exists")
                current_input = step_config["output_file"]
                continue
            step_start_time = time.time()
            result_file = self.process_scale_step(current_input, step_config)
            current_input = result_file
            step_duration = time.time() - step_start_time
            print(f"Step {i + 1} completed in {step_duration / 60:.1f} minutes")
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            if os.path.exists(self.temp_control_file):
                os.remove(self.temp_control_file)
        print("\n" + "=" * 70)
        print("🎉 FULL COLOR-GUIDED PIPELINE COMPLETED!")
        print("Results:")
        for step in self.scale_steps:
            if os.path.exists(step["output_file"]):
                file_size = os.path.getsize(step["output_file"]) / (1024 * 1024)
                print(f"  {step['name']}: {step['output_file']} ({file_size:.1f} MB)")


# ----------------------------
# Example main (unchanged; edit paths/placeholders in your workflow JSON)
# ----------------------------
def main():
    input_image = "input_512.png"
    color_map = "color_map_512.png"
    control_image = "ctrl.png"  # e.g. "C:/_Main/ai/control_depth_16k.png"
    mask_image = "mask.png"
    workflow_json = "mapWorkflow.json"  # must contain {placeholders}

    output_directory = "output"
    progress_save_interval = 300
    force_rebuild = False

    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found!")
        return
    if not os.path.exists(color_map):
        print(f"Error: Color map '{color_map}' not found!")
        return

    pipeline = MultiScaleGridInpainterResumeColorGuided(
        input_image_path=input_image,
        color_map_path=color_map,
        control_image_path=control_image,
        mask_path=mask_image,
        temp_file="q.png",
        comfy_server="127.0.0.1:8188",
        workflow_path=workflow_json,
        progress_save_interval=progress_save_interval,
        force_rebuild=force_rebuild,
        output_dir=output_directory,
        test_run=False,          # set False for full runs
        test_pos=(0.5, 0.5),    # center tile
    )
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
