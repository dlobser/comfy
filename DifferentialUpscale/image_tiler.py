#!/usr/bin/env python3
"""
Image Tiler for 8K -> 16x 2K Tile Processing
Splits 8K images into overlapping 2K tiles for ComfyUI processing
"""

import os
import json
import math
from PIL import Image
from pathlib import Path
import argparse

class ImageTiler:
    def __init__(self, input_size=8192, tile_size=2048, overlap=256):
        """
        Initialize the tiler
        
        Args:
            input_size: Size of input image (assuming square)
            tile_size: Size of each tile
            overlap: Overlap between tiles in pixels
        """
        self.input_size = input_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        
        # Calculate grid dimensions (4x4 for 8K->2K)
        self.grid_cols = math.ceil((input_size - overlap) / self.stride)
        self.grid_rows = math.ceil((input_size - overlap) / self.stride)
        
        print(f"Grid: {self.grid_rows}x{self.grid_cols} = {self.grid_rows * self.grid_cols} tiles")
        
    def split_image(self, image_path, output_dir):
        """
        Split image into overlapping tiles
        
        Args:
            image_path: Path to input 8K image
            output_dir: Directory to save tiles
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load image
        img = Image.open(image_path)
        
        # Resize to exact input size if needed
        if img.size != (self.input_size, self.input_size):
            print(f"Resizing from {img.size} to {self.input_size}x{self.input_size}")
            img = img.resize((self.input_size, self.input_size), Image.Resampling.LANCZOS)
        
        tiles_info = []
        tile_index = 0
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate tile position
                x = col * self.stride
                y = row * self.stride
                
                # Ensure we don't go beyond image bounds
                x = min(x, self.input_size - self.tile_size)
                y = min(y, self.input_size - self.tile_size)
                
                # Extract tile
                tile = img.crop((x, y, x + self.tile_size, y + self.tile_size))
                
                # Save tile
                tile_filename = f"tile_{tile_index:02d}_r{row}_c{col}.png"
                tile_path = output_path / tile_filename
                tile.save(tile_path)
                
                # Store tile info
                tile_info = {
                    'index': tile_index,
                    'row': row,
                    'col': col,
                    'x': x,
                    'y': y,
                    'width': self.tile_size,
                    'height': self.tile_size,
                    'filename': tile_filename
                }
                tiles_info.append(tile_info)
                
                print(f"Created tile {tile_index}: {tile_filename} at ({x}, {y})")
                tile_index += 1
        
        # Save metadata
        metadata = {
            'input_size': self.input_size,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'stride': self.stride,
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'total_tiles': len(tiles_info),
            'tiles': tiles_info
        }
        
        metadata_path = output_path / "tiles_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Split complete: {len(tiles_info)} tiles created")
        return metadata
    
    def create_blend_mask(self, tile_size, overlap):
        """
        Create a blend mask for seamless tile blending
        
        Args:
            tile_size: Size of the tile
            overlap: Overlap size
            
        Returns:
            PIL Image mask
        """
        mask = Image.new('L', (tile_size, tile_size), 255)
        
        # Create fade zones on edges
        fade_size = overlap // 2
        
        # Top fade
        for y in range(fade_size):
            alpha = int(255 * (y / fade_size))
            for x in range(tile_size):
                mask.putpixel((x, y), alpha)
        
        # Bottom fade
        for y in range(tile_size - fade_size, tile_size):
            alpha = int(255 * ((tile_size - y) / fade_size))
            for x in range(tile_size):
                mask.putpixel((x, y), alpha)
        
        # Left fade
        for x in range(fade_size):
            alpha = int(255 * (x / fade_size))
            for y in range(fade_size, tile_size - fade_size):
                mask.putpixel((x, y), alpha)
        
        # Right fade
        for x in range(tile_size - fade_size, tile_size):
            alpha = int(255 * ((tile_size - x) / fade_size))
            for y in range(fade_size, tile_size - fade_size):
                mask.putpixel((x, y), alpha)
        
        return mask
    
    def reassemble_image(self, tiles_dir, output_path, metadata_file=None):
        """
        Reassemble processed tiles back into full image
        
        Args:
            tiles_dir: Directory containing processed tiles
            output_path: Path for output image
            metadata_file: Path to metadata JSON (optional)
        """
        tiles_path = Path(tiles_dir)
        
        # Load metadata
        if metadata_file is None:
            metadata_file = tiles_path / "tiles_metadata.json"
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create output image
        output_img = Image.new('RGB', (self.input_size, self.input_size), (0, 0, 0))
        weight_map = Image.new('L', (self.input_size, self.input_size), 0)
        
        # Create blend mask
        blend_mask = self.create_blend_mask(self.tile_size, self.overlap)
        
        for tile_info in metadata['tiles']:
            # Load processed tile
            tile_filename = tile_info['filename']
            # Look for processed version (might have different naming)
            processed_filename = tile_filename.replace('.png', '_processed.png')
            
            tile_path = tiles_path / processed_filename
            if not tile_path.exists():
                # Try original filename
                tile_path = tiles_path / tile_filename
                if not tile_path.exists():
                    print(f"Warning: Tile {tile_filename} not found, skipping")
                    continue
            
            tile = Image.open(tile_path)
            
            # Get position
            x, y = tile_info['x'], tile_info['y']
            
            # Blend tile into output
            # Extract current region
            current_region = output_img.crop((x, y, x + self.tile_size, y + self.tile_size))
            current_weights = weight_map.crop((x, y, x + self.tile_size, y + self.tile_size))
            
            # Blend using weighted average
            blended = Image.composite(tile, current_region, blend_mask)
            
            # Update weight map
            new_weights = Image.new('L', (self.tile_size, self.tile_size), 255)
            combined_weights = Image.composite(new_weights, current_weights, blend_mask)
            
            # Paste back
            output_img.paste(blended, (x, y))
            weight_map.paste(combined_weights, (x, y))
            
            print(f"Blended tile {tile_info['index']}")
        
        # Save result
        output_img.save(output_path)
        print(f"Reassembled image saved to: {output_path}")
        
        return output_img

def main():
    parser = argparse.ArgumentParser(description='Split/reassemble images for tiled processing')
    parser.add_argument('command', choices=['split', 'reassemble'], help='Operation to perform')
    parser.add_argument('--input', required=True, help='Input image path (for split) or tiles directory (for reassemble)')
    parser.add_argument('--output', required=True, help='Output directory (for split) or output image path (for reassemble)')
    parser.add_argument('--input-size', type=int, default=8192, help='Input image size')
    parser.add_argument('--tile-size', type=int, default=2048, help='Tile size')
    parser.add_argument('--overlap', type=int, default=256, help='Overlap between tiles')
    
    args = parser.parse_args()
    
    tiler = ImageTiler(args.input_size, args.tile_size, args.overlap)
    
    if args.command == 'split':
        tiler.split_image(args.input, args.output)
    elif args.command == 'reassemble':
        tiler.reassemble_image(args.input, args.output)

if __name__ == '__main__':
    main()
