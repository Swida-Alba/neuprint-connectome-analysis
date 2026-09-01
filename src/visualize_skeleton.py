"""
3D Neuron Skeleton Visualization Module
========================================

This module provides the `VisualizeSkeleton` class for interactive 3D visualization
of neuron skeletons, synapses, and brain region meshes across multiple connectome datasets.

Supported Datasets
------------------
- **NeuPrint datasets**: hemibrain:v1.2.1, optic-lobe:v1.1, manc:v1.0, male-cns:v0.9
- **FlyWire/FAFB datasets**: flywire_FAFB_v783, flywire_BANC_v626

Key Features
------------
- **Multi-dataset support**: Seamlessly work with NeuPrint and FlyWire datasets
- **Skeleton visualization**: Render neurons as meshes or lines with automatic simplification
- **Synapse plotting**: Visualize synapses as scatter points, spheres, cones, or tetrahedrons
- **ROI meshes**: Display brain region meshes with automatic bilateral expansion
- **Soma-aware simplification**: Preserve cell body detail while simplifying branches (FAFB)
- **Extrusion auto-fix**: Detect and replace distorted skeletons automatically (FAFB)
- **Caching system**: Efficient caching for skeletons, meshes, and analysis results
- **Export options**: Save as HTML, PNG images, or video animations

Quick Start
-----------
```python
from coana import VisualizeSkeleton

# Basic neuron visualization
vs = VisualizeSkeleton(
    dataset='hemibrain:v1.2.1',
    neuron_layers=['DA1_lPN'],
)
vs.plot_skeleton()

# FlyWire with automatic extrusion fixes
vs = VisualizeSkeleton(
    dataset='flywire_FAFB_v783',
    neuron_layers=['MTe50'],
    auto_fix_extrusions=True,  # Auto-detect and fix distorted skeletons
)
vs.plot_skeleton()
```

Main Class
----------
VisualizeSkeleton : dataclass
    3D visualization of neuron skeletons with synapses and brain ROI meshes.
    
    Key attributes:
    - dataset: Dataset identifier (e.g., 'hemibrain:v1.2.1', 'flywire_FAFB_v783')
    - neuron_layers: List of neuron types or body IDs to visualize
    - skeleton_mode: 'tube' (mesh) or 'line' rendering
    - auto_fix_extrusions: Automatically detect and replace distorted skeletons (FAFB)
    - cache_neurons: Compatibility flag for the former visualization cache;
      all NeuPrint render modes use the shared raw SWC source
    
    Key methods:
    - plot_skeleton(): Generate 3D visualization
    - list_available_rois(): Show available brain regions for current dataset

FAFB-Specific Features
----------------------
For FlyWire/FAFB datasets, the system includes:
- **Soma-aware mesh simplification**: Preserves cell body detail
  (fast skeleton_mesh_simplification=0.90; fine/artistic=0.95;
  soma_mesh_simplification=0.8)
- **Automatic extrusion detection**: Identifies and replaces distorted skeletons
  (auto_fix_extrusions=True by default)
- **Parquet-based caching**: Efficient storage of extrusion check results
- **API fallback**: Fetches fresh meshes from CAVE API when needed, with a
  local branch-pruning fallback when an affected tree cannot be replaced

Performance Notes
-----------------
- First run downloads and caches data (may take several minutes)
- Subsequent runs use cached data (typically <10 seconds)
- For FAFB: Auto-extrusion detection adds ~30-60s on first run, cached thereafter
- Use skeleton_mesh_simplification=0.95-0.99 for faster rendering of large scenes

See Also
--------
- FAFB_INTEGRATION.md: Detailed guide for FlyWire/FAFB usage
- INSTALLATION.md: Setup and configuration instructions
- OUTPUT_FILES.md: Documentation of output formats and file structure
"""

import json
import os
import re
import sys
import shutil
import time
import signal
import gc
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import logging
import warnings
from contextlib import contextmanager
from tqdm import tqdm

# Suppress FutureWarning from neuprint about Series.__getitem__
warnings.filterwarnings("ignore", category=FutureWarning, module="neuprint")

# Suppress navis warnings about missing/invalid radii
# These are expected when skeletons have no radius info and are handled automatically
warnings.filterwarnings("ignore", message=".*radii are missing or <= 0.*")
warnings.filterwarnings("ignore", message=".*Mesh will look funny.*")

import numpy as np
import pandas as pd
import cv2
import navis
# Configure navis logger to suppress warnings about missing radii
logging.getLogger("navis").setLevel(logging.ERROR)

import navis.interfaces.neuprint as neu
from neuprint import Client, fetch_synapse_connections, SynapseCriteria, fetch_meta
import plotly.graph_objects as go
import bokeh.palettes

# Import color utilities
from utils.color_utils import (
    standardize_color,
    color_to_hex,
    color_to_rgba_string,
    color_has_explicit_alpha,
    extract_rgba_tuple,
    extract_rgb_tuple,
    is_dark_color,
    set_alpha,
)
from utils.flywire_readiness import (
    flywire_skeleton_readiness,
    require_flywire_skeleton_access,
)
try:
    from .flywire_ids import (
        body_id_to_api_int,
        is_flywire_dataset,
        normalize_flywire_body_id,
        normalize_flywire_body_ids,
        resolve_flywire_dataset_dir,
    )
except ImportError:
    from flywire_ids import (
        body_id_to_api_int,
        is_flywire_dataset,
        normalize_flywire_body_id,
        normalize_flywire_body_ids,
        resolve_flywire_dataset_dir,
    )
try:
    from flywire_mesh_cache import (
        FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        FlyWireMeshCache,
        flywire_mesh_cache_key,
        parse_soma_position,
        simplify_mesh_fine,
        simplify_mesh_with_soma_awareness,
    )
except ImportError:
    from .flywire_mesh_cache import (
        FLYWIRE_MESH_CACHE_SIMPLIFICATION,
        FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
        FlyWireMeshCache,
        flywire_mesh_cache_key,
        parse_soma_position,
        simplify_mesh_fine,
        simplify_mesh_with_soma_awareness,
    )
try:
    from skeleton_simplification import (
        FAFB_FAST_NODE_RETENTION,
        FAFB_LINE_NODE_REDUCTION,
        NEUPRINT_LINE_NODE_REDUCTION,
        simplify_skeleton_nodes,
    )
except ImportError:
    from .skeleton_simplification import (
        FAFB_FAST_NODE_RETENTION,
        FAFB_LINE_NODE_REDUCTION,
        NEUPRINT_LINE_NODE_REDUCTION,
        simplify_skeleton_nodes,
    )

try:
    from synapse_cache import SynapseCache
except ImportError:
    from .synapse_cache import SynapseCache


# Plotly's legend sorts by ``legendrank`` and coerces missing ranks to 0, so
# unranked traces cluster into the first legend group and appear BETWEEN two
# neuron entries. Pin every mesh family to an explicit rank band above all
# neuron/site ranks (which grow by 100 per group): ROI meshes first, then the
# brain mesh, with the VNC mesh always last.
ROI_MESH_LEGEND_RANK_BASE = 100_000_000
BRAIN_MESH_LEGEND_RANK = 200_000_000
VNC_MESH_LEGEND_RANK = 200_000_001

# Legend modes. 'tree' renders exactly like 'type' (same native legend,
# used by static exports) and additionally tags traces so the exported
# interactive HTML can embed a collapsible type -> neuron legend panel.
LEGEND_MODES = ('single', 'type', 'tree', 'layer')


def _configure_roi_mesh_traces(mesh_traces, roi_name, legend_rank=None):
    """Give one resolved ROI its own Plotly legend entry and trace group."""
    legend_group = f'roi_mesh:{roi_name}'
    display_name = f'brain region [{roi_name}]'

    for trace_index, trace in enumerate(mesh_traces):
        trace.legendgroup = legend_group
        trace.showlegend = trace_index == 0
        trace.name = display_name
        trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
        trace.hoverinfo = 'name'
        if legend_rank is not None:
            trace.legendrank = legend_rank

    return mesh_traces


# Timeout exception for PNG export

def _detect_content_bounds(img, background_color=(255, 255, 255)):
        """
        Detect the bounding box of content in an image.

        Parameters
        ----------
        img : PIL.Image
            Input image
        background_color : tuple
            RGB tuple for background color
            
        Returns
        -------
        tuple or None
            (row_min, row_max, col_min, col_max) or None if no content found
        """
        from PIL import Image
        import numpy as np
        
        # Convert to RGB if needed
        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, background_color)
            bg.paste(img, mask=img.split()[3])
            img_rgb = bg
        else:
            img_rgb = img.convert('RGB')
        
        arr = np.array(img_rgb)
        
        tolerance = 10
        bg_r, bg_g, bg_b = background_color
        
        non_bg_mask = (
            (np.abs(arr[:, :, 0].astype(int) - bg_r) > tolerance) |
            (np.abs(arr[:, :, 1].astype(int) - bg_g) > tolerance) |
            (np.abs(arr[:, :, 2].astype(int) - bg_b) > tolerance)
        )
        
        rows = np.any(non_bg_mask, axis=1)
        cols = np.any(non_bg_mask, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        
        return (row_min, row_max, col_min, col_max)

def _compute_unified_crop_bounds(image_paths, background_color=(255, 255, 255), sample_count=None):
        """
        Compute unified crop bounds across multiple images.
        
        This method finds the maximum extent (union) of content across all images
        to ensure consistent cropping during video rotation.
        
        Parameters
        ----------
        image_paths : list
            List of image file paths to analyze
        background_color : tuple
            RGB tuple for background color
        sample_count : int, optional
            Number of evenly-spaced frames to sample (None = all frames)
            
        Returns
        -------
        tuple or None
            (row_min, row_max, col_min, col_max) unified bounds or None if no content
        """
        from PIL import Image
        
        if not image_paths:
            return None
        
        # Determine which frames to sample
        if sample_count is not None and sample_count < len(image_paths):
            # Sample evenly across all frames
            indices = [int(i * len(image_paths) / sample_count) for i in range(sample_count)]
            sampled_paths = [image_paths[i] for i in indices]
        else:
            sampled_paths = image_paths
        
        # Initialize with None (will expand with each frame)
        unified_row_min = None
        unified_row_max = None
        unified_col_min = None
        unified_col_max = None
        
        for path in sampled_paths:
            try:
                img = Image.open(path)
                bounds = _detect_content_bounds(img, background_color)
                img.close()
                
                if bounds is None:
                    continue
                    
                row_min, row_max, col_min, col_max = bounds
                
                # Expand unified bounds
                if unified_row_min is None:
                    unified_row_min = row_min
                    unified_row_max = row_max
                    unified_col_min = col_min
                    unified_col_max = col_max
                else:
                    unified_row_min = min(unified_row_min, row_min)
                    unified_row_max = max(unified_row_max, row_max)
                    unified_col_min = min(unified_col_min, col_min)
                    unified_col_max = max(unified_col_max, col_max)
            except Exception:
                # Skip frames that can't be read
                continue
        
        if unified_row_min is None:
            return None
            
        return (unified_row_min, unified_row_max, unified_col_min, unified_col_max)

def _apply_consistent_crop(pic_folder, margin=20, background_color=(255, 255, 255)):
        """
        Apply consistent cropping to all images in a folder based on unified bounds.
        
        This method:
        1. Detects content bounds across all frames
        2. Computes the union (maximum extent) of all bounds
        3. Crops all frames to the same unified bounds + margin
        
        Parameters
        ----------
        pic_folder : str
            Folder containing the frame images
        margin : int
            Margin to preserve around content
        background_color : tuple
            RGB tuple for background color
            
        Returns
        -------
        tuple
            (width, height) of the cropped images, or None if failed
        """
        from PIL import Image
        import glob
        
        # Find all JPEG images in folder
        image_paths = sorted(glob.glob(os.path.join(pic_folder, 'deg_*.jpeg')))
        
        if not image_paths:
            return None
        
        # Compute unified bounds across all frames
        # Sample 36 frames (every 10 degrees) for efficiency on long videos
        unified_bounds = _compute_unified_crop_bounds(
            image_paths, background_color, sample_count=min(36, len(image_paths))
        )
        
        if unified_bounds is None:
            return None
        
        row_min, row_max, col_min, col_max = unified_bounds
        
        # Get image dimensions from first image
        with Image.open(image_paths[0]) as img:
            img_height = img.height
            img_width = img.width
        
        # Add margin to unified bounds
        row_min = max(0, row_min - margin)
        row_max = min(img_height - 1, row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(img_width - 1, col_max + margin)
        
        final_width = col_max - col_min + 1
        final_height = row_max - row_min + 1
        
        # Apply consistent crop to all images
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    # Crop to unified bounds (no additional margin - already added)
                    cropped = img.crop((col_min, row_min, col_max + 1, row_max + 1))
                    # Convert to RGB for JPEG
                    if cropped.mode == 'RGBA':
                        rgb_img = Image.new('RGB', cropped.size, background_color)
                        rgb_img.paste(cropped, mask=cropped.split()[3])
                        cropped = rgb_img
                    elif cropped.mode != 'RGB':
                        cropped = cropped.convert('RGB')
                    cropped.save(path, 'JPEG', quality=95)
            except Exception:
                # Log but continue
                pass
        
        return (final_width, final_height)

class PNGExportTimeout(Exception):
    """Raised when PNG export times out."""
    pass

def _timeout_handler(signum, frame):
    """Signal handler for PNG export timeout."""
    raise PNGExportTimeout("PNG export timed out")


class WebDriverExportSession:
    """
    Context manager for WebDriver-based exports that keeps the browser open
    for efficient multi-view or video frame exports.
    
    Opens the browser once and allows:
    - Loading HTML once and exporting multiple views by rotating camera
    - Exporting video frames without reopening browser
    
    File Loading
    ------------
    Files are served via a local HTTP server for maximum reliability.
    This approach works with files of any size (tested up to 200MB+).
    
    Export Method
    -------------
    Uses canvas.toDataURL() for fast, high-quality screenshots.
    This method is memory-efficient and produces excellent quality output.
    
    Usage:
    ------
    ```python
    with WebDriverExportSession(width=1200, height=900, scale=2) as session:
        session.load_html('/path/to/figure.html')
        for view_name, camera in cameras.items():
            session.set_camera(camera['eye'], camera['up'], camera.get('center'))
            session.screenshot(f'output_{view_name}.png')
    ```
    """
    
    def __init__(self, width=1200, height=900, scale=2, timeout=300, render_wait=None):
        """
        Initialize WebDriver session parameters.
        
        Parameters
        ----------
        width : int
            Browser viewport width
        height : int
            Browser viewport height  
        scale : int
            Scale factor for high DPI rendering (2 = 144 DPI, 3 = 216 DPI)
        timeout : int
            Timeout in seconds for page load (default 300s for large HTML files)
        render_wait : float, optional
            Fixed render wait time in seconds. If None, auto-calibrated.
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.timeout = timeout
        self.driver = None
        self._loaded_url = None
        self._render_wait = render_wait if render_wait is not None else 0.3
        self._render_wait_fixed = render_wait is not None  # Skip calibration if fixed
        self._initial_camera = None  # Store initial camera from HTML
        self._screenshot_count = 0  # Track screenshot count for periodic memory cleanup
        self._html_size_mb = 0.0  # Track loaded HTML file size for memory management
        self._cleanup_interval = 15  # Default cleanup interval (frames)
        self._http_server = None  # Local HTTP server for large files
        self._http_server_port = None
        
    def __enter__(self):
        """Initialize the WebDriver and return self."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service as ChromeService
            from selenium.webdriver.chrome.options import Options as ChromeOptions
        except ImportError:
            raise ImportError("selenium is required. Install with: pip install selenium webdriver-manager")
        
        # Scale the window size directly for high-resolution output
        # This is more reliable than deviceScaleFactor for WebGL content
        scaled_width = self.width * self.scale
        scaled_height = self.height * self.scale
        
        chrome_options = ChromeOptions()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'--window-size={scaled_width},{scaled_height}')
        
        # Use new headless mode (Chrome 109+) which supports WebGL better than old headless
        # This replaces the offscreen/position hack which can be flaky on macOS
        chrome_options.add_argument('--headless=new')
        
        # Enable high-quality rendering for WebGL
        chrome_options.add_argument('--use-gl=angle')  # Use ANGLE for better WebGL
        chrome_options.add_argument('--enable-webgl')
        chrome_options.add_argument('--enable-webgl2')
        chrome_options.add_argument('--disable-software-rasterizer')
        
        # Disable animations and GPU compositing to reduce flashing
        chrome_options.add_argument('--disable-gpu-compositing')
        chrome_options.add_argument('--disable-smooth-scrolling')
        # Memory and stability settings for large HTML files (>100MB)
        # --expose-gc enables window.gc() for explicit garbage collection
        chrome_options.add_argument('--js-flags=--max-old-space-size=8192 --expose-gc')
        chrome_options.add_argument('--disable-features=RendererCodeIntegrity')
        chrome_options.add_argument('--disable-backgrounding-occluded-windows')
        chrome_options.add_argument('--disable-renderer-backgrounding')
        chrome_options.add_argument('--memory-pressure-off')
        # Increase shared memory size (default /dev/shm is often too small)
        chrome_options.add_argument('--shm-size=2gb')
        # GPU memory for large WebGL scenes
        chrome_options.add_argument('--enable-gpu-rasterization')
        chrome_options.add_argument('--ignore-gpu-blocklist')
        # Limit GPU memory to prevent runaway allocation
        chrome_options.add_argument('--gpu-memory-buffer-count=4')
        
        # Initialize ChromeDriver using webdriver-manager (cross-platform) or system Chrome
        # webdriver-manager automatically downloads and caches the correct ChromeDriver version
        # Cache location: ~/.wdm/drivers/chromedriver/ (managed automatically)
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            service = ChromeService(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except ImportError:
            # webdriver-manager not installed, try system Chrome directly
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
            except Exception as e:
                raise RuntimeError(
                    f"Could not initialize Chrome WebDriver: {e}\n\n"
                    f"WebDriver export requires:\n"
                    f"  1. Google Chrome browser (version 109+)\n"
                    f"  2. selenium package: pip install selenium\n"
                    f"  3. webdriver-manager package: pip install webdriver-manager\n\n"
                    f"Install with: pip install selenium webdriver-manager\n"
                    f"Or use export_method='kaleido' as fallback."
                )
        except Exception as e:
            # webdriver-manager failed (version mismatch, network error, etc.)
            # Try system Chrome directly
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
            except Exception as e2:
                raise RuntimeError(
                    f"Could not initialize Chrome WebDriver: {e2}\n\n"
                    f"Common causes:\n"
                    f"  - Chrome not installed or not found in PATH\n"
                    f"  - Chrome version mismatch with ChromeDriver\n"
                    f"  - Network error downloading ChromeDriver\n\n"
                    f"Solutions:\n"
                    f"  1. Install/update Google Chrome (version 109+ required for --headless=new)\n"
                    f"  2. Ensure webdriver-manager is installed: pip install webdriver-manager\n"
                    f"  3. Check internet connection for ChromeDriver download\n"
                    f"  4. Use export_method='kaleido' as fallback\n\n"
                    f"Original error: {e}"
                )
        
        # Set up viewport size at initialization using CDP
        # Use scaled dimensions directly for accurate high-resolution WebGL export
        try:
            self.driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
                'width': self.width * self.scale,
                'height': self.height * self.scale,
                'deviceScaleFactor': 1,  # Use 1 since we already scaled the viewport
                'mobile': False
            })
            self._viewport_configured = True
        except Exception:
            self._viewport_configured = False
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up WebDriver and HTTP server."""
        # Stop HTTP server if running
        self._stop_http_server()
        
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
        return False
    
    def load_html(self, html_path, wait_for_render=True, render_wait=3, background_color='white'):
        """
        Load an HTML file into the browser via local HTTP server.
        
        Uses a local HTTP server to serve the file, which is more reliable than
        document.write() injection and works with files of any size.
        
        Parameters
        ----------
        html_path : str
            Path to the HTML file
        wait_for_render : bool
            If True, wait for Plotly to render
        render_wait : float
            Additional seconds to wait after Plotly detected
        background_color : str
            Background color for the page body (to match plot background)
        """
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        import time
        
        # Set page load and script timeouts for large HTML files
        self.driver.set_page_load_timeout(self.timeout)
        self.driver.set_script_timeout(self.timeout)
        
        # Track HTML file size for adaptive memory management
        # Larger files require more aggressive cleanup intervals
        self._html_size_mb = os.path.getsize(html_path) / (1024 * 1024)
        if self._html_size_mb > 30:
            # Very large files (>30MB): cleanup every 5 frames
            self._cleanup_interval = 5
        elif self._html_size_mb > 10:
            # Large files (>10MB): cleanup every 10 frames
            self._cleanup_interval = 10
        else:
            # Normal files: cleanup every 15 frames
            self._cleanup_interval = 15
        
        # Always use HTTP server for reliability with any file size
        self._start_http_server_for_file(html_path)
        url = f'http://127.0.0.1:{self._http_server_port}/{os.path.basename(html_path)}'
        self.driver.get(url)
        
        # Ensure no body margin/padding and set background color
        # This prevents white margins around the plot
        js_bg = f"document.body.style.margin = '0'; document.body.style.padding = '0'; document.body.style.backgroundColor = '{background_color}';"
        self.driver.execute_script(js_bg)
        
        self._loaded_url = os.path.abspath(html_path)
        
        if wait_for_render:
            wait = WebDriverWait(self.driver, self.timeout)
            # Wait for js-plotly-plot class (not just 'plotly') as this is added after render
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "js-plotly-plot")))
            time.sleep(render_wait)
            
            # Re-apply background color styling just in case page load reset it
            self.driver.execute_script(js_bg)
            
            # Set up the viewport and figure size ONCE to avoid flashing on every screenshot
            # This must be done after Plotly is loaded
            self._setup_viewport_size(background_color)
            
            # Force an initial draw to ensure WebGL canvas has content
            # This prevents blank images on first screenshot
            self._force_initial_draw()
            
            # Read initial camera from HTML - this is the "front view" reference
            self._initial_camera = self.get_current_camera()
            
            # Calibrate render wait time based on machine performance (unless fixed)
            if not self._render_wait_fixed:
                self._calibrate_render_wait()
    
    def _start_http_server_for_file(self, html_path):
        """
        Start a local HTTP server to serve a large HTML file.
        
        This is used for files ≥50MB where document.write() injection
        would exceed WebDriver protocol content-length limits.
        
        Parameters
        ----------
        html_path : str
            Path to the HTML file to serve
        """
        import http.server
        import socketserver
        import threading
        
        # Stop any existing server
        self._stop_http_server()
        
        # Get the directory containing the HTML file
        html_dir = os.path.dirname(os.path.abspath(html_path))
        
        # Find an available port
        for port in range(8765, 8865):
            try:
                # Create a handler that serves from the HTML file's directory
                handler = lambda *args, directory=html_dir, **kwargs: \
                    http.server.SimpleHTTPRequestHandler(*args, directory=directory, **kwargs)
                
                # Disable logging to avoid cluttering output
                class QuietHandler(http.server.SimpleHTTPRequestHandler):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, directory=html_dir, **kwargs)
                    def log_message(self, format, *args):
                        pass  # Suppress logging
                
                self._http_server = socketserver.TCPServer(("127.0.0.1", port), QuietHandler)
                self._http_server_port = port
                
                # Start server in a daemon thread
                server_thread = threading.Thread(target=self._http_server.serve_forever)
                server_thread.daemon = True
                server_thread.start()
                
                return
            except OSError:
                continue  # Port in use, try next
        
        raise RuntimeError("Could not find an available port for HTTP server")
    
    def _stop_http_server(self):
        """Stop the local HTTP server if running."""
        if self._http_server is not None:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                pass
            self._http_server = None
            self._http_server_port = None

    def _setup_viewport_size(self, background_color='white'):
        """
        Resize the Plotly figure to the full scaled resolution for high-quality export.
        This ensures the WebGL canvas renders at the target resolution.
        """
        import time
        
        # Resize the Plotly figure to fill the viewport at scaled resolution
        scaled_width = self.width * self.scale
        scaled_height = self.height * self.scale
        
        try:
            self.driver.execute_script("""
                var gd = document.querySelector('.js-plotly-plot');
                if (gd && Plotly) {
                    // Resize the figure to fill the viewport
                    Plotly.relayout(gd, {
                        width: arguments[0],
                        height: arguments[1],
                        'paper_bgcolor': arguments[2],
                        'plot_bgcolor': arguments[2]
                    });
                }
            """, scaled_width, scaled_height, background_color)
            
            # Wait for resize to complete
            time.sleep(0.5)
        except Exception:
            pass  # Non-critical, continue with original size
    
    def _force_initial_draw(self):
        """
        Force an initial WebGL draw to ensure canvas has content.
        This prevents blank images on first screenshot due to WebGL buffer clearing.
        """
        try:
            self.driver.execute_script("""
                var gd = document.querySelector('.js-plotly-plot');
                if (gd && gd._fullLayout && gd._fullLayout.scene) {
                    var scene = gd._fullLayout.scene._scene;
                    if (scene && scene.glplot) {
                        // Force WebGL to draw and keep the buffer
                        scene.glplot.draw();
                    }
                }
            """)
        except Exception:
            pass  # Not critical - screenshot will fall back to CDP
    
    def _calibrate_render_wait(self):
        """Calibrate render wait time based on machine performance."""
        import time
        
        if self._initial_camera is None:
            self._render_wait = 0.4
            return
        
        # Measure time for a small camera change
        eye = self._initial_camera.get('eye', {'x': 0, 'y': 0, 'z': -2.5})
        
        t0 = time.time()
        # Make a tiny camera adjustment
        self.driver.execute_script("""
            var gd = document.querySelector('.js-plotly-plot');
            if (gd && Plotly) {
                Plotly.relayout(gd, {'scene.camera.eye.x': arguments[0]});
            }
        """, eye.get('x', 0) + 0.001)
        
        # Wait for render to stabilize
        time.sleep(0.1)
        
        # Restore original
        self.driver.execute_script("""
            var gd = document.querySelector('.js-plotly-plot');
            if (gd && Plotly) {
                Plotly.relayout(gd, {'scene.camera.eye.x': arguments[0]});
            }
        """, eye.get('x', 0))
        
        elapsed = time.time() - t0
        
        # Set render wait based on observed performance
        # Minimum 0.2s, scale up for slower machines
        self._render_wait = max(0.2, min(0.6, elapsed * 1.5))
    
    def get_current_camera(self):
        """
        Get the current camera settings from the Plotly figure.
        
        Returns
        -------
        dict
            Camera settings with 'eye', 'up', 'center' keys
        """
        try:
            camera = self.driver.execute_script("""
                var gd = document.querySelector('.js-plotly-plot');
                if (gd && gd.layout && gd.layout.scene && gd.layout.scene.camera) {
                    return gd.layout.scene.camera;
                }
                return null;
            """)
            return camera if camera else {}
        except Exception:
            return {}
    
    def set_camera(self, eye, up=None, center=None):
        """
        Set the camera position using JavaScript.
        
        Parameters
        ----------
        eye : dict
            Camera eye position, e.g., {'x': 0, 'y': 2.5, 'z': 0}
        up : dict, optional
            Camera up vector, e.g., {'x': 0, 'y': 0, 'z': -1}
        center : dict, optional
            Camera center point
        """
        import time
        import json
        
        # Build camera object
        camera_obj = {}
        if eye:
            camera_obj['eye'] = eye
        if up:
            camera_obj['up'] = up
        if center:
            camera_obj['center'] = center
        
        camera_json = json.dumps(camera_obj)
        
        # Use Plotly.relayout which is the official API for camera updates
        # The key insight: flashing comes from deviceScaleFactor being applied
        # during relayout. We use relayout but ensure it's instant with no animation.
        js_code = f"""
        (function() {{
            var gd = document.querySelector('.js-plotly-plot');
            if (!gd || !Plotly) return;
            
            var camera = {camera_json};
            
            // Use relayout - the official Plotly API for updating camera
            // This is synchronous and doesn't cause resize if we only update camera
            Plotly.relayout(gd, {{'scene.camera': camera}});
        }})();
        """
        self.driver.execute_script(js_code)
        
        # Wait for rendering using calibrated time
        time.sleep(self._render_wait)
    
    def set_camera_for_rotation(self, eye_x, eye_y, eye_z, up_x=0, up_y=-1, up_z=0):
        """
        Set camera for rotation animation (convenience method).
        
        Parameters
        ----------
        eye_x, eye_y, eye_z : float
            Camera eye position
        up_x, up_y, up_z : float
            Camera up vector
        """
        self.set_camera(
            eye={'x': eye_x, 'y': eye_y, 'z': eye_z},
            up={'x': up_x, 'y': up_y, 'z': up_z}
        )
    
    def screenshot(self, output_path, convert_to_jpeg=False, jpeg_quality=100,
                   auto_crop=False, margin=20, background_color=(255, 255, 255),
                   use_webgl_export=True, fast_mode=True, auto_fast_mode=True):
        """
        Take a screenshot and save to file using canvas.toDataURL().
        
        Export method:
        1. WebGL canvas.toDataURL() - Fast, high-quality direct canvas capture
        2. CDP screenshot - Chrome DevTools Protocol screenshot as fallback
        
        Parameters
        ----------
        output_path : str
            Output file path
        convert_to_jpeg : bool
            If True, convert PNG to JPEG (smaller files)
        jpeg_quality : int
            JPEG quality (1-100)
        auto_crop : bool
            If True, automatically crop whitespace/background and add margin
        margin : int
            Margin (in pixels) to preserve around content when auto_crop=True
        background_color : tuple
            RGB tuple for background color detection (default white)
        use_webgl_export : bool, default True
            If True, try WebGL export method for higher quality.
            Falls back to CDP screenshot if WebGL export fails.
        fast_mode : bool, default True
            Legacy parameter, always True. Uses canvas.toDataURL() directly.
        auto_fast_mode : bool, default True
            Legacy parameter, ignored. Canvas mode is always used.
        """
        from PIL import Image
        import base64
        import time
        
        # Track screenshot count for periodic memory cleanup
        self._screenshot_count += 1
        
        # Always use fast_mode (canvas.toDataURL) - it's fast and high quality.
        # The former Method 1 (Plotly.toImage via execute_async_script) was only
        # reachable with fast_mode disabled, which never happens; it was removed
        # as dead code. Method 2 below handles all WebGL exports.
        fast_mode = True  # noqa: F841 - kept for signature/history clarity
        
        img = None
        temp_png = output_path.rsplit('.', 1)[0] + '_temp.png'
        scaled_width = self.width * self.scale
        scaled_height = self.height * self.scale
        
        # Method 2: Try WebGL canvas.toDataURL() with forced render
        if img is None and use_webgl_export:
            try:
                # Force render and immediately capture the WebGL canvas
                canvas_data = self.driver.execute_script("""
                    var gd = document.querySelector('.js-plotly-plot');
                    if (!gd) return null;
                    
                    // Force the scene to render
                    if (gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene) {
                        var scene = gd._fullLayout.scene._scene;
                        if (scene.render) scene.render();
                        
                        // Get the canvas
                        if (scene.glplot && scene.glplot.canvas) {
                            return scene.glplot.canvas.toDataURL('image/png');
                        }
                    }
                    
                    // Fallback: try direct canvas access from gl-container
                    var glCanvas = gd.querySelector('.gl-container canvas');
                    if (glCanvas) {
                        return glCanvas.toDataURL('image/png');
                    }
                    
                    return null;
                """)
                
                if canvas_data and canvas_data.startswith('data:image/png;base64,'):
                    base64_data = canvas_data.split(',')[1]
                    img_bytes = base64.b64decode(base64_data)
                    
                    if len(set(img_bytes[:1000])) > 1:
                        with open(temp_png, 'wb') as f:
                            f.write(img_bytes)
                        img = Image.open(temp_png)
                        
                        import numpy as np
                        arr = np.array(img)
                        if arr.max() == 0:
                            img = None
            except Exception:
                pass  # Will fall back to CDP
        
        # Method 3: Fall back to CDP screenshot (captures entire viewport)
        if img is None:
            try:
                # Wait a moment for any pending renders
                time.sleep(0.2)
                
                result = self.driver.execute_cdp_cmd('Page.captureScreenshot', {
                    'format': 'png',
                    'captureBeyondViewport': False,
                    'fromSurface': True,  # Capture from surface for better WebGL support
                    'clip': {
                        'x': 0,
                        'y': 0,
                        'width': scaled_width,
                        'height': scaled_height,
                        'scale': 1
                    }
                })
                screenshot_data = base64.b64decode(result['data'])
                with open(temp_png, 'wb') as f:
                    f.write(screenshot_data)
                img = Image.open(temp_png)
            except Exception:
                # Final fallback to regular screenshot
                self.driver.save_screenshot(temp_png)
                img = Image.open(temp_png)
        
        if auto_crop:
            img = self._auto_crop_image(img, margin, background_color)
        
        if convert_to_jpeg:
            img = img.convert('RGB')
            img.save(output_path, 'JPEG', quality=jpeg_quality)
        else:
            # For PNG output with auto_crop, save directly
            if auto_crop:
                img.save(output_path, 'PNG')
            else:
                # No processing needed, just rename
                os.rename(temp_png, output_path)
                return
        
        # Clean up temp file if it still exists
        if os.path.exists(temp_png):
            os.remove(temp_png)
        
        # Close PIL image to free memory
        if img is not None:
            try:
                img.close()
            except:
                pass
        img = None
        
        # Periodic memory cleanup to prevent Chrome crashes from accumulated WebGL resources
        # Chrome WebGL accumulates texture/buffer memory that isn't freed automatically
        # Interval is adaptive: more frequent for large HTML files (30MB+ = every 5 frames)
        # Note: For very large files (>30MB), using fast_mode=True is recommended
        if self._screenshot_count % self._cleanup_interval == 0:
            self._cleanup_chrome_memory()
            gc.collect()
    
    def _cleanup_chrome_memory(self):
        """
        Clean up Chrome memory to prevent crashes from accumulated WebGL resources.
        
        Chrome's WebGL implementation can accumulate texture and buffer memory
        that isn't automatically garbage collected. This forces cleanup.
        Also clears JavaScript variables that hold large data URLs.
        """
        try:
            # Clear WebGL context caches and force garbage collection in browser
            self.driver.execute_script("""
                // Clear any global variables that might hold image data
                if (window._lastImageData) {
                    window._lastImageData = null;
                }
                
                // Force WebGL context cleanup
                var gd = document.querySelector('.js-plotly-plot');
                if (gd && gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene) {
                    var scene = gd._fullLayout.scene._scene;
                    if (scene.glplot) {
                        var gl = scene.glplot.gl;
                        if (gl) {
                            // Flush pending GL commands
                            gl.flush();
                            gl.finish();
                            
                            // Clear WebGL caches by triggering texture/buffer cleanup
                            // This helps release GPU memory
                            var numTextureUnits = gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
                            for (var i = 0; i < numTextureUnits; i++) {
                                gl.activeTexture(gl.TEXTURE0 + i);
                                gl.bindTexture(gl.TEXTURE_2D, null);
                                gl.bindTexture(gl.TEXTURE_CUBE_MAP, null);
                            }
                            gl.bindBuffer(gl.ARRAY_BUFFER, null);
                            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
                            gl.bindRenderbuffer(gl.RENDERBUFFER, null);
                            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                        }
                    }
                }
                
                // Clear Plotly's internal image cache if it exists
                if (window.Plotly && window.Plotly.Plots && window.Plotly.Plots.resize) {
                    // Trigger a resize to flush internal caches
                    try {
                        window.Plotly.Plots.resize(gd);
                    } catch(e) {}
                }
                
                // Request browser garbage collection if available (Chrome with --js-flags=--expose-gc)
                if (window.gc) {
                    window.gc();
                }
            """)
        except Exception:
            pass  # Non-critical, continue if cleanup fails
    
    def _auto_crop_image(self, img, margin=20, background_color=(255, 255, 255)):
        """
        Auto-crop image by detecting content boundaries against background.
        
        Parameters
        ----------
        img : PIL.Image
            Input image
        margin : int
            Margin to preserve around content
        background_color : tuple
            RGB tuple for background color
            
        Returns
        -------
        PIL.Image
            Cropped image with margin
        """
        bounds = self._detect_content_bounds(img, background_color)
        if bounds is None:
            return img
        
        row_min, row_max, col_min, col_max = bounds
        
        # Add margin
        row_min = max(0, row_min - margin)
        row_max = min(img.height - 1, row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(img.width - 1, col_max + margin)
        
        # Crop the original image (preserve RGBA if present)
        cropped = img.crop((col_min, row_min, col_max + 1, row_max + 1))
        
        return cropped
    
    def _detect_content_bounds(self, img, background_color=(255, 255, 255)):
        return _detect_content_bounds(img, background_color)
    def _compute_unified_crop_bounds(self, image_paths, background_color=(255, 255, 255), 
                                      sample_count=None):
        return _compute_unified_crop_bounds(image_paths, background_color, sample_count)
    def _apply_consistent_crop(self, pic_folder, margin=20, background_color=(255, 255, 255)):
        return _apply_consistent_crop(pic_folder, margin, background_color)
    def set_trace_visibility(self, visible_indices: list, total_traces: int):
        """
        Set visibility of traces by index using JavaScript.
        
        Parameters
        ----------
        visible_indices : list
            List of trace indices to make visible (0-indexed)
        total_traces : int
            Total number of traces in the figure
        """
        import json
        import time
        
        # Build visibility array
        visibility = [False] * total_traces
        for idx in visible_indices:
            if 0 <= idx < total_traces:
                visibility[idx] = True
        
        visibility_json = json.dumps(visibility)
        
        js_code = f"""
        var gd = document.querySelector('.js-plotly-plot');
        if (gd && Plotly) {{
            Plotly.restyle(gd, {{'visible': {visibility_json}}});
        }}
        """
        self.driver.execute_script(js_code)
        time.sleep(0.15)  # Brief wait for restyle
    
    def update_layout(self, layout_update: dict):
        """
        Update layout properties using JavaScript.
        
        Parameters
        ----------
        layout_update : dict
            Dictionary of layout properties to update
        """
        import json
        import time
        
        layout_json = json.dumps(layout_update)
        
        js_code = f"""
        var gd = document.querySelector('.js-plotly-plot');
        if (gd && Plotly) {{
            Plotly.relayout(gd, {layout_json});
        }}
        """
        self.driver.execute_script(js_code)
        time.sleep(0.1)


def export_individuals_webdriver(
    html_path: str,
    output_dir: str,
    legend_entries: dict,
    background_indices: list,
    total_traces: int,
    views: list,
    view_cameras: dict,
    scale: int = 2,
    width: int = 900,
    height: int = 900,
    timeout: int = 60,
    verbose: bool = True,
    auto_crop: bool = True,
    crop_margin: int = 30,
    background_color: str = 'white'
) -> dict:
    """
    Export individual neuron plots efficiently using WebDriver.
    
    Opens the HTML once and toggles trace visibility via JavaScript
    to export each individual neuron/type with all requested views.
    This avoids reopening the browser for each export, significantly
    improving performance for large numbers of individuals.
    
    Parameters
    ----------
    html_path : str
        Path to the main HTML figure file
    output_dir : str
        Directory to save exported PNG files
    legend_entries : dict
        Dictionary mapping legend names to lists of trace indices
        e.g., {'MTe04': [0, 1, 2], 'MTe50': [3, 4, 5]}
    background_indices : list
        List of trace indices to always keep visible (meshes, ROIs)
    total_traces : int
        Total number of traces in the figure
    views : list
        List of view names to export, e.g., ['front', 'top']
    view_cameras : dict
        Dictionary mapping view names to camera settings
        e.g., {'front': {'eye': {...}, 'up': {...}}}
    scale : int, default 2
        Scale factor for export resolution
    width : int, default 900
        Base viewport width
    height : int, default 900
        Base viewport height
    timeout : int, default 60
        Timeout for page load
    verbose : bool, default True
        Whether to print progress messages
    auto_crop : bool, default True
        If True, automatically crop whitespace and preserve margin
    crop_margin : int, default 30
        Margin (in pixels) to preserve around content when auto_crop=True
    
    Returns
    -------
    dict
        Dictionary with results:
        {
            'success': bool,
            'files': {safe_name: [(png_path, view_name), ...]},
            'failed': [legend_name, ...],
            'error': str or None
        }
    """
    from tqdm import tqdm
    
    result = {
        'success': True,
        'files': {},
        'failed': [],
        'error': None
    }
    
    def _sanitize_filename(name):
        """Sanitize legend name for use as filename."""
        safe_name = "".join(c if c.isalnum() or c in '.+_- ' else '_' for c in str(name))
        safe_name = safe_name.strip().replace(' ', '_')
        while '__' in safe_name:
            safe_name = safe_name.replace('__', '_')
        return safe_name.rstrip('_')
    
    # Retry logic for Chrome crashes
    max_retries = 3
    last_error = None
    
    for retry_attempt in range(max_retries):
        if retry_attempt > 0:
            if verbose:
                print(f'   🔄 Retry attempt {retry_attempt + 1}/{max_retries}...')
            import time
            time.sleep(2)  # Brief pause before retry
            # Reset result for retry
            result['files'] = {}
            result['failed'] = []
            result['success'] = True
            result['error'] = None
        
        try:
            with WebDriverExportSession(
                width=width, 
                height=height, 
                scale=scale, 
                timeout=timeout
            ) as session:
                # Load HTML once with background color support
                session.load_html(html_path, wait_for_render=True, render_wait=2, background_color=background_color)
                
                # Hide legend and clean up layout for export
                # Remove interactive UI elements (view selector, controls hint)
                session.update_layout({
                    'showlegend': False,
                    'title.text': '',
                    'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                    'scene.domain': {'x': [0, 1], 'y': [0, 1]},  # Full viewport, no margin
                    'updatemenus': [],  # Remove view selection dropdown
                    'annotations': []   # Remove controls hint text
                })
                
                legend_names = list(legend_entries.keys())
                progress_iter = tqdm(legend_names, desc='Exporting individuals') if verbose else legend_names
                
                for legend_name in progress_iter:
                    trace_indices = legend_entries[legend_name]
                    safe_name = _sanitize_filename(legend_name)
                    
                    # Set visibility: show this legend's traces + background
                    visible_indices = list(trace_indices) + list(background_indices)
                    session.set_trace_visibility(visible_indices, total_traces)
                    
                    result['files'][safe_name] = []
                    
                    # Export each view
                    for view_name in views:
                        camera = view_cameras.get(view_name, {})
                        
                        # Set camera position
                        session.set_camera(
                            eye=camera.get('eye'),
                            up=camera.get('up'),
                            center=camera.get('center')
                        )
                        
                        # Export PNG with auto-crop
                        png_filename = f'{view_name}_{safe_name}.png'
                        png_path = os.path.join(output_dir, png_filename)
                        
                        # Decode the configured background through the shared
                        # parser so screenshot compositing follows the same
                        # rules as Plotly.
                        try:
                            bg_rgb = extract_rgb_tuple(background_color)
                        except (TypeError, ValueError):
                            bg_rgb = (255, 255, 255)
                        
                        session.screenshot(png_path, auto_crop=auto_crop, margin=crop_margin, background_color=bg_rgb)
                        
                        # Verify export
                        if os.path.exists(png_path) and os.path.getsize(png_path) > 1024:
                            result['files'][safe_name].append((png_path, view_name))
                        else:
                            result['failed'].append(legend_name)
                            if verbose:
                                print(f'   ⚠️  PNG export failed for {legend_name} ({view_name})')
                            break
            
            # If we got here without exception, export succeeded
            break  # Exit retry loop on success
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            # Check if this is a Chrome crash
            is_chrome_crash = (
                'Message: \n' in str(e) or 
                error_msg == '' or
                'chrome not reachable' in error_msg.lower() or
                'session deleted' in error_msg.lower() or
                'no such window' in error_msg.lower()
            )
            
            if is_chrome_crash and retry_attempt < max_retries - 1:
                if verbose:
                    print(f'   ⚠️  Chrome crashed unexpectedly. Will retry...')
                continue
            else:
                # Not a crash or out of retries
                result['success'] = False
                result['error'] = str(e)
                if verbose:
                    print(f'   ⚠️  WebDriver export failed: {e}')
                break
    
    return result


# Local imports
try:
    import statvis as sv
    import FAFB_file_converter
    import BANC_file_converter
except ImportError:
    # Fallback for when running from different context
    from . import statvis as sv
    from . import FAFB_file_converter
    from . import BANC_file_converter


# Legend color shared by every query_transformed_* overlay layer so the
# transformed source neurons read as one group across their per-neuron
# legend entries. Light blue: bokeh.palettes Category10, first item
# (#1f77b4), with alpha=0.2.
QUERY_OVERLAY_COLOR = 'rgba(31, 119, 180, 0.2)'


def dataset_native_space(dataset: str) -> str:
    """Native coordinate space of a connectome dataset.

    Mirrors the source-space mapping used by VisualizeSkeleton's
    ``_get_template_info`` so cross-dataset overlays resolve the same
    bridging transforms the ROI mesh pipeline uses
    (male-cns ROIs are transformed JRCFIB2022Mraw -> FLYWIRE).

    Raises ValueError for datasets without a known native space.
    """
    d = str(dataset or '').lower()
    if 'fafb' in d or ('flywire' in d and 'banc' not in d):
        return 'FLYWIRE'
    if 'banc' in d:
        return 'BANC'
    if 'male-cns' in d or 'malecns' in d or 'optic' in d:
        return 'JRCFIB2022Mraw'
    if 'hemibrain' in d:
        return 'JRCFIB2018Fraw'
    if 'manc' in d:
        return 'MANCraw'
    raise ValueError(
        f"No known native coordinate space for dataset '{dataset}'; "
        "pass explicit spaces to transform_neurons_to_space instead."
    )


def dataset_render_space(dataset: str) -> str:
    """Template-mode render space of a dataset's visualization scene.

    This is where skeletons are actually DRAWN: VisualizeSkeleton transforms
    a dataset's native-space skeletons to this space before rendering
    (``_get_template_info`` target, template mode). For overlays, transform
    the foreign neurons into this space — NOT the native space, which for
    NeuPrint datasets is a raw-voxel frame with different units.

    Raises ValueError for datasets without a known render space.
    """
    d = str(dataset or '').lower()
    if 'fafb' in d or ('flywire' in d and 'banc' not in d):
        return 'FLYWIRE'
    if 'banc' in d:
        return 'FLYWIRE'  # BANC scenes render in the FLYWIRE template today
    if ('male-cns' in d or 'malecns' in d or 'optic' in d):
        return 'JRCFIB2022M'
    if 'hemibrain' in d:
        return 'JRCFIB2018F'
    if 'manc' in d:
        return 'MANC'
    raise ValueError(
        f"No known render space for dataset '{dataset}'; "
        "pass explicit spaces to transform_neurons_to_space instead."
    )


def transform_neurons_to_space(
        neurons, source_space: str, target_space: str,
        validate_bounds: bool = True, verbose: bool = False) -> list:
    """Transform navis neurons between template spaces via the flybrains
    bridge registry (navis.xform per neuron, failure-isolated).

    For male-cns -> FAFB this resolves to the production ROI-mesh bridging
    (JRCFIB2022Mraw -affine-> JRCFIB2022M -landmark TPS-> FLYWIRE), which is
    fully offline: both transforms ship inside the flybrains package.

    When ``validate_bounds`` and the final hop is a landmark TPS, neurons
    whose soma lands outside the TPS target-landmark envelope (plus 20%
    margin) are dropped with a warning — a cheap guard against silent
    mis-registration. Neurons without a soma are kept.

    Returns a list of transformed navis neurons (failures excluded).
    """
    import navis
    import flybrains
    flybrains.register_transforms()
    from navis.transforms import registry as _registry

    if source_space == target_space:
        return list(navis.NeuronList(neurons))

    path, seq = _registry.shortest_bridging_seq(source_space, target_space)

    # TPS target landmarks bound the template space the bridge maps into.
    bounds_lo = bounds_hi = None
    if validate_bounds and seq.transforms:
        last = seq.transforms[-1]
        tgt = getattr(last, 'target', None)
        tgt = np.asarray(tgt) if tgt is not None else None
        if tgt is not None and tgt.ndim == 2 and len(tgt):
            span = tgt.max(axis=0) - tgt.min(axis=0)
            bounds_lo = tgt.min(axis=0) - 0.2 * span
            bounds_hi = tgt.max(axis=0) + 0.2 * span

    out = []
    dropped = []
    for n in navis.NeuronList(neurons):
        try:
            xf = navis.xform(n, transform=seq)
        except Exception as e:
            _vprint_safe(f'  ⚠️ Could not transform {getattr(n, "name", n)} '
                         f'({source_space} -> {target_space}): {e}')
            continue
        kept = True
        if bounds_lo is not None:
            try:
                soma = getattr(xf, 'soma', None)
                if soma is not None:
                    pos = np.array([float(soma.x), float(soma.y), float(soma.z)])
                    if np.any(pos < bounds_lo) or np.any(pos > bounds_hi):
                        kept = False
                        dropped.append(getattr(n, 'name', n))
            except Exception:
                pass  # advisory check only
        if kept:
            out.append(xf)
        else:
            _vprint_safe(f'  ⚠️ {getattr(n, "name", n)} transformed soma '
                         f'outside {target_space} bounds; excluded from overlay')
    if verbose:
        _vprint_safe(f'  Bridged {source_space} -> {target_space} via '
                     f'{" -> ".join(path)}: {len(out)} neuron(s)')
    if dropped:
        _vprint_safe(f'  {len(dropped)} transformed neuron(s) outside '
                     f'{target_space} landmark envelope: {dropped[:5]}')
    return out


def _vprint_safe(msg: str) -> None:
    """Module-level printer for helper warnings (no VisualizeSkeleton bound)."""
    try:
        tqdm.write(msg)
    except Exception:
        print(msg)


@dataclass
class VisualizeSkeleton:
    """
    Interactive 3D visualization of neuron skeletons, synapses, and brain regions.
    
    This class provides comprehensive tools for visualizing connectome data from multiple
    datasets including NeuPrint (hemibrain, male-cns, manc, optic-lobe) and FlyWire/FAFB.
    It handles data fetching, coordinate transformations, mesh simplification, caching,
    and export to various formats.
    
    Key Capabilities
    ----------------
    - **Multi-dataset support**: Seamlessly visualize neurons from different connectome datasets
    - **Flexible neuron selection**: Query by type, instance name, bodyId, or custom layers
    - **Mesh rendering**: High-quality tube meshes with automatic simplification
    - **Synapse visualization**: Multiple modes (scatter, sphere, cone, tetrahedron)
    - **ROI meshes**: Display brain region boundaries with bilateral auto-expansion
    - **Smart caching**: Efficient storage of raw compressed-SWC skeletons
      and analysis results
    - **Export options**: HTML (interactive), PNG (multiple views), video animations
    
    FAFB/FlyWire Features
    ---------------------
    For FlyWire datasets, specialized features handle high-resolution mesh data:
    
    - **Soma-aware simplification**: Preserves cell body detail while simplifying branches
    - **Automatic extrusion detection**: Identifies and replaces distorted skeletons
    - **Parquet caching**: Efficient storage of large-scale analysis results
    - **API fallback**: Fetches fresh meshes from CAVE when ZIP data has artifacts
    
    Attributes
    ----------
    dataset : str, default='hemibrain:v1.2.1'
        Dataset identifier. Supported values:
        - NeuPrint: 'hemibrain:v1.2.1', 'optic-lobe:v1.1', 'manc:v1.0', 'male-cns:v0.9'
        - FlyWire: 'flywire_FAFB_v783', 'flywire_BANC_v626'
    
    neuron_layers : str | list
        Neuron layers to visualize. Can be:
        - List of layers: ['L1', 'L2', 'L3']
        - String with '->' separator: 'L1->L2->L3'
        - Each layer can contain types, instances (regex), bodyIds, or lists thereof
    
    backend : str, default='plotly'
        Visualization backend. Options: 'plotly' (interactive HTML), 'k3d' (WebGL)
    
    skeleton_mode : str, default='tube'
        Rendering mode: 'tube' (mesh with radius), 'line' (simple lines)
    
    auto_fix_extrusions : bool, default=True
        (FAFB only) Automatically detect and replace distorted skeletons.
        Uses edge length analysis to identify extrusion artifacts from aggressive
        simplification. Results cached in parquet format for fast subsequent runs.
    
    skeleton_mesh_simplification : float, optional
        Mesh simplification factor (0.0-1.0). Higher = more simplification.
        Default: 0.90 for the fast/direct pipeline and 0.95 for fine/artistic
        renders in both the dedicated Skeleton tab and analysis renderers.
        Example: 0.95 removes 95% of faces, keeping 5%.
    
    soma_mesh_simplification : float, default=0.8
        (FAFB only) Gentler simplification for soma region to prevent artifacts.
    Applied within soma_region_radius (default 20µm) of detected soma position.
    
    cache_neurons : bool, default=False
        Compatibility flag retained for callers that used the old
        visualization cache switch. NeuPrint fetches now use the shared raw
        compressed-SWC cache for every render mode; fast/direct and
        fine/artistic simplification happens in memory and meshes are never
        persisted by this option.
    
    mesh_roi : list, default=[]
        List of brain region names to display. ROI names without (L)/(R) suffix
        are auto-expanded to bilateral variants. Use list_available_rois() to see options.
    
    brain_mesh : str, default='none'
        Brain/VNC envelope mesh. Options:
        - 'none': No envelope (only ROIs in mesh_roi)
        - 'template': Dataset's native template (fast, no H5 transforms needed)
        - 'whole': Standard brain mesh (may require H5 transform download)
    
    legend_mode : str, default='layer'
        Controls how neurons appear in the legend. Options:
        - 'single': Each neuron gets its own legend entry ({bodyId}_{layer_name})
        - 'type': Group by neuron type within each layer. If a layer has multiple types,
                  each type gets a separate legend entry.
        - 'layer': Merge all neurons in a layer into one legend entry.
                   Auto-named as {type1}_{type2}_etc if 3+ types present.
    
    expand_colors : str, default='interpolation'
        Method for generating extra colors when more layers than colors available.
        - 'interpolation': Create a smooth colormap and sample extra colors (recommended)
        - 'darken': Recycle colors with progressive darkening (100% to 70% brightness)
        - 'cycle': Simply cycle through colors (color1, color2, ..., color1, color2, ...)
    
    min_synapse_num : int, default=10
        Minimum synapse count threshold for fetching/plotting connections.
    
    synapse_mode : str, default='scatter'
        Synapse rendering mode: 'scatter', 'sphere', 'cone', 'tetrahedron'
    
    skip_synapse : bool, default=False
        Skip all synapse operations for faster initialization and smaller files.

    output_format : str, default='csv'
        Synapse data export format. Use ``'csv'`` for one merged CSV file or
        ``'xlsx'`` for one merged Excel sheet.
    
    export_views : bool | list, default=True
        PNG export configuration:
        - True: Export all 6 views (front, back, top, bottom, left, right)
        - False: No PNG export
        - List[str]: Specific views to export, e.g., ['front', 'top']
    
    export_scale : int, default=3
        PNG resolution scale factor (1-4). Higher = larger/better quality but slower.
        scale=3 produces ~3600x2700 pixel images.
    
    verbose : bool | str, default='full'
        Verbosity level: 'full' (all messages), 'simple' (essential only), False (silent)
    
    output_dir : str, optional
        Directory for output files. If None, uses data_folder.
    
    include_timestamp : bool, default=True
        Include timestamp in output folder names for unique versioning.
    
    Methods
    -------
    plot_skeleton()
        Generate and display/save the 3D visualization.
    
    list_available_rois(refresh=False, fetch_online=True)
        List all brain regions available for the current dataset.
    
    Examples
    --------
    Basic visualization with hemibrain data:
    
    >>> vs = VisualizeSkeleton(
    ...     dataset='hemibrain:v1.2.1',
    ...     neuron_layers=['DA1_lPN', 'DA1_adPN'],
    ...     mesh_roi=['AL', 'LH', 'MB'],
    ... )
    >>> vs.plot_skeleton()
    
    FlyWire with automatic artifact correction:
    
    >>> vs = VisualizeSkeleton(
    ...     dataset='flywire_FAFB_v783',
    ...     neuron_layers=['MTe50', 'MTe51'],
    ...     auto_fix_extrusions=True,  # Auto-detect distorted neurons
    ...     skeleton_mesh_simplification=0.95,
    ...     soma_mesh_simplification=0.8,  # Gentler on cell bodies
    ... )
    >>> vs.plot_skeleton()
    
    Custom layer mapping from CSV:
    
    >>> vs = VisualizeSkeleton(
    ...     dataset='male-cns:v0.9',
    ...     layer_map_csv='my_layers.csv',  # CSV with 'layer' and 'id_type_instance' columns
    ...     cache_neurons=True,
    ... )
    >>> vs.plot_skeleton()
    
    Video export with multiple views:
    
    >>> vs = VisualizeSkeleton(
    ...     dataset='hemibrain:v1.2.1',
    ...     neuron_layers=['MBON14'],
    ...     export_views=['front', 'top', 'lateral'],
    ...     export_scale=2,
    ... )
    >>> vs.plot_skeleton()
    
    Notes
    -----
    - First run downloads and caches data (may take several minutes)
    - Subsequent runs use cached data (typically <10 seconds)
    - For FAFB: First run with auto_fix_extrusions takes ~30-60s for analysis,
      then results are cached in parquet format
    - Use higher simplification (0.95-0.99) for scenes with many neurons
    - Set legend_mode='single' to see individual neurons in legend
    
    See Also
    --------
    FAFB_INTEGRATION.md : Detailed guide for FlyWire/FAFB usage
    INSTALLATION.md : Setup and configuration instructions
    OUTPUT_FILES.md : Documentation of output formats
    """
    
    backend: str = 'plotly'
    '''
    visualization backend: 'plotly' (default) or 'k3d'
    'plotly': interactive HTML with plotly (good for small/medium scenes)
    'k3d': WebGL-based, faster for large scenes, supports binary export
    '''

    dataset: str = 'hemibrain:v1.2.1'
    '''dataset to use, default is hemibrain:v1.2.1'''

    client_type: str = 'neuprint'
    '''client type: 'neuprint' (default) or 'flywire' '''

    client_flywire: object = None
    '''flywire client instance'''

    client: object = None
    '''neuprint client instance (optional, to reuse existing client)'''

    server: str = 'https://neuprint.janelia.org'
    '''the neuprint server to visit'''
    
    token: str = None
    '''neuprint auth token'''

    version: int | None = None
    '''Materialization version for FlyWire (e.g. 783). If None, uses default/latest.'''

    source_path: str = os.path.dirname(os.path.abspath(__file__))
    '''absolute path to the src/ directory where coana.py is located'''
    
    script_path: str = os.path.dirname(source_path)
    '''absolute path to the project root directory (parent of src/)'''

    data_folder: str = os.path.join(script_path, 'local_data', 'connectome_analysis')
    '''
    folder to save all data (subfolders auto-generated based on neuron_layers)
    Default: <project root>/local_data/connectome_analysis/
    '''

    output_dir: str = None
    '''Directory to save output. If None, uses data_folder.'''

    verbose: bool | str = 'full'
    '''
    Verbosity level:
    'full' or True: Print all messages
    'simple': Print only essential progress
    False: Silent
    '''
    
    save_folder: str = ''
    '''
    folder to save the current data (auto-generated from neuron_layers)
    # initialized in __post_init__
    # You can set the "saveas" parameter to customize the folder name'''

    neuron_layers: str | list = ''
    '''
    layers of neurons to plot, can be:
        list of neuron layers: e.g. ['L1', 'L2', 'L3']; or \n
        str of neuron layers separated by '->': \n
        e.g. 'L1->L2->L3'. All type, instance (in regular expression), and bodyId are compatible.\n
    when use list, each layer can be neuron bodyIds, types, instances in regular expressions, or a list of them\n
    e.g. [['L1_0','L1_1'], ['L2_0','L2_1'], ['L3_0','L3_1']]
    '''

    custom_neurons: list = None
    '''
    Pre-built neurons rendered as extra layers, already in this dataset's
    coordinate space. List of (layer_name, neurons) tuples where neurons is a
    navis TreeNeuron / NeuronList / list of TreeNeurons. These layers bypass
    dataset-bound fetching and coordinate transforms entirely (rendered
    as-is), so callers must transform them into the scene space first —
    see ``transform_neurons_to_space``. Used for cross-dataset overlays such
    as the homolog visualizer's ``query_transformed_{neuron name}`` layer.
    '''

    search_columns: str = 'auto'
    '''
    Which columns to search when resolving neuron names (same backend as
    pathfinding, statvis.getNeurons):
        'auto' (default): search all columns with priority
        bodyId -> type -> instance -> other string columns
        (e.g. flywireType, hemibrainType, mancType); 'type', 'instance'
        and 'bodyId' restrict the search to that single column.
    '''

    hemisphere: str = 'both'
    '''
    Restrict the plotted neurons to one hemisphere:
        'both' (default): all neurons.
        'left': only left-hemisphere neurons.
        'right': only right-hemisphere neurons.
    Hemisphere assignment comes from the 'Soma side' / 'hemisphere' columns
    or the instance suffix (_L/_R).  Neurons WITHOUT an explicit hemisphere
    are treated as unclassified ('U') and are ALWAYS included - in 'left',
    'right' AND 'both' - so data that lacks hemisphere notation is never
    silently dropped.
    '''

    custom_layer_names: list = field(default_factory=list)
    '''
    Optional custom names for layers. If empty, auto-generates smart names from neuron types.
    
    Supports partial specification:
    - If list is shorter than neuron_layers, remaining layers use auto-generated names
    - Use empty string '' to skip a layer and use auto-naming for that position
    
    Examples:
        custom_layer_names=['DN1', 'LN', 'Output']  # Full specification for 3 layers
        custom_layer_names=['MyLayer1']             # Only name first layer, auto-name rest
        custom_layer_names=['', '', 'Output']       # Skip first two, name only third layer
    '''

    layer_map_csv: str = None
    '''
    Path to CSV file that defines neuron layers mapping.
    CSV format: columns 'layer' and 'id_type_instance'
    - 'layer': custom layer name (neurons with same layer value are grouped together)
    - 'id_type_instance': neuron identifier (bodyId, type, or instance name)
    
    When provided, this overrides `neuron_layers` and `custom_layer_names`.
    The CSV is parsed to construct layers automatically.
    
    Example CSV:
        layer,id_type_instance
        DN1p,DN1pA
        DN1p,DN1pB
        DN2,DN2
        l-LNv,l-LNv
    
    This creates 3 layers: DN1p (with DN1pA, DN1pB), DN2 (with DN2), l-LNv (with l-LNv)
    '''

    soma_radius_cap: float = None
    '''
    Maximum radius for soma node (in nm) to prevent extrusion artifacts.
    When set, skeleton nodes near the soma with radius > soma_radius_cap will be capped.
    Useful for FAFB skeletons where soma detection may create exaggerated radii.
    Example: soma_radius_cap=2000 caps soma radius to 2 microns
    None (default): No capping, use original skeleton radii
    '''

    smooth_skeleton: bool = False
    '''
    Whether to apply iterative smoothing to skeleton radii.
    When True, applies aggressive smoothing to prevent extrusion artifacts from
    chains of large-radius nodes. Requires soma_radius_cap to be set.
    False (default): Only apply hard cap without smoothing.
    '''

    min_synapse_num: int = 10
    '''minimum number of synapses to fetch and plot'''

    saveas: str = None
    '''filename to save the plot, if an absolute path is given, ignore data_folder'''

    export_views: bool | list = True
    '''
    Which views to export as PNG.
    True: Export all 6 views (front, back, top, bottom, left, right)
    False: Do not export PNGs
    List[str]: List of views to export, e.g. ['front', 'top']
    Default: True (all 6 views)
    '''

    export_scale: int = 3
    '''
    Scale factor for PNG export resolution (1-4 recommended).
    Higher values produce larger, higher-quality images but take longer to export.
    - scale=1: 1200x900 pixels (fast, ~135KB)
    - scale=2: 2400x1800 pixels (~400KB)
    - scale=3: 3600x2700 pixels (default, ~1MB+)
    - scale=4: 4800x3600 pixels (high quality, ~2MB+)
    
    Note: PNG export uses a 60-second timeout per image. If export times out:
    1. Automatic retry with scale=1
    2. If still fails, remaining exports are skipped
    3. Recommendation: increase skeleton_mesh_simplification (e.g., 0.95)
    
    For large figures (>50MB HTML), consider using scale=2 to avoid timeouts.
    For export_video, scale is capped at 3 by default. Use scale=4 explicitly to override.
    '''

    export_method: str = 'webdriver'
    '''
    PNG export method. Options:
    - 'webdriver': Use Selenium WebDriver with Chrome (default, fast and high quality)
    - 'kaleido': Use kaleido engine (slower but reliable fallback if WebDriver fails)
    
    Note: 'webdriver' requires selenium and webdriver-manager packages and Chrome 109+.
    If WebDriver fails, the export automatically falls back to kaleido.
    
    The 'webdriver' method uses canvas.toDataURL() which is fast (~0.3s/frame at scale 1)
    and produces high-quality output. Files are served via local HTTP server for reliability.
    
    Performance Guide:
    ┌───────────────────────────────────────────────────────────────────────┐
    │ Scale   │ webdriver       │ kaleido   │ Recommended                   │
    ├─────────┼─────────────────┼───────────┼───────────────────────────────┤
    │ 1-3     │ 0.3-0.9s ✓      │ 1.6-2.2s  │ webdriver (2-7x faster)       │
    │ 4-10    │ 1.5-2.5s ✓      │ 2.4-3.8s  │ webdriver (faster, stable)    │
    └───────────────────────────────────────────────────────────────────────┘
    
    Recommendations:
    - General use: 'webdriver' (default, fast and reliable)
    - Video export: 'webdriver' with scale≤3 (best throughput)
    - Maximum reliability: 'kaleido' (1.5-3x slower but works without Chrome)
    
    Legacy note: 'webdriver-fast' is still accepted but maps to 'webdriver'.
    '''

    webdriver_render_wait: float = 0
    '''
    Render wait time (seconds) between camera updates in WebDriver export.
    Default: 0 (fastest, works well with modern Chrome).
    Set to None for auto-calibration, or a larger value (e.g., 0.3, 0.5) if
    export aborts mid-way through (not at the beginning).
    '''

    export_timeout: int = 60
    '''
    Timeout in seconds for kaleido-based PNG/video export per frame.
    Default: 60 seconds for first frame, subsequent frames use 2x this value
    if frame 1 exports successfully (verified working).
    
    For large/complex figures, increase this value or use export_method='webdriver'.
    '''

    html_size_cap: int = None
    '''
    HTML file size threshold (in MB) for automatic figure simplification before export.
    When HTML file size exceeds this threshold, the figure is simplified before PNG/video export.
    
    Options:
    - None (default): Auto-determine based on export_method:
        • kaleido: 100 MB (kaleido struggles with large figures)
        • webdriver: 200 MB (Chrome can handle larger files but may still crash)
    - int: Explicit threshold in MB (e.g., 50, 100, 200)
    
    When threshold is exceeded:
    1. First export attempt uses original figure
    2. If export fails (timeout/crash), automatically simplifies and retries
    3. Simplified HTML is saved as {name}_simplified.html for reference
    
    To disable automatic simplification, set a very high value (e.g., 9999).
    '''

    export_simplified_png: bool | int = False
    '''
    Whether to create a simplified figure for PNG export (reduces mesh complexity).
    
    Options:
    - False: Never simplify. Export the full figure directly (default).
            If export times out, will retry with simplified figure.
    - True: Always simplify figures > 50MB HTML before export.
    - int (e.g., 50): Target size threshold in MB. Simplify figures larger than this.
    
    Simplification reduces mesh vertex/face count by ~90% for faster export.
    The original HTML is always preserved at full quality.
    
    Note: Simplification only affects PNG export, not the interactive HTML.
    '''

    include_timestamp: bool = True
    '''Whether to include timestamp in the output folder name. Default True for unique folders.'''

    interactive_html: bool = True
    '''
    Enable interactive controls in exported HTML files.
    
    When True (default):
    - Displays the Plotly mode bar (toolbar) with pan/zoom/rotate buttons
    - Adds a "View" dropdown menu for preset camera angles (Front, Back, Top, Bottom, Left, Right)
    - Enables scroll zoom
    - Shows download button for saving PNG
    - Displays camera angle indicator (azimuth/elevation)
    
    When False:
    - Hides the mode bar for cleaner appearance
    - No view selection dropdown
    - Simpler HTML file (slightly smaller)
    
    Note: Mouse controls (drag to rotate, scroll to zoom) always work regardless of this setting.
    '''

    background_color: str = 'white'
    '''
    Background color for HTML and exported PNG/video views.
    
    Options:
    - 'white' (default): White background, good for publications and printing
    - 'black': Black background, good for presentations and dark mode
    - Any valid CSS color: '#f0f0f0', 'rgb(240,240,240)', 'lightgray', etc.
    
    The background color is applied to:
    - Interactive HTML files
    - Exported PNG images
    - Video frames
    '''

    html_theme_toggle: bool = True
    '''
    Embed a light/dark theme switch into the main exported HTML viewer.

    The switch is a small floating button (top-right corner, also bound to
    the 'T' key) that flips the page background, legend/dropdown/hint text
    colors, and the auto-adaptive brain/VNC mesh color at runtime via Plotly
    relayout/restyle. The theme shown on load always matches background_color;
    the switch only affects interactive viewing.

    The switch is embedded in {saveas}.html and its {saveas}_simplified.html
    viewer copies. Per-neuron HTML files and every static export (view PNGs,
    individual PNG/HTML/PDF/PPTX, video) keep the fixed background_color set
    before generation. Only the Plotly backend is supported.
    '''

    neuron_colors: tuple | list | str = None
    '''
    Colors for neuron layers. Supports multiple input formats that are automatically
    standardized to rgba format internally.
    
    None (default): auto-pick a categorical palette from the background -
    bokeh Category10 on a white background, bokeh Set3 on a black background.
    
    Supported Formats
    -----------------
    - **Named colors**: 'red', 'blue', 'lightgray', 'darkslategray', etc.
    - **Hex colors**: '#RGB', '#RGBA', '#RRGGBB', '#RRGGBBAA'
    - **RGB(A) tuples/lists**: 0-255 integers or normalized 0-1 floats
    - **CSS rgb/rgba strings**: comma or space/slash syntax, percentages,
      and normalized decimal channels
    - **CSS hsl/hsla strings**: e.g. 'hsl(210 100% 50% / 50%)'
    - **Bokeh palettes**: bokeh.palettes.Category10[10], Category20[20], etc.
    
    Usage
    -----
    - List/tuple of colors: Each color corresponds to a neuron layer
    - Single color: Applied to all layers
    - If fewer colors than layers, colors are expanded via `expand_colors` method
    
    Examples
    --------
    >>> neuron_colors = ['red', 'blue', 'green']  # Named colors
    >>> neuron_colors = ['#ff0000', '#00ff00', '#0000ff']  # Hex colors
    >>> neuron_colors = [(255, 0, 0), (0, 255, 0)]  # RGB tuples
    >>> neuron_colors = [(255, 0, 0, 0.5), (0, 255, 0, 0.8)]  # RGBA tuples
    >>> neuron_colors = bokeh.palettes.Category20[20]  # Bokeh palette
    >>> neuron_colors = 'rgba(255, 0, 0, 0.5)'  # Single RGBA color for all
    '''

    color_mode: str = 'per_layer'
    '''
    Controls how `neuron_colors` are assigned to neurons.

    Options:
    - 'per_layer' (default): One color per layer. All neurons in the same layer share a color.
    - 'per_neuron': One color per neuron across the full selection. Colors are assigned
      in layer order, then neuron order within each layer.

    This is independent of `legend_mode`, which still controls how legend entries are grouped.
    '''

    neuron_alpha: float = 0.2
    '''Alpha (transparency) for neurons. 0.0 = transparent, 1.0 = opaque.
    
    This is a single value applied uniformly to all neuron layers.
    
    **Override Behavior**: An explicit alpha on an individual color (e.g.,
    'rgba(255,0,0,0.5)', '#ff000080', or (255,0,0,0.5)) overrides this
    setting for that layer. In a mixed palette, colors without alpha continue
    to use `neuron_alpha`.
    
    **Per-Layer Alpha**: Use embedded alpha to set different transparencies
    per layer while leaving other layers on the global fallback:
        neuron_colors = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.7)', 'rgba(0,0,255,0.5)']
    
    This applies to both skeleton modes. In line mode it is passed as the
    Plotly/K3D trace opacity; in tube mode it is passed to the mesh renderer.
    '''

    synapse_colors: tuple | list | str = bokeh.palettes.Category10[10]
    '''
    Colors for synapse connections between layers. Same format options as neuron_colors.
    
    Supported Formats
    -----------------
    - **Named colors**: 'red', 'blue', 'lightgray', etc.
    - **Hex colors**: '#RGB', '#RGBA', '#RRGGBB', '#RRGGBBAA'
    - **RGB(A) tuples/lists**: 0-255 integers or normalized 0-1 floats
    - **CSS rgb/rgba and hsl/hsla strings**, including percentage alpha
    - **Bokeh palettes**: bokeh.palettes.Category10[10], etc.
    
    Note: Number of synapse colors needed = number of neuron layers - 1
    (for connections between adjacent layers)
    '''

    synapse_size: int | float | str = 1
    '''
    size of synapse (1-12, default 1).\n
    when synapse_mode='scatter': this is the visible marker size in pixels.
    In the exported figure the size slider offers 1-12 px (step 1).\n
    when synapse_mode='sphere'/'cone'/'tetrahedron' or the solid pre/post
    meshes: the number is used as a size multiplier over the real pre->post
    distance (1 = 1x), so mesh sizing follows the same value.\n
    when synapse_mode='pre_post': the same multiplier of one uniform
    pseudo-real site size, based on the mean unfiltered real connector
    distance sampled for the visualization (or a sampled upstream/downstream
    estimate for a one-layer view). The display threshold only selects which
    sites are rendered.\n
    Only the integer values 1-12 are accepted (the UI offers 1,2,3,...).\n
    '''

    uniform_synapse_size: bool = False
    '''
    when True, every synapse marker in sphere/cone/tetrahedron modes uses the
    median pre->post distance of the plotted connections instead of its own
    distance, so all markers share one uniform size\n
    '''

    synapse_criteria: SynapseCriteria = None
    '''criteria to filter synapses'''

    synapse_mode: str = 'scatter'
    '''
    mode to plot synapses, 'scatter', 'sphere', 'cone', 'tetrahedron', or 'pre_post'\n
    'scatter': plot synapses as scatter points using a visible pixel scale\n
    'sphere': plot synapses as spheres, absolute size in the figure \n
    'cone': plot synapses as cones pointing from pre to post\n
    'tetrahedron': plot synapses as tetrahedrons pointing from pre to post\n
    'pre_post': plot every queried neuron's pre- and post-synaptic SITES (not
        the paired inter-layer synapses). Post-synaptic sites (inputs) render as
        spheres (circles in scatter) and pre-synaptic sites (outputs) as cones
        (diamonds in scatter), colored by their neuron layer with separate legends.
        Scatter markers are the default (``pre_post_scatter=True``) so the exported
        HTML stays small.\n
    '''
    
    synapse_alpha: float = 0.6
    '''Alpha (transparency) for synapses. 0.0 = transparent, 1.0 = opaque.
    
    This is the fallback value applied uniformly to synapse layers when their
    colors do not include an explicit alpha.
    
    **Override Behavior**: An explicit alpha on an individual synapse color
    overrides this setting for that connection layer. Colors without alpha
    continue to use `synapse_alpha`, so mixed formats are supported.
    
    **Per-Layer Alpha**: Use embedded alpha to set different transparencies
    per synapse layer while leaving alpha-less colors on the global fallback:
        synapse_colors = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.7)']
    '''

    pre_post_scatter: bool = True
    '''When ``synapse_mode='pre_post'``, render the pre/post-synaptic sites as
    flat scatter markers instead of solid meshes: post-synaptic (input) sites
    as circles and pre-synaptic (output) sites as diamonds. On by default
    because scatter markers keep the exported HTML much smaller than solid
    meshes. Ignored for the other synapse modes.
    '''

    mesh_roi: list | str = field(default_factory=list)
    '''
    Meshes of brain ROIs to plot. Accepts a list of ROI names, special keywords,
    regex patterns, or nested lists for color grouping. Also accepts a single string.
    
    Input Formats
    -------------
    - **Single ROI**: 'EB' or ['EB']
    - **Multiple ROIs**: ['LH', 'AL', 'EB']
    - **Special Keywords**:
        - 'primary': All primary brain regions (~50-100 major ROIs)
        - 'all': Every available ROI for the current dataset
    - **Regex patterns**: 'ME.*' matches all ROIs starting with 'ME'
    - **Nested lists for color grouping**: ['AME', ['aL', 'bL', 'gL'], 'EB']
        ROIs in nested lists share the same color but keep separate legend entries.
    
    Auto-Expansion
    --------------
    ROI names without (L)/(R) suffix are automatically expanded to include
    both bilateral variants if available:
    - 'LH' → ['LH(L)', 'LH(R)']
    - 'AL' → ['AL(L)', 'AL(R)']
    - 'EB' → ['EB'] (unpaired, no expansion)
    
    Examples
    --------
    >>> mesh_roi = 'EB'                    # Single ROI
    >>> mesh_roi = ['LH', 'AL', 'EB']      # Multiple ROIs
    >>> mesh_roi = ['primary']             # All primary brain regions
    >>> mesh_roi = ['all']                 # All available ROIs
    >>> mesh_roi = ['ME.*', 'LO.*']        # Regex patterns
    >>> mesh_roi = ['AME', ['aL', 'bL', 'gL'], 'EB']  # Nested for shared color
    >>> mesh_roi = ['.*\\(R\\)']            # All right-hemisphere ROIs
    
    FAFB/FlyWire Note
    -----------------
    FAFB/FlyWire datasets do not have native ROI meshes. When visualizing FAFB data,
    ROI meshes from male-cns are automatically transformed to FAFB coordinates.
    This allows ROI context visualization but may have minor alignment differences.
    
    Finding Available ROIs
    ----------------------
    - Use `vs.list_available_rois()` to query available ROIs programmatically
    - Check the cached list at: `cache/{dataset}/available_rois.json`
    - For FAFB: Uses `cache/male-cns_v0_9/available_rois.json` (transformed)
    
    Common ROIs
    -----------
    - Central Complex (unpaired): EB, FB, PB, NO, AB
    - Mushroom Body: CA, PED, aL, bL, gL, a'L, b'L (bilateral)
    - Optic Lobe: ME, LO, LOP, AME (bilateral)
    - Antennal Lobe: AL (bilateral)
    - Lateral Horn: LH (bilateral)
    
    Set mesh_roi=None or [] to hide all ROI meshes.
    Use brain_mesh parameter to show whole brain/hemibrain envelope.
    '''

    mesh_color: tuple | list | str = (100, 100, 100)
    '''
    Colors for brain ROI meshes. Supports multiple input formats.
    
    Supported Formats
    -----------------
    - **Named colors**: 'red', 'blue', 'lightgray', etc.
    - **Hex colors**: '#RGB', '#RGBA', '#RRGGBB', '#RRGGBBAA'
    - **RGB(A) tuples/lists**: 0-255 integers or normalized 0-1 floats
    - **CSS rgb/rgba and hsl/hsla strings**, including percentage alpha
    - **Bokeh palettes**: bokeh.palettes.Category10[10], etc.
    - **List of colors**: Each color for corresponding ROI in mesh_roi
    
    **Override Behavior**: An explicit alpha on an individual ROI color
    overrides `mesh_alpha` for that ROI. Colors without alpha continue to use
    `mesh_alpha`, so a list can mix both forms.
    
    Note: ROI mesh colors are separate from brain_mesh_color and vnc_mesh_color.
    Use brain_mesh_color and vnc_mesh_color to customize the brain/VNC outline meshes.
    
    Examples
    --------
    >>> mesh_color = (100, 100, 100)  # Gray (uses mesh_alpha for transparency)
    >>> mesh_color = 'gray'  # Named color
    >>> mesh_color = 'rgba(100, 0, 100, 0.05)'  # Purple with explicit 5% opacity
    >>> mesh_color = [(255, 0, 0, 0.2), (0, 0, 255, 0.2)]  # Red and blue for 2 ROIs
    '''

    mesh_alpha: float = 0.1
    '''Alpha (transparency) for ROI meshes. 0.0 = transparent, 1.0 = opaque.
    
    This is a single value applied uniformly to all ROI meshes.
    
    **Override Behavior**: An explicit alpha on an individual ROI color
    overrides `mesh_alpha` for that ROI. Colors without alpha continue to use
    `mesh_alpha`, so a list can mix both forms.
    
    **Per-ROI Alpha**: Since `mesh_alpha` only supports a single value,
    use `mesh_color` with embedded alpha to set different transparencies
    per ROI:
        mesh_color = ['rgba(255,0,0,0.1)', 'rgba(0,255,0,0.2)', 'rgba(0,0,255,0.05)']
    
    Note: This setting only affects ROI meshes (mesh_roi). Brain mesh and VNC mesh
    transparency are controlled by brain_mesh_color and vnc_mesh_color respectively.
    
    Recommended values: 0.05-0.2 for subtle background meshes.
    '''

    legend_mode: str = 'layer'
    '''
    Controls how neurons appear in the legend. Options:
    - 'single': Each neuron gets its own legend entry ({bodyId}_{layer_name}).
                Full detail for identifying individual neurons.
    - 'type': Group by neuron type within each layer. If a layer has multiple
              neuron types, each type gets a separate legend entry.
    - 'tree': Same legend as 'type', plus the exported interactive HTML
                   embeds a collapsible type -> neuron legend panel at the
                   original legend position (top-right; a type row expands to
                   its bodyId rows, each toggleable).
                   Static exports keep the plain type-level legend.
    - 'layer': Merge all neurons in a layer into one legend entry.
               Auto-named as {type1}_{type2}_etc if 3+ types present.
    '''

    expand_colors: str = 'interpolation'
    '''
    Method for generating extra colors when more layers than colors available.
    - 'interpolation': Create a smooth colormap and sample extra colors (recommended).
                       Ensures visually distinct colors even with many layers.
    - 'darken': Recycle colors with progressive darkening (100% to 70% brightness).
                Same base colors but darker on each cycle.
    - 'cycle': Simple color cycling without modification. Same colors repeat.
    '''

    mirror_on_contralateral: bool = False
    '''
    Whether to mirror neurons and ROIs to the contralateral hemisphere.
    True: Mirror neurons and ROIs (e.g. 'ME(R)' -> 'ME(L)') to the other side.
          Useful for visualizing the full brain structure from hemibrain data.
    False: Only show the original data (default).
    '''

    skeleton_mesh_simplification: float = None
    '''
    Mesh simplification factor for neuron skeletons (0.0 to 1.0).
    Only applies when skeleton_mode='tube'.
    0.0: No simplification (keep all faces).
    0.8: Remove 80% of faces (keep 20%).
    Higher values reduce file size but may lose detail.
    
    Default:
    - ``fast``/direct: 0.90 (remove 90% of faces)
    - ``fine``/``artistic``: 0.95 (remove 95% of faces)
      for both NeuPrint and FAFB/FlyWire tube renders
    
    Recommended: 0.5 - 0.95 for large populations.
    
    NeuPrint tube mode has three user-facing methods:

    - ``fast``: the original direct path. It starts from the shared raw
      compressed-SWC cache and applies direct mesh decimation in memory.
    - ``fine`` (default): the FAFB-style NeuPrint pipeline. It loads RAW
      skeletons from the standalone compressed-SWC cache, transforms them to
      a smoothed/resampled path with the FAFB-style radius profile, and then
      builds/decimates the tube mesh during visualization. Fine meshes are
      never persisted; the raw source is the reusable cache contract.
    - ``artistic``: the same FAFB-style transform and shared batched
      online-fetch path as ``fine``, but with fast Open3D vertex-cluster
      decimation. Its result is also generated transiently from the raw
      skeleton cache.

    All three methods use the same raw skeleton source. No fetch-level
    simplified skeleton pickle or ``.level`` marker is created.
    '''
    
    soma_mesh_simplification: float = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION
    '''
    Mesh simplification factor specifically for the soma (cell body) region (0.0 to 1.0).
    Only applies when skeleton_mode='tube' and for FAFB/FlyWire datasets.
    
    The soma region often has high vertex density that can cause extrusion artifacts
    when using high simplification levels. This parameter allows applying gentler 
    simplification to the soma while keeping higher simplification for the skeleton.
    
    None (default): Use skeleton_mesh_simplification for the entire neuron.
    0.8: Recommended for FAFB - reduces soma artifacts while preserving shape.
    
    Example usage:
        skeleton_mesh_simplification=0.95,  # Aggressive on skeleton branches
        soma_mesh_simplification=0.8,       # Gentler on cell body
    
    Note: The soma region is defined as vertices within soma_region_radius (default 20µm)
    of the detected soma position.
    '''
    
    soma_region_radius: float = FLYWIRE_MESH_CACHE_SOMA_RADIUS
    '''
    Radius (in nm) around the soma position to define the soma region for 
    differential simplification. Default is 20000nm (20 microns).
    
    Vertices within this distance from the soma center will use soma_mesh_simplification.
    Vertices outside will use skeleton_mesh_simplification.
    '''

    roi_mesh_simplification: float = 0.9
    '''
    Mesh simplification factor for ROI meshes (0.0 to 1.0).
    0.0: No simplification (keep all faces).
    0.9: Remove 90% of faces (keep 10%).
    Higher values reduce file size but may lose detail.
    Recommended: 0.9 - 0.99 for large ROI meshes.
    '''

    FAFB_template_correction: bool = True
    '''
    Whether to apply tilt correction for FAFB/FlyWire datasets when using 'template' brain mesh.
    
    The FAFB/FlyWire template mesh has a slight tilt relative to the standard view axes.
    When True (default), a rotation correction is applied to align the brain:
    - Z-axis rotation: -4 degrees (corrects left-right tilt in front view)
    - Y-axis rotation: -3 degrees (corrects tilt in top view)
    - Rotation is applied around the brain center to preserve position.
    
    Set to False to use original coordinates (no rotation), which may be preferred if
    aligning with other raw FAFB data or if the rotation causes issues.
    '''

    show_soma: bool = True
    '''whether to show soma'''

    show_fig: bool = True
    '''whether to show the figure'''

    skeleton_mode: str = 'tube'
    '''
    whether to plot the radius of skeleton or only skeleton lines\n
    'tube': plot the radius of skeleton\n
    'line': only plot skeleton lines\n
    when 'line', the file size will be significantly smaller and the rendering will be faster
    '''

    skeleton_radius_style: str = 'auto'
    '''
    How the skeleton radius is interpreted when skeleton_mode='tube'.\n
    'auto' (default): NeuPrint datasets use 'fafb', FlyWire/FAFB use 'source'.\n
    'fafb': synthesize a FAFB-style thickness profile (fat base, tapered\n
            tips, capped soma) because NeuPrint radius columns are a\n
            near-constant placeholder; this is a visualization convention.\n
    'source': use the radius values from the data source unchanged.\n
    'constant': use one NeuPrint baseline radius for every node. This keeps\n
                fine-path smoothing/resampling but skips the graph-based\n
                FAFB radius-profile calculation; it is intended for\n
                comparison/experimentation.\n
    Only applies to NeuPrint datasets in tube mode.
    '''

    neuprint_skeleton_pipeline: str = 'fast'
    '''
    NeuPrint tube-mode skeleton pipeline (also selects the FAFB tube path).\n
    - ``'fast'`` (default): the original direct simplification path. Read the
      shared raw level-0 ``.swc.zst`` cache and apply direct mesh decimation in
      memory at render time. For FAFB, applies the fast node-reduction stage
      before tube meshing.
    - ``'fine'``: smooth and resample raw NeuPrint skeletons,
      synthesize a FAFB-style radius profile, then build and simplify a tube
      mesh transiently at the 0.95 fine-render target using the accelerated
      implementation. The cached source is the same shared raw level-0
      ``.swc.zst`` used by the fast path.
    - ``'artistic'``: the same cached-source contract as ``fine``, using Open3D
      vertex clustering for accelerated transient mesh decimation. It is
      retained as an explicit method name while all NeuPrint methods use the
      shared batched, parallel online skeleton prefetch.

    The legacy values ``'direct'``, ``'fine_opt'``, ``'fine_opt1'``, and
    ``'fafb'`` remain accepted as compatibility aliases for ``'fast'``,
    ``'fine'``, ``'artistic'``, and ``'fine'`` respectively.
    This is separate from ``skeleton_radius_style``. In ``'fast'`` mode,
    source radii are used because the FAFB-format transform is not applied.
    '''

    show_connectors: bool = False
    '''whether to fetch and plot the connectors, all pre- and post-synaptic sites of the neurons, for single layer of neurons'''

    skip_synapse: bool = False
    '''
    whether to skip synapse fetching and plotting between layers
    True: skip all synapse operations (faster initialization, smaller file size)
    False: fetch and plot synapses between layers (default behavior)
    Note: This only affects inter-layer synapses, not show_connectors (neuron connectors)
    '''

    output_format: str = 'csv'
    '''
    format for the merged inter-layer synapse export: 'xlsx' or 'csv'
    '''


    
    transforms_dir: str = '~/flybrain-data'
    '''
    Directory for brain transform files (used by flybrains package)\n
    Default: ~/flybrain-data (flybrains default location)\n
    To use a custom location:\n
    1. Set this attribute to your preferred path\n
    2. Ensure the flybrains package uses this path\n
    Note: Changing this requires setting the FLYBRAINS_DATA environment variable\n
    before importing flybrains, or manually moving existing transform files.\n
    '''
    
    cache_neurons: bool = False
    '''
    Compatibility flag for the former visualization skeleton cache. All
    NeuPrint fetches now persist raw skeletons as compressed SWC in the shared
    raw cache, regardless of this flag. Fast/fine simplification and tube
    meshes are transient render products and are never cached here.
    '''
    
    cache_synapses: bool = False
    '''
    Whether to cache fetched synapse data to disk\n
    True: Use synapse table from datasets/{dataset}/*_synapse_table.parquet if available\n
    False: Fetch from NeuPrint every time (default)\n
    For FlyWire/FAFB: Always uses datasets/{dataset}/flywire_FAFB_v783_synapse_table.parquet\n
    '''
    
    force_API_fetching: bool = False
    '''
    Force fetching skeletons from CAVE API instead of downloaded ZIP files (FAFB only).\n
    \n
    True: Fetch native meshes from CloudVolume API without skeletonizing.\n
          Avoids extrusion artifacts from pre-generated ZIP data.\n
          Prepared meshes use the dedicated FlyWire mesh cache.\n
    False: Use downloaded ZIP skeletons if available (default).\n
    \n
    Note: API fetching is slower (~5-10s per neuron) but produces cleaner skeletons.\n
    Only applies to FlyWire/FAFB datasets. Has no effect on NeuPrint datasets.\n
    '''
    
    auto_fix_extrusions: bool = True
    '''
    Automatically detect and replace skeletons with extrusion artifacts (FAFB only).\n
    \n
    True: When loading skeletons from ZIP, automatically check for extrusion artifacts\n
          (spiky protrusions from aggressive mesh simplification). If detected, fetch\n
          fresh meshes from CAVE API to replace them. If CAVE is unavailable,\n
          prune only a safely diagnosed local extrusion branch in memory.\n
          Results are cached for future use; derived local repairs are not\n
          written as raw data.\n
          This is the default to ensure visual quality.\n
    False: Skip extrusion detection. Faster but may show distorted neurons.\n
    \n
    Extrusion check results are cached in cache/{dataset}/extrusion_check_results.parquet\n
    to avoid repeated analysis. Previously checked neurons are skipped.\n
    \n
    Note: First run with this enabled may take longer due to mesh analysis.\n
    Subsequent runs use cached results and only check new neurons.\n
    Only applies to FlyWire/FAFB datasets. Has no effect on NeuPrint datasets.\n
    '''
    
    layer_sample_notes: Optional[List[str]] = None
    '''
    Layer member-sample truncation notes (type-level renders).

    Set by type-level callers (e.g. the morphology comparer's top-results
    render) when a result layer shows fewer neurons than its type has:
    each note names the layer, the rendered count, and the type's member
    count. When non-empty, the saved HTML page carries a single-row
    truncation warning inside the combined in-page banner, and the
    per-layer details are written to user_warning_notes.txt. Has no
    effect on scoring.
    '''

    brain_mesh: str = 'none'
    '''
    Brain/VNC mesh visualization options (dataset-specific):\n
    - 'none': Only plot meshes specified in mesh_roi parameter\n
    - 'template': Plot the dataset's native template mesh (EM resolution)\n
      • hemibrain → JRCFIB2018F (affine transform only, fast)\n
      • optic-lobe → JRCFIB2022M (affine transform only, fast)\n
      • manc → MANC (male adult nerve cord VNC, affine transform only, fast)\n
      • male-cns → JRCFIB2022M (full male CNS: brain + VNC, affine transform only, fast)\n
      • flywire/FAFB → FLYWIRE (native FAFB coordinates, NO transform needed)\n
    - 'whole': Plot standard whole-brain/VNC envelope mesh\n
      • hemibrain → JRC2018F (REQUIRES H5 transforms ~13GB download)\n
      • optic-lobe → JRCFIB2022M (affine transform only, fast)\n
      • manc → MANC VNC envelope (affine transform only, fast)\n
      • male-cns → JRCFIB2022M CNS envelope (affine transform only, fast)\n
      • flywire/FAFB → JRC2018F (REQUIRES H5 transforms, standard female brain)\n
    \n
    ⚡ Performance Tip: Use 'template' for fast visualization with native coordinates.\n
    Only hemibrain and FAFB with brain_mesh='whole' require H5 transform downloads.\n
    \n
    See https://github.com/navis-org/navis-flybrains
    '''
    
    brain_mesh_color: str | tuple | list = 'auto'
    '''
    Color of the brain/VNC mesh. Supports multiple input formats.
    
    Supported Formats
    -----------------
    - **'auto'** (default): Automatically selects optimal color based on background_color:
        • White background: 'rgba(200, 230, 240, 0.1)' (light blue, 10% opacity)
        • Black background: 'rgba(60, 60, 70, 0.1)' (dark gray, 10% opacity)
    - **Named colors**: 'lightblue', 'gray', etc.
    - **Hex colors**: '#RGB', '#RGBA', '#RRGGBB', '#RRGGBBAA'
    - **RGB(A) tuples/lists**: 0-255 integers or normalized 0-1 floats
    - **CSS rgb/rgba and hsl/hsla strings**, including percentage alpha

    An explicit alpha overrides the mesh default (0.1); a color without alpha
    uses that default.
    
    Recommendations
    ---------------
    - White background: Light blue/gray with 5-15% opacity
    - Black background: Dark gray with 5-15% opacity to avoid mesh fragment highlights
    
    Examples
    --------
    >>> brain_mesh_color = 'auto'  # Adaptive (default)
    >>> brain_mesh_color = 'rgba(200, 230, 240, 0.1)'  # Light blue, 10% opacity
    >>> brain_mesh_color = (60, 60, 70, 0.1)  # Dark gray tuple
    >>> brain_mesh_color = 'rgba(40, 40, 50, 0.05)'  # Very subtle for dark backgrounds
    '''
    
    vnc_mesh: bool = False
    '''
    Whether to show the VNC (Ventral Nerve Cord) mesh.\n
    Available for datasets with VNC data (requires flybrains >= 0.6.3):\n
    - male-cns → JRCFIB2022M.mesh_vnc (VNC portion of male CNS)\n
    - manc → MANC template (native VNC mesh)\n
    For other datasets (hemibrain, optic-lobe, flywire), this option is ignored.\n
    Note: For MANC with brain_mesh='template', the VNC is already shown\n
    (MANC template IS the VNC, so it ignores vnc_mesh value). 
    Use brain_mesh='none' to hide brain and VNC mesh.\n
    Default: False\n
    '''
    
    vnc_mesh_color: str | tuple | list = 'auto'
    '''
    Color of the VNC mesh. Supports multiple input formats (same as brain_mesh_color).
    
    Supported Formats
    -----------------
    - **'auto'** (default): Automatically selects optimal color based on background_color:
        • White background: 'rgba(200, 230, 240, 0.1)' (light green, 10% opacity)
        • Black background: 'rgba(60, 60, 70, 0.1)' (dark green-gray, 10% opacity)
    - **Named colors**: 'lightgreen', 'gray', etc.
    - **Hex colors**: '#RGB', '#RGBA', '#RRGGBB', '#RRGGBBAA'
    - **RGB(A) tuples/lists**: 0-255 integers or normalized 0-1 floats
    - **CSS rgb/rgba and hsl/hsla strings**, including percentage alpha

    An explicit alpha overrides the mesh default (0.1); a color without alpha
    uses that default.
    
    Note: Default 'auto' uses slightly different hue from brain_mesh_color to distinguish.
    '''

    progress_total: int | None = None
    '''
    Total number of [DROCAT][progress] step events for the combined run
    (plot_neurons + optional plot_individuals / export_video), set by the
    web UI's generated script.  When None, plot_neurons emits no structured
    step events (script/API callers keep the previous quiet behavior).
    '''

    @staticmethod
    def _auto_mesh_theme_colors():
        """Auto brain/VNC mesh colors keyed by page theme.

        'light' targets a white background, 'dark' a black one. Shared by
        _get_effective_mesh_color (generation time) and the exported HTML
        theme toggle (viewing time) so the two can never drift apart.
        """
        return {
            'light': 'rgba(200, 230, 240, 0.1)',  # Light blue, 10% opacity
            'dark': 'rgba(60, 60, 70, 0.1)',  # Subtle dark gray, 10% opacity
        }

    def _get_effective_mesh_color(self, mesh_type='brain'):
        """
        Get the effective mesh color, resolving 'auto' based on background.

        Parameters
        ----------
        mesh_type : str
            'brain' or 'vnc' to select which mesh color to resolve

        Returns
        -------
        str
            RGBA color string
        """
        theme = 'dark' if self._is_dark_background() else 'light'
        if mesh_type == 'brain':
            color = self.brain_mesh_color
            if color == 'auto':
                return self._auto_mesh_theme_colors()[theme]
            return color
        else:  # vnc
            color = self.vnc_mesh_color
            if color == 'auto':
                return self._auto_mesh_theme_colors()[theme]
            return color

    def list_available_rois(self, refresh=False, fetch_online=True):
        """List all available ROIs for the current dataset.
        
        Parameters
        ----------
        refresh : bool
            If True, force refresh from NeuPrint API. If False, use cached data if available.
        fetch_online : bool
            If True, attempt to fetch from NeuPrint online database. If False, only use local cache.
        
        Returns
        -------
        list
            Sorted list of available ROI names.
        
        Examples
        --------
        >>> vs = VisualizeSkeleton(dataset='hemibrain:v1.2.1', neuron_layers=['EB'])
        >>> available_rois = vs.list_available_rois()
        >>> print(f"Found {len(available_rois)} available ROIs")
        >>> print(available_rois[:10])  # Show first 10 ROIs
        
        >>> # Force refresh from online database
        >>> fresh_rois = vs.list_available_rois(refresh=True, fetch_online=True)
        """
        self._vprint(f'\\n' + '='*70)
        self._vprint(f'Available ROIs for {self.dataset}')
        self._vprint('='*70)
        
        rois = self._get_available_rois(use_cache=not refresh, fetch_online=fetch_online)
        
        if rois:
            self._vprint(f'\\n📊 Total: {len(rois)} ROIs')
            self._vprint(f'\\n🔹 First 30 ROIs:')
            for i in range(0, min(30, len(rois)), 5):
                self._vprint('  ' + ', '.join(rois[i:i+5]))
            if len(rois) > 30:
                self._vprint(f'  ... and {len(rois) - 30} more')
            self._vprint(f'\\n💡 Use these ROI names in the mesh_roi parameter')
            self._vprint('='*70)
        else:
            self._vprint('⚠️  No ROIs found')
            self._vprint('='*70)
        
        return rois
    
    def _progress(self, step: int, total: int, label: str = ''):
        """Emit a structured step-progress event consumed by the web UI.

        The line ``[DROCAT][progress] <step>/<total> <label>`` drives the
        determinate progress bar + step label in the results panel; it is a
        control event, never shown in the execution log.  Uses ``tqdm.write``
        so it never interleaves with an active bar.
        """
        if not self.verbose:
            return
        msg = f"[DROCAT][progress] {int(step)}/{int(total)} {label}".rstrip()
        try:
            from tqdm import tqdm
            tqdm.write(msg)
        except Exception:
            print(msg, flush=True)

    def _vprint(self, msg, level='simple', use_tqdm=False, **kwargs):
        """
        Print message based on verbosity level.
        level: 'simple' (default) or 'full'
        use_tqdm: if True, use tqdm.write() to avoid progress bar conflicts
        """
        if not self.verbose:
            return
        
        # If verbose is 'simple', only print 'simple' messages
        if self.verbose == 'simple' and level == 'full':
            return
            
        # If verbose is 'full', print everything
        if use_tqdm:
            from tqdm import tqdm
            # tqdm.write doesn't support 'end' kwarg, handle it separately
            end = kwargs.pop('end', '\n')
            if end != '\n':
                # For partial lines, just print normally (will be on same line)
                print(msg, end=end, **kwargs)
            else:
                tqdm.write(msg, **kwargs)
        else:
            print(msg, **kwargs)

    def _warn_neuprint_token_rejected(self, from_env=False):
        """Print actionable guidance when NeuPrint rejects the token (401).

        NeuPrint can revoke a token server-side even before its JWT expiry.
        The run continues without a live connection so cached data can still
        be used; fetches that require the client will fail afterwards.
        """
        source = (
            'the NEUPRINT_APPLICATION_CREDENTIALS / NEUPRINT_TOKEN environment'
            ' variable' if from_env
            else "the 'tokens.neuprint' entry in config.json / config_local.json"
        )
        print('\n\033[33m⚠️  NeuPrint rejected the configured token'
              ' (401 Unauthorized).\033[0m')
        print(f'   The token configured via {source} was refused by'
              ' https://neuprint.janelia.org — it may have been revoked')
        print('   even though it has not expired. Generate a fresh token at'
              ' https://neuprint.janelia.org/account and update it there.')
        print('   Continuing without a live connection; cached data will be'
              ' used where available, but fetches that need the client will'
              ' fail.')

    def _add_view_selection_menu(self):
        """
        Add interactive view selection dropdown and camera angle display to the figure.
        
        Adds a dropdown menu with preset camera positions (Front, Back, Top, Bottom, Left, Right)
        and annotations showing current camera angles. The camera presets are adjusted based on
        the dataset coordinate system.
        """
        # Define camera positions based on dataset
        # Default: Standard fly brain - X: Left-Right, Y: Dorsal-Ventral, Z: Anterior-Posterior
        view_cameras = {
            'Front': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            'Back': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            'Top': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
            'Bottom': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
            'Left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            'Right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
        }
        
        # Adjust for MANC (Male Adult Nerve Cord)
        if 'manc' in self.dataset.lower():
            view_cameras = {
                'Front': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                'Back': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                'Top': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                'Bottom': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                'Left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                'Right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            }
        
        # Adjust for hemibrain template (JRCFIB2018F)
        if 'hemibrain' in self.dataset.lower() and self.brain_mesh == 'template':
            view_cameras = {
                'Front': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                'Back': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                'Top': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
                'Bottom': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
                'Left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                'Right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
            }
        
        # Create dropdown buttons for view selection
        view_buttons = []
        for view_name, camera in view_cameras.items():
            view_buttons.append(
                dict(
                    args=[{'scene.camera': camera}],
                    label=view_name,
                    method='relayout'
                )
            )
        
        # Determine text color based on background
        is_dark_bg = self._is_dark_background()
        text_color = 'white' if is_dark_bg else 'black'
        hint_color = 'lightgray' if is_dark_bg else 'gray'
        dropdown_bg = 'rgba(50,50,50,0.9)' if is_dark_bg else 'rgba(255,255,255,0.9)'
        hint_bg = 'rgba(50,50,50,0.7)' if is_dark_bg else 'rgba(255,255,255,0.7)'
        
        # Add dropdown menu for view selection
        self.fig_3d.update_layout(
            updatemenus=[
                dict(
                    type='dropdown',
                    showactive=True,
                    active=0,  # Default to Front view
                    buttons=view_buttons,
                    x=0.07,  # Offset to make room for title
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    bgcolor=dropdown_bg,
                    bordercolor='rgba(128,128,128,0.5)',
                    borderwidth=1,
                    font=dict(size=12, color=text_color),
                    pad=dict(l=5, r=5, t=5, b=5),
                )
            ],
            # Add annotations: title for dropdown and controls hint
            annotations=[
                # Title for view selection dropdown
                dict(
                    text="<b>View:</b>",
                    x=0.01, y=0.975,
                    xref='paper', yref='paper',
                    xanchor='left', yanchor='top',
                    showarrow=False,
                    font=dict(size=12, color=text_color),
                ),
                # Controls hint at bottom
                dict(
                    text="🖱️ Drag: Rotate | Scroll: Zoom | Ctrl+Drag: Pan",
                    x=0.5, y=0.01,
                    xref='paper', yref='paper',
                    xanchor='center', yanchor='bottom',
                    showarrow=False,
                    font=dict(size=10, color=hint_color),
                    bgcolor=hint_bg,
                    borderpad=3,
                )
            ],
        )

    def _is_dark_background(self, color=None):
        """
        Check if the background color is dark.
        
        Parameters
        ----------
        color : str, optional
            Color to check. If None, uses self.background_color.
        
        Returns True if the background color is considered dark (for text contrast).
        """
        try:
            return is_dark_color(color if color is not None else self.background_color)
        except (TypeError, ValueError):
            return False

    def _get_html_size_cap(self):
        """
        Get the effective HTML size cap in MB for figure simplification.
        
        Returns the configured html_size_cap, or auto-determines based on export_method:
        - kaleido: 100 MB (kaleido struggles with large figures)
        - webdriver: 200 MB (Chrome can handle larger files)
        
        Returns
        -------
        int
            HTML size threshold in MB
        """
        if self.html_size_cap is not None:
            return self.html_size_cap
        
        # Auto-determine based on export method
        if self.export_method in ('webdriver', 'webdriver-fast'):
            return 200
        else:  # kaleido
            return 100

    @staticmethod
    def _skeleton_simplification_warning(dataset, client_type,
                                         skeleton_mode, simplification,
                                         neuprint_skeleton_pipeline='fast'):
        """Return warning metadata for an aggressively simplified tube render.

        The simplification value is a rendering setting, not an analysis
        setting. Fast NeuPrint keeps its historical 0.90 warning threshold;
        the optimized FAFB-style fine pipeline warns only above its 0.95
        fine-render target.

        Returns
        -------
        dict or None
            Dataset family, value, and threshold for the warning, or ``None``
            when the warning does not apply.
        """
        if str(skeleton_mode or '').lower() != 'tube':
            return None

        try:
            value = float(simplification)
        except (TypeError, ValueError):
            return None

        dataset_name = str(dataset or '')
        normalized_dataset = dataset_name.lower()
        is_fafb = 'fafb' in normalized_dataset
        is_flywire = 'flywire' in normalized_dataset

        if is_fafb:
            family = 'FlyWire FAFB'
            threshold = 0.95
        elif not is_flywire:
            family = 'NeuPrint'
            pipeline = str(neuprint_skeleton_pipeline or 'fast').strip().lower()
            threshold = 0.95 if pipeline in {
                'fine', 'artistic', 'fine_opt', 'fine_opt1', 'fafb'
            } else 0.90
        else:
            # BANC and other FlyWire datasets do not share the FAFB mesh
            # pipeline, so do not apply the FAFB warning to them.
            return None

        if value <= threshold:
            return None

        return {
            'dataset': dataset_name,
            'family': family,
            'value': value,
            'threshold': threshold,
        }

    def _skeleton_simplification_warning_html(self):
        """Build the in-page warning for the current visualization settings."""
        warning = self._skeleton_simplification_warning(
            getattr(self, 'dataset', ''),
            getattr(self, 'client_type', ''),
            getattr(self, 'skeleton_mode', ''),
            getattr(self, 'skeleton_mesh_simplification', None),
            getattr(self, 'neuprint_skeleton_pipeline', 'fast'),
        )
        if warning is None:
            return ''

        # Escape dataset text before inserting it into generated HTML.  The
        # warning is generated from user-provided dataset/configuration input.
        from html import escape

        family = escape(warning['family'])
        dataset = escape(warning['dataset'])
        value = warning['value']
        threshold = warning['threshold']
        if warning['family'] == 'NeuPrint':
            pipeline = getattr(self, 'neuprint_skeleton_pipeline', 'fast')
            resolved_pipeline = {
                'direct': 'fast',
                'fine_opt': 'fine',
                'fine_opt1': 'artistic',
                'fafb': 'fine',
            }.get(
                str(pipeline or 'fast').strip().lower(),
                str(pipeline or 'fast').strip().lower(),
            )
            if resolved_pipeline in {'fine', 'artistic'}:
                pipeline_note = (
                    ' Fine NeuPrint renders rebuild the FAFB-style tube mesh '
                    'from the raw .swc.gz source on each visualization; '
                    'values above 0.95 apply additional transient decimation.'
                )
            else:
                pipeline_note = (
                    ' Fast NeuPrint renders start from the shared raw level-0 '
                    '.swc.zst source; the selected value is applied once to '
                    'the in-memory tube mesh.'
                )
        else:
            pipeline_note = (
                ' FAFB renders start from a soma-aware mesh cache, so the '
                'same percentage is not directly comparable with NeuPrint '
                'skeleton-node reduction.'
            )
        return (
            '<strong>Skeleton simplification warning.</strong> '
            f'{family} dataset <code>{dataset}</code> is rendered with '
            f'<code>skeleton_mesh_simplification={value:.3f}</code>, above '
            f'the {threshold:.2f} detail-warning threshold. Higher values '
            'remove more mesh faces and can hide fine branches. Lower the '
            'simplification value in the visualization settings for more '
            f'detail.{pipeline_note} This affects visualization only; analysis graphs and '
            'query results are unchanged.'
        )

    def _line_mode_export_warning_html(self):
        """Build the in-page warning text raised for line-mode renders.

        Line plots bypass the tube-mesh pipeline entirely, so the
        high-quality export pipeline (exported PNG views, rotating videos,
        and individual PDF/PPTX profiles) captures the same lightweight
        line rendering instead of detailed tube meshes.  Every line-mode
        page carries this warning so exported artifacts are not mistaken
        for high-quality tube renders.  Returns the banner paragraph only;
        ``_in_page_warning_html`` supplies the shared single-banner styling.
        """
        if str(getattr(self, 'skeleton_mode', 'tube') or 'tube').lower() != 'line':
            return ''

        # Escape dataset text before inserting it into generated HTML.  The
        # warning is generated from user-provided dataset/configuration input.
        from html import escape

        dataset = escape(str(getattr(self, 'dataset', '') or ''))
        return (
            '<strong>Line-mode skeleton rendering.</strong> '
            f'<code>{dataset}</code> is plotted with line-mode skeletons: '
            'thin, radius-less lines that bypass the tube-mesh pipeline. '
            'The high-quality export pipeline (exported PNG views, rotating '
            'videos, and individual PDF/PPTX profiles) therefore captures '
            'this same lightweight line rendering; mesh simplification, '
            'radius, and tube detail do not apply to the exports. Switch '
            "Skeleton Mode to 'tube' for high-quality exports. This affects "
            'visualization only; analysis graphs and query results are '
            'unchanged.'
        )

    def _pre_post_mode_warning_html(self):
        """Build the in-page warning text for pre/post-site synapse mode.

        Pre/post-site mode shows the input/output SITES of the queried neurons
        rather than the paired inter-layer synapses; the warning makes that
        distinction explicit so exported artifacts are not mis-read as paired
        connector view.  Returns the banner paragraph only;
        ``_in_page_warning_html`` supplies the shared single-banner styling.
        """
        if str(getattr(self, 'synapse_mode', '') or '').lower() != 'pre_post':
            return ''
        from html import escape
        dataset = escape(str(getattr(self, 'dataset', '') or ''))
        return (
            '<strong>Pre/post-synaptic site mode.</strong> '
            f'<code>{dataset}</code> is plotted with <code>synapse_mode=pre_post</code>: '
            'this shows the pre- and post-synaptic SITES of the queried neurons '
            '(post-synaptic/input sites as spheres, pre-synaptic/output sites as '
            'cones) - or as circle/square scatter markers when '
            '<code>pre_post_scatter</code> is enabled - not the paired synapses '
            'between adjacent layers. The sites of every queried neuron are shown '
            'regardless of which other neuron they connect to, with a separate '
            'legend for pre and post sites per layer. This affects visualization '
            'only; analysis graphs and query results are unchanged.'
        )

    def _layer_sampling_warning_html(self):
        """One-row in-page note for type layers whose sample was capped.

        Type-level callers (e.g. the morphology comparer's top-results
        render) resolve each result layer's members from the type's member
        list bounded by a per-layer render cap. When a layer shows fewer
        neurons than its type has, the caller passes human-readable notes
        via ``layer_sample_notes``; the banner stays a single row that
        points to ``user_warning_notes.txt``, which carries the per-layer
        details.  Returns the banner paragraph only;
        ``_in_page_warning_html`` supplies the shared single-banner styling.
        """
        notes = list(getattr(self, 'layer_sample_notes', None) or [])
        if not notes:
            return ''

        count = len(notes)
        layer_word = 'layer' if count == 1 else 'layers'
        verb = 'shows' if count == 1 else 'show'
        return (
            '<strong>Truncated type layers.</strong> '
            f'{count} type {layer_word} {verb} a capped sample of their '
            'type\'s members, so they render fewer neurons than their type '
            'has; per-layer details are in <code>user_warning_notes.txt'
            '</code>. This affects visualization only; analysis graphs and '
            'query results are unchanged.'
        )

    def _in_page_warning_html(self):
        """Fold every applicable rendering caveat into ONE warning banner.

        Mesh simplification, line-mode exports, pre/post-site mode, and
        capped type-layer samples each contribute a paragraph; a page
        never carries more than a single styled warning box.
        """
        sections = [
            section
            for section in (
                self._skeleton_simplification_warning_html(),
                self._line_mode_export_warning_html(),
                self._pre_post_mode_warning_html(),
                self._layer_sampling_warning_html(),
            )
            if section
        ]
        if not sections:
            return ''

        paragraphs = ''.join(
            f'<p style="{"margin:0;" if i == 0 else "margin:6px 0 0 0;"}">'
            f'{section}</p>'
            for i, section in enumerate(sections)
        )
        return (
            '<div id="drocat-in-page-warning" role="alert" '
            'style="box-sizing:border-box;width:100%;padding:10px 14px;'
            'margin:0 0 8px 0;border:1px solid #d99a2b;border-left:5px solid '
            '#d97706;border-radius:6px;background:#fff8e6;color:#5b4300;'
            'font:14px/1.45 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;">'
            f'{paragraphs}</div>'
        )

    def _collect_adaptive_mesh_trace_indices(self, figure):
        """Find Mesh3d traces carrying an auto brain/VNC mesh color.

        The exported HTML theme switch bakes these indices at write time so
        it can restyle the brain/VNC meshes together with the background.
        Explicit user mesh colors never match the auto pair and are left
        untouched.
        """
        auto_hex = set()
        for rgba in self._auto_mesh_theme_colors().values():
            try:
                auto_hex.add(color_to_hex(rgba).lower())
            except (TypeError, ValueError):
                pass
        indices = []
        for idx, trace in enumerate(figure.data):
            if getattr(trace, 'type', None) != 'mesh3d':
                continue
            trace_color = getattr(trace, 'color', None)
            if isinstance(trace_color, str) and trace_color.lower() in auto_hex:
                indices.append(idx)
        return indices

    def _theme_toggle_html(self, mesh_indices=None):
        """Build the floating light/dark switch injected into viewer HTML.

        The switch starts in the theme matching background_color and flips
        page background, plotly chrome colors, and the auto brain/VNC mesh
        color at runtime. Neuron/synapse trace colors are left alone — those
        palettes are chosen per background before generation. Everything is
        guarded so a script error can never break the page, and the widget
        hides itself under webdriver (navigator.webdriver) so automated
        exports never show or trigger it.
        """
        initial = 'dark' if self._is_dark_background() else 'light'
        mesh_colors = {}
        for theme, rgba in self._auto_mesh_theme_colors().items():
            mesh_colors[theme] = {
                'color': self._rgba_to_hex(rgba),
                'opacity': self._extract_alpha_from_color(rgba),
            }
        baked = {
            'initial': initial,
            'meshTraces': list(mesh_indices or []),
            'meshColors': mesh_colors,
        }

        button_html = (
            '<button id="drocat-theme-toggle" type="button"'
            ' aria-label="Toggle light/dark theme"'
            ' title="Toggle light/dark theme (T)"></button>'
        )
        style_html = (
            '<style>'
            '#drocat-theme-toggle{position:fixed;right:14px;top:14px;'
            'z-index:10000;width:36px;height:36px;border-radius:50%;'
            'font-size:16px;line-height:1;display:flex;align-items:center;'
            'justify-content:center;cursor:pointer;opacity:.85;'
            'transition:opacity .15s,background .15s,color .15s;}'
            '#drocat-theme-toggle:hover{opacity:1;}'
            # Keep the hover mode bar clear of the corner button.
            '.modebar-container{right:52px !important;}'
            '</style>'
        )
        script_html = """
<script>
(function(){
  if (window.__drocatThemeToggle) { return; }
  var CONFIG = __DROCAT_CONFIG__;
  var THEMES = {
    light: {bg:'#ffffff', text:'#000000', menuBg:'rgba(255,255,255,0.9)',
            hintColor:'gray', hintBg:'rgba(255,255,255,0.7)',
            btnBg:'rgba(255,255,255,0.85)', btnBorder:'rgba(0,0,0,0.3)',
            icon:'\\uD83C\\uDF19'},
    dark:  {bg:'#000000', text:'#ffffff', menuBg:'rgba(50,50,50,0.9)',
            hintColor:'lightgray', hintBg:'rgba(50,50,50,0.7)',
            btnBg:'rgba(50,50,50,0.85)', btnBorder:'rgba(255,255,255,0.35)',
            icon:'\\u2600\\uFE0F'}
  };
  var current = CONFIG.initial;
  var btn = document.getElementById('drocat-theme-toggle');

  function decorateButton(theme) {
    if (!btn) { return; }
    var t = THEMES[theme];
    btn.style.background = t.btnBg;
    btn.style.border = '1px solid ' + t.btnBorder;
    btn.style.color = t.text;
    btn.textContent = t.icon;
  }

  function graphDiv() {
    return document.querySelector('.js-plotly-plot')
      || document.querySelector('.plotly-graph-div');
  }

  function apply(name) {
    var t = THEMES[name];
    if (!t) { return; }
    current = name;
    document.body.style.backgroundColor = t.bg;
    document.body.classList.toggle('drocat-theme-dark', name === 'dark');
    decorateButton(name);
    try {
      var gd = graphDiv();
      if (!gd || !window.Plotly || !gd._fullLayout) { return; }
      var layout = gd.layout || {};
      var update = {
        paper_bgcolor: t.bg,
        plot_bgcolor: t.bg,
        'scene.bgcolor': t.bg,
        'legend.font.color': t.text,
        'title.font.color': t.text,
        'font.color': t.text
      };
      var menus = layout.updatemenus || [];
      for (var i = 0; i < menus.length; i++) {
        update['updatemenus[' + i + '].bgcolor'] = t.menuBg;
        update['updatemenus[' + i + '].font.color'] = t.text;
      }
      var anns = layout.annotations || [];
      for (var j = 0; j < anns.length; j++) {
        if (anns[j] && anns[j].bgcolor) {
          update['annotations[' + j + '].font.color'] = t.hintColor;
          update['annotations[' + j + '].bgcolor'] = t.hintBg;
        } else {
          update['annotations[' + j + '].font.color'] = t.text;
        }
      }
      var sliders = layout.sliders || [];
      for (var k = 0; k < sliders.length; k++) {
        update['sliders[' + k + '].font.color'] = t.text;
        update['sliders[' + k + '].currentvalue.font.color'] = t.text;
      }
      Plotly.relayout(gd, update);
      if (CONFIG.meshTraces.length) {
        Plotly.restyle(gd,
          {color: CONFIG.meshColors[name].color,
           opacity: CONFIG.meshColors[name].opacity},
          CONFIG.meshTraces);
      }
    } catch (err) { /* the switch must never break the page */ }
  }

  function toggle() {
    apply(current === 'dark' ? 'light' : 'dark');
  }

  window.__drocatThemeToggle = true;
  if (navigator.webdriver) {
    if (btn) { btn.style.display = 'none'; }
    return;
  }
  document.body.style.backgroundColor = THEMES[current].bg;
  document.body.classList.toggle('drocat-theme-dark', current === 'dark');
  decorateButton(current);
  if (!btn) { return; }
  btn.addEventListener('click', toggle);
  document.addEventListener('keydown', function(e){
    var tag = (e.target && e.target.tagName) || '';
    if (/input|textarea|select/i.test(tag)) { return; }
    if (e.key === 't' || e.key === 'T') { toggle(); }
  });

  // The switch is fixed to the viewport's top-right corner, which would
  // paint over a full-width warning banner. When a banner is present,
  // drop the switch below it instead (its height varies with content).
  function positionBelowBanner() {
    var banner = document.querySelector('.drocat-warning-container');
    if (!banner) { return; }
    var bannerBottom = banner.getBoundingClientRect().bottom;
    if (bannerBottom > 0) {
      btn.style.top = (Math.round(bannerBottom) + 14) + 'px';
    }
  }
  positionBelowBanner();
  window.addEventListener('resize', positionBelowBanner);
  window.addEventListener('load', positionBelowBanner);
})();
</script>
"""
        script_html = script_html.replace(
            '__DROCAT_CONFIG__', json.dumps(baked, ensure_ascii=True)
        )
        return button_html + style_html + script_html

    def _tree_uses_custom_groups(self):
        """Whether tree mode organizes the legend by custom group first.

        Custom groups come from custom_layer_names (or a layer_map_csv,
        which fills the same field); each layer then holds a mixed set of
        neurons, so the tree becomes group > type > bodyId.
        """
        return bool(getattr(self, 'custom_layer_names', None))

    def _legend_tree_html(self):
        """Build the collapsible type -> neuron legend panel for viewer HTML.

        Injected only for ``legend_mode='tree'`` on the permanent
        viewer copies. The panel is built client-side from the
        ``drocatLegend`` meta tags written at legend-assignment time, so
        the row-to-trace mapping can never drift from the figure. The
        native Plotly legend is hidden on these pages; static exports
        never receive this injection and keep the native type-level
        legend. Type toggles cover the type's skeleton plus its pre/post
        site traces; synapse-group and mesh traces get their own rows,
        with meshes pinned last.
        """
        baked = {'meshRankBase': ROI_MESH_LEGEND_RANK_BASE}
        panel_html = '<div id="drocat-legend-tree" style="display:none"></div>'
        style_html = (
            '<style>'
            '#drocat-legend-tree{position:fixed;right:10px;top:60px;'
            'z-index:9999;max-width:280px;max-height:65vh;overflow-y:auto;'
            'overflow-x:hidden;font:12px/1.6 -apple-system,BlinkMacSystemFont,'
            'Segoe UI,sans-serif;border-radius:8px;padding:6px 8px;'
            'background:rgba(255,255,255,0.92);color:#000;'
            'border:1px solid rgba(0,0,0,0.25);user-select:none;'
            '-webkit-user-select:none;}'
            'body.drocat-theme-dark #drocat-legend-tree{'
            'background:rgba(28,28,30,0.92);color:#fff;'
            'border-color:rgba(255,255,255,0.28);}'
            '.drocat-lt-row{display:flex;align-items:center;gap:6px;'
            'padding:1px 2px;border-radius:4px;cursor:pointer;'
            'white-space:nowrap;}'
            '.drocat-lt-row:hover{background:rgba(128,128,128,0.18);}'
            '.drocat-lt-caret{width:10px;flex:0 0 auto;font-size:9px;'
            'display:inline-block;transition:transform .15s;}'
            '.drocat-lt-expanded .drocat-lt-caret{transform:rotate(90deg);}'
            '.drocat-lt-swatch{width:11px;height:11px;border-radius:2px;'
            'flex:0 0 auto;display:inline-block;}'
            '.drocat-lt-swatch-item{width:8px;height:8px;margin-left:10px;}'
            '.drocat-lt-label{overflow:hidden;text-overflow:ellipsis;'
            'flex:1 1 auto;}'
            '.drocat-lt-count{opacity:.6;font-size:10px;flex:0 0 auto;}'
            '.drocat-lt-eye{flex:0 0 auto;font-size:10px;opacity:.85;}'
            '.drocat-lt-group-row{font-weight:600;}'
            '.drocat-lt-off{opacity:.4;}'
            '.drocat-lt-section{font-weight:600;opacity:.65;'
            'margin:6px 0 2px 2px;font-size:10px;text-transform:uppercase;'
            'letter-spacing:.4px;}'
            # tree replaces the native legend on these pages
            '.js-plotly-plot .legend{display:none !important;}'
            '</style>'
        )
        script_html = """
<script>
(function(){
  if (window.__drocatLegendTree) { return; }
  window.__drocatLegendTree = true;
  var CONFIG = __DROCAT_TREE_CONFIG__;
  var panel = document.getElementById('drocat-legend-tree');
  if (!panel) { return; }
  if (navigator.webdriver) { return; }

  function graphDiv() {
    return document.querySelector('.js-plotly-plot')
      || document.querySelector('.plotly-graph-div');
  }
  function isVisible(tr) { return !tr || tr.visible !== false; }
  function traceColor(tr) {
    if (tr.marker && tr.marker.color) { return tr.marker.color; }
    if (tr.line && tr.line.color) { return tr.line.color; }
    return tr.color || null;
  }

  var records = [];  /* {row, eye, indices} kept in sync with the plot */

  function buildModel(data) {
    var groups = {}, groupOrder = [];
    var meshes = {}, meshOrder = [];
    var synapses = {}, synOrder = [];
    data.forEach(function(tr, i) {
      var lg = tr.meta && tr.meta.drocatLegend;
      if (lg && lg.group) {
        if (!groups[lg.group]) {
          groups[lg.group] = {rank: Infinity, types: {}, typeOrder: [],
                              direct: {}, directOrder: [], sites: [],
                              indices: []};
          groupOrder.push(lg.group);
        }
        var g = groups[lg.group];
        if (typeof tr.legendrank === 'number') {
          g.rank = Math.min(g.rank, tr.legendrank);
        }
        g.indices.push(i);
        if (lg.kind === 'site') {
          /* Sites toggle with their type sub-row when one exists. */
          g.sites.push({type: lg.type || null, index: i});
          return;
        }
        if (lg.type) {
          /* Custom-group hierarchy: group > type > bodyId. */
          if (!g.types[lg.type]) {
            g.types[lg.type] = {items: {}, itemOrder: [], indices: []};
            g.typeOrder.push(lg.type);
          }
          g.types[lg.type].indices.push(i);
          if (!g.types[lg.type].items[lg.item]) {
            g.types[lg.type].items[lg.item] = [];
            g.types[lg.type].itemOrder.push(lg.item);
          }
          g.types[lg.type].items[lg.item].push(i);
        } else {
          /* Flat (no custom groups): the group itself is the type and the
             items are its neurons. */
          if (!g.direct[lg.item]) {
            g.direct[lg.item] = [];
            g.directOrder.push(lg.item);
          }
          g.direct[lg.item].push(i);
        }
        return;
      }
      if ((tr.legendgroup || '').indexOf('synapses ') === 0) {
        if (!synapses[tr.legendgroup]) {
          synapses[tr.legendgroup] = {name: tr.name || tr.legendgroup,
                                      color: traceColor(tr), indices: []};
          synOrder.push(tr.legendgroup);
        }
        synapses[tr.legendgroup].indices.push(i);
        return;
      }
      if (typeof tr.legendrank === 'number'
          && tr.legendrank >= CONFIG.meshRankBase) {
        var mk = tr.name || ('mesh ' + i);
        if (!meshes[mk]) {
          meshes[mk] = {color: traceColor(tr), rank: tr.legendrank,
                        indices: []};
          meshOrder.push(mk);
        }
        meshes[mk].indices.push(i);
      }
    });
    groupOrder.sort(function(a, b) { return groups[a].rank - groups[b].rank; });
    meshOrder.sort(function(a, b) { return meshes[a].rank - meshes[b].rank; });
    return {groups: groups, groupOrder: groupOrder, meshes: meshes,
            meshOrder: meshOrder, synapses: synapses, synOrder: synOrder};
  }

  function groupColor(data, g, groupName) {
    /* Prefer the opaque legend-swatch trace (marker color, group name). */
    for (var i = 0; i < data.length; i++) {
      var tr = data[i];
      if (tr.name === groupName && tr.marker && tr.marker.color) {
        return tr.marker.color;
      }
    }
    for (var j = 0; j < g.indices.length; j++) {
      var c = traceColor(data[g.indices[j]]);
      if (c) { return c; }
    }
    return '#7f7f7f';
  }

  function makeEl(tag, className, text) {
    var el = document.createElement(tag);
    if (className) { el.className = className; }
    if (text !== undefined) { el.textContent = text; }
    return el;
  }

  function sync() {
    var gd = graphDiv();
    if (!gd) { return; }
    records.forEach(function(rec) {
      var on = rec.indices.every(function(i) { return isVisible(gd.data[i]); });
      rec.eye.textContent = on ? '\\u25CF' : '\\u25CB';
      rec.row.classList.toggle('drocat-lt-off', !on);
    });
  }

  function addToggleRow(container, label, color, indices, extraIcon) {
    var row = makeEl('div', 'drocat-lt-row' + (extraIcon ? '' : ' drocat-lt-mesh-row'));
    var swatch = makeEl('span', 'drocat-lt-swatch');
    swatch.style.background = color;
    var text = makeEl('span', 'drocat-lt-label', label);
    text.title = label;
    var eye = makeEl('span', 'drocat-lt-eye');
    if (extraIcon) {
      row.appendChild(extraIcon);
      row.classList.add('drocat-lt-group-row');
    }
    row.appendChild(swatch);
    row.appendChild(text);
    row.appendChild(eye);
    container.appendChild(row);
    records.push({row: row, eye: eye, indices: indices});
    eye.addEventListener('click', function(e) {
      e.stopPropagation();
      var gd = graphDiv();
      if (!gd) { return; }
      var on = indices.every(function(i) { return isVisible(gd.data[i]); });
      Plotly.restyle(gd, {visible: !on}, indices);
      sync();
    });
    return {row: row, eye: eye};
  }

  function attachExpandable(parent, labelText, color, count, eyeIndices) {
    /* One expandable row (caret + swatch + label + count + eye) with a
       hidden container for its children. Returns the children container;
       the caller fills it with leaf rows or nested expandables. */
    var groupEl = makeEl('div', 'drocat-lt-group');
    var caret = makeEl('span', 'drocat-lt-caret', '\\u25B6');
    var row = makeEl('div', 'drocat-lt-row drocat-lt-group-row');
    var swatch = makeEl('span', 'drocat-lt-swatch');
    swatch.style.background = color;
    var label = makeEl('span', 'drocat-lt-label', labelText);
    label.title = labelText;
    var countEl = makeEl('span', 'drocat-lt-count', count);
    var eye = makeEl('span', 'drocat-lt-eye');
    row.appendChild(caret);
    row.appendChild(swatch);
    row.appendChild(label);
    row.appendChild(countEl);
    row.appendChild(eye);
    groupEl.appendChild(row);
    var itemsEl = makeEl('div', 'drocat-lt-items');
    itemsEl.style.display = 'none';
    groupEl.appendChild(itemsEl);
    parent.appendChild(groupEl);
    records.push({row: row, eye: eye, indices: eyeIndices});
    eye.addEventListener('click', function(e) {
      e.stopPropagation();
      var gd = graphDiv();
      if (!gd) { return; }
      var on = eyeIndices.every(function(i) { return isVisible(gd.data[i]); });
      Plotly.restyle(gd, {visible: !on}, eyeIndices);
      sync();
    });
    function toggleExpand() {
      var open = itemsEl.style.display === 'none';
      itemsEl.style.display = open ? 'block' : 'none';
      groupEl.classList.toggle('drocat-lt-expanded', open);
    }
    caret.addEventListener('click', toggleExpand);
    label.addEventListener('click', toggleExpand);
    return itemsEl;
  }

  function attachLeaf(parent, labelText, color, indices) {
    var irow = makeEl('div', 'drocat-lt-row drocat-lt-item-row');
    var isw = makeEl('span', 'drocat-lt-swatch drocat-lt-swatch-item');
    isw.style.background = color;
    var ilabel = makeEl('span', 'drocat-lt-label', labelText);
    ilabel.title = labelText;
    var ieye = makeEl('span', 'drocat-lt-eye');
    irow.appendChild(isw);
    irow.appendChild(ilabel);
    irow.appendChild(ieye);
    parent.appendChild(irow);
    records.push({row: irow, eye: ieye, indices: indices});
    irow.addEventListener('click', function() {
      var gd = graphDiv();
      if (!gd) { return; }
      var on = indices.every(function(i) { return isVisible(gd.data[i]); });
      Plotly.restyle(gd, {visible: !on}, indices);
      sync();
    });
  }

  function render(model, data) {
    model.groupOrder.forEach(function(name) {
      var g = model.groups[name];
      var color = groupColor(data, g, name);
      var neuronCount = g.indices.length - g.sites.length;
      var itemsEl = attachExpandable(panel, name, color, neuronCount,
                                     g.indices);

      if (!g.typeOrder.length) {
        /* Flat tree: the group is the type, items are its neurons. */
        g.directOrder.forEach(function(itemName) {
          attachLeaf(itemsEl, itemName, color, g.direct[itemName]);
        });
      } else {
        /* Custom-group hierarchy: type sub-rows only when a group holds
           2+ neurons of that type; singletons and untyped neurons become
           direct bodyId/instance leaves. Sites toggle with their type. */
        var sitesByType = {};
        g.sites.forEach(function(s) {
          (sitesByType[s.type] = sitesByType[s.type] || []).push(s.index);
        });
        g.typeOrder.forEach(function(t) {
          var tt = g.types[t];
          if (tt.indices.length < 2) {
            /* Singleton type: no sub-row, show its bodyId directly. */
            tt.itemOrder.forEach(function(itemName) {
              attachLeaf(itemsEl, itemName, color, tt.items[itemName]);
            });
            return;
          }
          var eyeIdx = tt.indices.concat(sitesByType[t] || []);
          var subEl = attachExpandable(itemsEl, t, color,
                                       tt.indices.length, eyeIdx);
          tt.itemOrder.forEach(function(itemName) {
            attachLeaf(subEl, itemName, color, tt.items[itemName]);
          });
        });
        g.directOrder.forEach(function(itemName) {
          attachLeaf(itemsEl, itemName, color, g.direct[itemName]);
        });
      }
    });

    if (model.synOrder.length) {
      panel.appendChild(makeEl('div', 'drocat-lt-section', 'Synapses'));
      model.synOrder.forEach(function(key) {
        var s = model.synapses[key];
        addToggleRow(panel, s.name, s.color || '#7f7f7f', s.indices);
      });
    }
    if (model.meshOrder.length) {
      panel.appendChild(makeEl('div', 'drocat-lt-section', 'Meshes'));
      model.meshOrder.forEach(function(name) {
        var m = model.meshes[name];
        addToggleRow(panel, name, m.color || '#7f7f7f', m.indices);
      });
    }
  }

  function positionPanel() {
    /* Sit below the theme switch on the right, and below the warning
       banner (which shifts the whole plot down) when one is present. */
    var top = 60;
    var banner = document.querySelector('.drocat-warning-container');
    if (banner) {
      var bottom = banner.getBoundingClientRect().bottom;
      if (bottom > 0) { top += Math.round(bottom); }
    }
    panel.style.top = top + 'px';
  }

  var tries = 0;
  (function whenPlotly() {
    var gd = graphDiv();
    if (gd && window.Plotly && gd._fullLayout) {
      var model = buildModel(gd.data || []);
      render(model, gd.data || []);
      panel.style.display = 'block';
      positionPanel();
      sync();
      try { gd.on('plotly_restyle', function() { sync(); }); } catch (err) {}
      window.addEventListener('resize', positionPanel);
      window.addEventListener('load', positionPanel);
      return;
    }
    if (++tries < 300) { setTimeout(whenPlotly, 200); }
  })();
})();
</script>
"""
        script_html = script_html.replace(
            '__DROCAT_TREE_CONFIG__', json.dumps(baked, ensure_ascii=True)
        )
        return panel_html + style_html + script_html

    def _inject_page_extras(self, html_path, theme_toggle=False,
                            mesh_indices=None, legend_tree=False):
        """Insert the warning banner and/or viewer extras at the top of a page.

        All extras share one read/insert/write pass so large HTML files are
        not rewritten twice.  Each extra is skipped when already present,
        keeping the pass idempotent against retry/export paths.
        """
        if not os.path.exists(html_path):
            return

        warning_html = self._in_page_warning_html()
        theme_html = (
            self._theme_toggle_html(mesh_indices) if theme_toggle else ''
        )
        legend_html = self._legend_tree_html() if legend_tree else ''
        if not warning_html and not theme_html and not legend_html:
            return

        try:
            with open(html_path, 'r', encoding='utf-8') as handle:
                html = handle.read()

            blocks = []
            # Avoid duplicate banners when an existing file is decorated by a
            # retry/export path more than once.  The legacy ids keep pages
            # written by older versions from being decorated a second time.
            if warning_html and not any(
                    marker in html
                    for marker in (
                        'drocat-in-page-warning',
                        'drocat-skeleton-simplification-warning',
                        'drocat-line-mode-export-warning',
                        'drocat-pre-post-sites-warning',
                        'drocat-layer-sampling-warning',
                    )):
                # Wrap the banners in a flex-shrinking container and run the
                # page as a full-height flex column so the plot (with its
                # bottom size slider) fills the remaining viewport instead
                # of pushing a scrollbar.
                blocks.append(
                    '<div class="drocat-warning-container">'
                    + warning_html
                    + '</div>'
                    '<style>'
                    'html,body{height:100%;margin:0;overflow:hidden;}'
                    'body{display:flex;flex-direction:column;}'
                    '.drocat-warning-container{flex:0 0 auto;}'
                    'body > div:not(.drocat-warning-container)'
                    '{flex:1 1 auto;min-height:0;position:relative;}'
                    'body > div:not(.drocat-warning-container) .plotly-graph-div'
                    '{height:100% !important;width:100% !important;}'
                    '</style>'
                )
            if theme_html and 'drocat-theme-toggle' not in html:
                blocks.append(theme_html)
            if legend_html and 'drocat-legend-tree' not in html:
                blocks.append(legend_html)
            if not blocks:
                return

            match = re.search(r'<body\b[^>]*>', html, flags=re.IGNORECASE)
            if match is None:
                return

            html = (html[:match.end()] + '\n' + '\n'.join(blocks)
                    + html[match.end():])
            with open(html_path, 'w', encoding='utf-8') as handle:
                handle.write(html)
        except Exception as exc:
            self._vprint(
                f'  Warning: could not decorate HTML page: {exc}',
                level='full',
            )

    def _write_plotly_html(self, figure, html_path, theme_toggle=False,
                           legend_tree=False, **kwargs):
        """Write a self-contained visualization HTML.

        Plotly's JavaScript runtime is embedded in every page so an HTML file
        remains portable when copied without its output directory.  A warning
        banner is added after Plotly has generated the document so it remains
        visible in both the main and per-neuron pages, and permanent viewer
        copies can additionally carry the light/dark theme switch and the
        collapsible tree legend panel.
        """
        kwargs.setdefault('auto_open', False)
        kwargs.setdefault('full_html', True)
        kwargs.setdefault('include_plotlyjs', True)
        figure.write_html(html_path, **kwargs)
        mesh_indices = (
            self._collect_adaptive_mesh_trace_indices(figure)
            if theme_toggle else None
        )
        self._inject_page_extras(
            html_path, theme_toggle=theme_toggle, mesh_indices=mesh_indices,
            legend_tree=legend_tree,
        )
        self._record_large_html_warning(html_path)

    def _simplify_mesh_open3d(self, trimesh_obj, target_faces):
        """
        Simplify a trimesh mesh using open3d's quadric decimation.
        
        This uses open3d directly instead of trimesh's wrapper because:
        - trimesh 4.x uses fast_simplification which achieves ~60% max reduction on tube meshes
        - open3d achieves exact target face count (true 90%+ reduction)
        
        Parameters
        ----------
        trimesh_obj : trimesh.Trimesh
            The mesh to simplify.
        target_faces : int
            Target number of faces after simplification.
            
        Returns
        -------
        trimesh.Trimesh
            Simplified mesh, or original if simplification fails.
        """
        try:
            import open3d as o3d
            import trimesh
            import numpy as np
            
            # Convert trimesh to open3d
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
            
            # Simplify with open3d (achieves exact target)
            simplified = o3d_mesh.simplify_quadric_decimation(int(target_faces))
            
            # Convert back to trimesh
            return trimesh.Trimesh(
                vertices=np.asarray(simplified.vertices),
                faces=np.asarray(simplified.triangles)
            )
        except ImportError:
            # Fallback to trimesh if open3d not available
            try:
                return trimesh_obj.simplify_quadric_decimation(face_count=target_faces)
            except Exception:
                return trimesh_obj
        except Exception:
            # Return original on any error
            return trimesh_obj

    def _simplify_mesh_vertex_clustering(self, trimesh_obj, target_faces):
        """Simplify a mesh with fast Open3D vertex clustering.

        Vertex clustering is the accelerated ``artistic`` decimator. It
        pools vertices into a spatial grid using average contraction, then
        searches for the voxel size that gets closest to the requested face
        count. The count is intentionally approximate: unlike QEM, vertex
        clustering does not promise an exact target, but it avoids the much
        more expensive global edge-collapse optimization.

        If Open3D clustering is unavailable or fails, fall back to the
        established QEM implementation so a visualization never loses its
        mesh solely because the optional accelerator is unavailable.
        """
        try:
            import open3d as o3d
            import trimesh
            import numpy as np

            target_faces = max(1, int(target_faces))
            source_faces = len(trimesh_obj.faces)
            if source_faces == 0 or target_faces >= source_faces:
                return trimesh_obj

            vertices = np.asarray(trimesh_obj.vertices, dtype=np.float64)
            faces = np.asarray(trimesh_obj.faces, dtype=np.int32)
            if len(vertices) == 0 or len(faces) == 0:
                return trimesh_obj

            base = o3d.geometry.TriangleMesh()
            base.vertices = o3d.utility.Vector3dVector(vertices)
            base.triangles = o3d.utility.Vector3iVector(faces)
            base.remove_duplicated_vertices()
            base.remove_duplicated_triangles()
            base.remove_degenerate_triangles()
            base.remove_unreferenced_vertices()

            source_faces = len(base.triangles)
            if source_faces == 0 or target_faces >= source_faces:
                return trimesh_obj

            bounds = np.asarray(base.get_max_bound()) - np.asarray(
                base.get_min_bound())
            max_extent = float(np.max(bounds))
            if not np.isfinite(max_extent) or max_extent <= 0:
                return trimesh_obj

            try:
                surface_area = float(trimesh_obj.area)
            except Exception:
                surface_area = 0.0
            if not np.isfinite(surface_area) or surface_area <= 0:
                surface_area = max_extent * max_extent

            # Surface area gives a useful first estimate for the grid size;
            # the short search below corrects for tube topology and branch
            # density. The factor was calibrated on the aMe12 probe mesh.
            estimate = np.sqrt(surface_area / target_faces) * float(
                self.NEUPRINT_FINE_OPT1_CLUSTER_VOXEL_FACTOR)
            estimate = max(estimate, max_extent * 1e-5, 1e-6)

            contraction = o3d.geometry.SimplificationContraction.Average

            def simplify_at(voxel_size):
                candidate = base.simplify_vertex_clustering(
                    float(voxel_size), contraction=contraction)
                candidate.remove_duplicated_triangles()
                candidate.remove_degenerate_triangles()
                candidate.remove_unreferenced_vertices()
                return candidate, len(candidate.triangles)

            low = estimate * 0.25
            high = estimate * 4.0
            low_mesh, low_count = simplify_at(low)
            high_mesh, high_count = simplify_at(high)

            # Expand the bracket only when the initial surface-area estimate
            # missed the target regime. This keeps the common case to a
            # handful of inexpensive clustering passes.
            for _ in range(5):
                if low_count < target_faces:
                    low *= 0.5
                    low_mesh, low_count = simplify_at(low)
                    continue
                if high_count > target_faces:
                    high *= 2.0
                    high_mesh, high_count = simplify_at(high)
                    continue
                break

            best_mesh = low_mesh
            best_count = low_count
            if abs(high_count - target_faces) < abs(best_count - target_faces):
                best_mesh = high_mesh
                best_count = high_count

            for _ in range(int(self.NEUPRINT_FINE_OPT1_CLUSTER_SEARCH_STEPS)):
                voxel_size = (low + high) * 0.5
                candidate, candidate_count = simplify_at(voxel_size)
                if candidate_count and abs(candidate_count - target_faces) < abs(
                        best_count - target_faces):
                    best_mesh = candidate
                    best_count = candidate_count
                if candidate_count > target_faces:
                    low = voxel_size
                else:
                    high = voxel_size

            if best_count <= 0:
                return trimesh_obj
            # Let trimesh repair winding and compute consistent face normals.
            # Open3D's clustered triangles are valid geometry, but leaving
            # them unprocessed makes Plotly's lighting treat much of the
            # surface as back-facing, producing an artificially pale render.
            return trimesh.Trimesh(
                vertices=np.asarray(best_mesh.vertices),
                faces=np.asarray(best_mesh.triangles),
                process=True,
            )
        except ImportError:
            return self._simplify_mesh_open3d(trimesh_obj, target_faces)
        except Exception:
            return self._simplify_mesh_open3d(trimesh_obj, target_faces)

    def _simplify_mesh_for_neuprint_pipeline(self, trimesh_obj, target_faces):
        """Dispatch mesh decimation for the active public method.

        ``artistic`` selects the accelerated Open3D vertex clustering
        decimator for every dataset, including FAFB artistic tube renders.
        All other pipelines use the established QEM decimation.
        """
        pipeline = self._resolved_neuprint_skeleton_pipeline()
        if pipeline == 'artistic':
            return self._simplify_mesh_vertex_clustering(
                trimesh_obj, target_faces)
        return self._simplify_mesh_open3d(trimesh_obj, target_faces)

    def _simplify_mesh_with_soma_awareness(
            self, trimesh_obj, skeleton_simp, soma_simp, soma_pos,
            soma_radius=FLYWIRE_MESH_CACHE_SOMA_RADIUS, decimator=None):
        """Apply the shared FAFB soma-aware mesh decimation policy."""
        return simplify_mesh_with_soma_awareness(
            trimesh_obj,
            skeleton_simp=skeleton_simp,
            soma_simp=soma_simp,
            soma_pos=soma_pos,
            soma_radius=soma_radius,
            # FAFB's large ZIP/CAVE surfaces use the accelerated fine
            # prefilter + final QEM decimator by default; the artistic
            # pipeline passes vertex clustering instead.  The final target
            # and soma policy remain unchanged.
            decimator=decimator or self._simplify_mesh_fafb_fine,
        )

    @staticmethod
    def _simplify_mesh_fafb_fine(trimesh_obj, target_faces):
        """Decimate a FAFB mesh with the shared accelerated fine path."""
        return simplify_mesh_fine(trimesh_obj, target_faces)

    def _export_png_with_timeout(self, fig, output_path, width=1200, height=900, scale=3, timeout=120,
                                   auto_crop=False, crop_margin=30):
        """
        Export PNG using kaleido with timeout protection and optional auto-crop.
        
        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            The figure to export.
        output_path : str
            Path to save the PNG file.
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
        scale : int
            Scale factor for resolution.
        timeout : int
            Timeout in seconds (default 120).
        auto_crop : bool, default False
            If True, automatically crop whitespace/background from the exported image.
        crop_margin : int, default 30
            Margin (in pixels) to preserve around content when auto_crop=True.
            
        Returns
        -------
        tuple
            (success: bool, message: str, final_scale: int)
            If success, message contains file size info.
            If failed, message contains error/recommendation.
        """
        import signal
        
        # Get export_timeout from attribute if set, otherwise use parameter
        actual_timeout = getattr(self, 'export_timeout', timeout)
        
        # Set up timeout handler (Unix only)
        old_handler = None
        has_alarm_support = False
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            has_alarm_support = True
        except (AttributeError, ValueError):
            has_alarm_support = False
        
        try:
            if has_alarm_support:
                signal.alarm(actual_timeout)

            fig.write_image(output_path, width=width, height=height, scale=scale, validate=False)

            if has_alarm_support:
                signal.alarm(0)

            if not os.path.exists(output_path):
                return (False, "Export failed - no file created", scale)

            size_kb = os.path.getsize(output_path) / 1024
            if size_kb < 10:
                return (False, f"Export produced blank image ({size_kb:.1f}KB)", scale)

            if auto_crop:
                try:
                    from PIL import Image

                    img = Image.open(output_path)
                    bounds = self._detect_content_bounds(img, (255, 255, 255))
                    if bounds:
                        row_min, row_max, col_min, col_max = bounds
                        row_min = max(0, row_min - crop_margin)
                        row_max = min(img.height - 1, row_max + crop_margin)
                        col_min = max(0, col_min - crop_margin)
                        col_max = min(img.width - 1, col_max + crop_margin)
                        cropped = img.crop((col_min, row_min, col_max + 1, row_max + 1))
                        cropped.save(output_path, 'PNG')
                        size_kb = os.path.getsize(output_path) / 1024
                except Exception:
                    pass

            return (True, f"{size_kb:.1f}KB", scale)

        except PNGExportTimeout:
            return (False, f"Export timed out after {actual_timeout}s", scale)
        except Exception as e:
            return (False, str(e), scale)
        finally:
            if has_alarm_support:
                try:
                    signal.alarm(0)
                except Exception:
                    pass
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

    def _export_views_with_webdriver_session(self, export_fig, views_to_export, view_cameras, output_folder):
        """
        Export multiple views efficiently using a single WebDriver session.
        
        Opens the browser once, loads the HTML, then rotates camera via JavaScript
        to export all views without reopening the browser.
        
        Parameters
        ----------
        export_fig : plotly.graph_objects.Figure
            The figure to export
        views_to_export : list
            List of view names to export (e.g., ['front', 'back', 'top'])
        view_cameras : dict
            Dict mapping view names to camera dicts with 'eye', 'up', 'center' keys
        output_folder : str
            Folder to save the PNG files
            
        Returns
        -------
        list
            List of successfully exported view names
        """
        exported_views = []
        
        # Save figure to temporary HTML
        temp_html = os.path.join(output_folder, "_temp_export.html")
        self._write_plotly_html(
            export_fig,
            temp_html, 
            auto_open=False, 
            include_plotlyjs=True,
            config={'displayModeBar': False}
        )
        
        try:
            self._vprint(f'     Using WebDriver session for {len(views_to_export)} views...')
            
            # Get render_wait from attribute (None = auto-calibrate)
            render_wait = getattr(self, 'webdriver_render_wait', None)
            
            # Retry logic for Chrome crashes
            max_retries = 3
            last_error = None
            
            for retry_attempt in range(max_retries):
                if retry_attempt > 0:
                    self._vprint(f'      🔄 Retry attempt {retry_attempt + 1}/{max_retries}...')
                    import time
                    time.sleep(2)  # Brief pause before retry
                
                try:
                    with WebDriverExportSession(
                        width=1200, height=900, 
                        scale=self.export_scale, 
                        timeout=300,
                        render_wait=render_wait
                    ) as session:
                        # Load HTML once
                        session.load_html(temp_html, wait_for_render=True, render_wait=3, background_color=self.background_color)
                        self._vprint(f'      ✓ HTML loaded in browser (render_wait={session._render_wait:.2f}s)')
                        
                        for view_name in views_to_export:
                            if view_name not in view_cameras:
                                self._vprint(f'      ⚠️  Skipping invalid view: {view_name}')
                                continue
                            
                            camera = view_cameras[view_name]
                            view_path = os.path.join(output_folder, f"{self.saveas}_{view_name}.png")
                            
                            try:
                                # Rotate camera via JavaScript
                                session.set_camera(
                                    eye=camera.get('eye'),
                                    up=camera.get('up'),
                                    center=camera.get('center')
                                )
                                
                                # Take screenshot
                                session.screenshot(view_path)
                                
                                # Verify file
                                if os.path.exists(view_path):
                                    size_kb = os.path.getsize(view_path) / 1024
                                    if size_kb > 10:
                                        exported_views.append(view_name)
                                        self._vprint(f'      {view_name}: {size_kb:.1f}KB', level='full')
                                    else:
                                        self._vprint(f'      ⚠️  {view_name}: blank image ({size_kb:.1f}KB)')
                                else:
                                    self._vprint(f'      ⚠️  {view_name}: no file created')
                                    
                            except Exception as e:
                                self._vprint(f'      ⚠️  {view_name} export failed: {e}')
                        
                        # If we got any views, consider it a success
                        if exported_views:
                            break  # Exit retry loop
                            
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    # Check if this is a Chrome crash
                    is_chrome_crash = (
                        'Message: \n' in str(e) or 
                        error_msg == '' or
                        'chrome not reachable' in error_msg.lower() or
                        'session deleted' in error_msg.lower() or
                        'no such window' in error_msg.lower()
                    )
                    
                    if is_chrome_crash and retry_attempt < max_retries - 1:
                        self._vprint(f'      ⚠️  Chrome crashed unexpectedly. Will retry...')
                        continue
                    else:
                        raise  # Re-raise for outer exception handler
                
        except Exception as e:
            self._vprint(f'      ⚠️  WebDriver session failed: {e}')
            
            # Check HTML size and try with simplified figure
            html_size_mb = os.path.getsize(temp_html) / 1024 / 1024 if os.path.exists(temp_html) else 0
            html_size_cap = self._get_html_size_cap()
            
            if html_size_mb > html_size_cap:
                # First check if we already have a simplified figure
                existing_simplified = getattr(self, '_simplified_export_fig', None)
                existing_html = getattr(self, '_simplified_html_path', None)
                
                if existing_simplified is not None and existing_html and os.path.exists(existing_html):
                    self._vprint(f'      💡 Trying with existing simplified figure...')
                    try:
                        with WebDriverExportSession(
                            width=1200, height=900, 
                            scale=self.export_scale, 
                            timeout=300,
                            render_wait=render_wait
                        ) as session:
                            session.load_html(existing_html, wait_for_render=True, render_wait=3, background_color=self.background_color)
                            self._vprint(f'      ✓ Existing simplified HTML loaded in browser')
                            
                            for view_name in views_to_export:
                                if view_name not in view_cameras:
                                    continue
                                
                                camera = view_cameras[view_name]
                                view_path = os.path.join(output_folder, f"{self.saveas}_{view_name}.png")
                                
                                try:
                                    session.set_camera(
                                        eye=camera.get('eye'),
                                        up=camera.get('up'),
                                        center=camera.get('center')
                                    )
                                    session.screenshot(view_path)
                                    
                                    if os.path.exists(view_path):
                                        size_kb = os.path.getsize(view_path) / 1024
                                        if size_kb > 10:
                                            exported_views.append(view_name)
                                            self._vprint(f'      {view_name}: {size_kb:.1f}KB', level='full')
                                except Exception as inner_e:
                                    self._vprint(f'      ⚠️  {view_name} failed: {inner_e}')
                    except Exception as retry_e:
                        self._vprint(f'      ⚠️  Existing simplified failed: {retry_e}')
                
                # If still no success, try creating new simplified figure with 50% target, then 25%
                if not exported_views:
                    html_overhead_mb = 25  # Estimated fixed overhead
                    for reduction_target in [0.5, 0.25]:  # 50% then 25% of original
                        if exported_views:
                            break
                        
                        target_size_mb = html_size_mb * reduction_target
                        data_size = html_size_mb - html_overhead_mb
                        target_data_size = target_size_mb - html_overhead_mb
                        simplification_factor = max(0.1, target_data_size / data_size) if data_size > 0 else reduction_target
                        
                        self._vprint(f'      💡 Retrying with simplified figure (target ~{target_size_mb:.0f}MB, factor={simplification_factor:.2f})...')
                        
                        try:
                            simplified_fig = self._simplify_figure_for_kaleido(export_fig, simplification_factor)
                            
                            # Save simplified HTML to permanent file for reuse
                            simplified_html_path = os.path.join(output_folder, f"{self.saveas}_simplified.html")
                            self._write_plotly_html(
                                simplified_fig,
                                simplified_html_path,
                                theme_toggle=self.html_theme_toggle,
                                legend_tree=(self.legend_mode == 'tree'),
                                auto_open=False,
                                include_plotlyjs=True,
                                config={'displayModeBar': False},
                            )
                            
                            new_size_mb = os.path.getsize(simplified_html_path) / 1024 / 1024
                            self._vprint(f'      ✓ Simplified HTML saved: {os.path.basename(simplified_html_path)} ({new_size_mb:.0f}MB)')
                            
                            # Store simplified figure and path for reuse in subsequent exports
                            self._simplified_export_fig = simplified_fig
                            self._simplified_html_path = simplified_html_path
                            
                            with WebDriverExportSession(
                                width=1200, height=900, 
                                scale=self.export_scale, 
                                timeout=300,
                                render_wait=render_wait
                            ) as session:
                                session.load_html(simplified_html_path, wait_for_render=True, render_wait=3, background_color=self.background_color)
                                self._vprint(f'      ✓ Simplified HTML loaded in browser')
                                
                                for view_name in views_to_export:
                                    if view_name not in view_cameras:
                                        continue
                                    
                                    camera = view_cameras[view_name]
                                    view_path = os.path.join(output_folder, f"{self.saveas}_{view_name}.png")
                                    
                                    try:
                                        session.set_camera(
                                            eye=camera.get('eye'),
                                            up=camera.get('up'),
                                            center=camera.get('center')
                                        )
                                        session.screenshot(view_path)
                                        
                                        if os.path.exists(view_path):
                                            size_kb = os.path.getsize(view_path) / 1024
                                            if size_kb > 10:
                                                exported_views.append(view_name)
                                                self._vprint(f'      {view_name}: {size_kb:.1f}KB', level='full')
                                    except Exception as inner_e:
                                        self._vprint(f'      ⚠️  {view_name} failed: {inner_e}')
                            
                            if exported_views:
                                break  # Success with this simplification level
                                
                        except Exception as retry_e:
                            self._vprint(f'      ⚠️  Retry failed: {retry_e}')
                            continue
            
            if not exported_views:
                self._vprint(f'      ⚠️  All WebDriver export attempts failed')
                self._vprint(f'      💡 Try: export_method="kaleido" or increase skeleton_mesh_simplification')
        
        finally:
            # Clean up temp HTML (but not the permanent simplified HTML)
            if temp_html and not temp_html.endswith('_simplified.html'):
                try:
                    os.remove(temp_html)
                except:
                    pass
        
        return exported_views

    def _create_simplified_export_figure(self):
        """
        Create a simplified copy of the figure for PNG export.
        
        For large/complex figures, this dramatically reduces export time by:
        1. Reducing mesh complexity (decimating vertices/faces)
        2. Simplifying line traces (reducing point count)
        3. Keeping visual appearance similar but with less data
        
        Returns
        -------
        plotly.graph_objects.Figure
            Simplified figure optimized for static image export.
        """
        import trimesh
        
        self._vprint('      Creating simplified figure for PNG export...')
        
        # Create a new figure with same layout
        simplified_fig = go.Figure()
        simplified_fig.update_layout(self.fig_3d.layout)
        
        # Statistics
        total_original_vertices = 0
        total_simplified_vertices = 0
        mesh_count = 0
        
        for trace in self.fig_3d.data:
            trace_dict = trace.to_plotly_json()
            trace_type = trace_dict.get('type', '')
            
            if trace_type == 'mesh3d':
                # Simplify mesh traces - this is the main bottleneck
                x = trace_dict.get('x', [])
                y = trace_dict.get('y', [])
                z = trace_dict.get('z', [])
                i = trace_dict.get('i', [])
                j = trace_dict.get('j', [])
                k = trace_dict.get('k', [])
                
                if x is not None and len(x) > 0 and i is not None and len(i) > 0:
                    original_verts = len(x)
                    original_faces = len(i)
                    total_original_vertices += original_verts
                    mesh_count += 1
                    
                    # Only simplify large meshes
                    if original_faces > 5000:
                        try:
                            import numpy as np
                            vertices = np.array([x, y, z]).T
                            faces = np.array([i, j, k]).T
                            
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                            
                            # Target: reduce to ~10% of faces (aggressive for export)
                            target_faces = max(500, int(original_faces * 0.1))
                            simplified_mesh = self._simplify_mesh_open3d(mesh, target_faces)
                            
                            # Create new trace with simplified data
                            new_trace_dict = {k: v for k, v in trace_dict.items() 
                                             if k not in ['x', 'y', 'z', 'i', 'j', 'k', 'type'] and v is not None}
                            new_trace_dict['x'] = simplified_mesh.vertices[:, 0].tolist()
                            new_trace_dict['y'] = simplified_mesh.vertices[:, 1].tolist()
                            new_trace_dict['z'] = simplified_mesh.vertices[:, 2].tolist()
                            new_trace_dict['i'] = simplified_mesh.faces[:, 0].tolist()
                            new_trace_dict['j'] = simplified_mesh.faces[:, 1].tolist()
                            new_trace_dict['k'] = simplified_mesh.faces[:, 2].tolist()
                            
                            simplified_fig.add_trace(go.Mesh3d(**new_trace_dict))
                            total_simplified_vertices += len(simplified_mesh.vertices)
                            continue
                            
                        except Exception as e:
                            # Fall back to original if simplification fails
                            self._vprint(f'        ⚠️ Mesh simplification failed: {e}', level='full')
                    
                    total_simplified_vertices += original_verts
                
                # Keep original if not simplified
                simplified_fig.add_trace(trace)
                
            elif trace_type == 'scatter3d':
                # For scatter/line traces, reduce point count
                x = trace_dict.get('x', [])
                y = trace_dict.get('y', [])
                z = trace_dict.get('z', [])
                
                if x is not None and len(x) > 1000:
                    # Subsample large traces
                    step = max(1, len(x) // 500)
                    trace_dict['x'] = list(x[::step])
                    trace_dict['y'] = list(y[::step])
                    trace_dict['z'] = list(z[::step])
                    
                    new_trace = go.Scatter3d(**{k: v for k, v in trace_dict.items() 
                                               if k != 'type' and v is not None})
                    simplified_fig.add_trace(new_trace)
                else:
                    simplified_fig.add_trace(trace)
                
            else:
                # Keep other trace types as-is
                simplified_fig.add_trace(trace)
        
        if total_original_vertices > 0:
            reduction = (1 - total_simplified_vertices / total_original_vertices) * 100
            self._vprint(f'      Simplified {mesh_count} meshes: {total_original_vertices:,} → {total_simplified_vertices:,} vertices ({reduction:.0f}% reduction)')
        
        return simplified_fig

    def _simplify_figure_for_kaleido(self, fig, simplification_factor):
        """
        Simplify a figure for kaleido export based on HTML size ratio.
        
        For figures where HTML size > 100MB, this applies proportional simplification
        to reduce memory/time requirements for kaleido rendering.
        
        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            The figure to simplify
        simplification_factor : float
            Target ratio for data reduction (e.g., 0.25 = reduce to 25% of original)
            
        Returns
        -------
        plotly.graph_objects.Figure
            Simplified figure
        """
        import trimesh
        import numpy as np
        
        # Create new figure with same layout
        simplified_fig = go.Figure()
        simplified_fig.update_layout(fig.layout)
        
        mesh_count = 0
        total_original_faces = 0
        total_simplified_faces = 0
        
        for trace in fig.data:
            trace_dict = trace.to_plotly_json()
            trace_type = trace_dict.get('type', '')
            
            if trace_type == 'mesh3d':
                x = trace_dict.get('x', [])
                y = trace_dict.get('y', [])
                z = trace_dict.get('z', [])
                i = trace_dict.get('i', [])
                j = trace_dict.get('j', [])
                k = trace_dict.get('k', [])
                
                if x is not None and len(x) > 0 and i is not None and len(i) > 0:
                    original_faces = len(i)
                    total_original_faces += original_faces
                    mesh_count += 1
                    
                    # Target faces based on simplification factor
                    target_faces = max(100, int(original_faces * simplification_factor))
                    
                    # Only simplify if it's worth it
                    if target_faces < original_faces * 0.8:
                        try:
                            vertices = np.array([x, y, z]).T
                            faces = np.array([i, j, k]).T
                            
                            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                            simplified_mesh = self._simplify_mesh_open3d(mesh, target_faces)
                            
                            # Create new trace with simplified data
                            new_trace_dict = {k: v for k, v in trace_dict.items() 
                                             if k not in ['x', 'y', 'z', 'i', 'j', 'k', 'type'] and v is not None}
                            new_trace_dict['x'] = simplified_mesh.vertices[:, 0].tolist()
                            new_trace_dict['y'] = simplified_mesh.vertices[:, 1].tolist()
                            new_trace_dict['z'] = simplified_mesh.vertices[:, 2].tolist()
                            new_trace_dict['i'] = simplified_mesh.faces[:, 0].tolist()
                            new_trace_dict['j'] = simplified_mesh.faces[:, 1].tolist()
                            new_trace_dict['k'] = simplified_mesh.faces[:, 2].tolist()
                            
                            simplified_fig.add_trace(go.Mesh3d(**new_trace_dict))
                            total_simplified_faces += len(simplified_mesh.faces)
                            continue
                            
                        except Exception as e:
                            # Fall back to original if simplification fails
                            pass
                    
                    total_simplified_faces += original_faces
                
                simplified_fig.add_trace(trace)
                
            elif trace_type == 'scatter3d':
                # Reduce point count for line traces
                x = trace_dict.get('x', [])
                
                if x is not None and len(x) > 500:
                    # Subsample based on factor
                    target_points = max(100, int(len(x) * simplification_factor))
                    step = max(1, len(x) // target_points)
                    
                    for key in ['x', 'y', 'z']:
                        if key in trace_dict and trace_dict[key] is not None:
                            trace_dict[key] = list(trace_dict[key][::step])
                    
                    # Also handle marker colors if present
                    if 'marker' in trace_dict and isinstance(trace_dict['marker'], dict):
                        if 'color' in trace_dict['marker'] and isinstance(trace_dict['marker']['color'], (list, tuple)):
                            trace_dict['marker']['color'] = list(trace_dict['marker']['color'][::step])
                    
                    new_trace = go.Scatter3d(**{k: v for k, v in trace_dict.items() 
                                               if k != 'type' and v is not None})
                    simplified_fig.add_trace(new_trace)
                else:
                    simplified_fig.add_trace(trace)
            else:
                simplified_fig.add_trace(trace)
        
        if total_original_faces > 0:
            reduction = (1 - total_simplified_faces / total_original_faces) * 100
            self._vprint(f'   Simplified {mesh_count} meshes: {total_original_faces:,} → {total_simplified_faces:,} faces ({reduction:.0f}% reduction)')
        
        return simplified_fig

    def _detect_content_bounds(self, img, background_color=(255, 255, 255)):
        return _detect_content_bounds(img, background_color)
    def _compute_unified_crop_bounds(self, image_paths, background_color=(255, 255, 255), 
                                      sample_count=None):
        return _compute_unified_crop_bounds(image_paths, background_color, sample_count)
    def _apply_consistent_crop(self, pic_folder, margin=20, background_color=(255, 255, 255)):
        return _apply_consistent_crop(pic_folder, margin, background_color)
    def _filter_neuron_df_by_hemisphere(self, ndf, rdf=None):
        """Keep only neurons of the selected hemisphere (self.hemisphere).

        Hemisphere assignment mirrors the pathfinding backend: the
        'Soma side' / 'hemisphere' column wins, then the instance suffix
        (_L/_R).  Neurons WITHOUT an explicit hemisphere are treated as
        unclassified ('U') and are ALWAYS kept - in 'left', 'right' AND
        'both' - so data that lacks hemisphere notation is never silently
        dropped.  Returns (filtered_ndf, filtered_rdf).
        """
        df = ndf.copy()
        code = pd.Series('U', index=df.index, dtype=object)

        # Locate the side column regardless of naming variant ('Soma side',
        # 'somaSide', 'rootSide', 'hemisphere') - male-cns CSVs use camelCase.
        side_col = None
        lowered = {str(c).strip().lower(): c for c in df.columns}
        for candidate in ('hemisphere', 'soma side', 'somaside', 'rootside'):
            if candidate in lowered:
                side_col = lowered[candidate]
                break
        if side_col is not None:
            side = df[side_col].fillna('').astype(str).str.strip().str.lower()
            code[side.isin(['l', 'left', 'lhs', 'left hemisphere'])] = 'L'
            code[side.isin(['r', 'right', 'rhs', 'right hemisphere'])] = 'R'

        if 'instance' in df.columns:
            inst = df['instance'].fillna('').astype(str)
            unassigned = code == 'U'
            code[unassigned & inst.str.endswith('_R')] = 'R'
            code[unassigned & inst.str.endswith('_L')] = 'L'

        if self.hemisphere == 'left':
            df = df[code.isin(['L', 'U'])].copy()
        elif self.hemisphere == 'right':
            df = df[code.isin(['R', 'U'])].copy()

        if rdf is not None and not rdf.empty and 'bodyId' in df.columns:
            rdf = rdf[rdf.index.isin(df['bodyId'].astype(str))].copy() if rdf.index.name == 'bodyId' else rdf
        return df, rdf

    def _read_layer_csv_tolerant(self, csv_path):
        """Read a layer-map CSV, tolerating unquoted CSS color functions.

        Color values such as ``rgba(74,144,226,0.3)`` contain commas. Pandas and
        the CSV dialect would normally split them across columns unless quoted;
        this helper re-quotes any ``rgba/rgb/hsla/hsl(...)`` token that is not
        already quoted, then parses with ``pandas.read_csv``.
        """
        import io
        import re
        import pandas as pd
        with open(csv_path, 'r', encoding='utf-8') as handle:
            raw = handle.read()
        # Lookbehind avoids re-quoting an already-quoted value; only color
        # functions that begin a field (after a comma/newline) are wrapped.
        raw = re.sub(
            r'(?<![\"\w])(rgba|rgb|hsla|hsl)\s*\([^)]*\)',
            lambda m: '"' + m.group(0) + '"',
            raw,
        )
        return pd.read_csv(io.StringIO(raw))

    def _parse_layer_map_csv(self):
        """
        Parse layer_map_csv file to construct neuron_layers and custom_layer_names.

        The CSV must have a 'layer' column and a neuron column named 'neuron'
        (the legacy 'id_type_instance' name is accepted as an alias). Rows with
        the same 'layer' value are grouped together into a single layer. The
        layer column may be numeric, in which case the base is auto-detected
        (minimum 0 => 0-based, minimum 1 => 1-based and shifted to 0
        internally), or it may hold group-name labels which are used verbatim.

        Optional color columns:
        - 'color': per-neuron display color (alpha overrides neuron_alpha).
        - 'synapse_color': per-neuron color for the inter-layer (connector)
          synapse whose pre is this neuron; alpha overrides synapse_alpha.
        - 'pre_synaptic_color': per-neuron color for this neuron's pre sites.
        - 'post_synaptic_color': per-neuron color for this neuron's post sites.

        This method overrides self.neuron_layers and self.custom_layer_names and
        populates the per-neuron color override dicts used during rendering.
        """
        import io
        import re
        import pandas as pd
        # standardize_color already imported at module level from utils.color_utils

        csv_path = self.layer_map_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"layer_map_csv not found: {csv_path}")

        self._vprint(f"Loading layer map from: {csv_path}", level='full')

        # Color values like ``rgba(74,144,226,0.3)`` contain commas; pandas would
        # split them into extra columns unless they are quoted. Quote any CSS color
        # function in the raw text so hand-written CSVs with unquoted colors parse
        # correctly. Already-quoted values are left alone.
        df = self._read_layer_csv_tolerant(csv_path)

        # Validate columns: a 'layer' column plus a neuron column ('neuron' or
        # the legacy 'id_type_instance' alias).
        if 'layer' not in df.columns:
            raise ValueError(f"layer_map_csv must have column 'layer'. Found: {list(df.columns)}")
        neuron_col = next(
            (c for c in ('neuron', 'id_type_instance') if c in df.columns), None
        )
        if neuron_col is None:
            raise ValueError(
                "layer_map_csv must have a 'neuron' (or legacy 'id_type_instance') "
                f"column. Found: {list(df.columns)}"
            )

        # Preserve the source rows before numeric layer normalization and
        # neuron resolution.  This is the human-readable/reusable form that
        # the export writer should round-trip; the resolved DataFrames below
        # remain the canonical form for rendering and connector queries.
        self._viz_layer_info_input_rows = []
        for record in df.to_dict('records'):
            self._viz_layer_info_input_rows.append({
                'layer': record.get('layer', ''),
                'neuron': record.get(neuron_col, ''),
                'color': record.get('color', ''),
                'synapse_color': record.get('synapse_color', ''),
                'pre_synaptic_color': record.get('pre_synaptic_color', ''),
                'post_synaptic_color': record.get('post_synaptic_color', ''),
            })

        # The layer column is either numeric (base auto-detect for 0/1) or a
        # set of group-name labels (legacy behavior: group by the label).
        layer_values = df['layer'].dropna().tolist()
        numeric_layers = bool(layer_values)
        for value in layer_values:
            try:
                float(value)
            except (TypeError, ValueError):
                numeric_layers = False
                break
        if numeric_layers:
            self._layer_map_numeric_layers = True
            # Base-0/1 auto-detect: subtract the lowest layer value so the
            # first layer becomes index 0 internally, keeping the original
            # label only as the source index. Numeric indices are not useful
            # display names and must not produce folders such as
            # ``plot-3d_MCNS_1_<timestamp>``.
            layer_offset = int(min(float(v) for v in layer_values))
            df['_layer_idx'] = (
                df['layer'].astype(float).round().astype(int) - layer_offset
            )
        else:
            self._layer_map_numeric_layers = False
            layer_offset = None
            df['_layer_idx'] = [str(v) for v in df['layer'].fillna('')]

        # Keep the CSV's layer identity alongside each override.  The public
        # resolver still has a global fallback for legacy callers, but a
        # layer-aware map is necessary when the same type/body label appears
        # in more than one editor row with different site colors.
        layer_groups = df.groupby('_layer_idx', sort=True)[neuron_col].apply(list).to_dict()
        self._layer_map_layer_indices = {
            layer_key: index for index, layer_key in enumerate(layer_groups)
        }

        # Optional color columns.
        color_col = 'color' if 'color' in df.columns else None
        synapse_color_col = 'synapse_color' if 'synapse_color' in df.columns else None
        pre_color_col = 'pre_synaptic_color' if 'pre_synaptic_color' in df.columns else None
        post_color_col = 'post_synaptic_color' if 'post_synaptic_color' in df.columns else None
        optional_cols = [
            c for c in (color_col, synapse_color_col, pre_color_col, post_color_col) if c
        ]
        if optional_cols:
            self._vprint(
                f"  Found color columns {optional_cols} - will apply per-neuron colors",
                level='simple',
            )

        # Build per-neuron color overrides for each present color column.
        self._neuron_color_overrides = {}
        self._neuron_synapse_color_overrides = {}
        self._neuron_pre_color_overrides = {}
        self._neuron_post_color_overrides = {}
        self._neuron_color_overrides_by_layer = {}
        self._neuron_synapse_color_overrides_by_layer = {}
        self._neuron_pre_color_overrides_by_layer = {}
        self._neuron_post_color_overrides_by_layer = {}
        override_targets = [
            (
                color_col, self._neuron_color_overrides,
                self._neuron_color_overrides_by_layer,
                self.neuron_alpha, 'color',
            ),
            (
                synapse_color_col, self._neuron_synapse_color_overrides,
                self._neuron_synapse_color_overrides_by_layer,
                self.synapse_alpha, 'synapse_color',
            ),
            (
                pre_color_col, self._neuron_pre_color_overrides,
                self._neuron_pre_color_overrides_by_layer,
                self.synapse_alpha, 'pre_synaptic_color',
            ),
            (
                post_color_col, self._neuron_post_color_overrides,
                self._neuron_post_color_overrides_by_layer,
                self.synapse_alpha, 'post_synaptic_color',
            ),
        ]
        for col, store, layered_store, default_alpha, label in override_targets:
            if col is None:
                continue
            for _, row in df.iterrows():
                id_val = row[neuron_col]
                color_val = row[col]
                if pd.isna(color_val) or not str(color_val).strip():
                    continue
                id_str = str(id_val).strip()
                id_key = (
                    normalize_flywire_body_id(id_str)
                    if is_flywire_dataset(self.dataset) and id_str.isdigit()
                    else (int(id_str) if id_str.isdigit() else id_str)
                )
                try:
                    rgba_str = standardize_color(str(color_val).strip(), default_alpha=default_alpha)
                    # Store every safe alias, including the ``15832.0`` ->
                    # ``15832`` normalization used by connector readers.
                    lookup_keys = self._normalize_neuron_lookup_keys(id_val)
                    if id_key not in lookup_keys:
                        lookup_keys.insert(0, id_key)
                    if str(id_key) not in lookup_keys:
                        lookup_keys.append(str(id_key))
                    layer_index = self._layer_map_layer_indices.get(row['_layer_idx'])
                    layer_values = layered_store.setdefault(layer_index, {})
                    for key in lookup_keys:
                        store[key] = rgba_str
                        layer_values[key] = rgba_str
                except Exception as e:
                    self._vprint(
                        f"  ⚠️ Failed to parse {label} '{color_val}' for {id_val}: {e}",
                        level='full',
                    )
            if store:
                self._vprint(
                    f"  Loaded {len(store) // 2} per-neuron {label} overrides",
                    level='simple',
                )

        # Construct neuron_layers and custom_layer_names.
        self.neuron_layers = []
        self.custom_layer_names = []
        for layer_idx in sorted(layer_groups):
            identifiers = layer_groups[layer_idx]
            processed_ids = []
            for id_val in identifiers:
                id_str = str(id_val).strip()
                if id_str.isdigit():
                    processed_ids.append(
                        normalize_flywire_body_id(id_str)
                        if is_flywire_dataset(self.dataset)
                        else int(id_str)
                    )
                else:
                    processed_ids.append(id_str)
            if len(processed_ids) == 1:
                self.neuron_layers.append(processed_ids[0])
            else:
                self.neuron_layers.append(processed_ids)
            label = layer_idx + layer_offset if layer_offset is not None else str(layer_idx)
            self.custom_layer_names.append(str(label))

        self._vprint(f"  Loaded {len(self.neuron_layers)} layers from CSV:")
        for i, (name, neurons) in enumerate(zip(self.custom_layer_names, self.neuron_layers)):
            n_count = len(neurons) if isinstance(neurons, list) else 1
            self._vprint(f"    Layer {i}: {name} ({n_count} neurons)")

    def _apply_soma_radius_cap(self, neuron_vols):
        """
        Apply radius capping and optional smoothing to skeleton radii.
        
        When smooth_skeleton=False (default): Only applies hard cap to radii.
        When smooth_skeleton=True: Also applies iterative smoothing to prevent
        extrusion artifacts from chains of large-radius nodes.
        
        Parameters
        ----------
        neuron_vols : navis.NeuronList
            List of neurons to process (modified in place)
        """
        cap = self.soma_radius_cap
        total_capped = 0
        total_smoothed = 0
        
        for n in neuron_vols:
            if not hasattr(n, 'nodes') or not isinstance(n.nodes, pd.DataFrame):
                continue
            if 'radius' not in n.nodes.columns:
                continue
            
            nodes = n.nodes
            radii = nodes['radius'].values.copy().astype(float)
            original_radii = radii.copy()
            
            # Step 1: Hard cap all radii above threshold
            over_cap = radii > cap
            if over_cap.any():
                radii[over_cap] = cap
                total_capped += over_cap.sum()
            
            # Step 2: Optional iterative smoothing (only if smooth_skeleton=True)
            if self.smooth_skeleton and 'parent_id' in nodes.columns and 'node_id' in nodes.columns:
                node_ids = nodes['node_id'].values
                parent_ids = nodes['parent_id'].values
                id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
                
                # Build child map
                children = {idx: [] for idx in range(len(radii))}
                for idx, pid in enumerate(parent_ids):
                    if pid in id_to_idx:
                        children[id_to_idx[pid]].append(idx)
                
                # Aggressive smoothing: 20 passes with very strong neighbor influence
                for pass_num in range(20):
                    new_radii = radii.copy()
                    for idx in range(len(radii)):
                        if radii[idx] <= 0:
                            continue
                        
                        # Collect neighbor radii (parent + children)
                        neighbors = []
                        pid = parent_ids[idx]
                        if pid in id_to_idx:
                            neighbors.append(radii[id_to_idx[pid]])
                        for child_idx in children[idx]:
                            if radii[child_idx] > 0:
                                neighbors.append(radii[child_idx])
                        
                        if neighbors:
                            # Very strong neighbor influence: 10% self, 90% neighbors
                            neighbor_avg = np.mean(neighbors)
                            new_radii[idx] = 0.1 * radii[idx] + 0.9 * neighbor_avg
                    
                    radii = new_radii
                
                # Count how many were significantly changed
                total_smoothed += np.sum(np.abs(radii - original_radii) > 1)
                
                # Final cap check after smoothing
                radii = np.minimum(radii, cap)
            
            n.nodes['radius'] = radii
        
        if total_capped > 0:
            if self.smooth_skeleton:
                self._vprint(f"  ✓ Radius capping: capped {total_capped}, smoothed {total_smoothed} nodes (cap={cap:.0f}nm)", level='full')
            else:
                self._vprint(f"  ✓ Radius capping: capped {total_capped} nodes (cap={cap:.0f}nm)", level='full')

    @contextmanager
    def _suppress_output(self):
        """Suppress stdout and stderr if verbose is not full."""
        if self.verbose == 'full':
            yield
        else:
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                try:  
                    yield
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

    @contextmanager
    def _suppress_navis_progress(self):
        """Hide navis' internal tqdm bars while a shared bar is active.

        navis controls its progress bars through ``navis.config`` rather than
        the current stdout/stderr streams, so redirecting output alone does
        not prevent nested bars such as ``Smoothing`` from being emitted.
        Restore the caller's setting on exit because the navis configuration
        is process-global.
        """
        previous = getattr(navis.config, 'pbar_hide', False)
        navis.config.pbar_hide = True
        try:
            yield
        finally:
            navis.config.pbar_hide = previous

    def _xform_neurons_safe(self, neuron_vols, source, target, layer_label='layer',
                            progress_bar=None, compact_progress=False):
        """
        Transform neurons one at a time with explicit progress and per-neuron fallback.

        ``navis.xform_brain(neuron_vols, ...)`` is a single black-box call: when
        one pathological neuron (or a stall inside the transform backend) freezes
        the run, the only clue is a tqdm bar stuck at some percentage. This helper
        instead:

          * resolves the bridging transform sequence once and shows it in the
            existing layer progress bar,
          * shows the neuron currently being transformed in that bar's postfix,
            so a stall immediately identifies the culprit neuron,
          * keeps the untransformed original on a per-neuron failure instead of
            aborting the whole layer,
          * releases each original neuron as soon as its copy is transformed to
            keep peak memory flat on very large layers.

        ``progress_bar`` is the outer layer bar used by ``plot_neurons``. Keeping
        transformation state in that bar avoids nested bars and prevents every
        status update from becoming a permanent log line. When the helper is
        called directly, it creates a private single-line bar as a fallback.
        """
        from navis.transforms import registry as _registry

        # flybrains registers the template brains + transforms with navis;
        # make sure it is imported before resolving the bridging sequence.
        try:
            import flybrains  # noqa: F401
        except ImportError:
            pass

        neurons = list(neuron_vols) if isinstance(neuron_vols, navis.NeuronList) else [neuron_vols]
        if len(neurons) == 1 and isinstance(neurons[0], (list, tuple)):
            # A plain Python list of neurons is a valid layer shape; unwrap it
            # so navis.xform is called per neuron (a whole-list call would
            # spawn navis' own per-neuron progress bar that prints on
            # alternate rows instead of refreshing this shared bar).
            neurons = list(neurons[0])
        total = len(neurons)

        n_nodes_total = 0
        for n in neurons:
            # MeshNeurons report ``n_nodes`` as None (navis does not
            # implement it for meshes); count vertices instead so the
            # progress display never crashes on tube meshes.
            if hasattr(n, 'n_nodes') and n.n_nodes is not None:
                n_nodes_total += int(n.n_nodes)
            elif hasattr(n, 'vertices') and getattr(n, 'vertices', None) is not None:
                n_nodes_total += len(n.vertices)

        # Resolve the transform sequence once (raises if no path exists, so the
        # caller can trigger the transform download + retry flow).
        path, seq = _registry.shortest_bridging_seq(source, target)
        path_label = '→'.join(str(p) for p in path)

        start = time.time()
        xf_neurons = []
        failed = []
        owns_progress_bar = progress_bar is None
        if progress_bar is not None:
            pbar = progress_bar
        else:
            pbar = tqdm(
                total=total,
                desc=f'  Transforming {layer_label}',
                unit='neuron',
                mininterval=0.5,
                disable=not self.verbose,
                leave=False,
                file=sys.stdout,
            )

        def set_status(status, refresh=True):
            """Refresh the shared bar without emitting a standalone log line."""
            if pbar is not None and hasattr(pbar, 'set_postfix_str'):
                pbar.set_postfix_str(status, refresh=refresh)

        if compact_progress:
            set_status(f'{layer_label} (0/{total})')
        else:
            set_status(
                f'{layer_label}: xform 0/{total} | '
                f'{n_nodes_total:,} pts | {path_label}',
            )
        try:
            for i, n in enumerate(neurons):
                label = getattr(n, 'name', None) or str(getattr(n, 'id', i))
                # Refresh before the potentially slow transform so a stalled
                # neuron remains visible in the one-line progress display.
                if compact_progress:
                    set_status(f'{layer_label} ({i}/{total})')
                else:
                    set_status(
                        f'{layer_label}: xform {i}/{total} | current={label}')
                if not isinstance(n, navis.BaseNeuron):
                    # Volumes or other non-neuron objects pass through untouched
                    xf_neurons.append(n)
                    neurons[i] = None
                    if owns_progress_bar:
                        pbar.update(1)
                    continue
                try:
                    xf_neurons.append(navis.xform(n, transform=seq))
                except Exception as e:
                    failed.append((label, str(e)))
                    xf_neurons.append(n)
                # Free the original as soon as its copy is done to keep peak
                # memory flat (large layers double memory inside navis.xform).
                neurons[i] = None
                if owns_progress_bar:
                    pbar.update(1)

            if failed:
                failed_labels = ', '.join(label for label, _ in failed[:3])
                if len(failed) > 3:
                    failed_labels += ', …'
                set_status(
                    f'{layer_label} ({total}/{total})'
                    if compact_progress else
                    f'{layer_label}: xform {total}/{total} | '
                    f'⚠ {len(failed)} failed',
                )
            else:
                elapsed = time.time() - start
                set_status(
                    f'{layer_label} ({total}/{total})'
                    if compact_progress else
                    f'{layer_label}: xform {total}/{total} | '
                    f'done {elapsed:.1f}s',
                )
        finally:
            if owns_progress_bar:
                pbar.close()

        if failed:
            if self.verbose:
                tqdm.write(f'  ⚠️  {layer_label}: {len(failed)}/{total} neurons could not '
                           f'be transformed ({failed_labels}); kept in source coordinates')
        return navis.NeuronList(xf_neurons)

    def _validate_inputs(self):
        """
        Validate all input parameters before processing.
        Raises ValueError with descriptive messages for invalid inputs.
        """
        errors = []
        
        # === String validations ===
        if not isinstance(self.backend, str):
            errors.append(f"backend must be a string, got {type(self.backend).__name__}")
        elif self.backend not in ('plotly', 'k3d'):
            errors.append(f"backend must be 'plotly' or 'k3d', got '{self.backend}'")
            
        if not isinstance(self.dataset, str):
            errors.append(f"dataset must be a string, got {type(self.dataset).__name__}")
        elif not self.dataset:
            errors.append("dataset cannot be empty")
            
        if not isinstance(self.client_type, str):
            errors.append(f"client_type must be a string, got {type(self.client_type).__name__}")
        elif self.client_type not in ('neuprint', 'flywire'):
            errors.append(f"client_type must be 'neuprint' or 'flywire', got '{self.client_type}'")

        if not isinstance(self.output_format, str):
            errors.append(f"output_format must be a string, got {type(self.output_format).__name__}")
        elif self.output_format.lower() not in ('xlsx', 'csv'):
            errors.append(f"output_format must be 'xlsx' or 'csv', got '{self.output_format}'")

        if not isinstance(self.search_columns, str):
            errors.append(f"search_columns must be a string, got {type(self.search_columns).__name__}")
        elif self.search_columns not in ('auto', 'type', 'instance', 'bodyId'):
            errors.append(
                f"search_columns must be 'auto', 'type', 'instance' or 'bodyId', "
                f"got '{self.search_columns}'"
            )

        if not isinstance(self.hemisphere, str):
            errors.append(f"hemisphere must be a string, got {type(self.hemisphere).__name__}")
        elif self.hemisphere not in ('both', 'left', 'right'):
            errors.append(
                f"hemisphere must be 'both', 'left' or 'right', got '{self.hemisphere}'"
            )
            
        if not isinstance(self.server, str):
            errors.append(f"server must be a string, got {type(self.server).__name__}")
            
        if self.token is not None and not isinstance(self.token, str):
            errors.append(f"token must be a string or None, got {type(self.token).__name__}")
            
        if self.version is not None and not isinstance(self.version, int):
            errors.append(f"version must be an integer or None, got {type(self.version).__name__}")
        elif self.version is not None and self.version < 0:
            errors.append(f"version must be positive, got {self.version}")
            
        # === Path validations ===
        if not isinstance(self.data_folder, str):
            errors.append(f"data_folder must be a string, got {type(self.data_folder).__name__}")
            
        if self.output_dir is not None and not isinstance(self.output_dir, str):
            errors.append(f"output_dir must be a string or None, got {type(self.output_dir).__name__}")
            
        if self.saveas is not None and not isinstance(self.saveas, str):
            errors.append(f"saveas must be a string or None, got {type(self.saveas).__name__}")
            
        if self.layer_map_csv is not None:
            if not isinstance(self.layer_map_csv, str):
                errors.append(f"layer_map_csv must be a string or None, got {type(self.layer_map_csv).__name__}")
            elif not os.path.exists(self.layer_map_csv):
                errors.append(f"layer_map_csv file not found: {self.layer_map_csv}")
        
        # === Neuron layers validation ===
        if not isinstance(self.neuron_layers, (str, list)):
            errors.append(f"neuron_layers must be a string or list, got {type(self.neuron_layers).__name__}")
        else:
            # Check if neuron_layers is empty (empty string or empty list)
            is_empty = (isinstance(self.neuron_layers, str) and not self.neuron_layers) or \
                       (isinstance(self.neuron_layers, list) and len(self.neuron_layers) == 0)
            
            if is_empty and self.layer_map_csv is None:
                # Allow empty neuron_layers if mesh_roi or brain_mesh is specified (mesh-only mode)
                has_mesh = self.mesh_roi or (self.brain_mesh and self.brain_mesh.lower() != 'none') or self.vnc_mesh
                if not has_mesh:
                    errors.append("neuron_layers cannot be empty (or provide layer_map_csv, mesh_roi, or brain_mesh)")
            
        if not isinstance(self.custom_layer_names, list):
            errors.append(f"custom_layer_names must be a list, got {type(self.custom_layer_names).__name__}")
            
        # === Boolean validations ===
        bool_params = [
            ('cache_neurons', self.cache_neurons),
            ('cache_synapses', self.cache_synapses),
            ('skip_synapse', self.skip_synapse),
            ('include_timestamp', self.include_timestamp),
            ('auto_fix_extrusions', self.auto_fix_extrusions),
            ('smooth_skeleton', self.smooth_skeleton),
        ]
        for name, value in bool_params:
            if not isinstance(value, bool):
                errors.append(f"{name} must be a boolean, got {type(value).__name__}")
        
        # === Numeric validations ===
        if not isinstance(self.min_synapse_num, int):
            errors.append(f"min_synapse_num must be an integer, got {type(self.min_synapse_num).__name__}")
        elif self.min_synapse_num < 0:
            errors.append(f"min_synapse_num must be non-negative, got {self.min_synapse_num}")
            
        if not isinstance(self.export_scale, int):
            errors.append(f"export_scale must be an integer, got {type(self.export_scale).__name__}")
        elif self.export_scale < 1 or self.export_scale > 10:
            errors.append(f"export_scale must be between 1 and 10, got {self.export_scale}")
            
        if not isinstance(self.export_timeout, int):
            errors.append(f"export_timeout must be an integer, got {type(self.export_timeout).__name__}")
        elif self.export_timeout < 1:
            errors.append(f"export_timeout must be at least 1 second, got {self.export_timeout}")
            
        # === Float validations ===
        if not isinstance(self.neuron_alpha, (int, float)):
            errors.append(f"neuron_alpha must be a number, got {type(self.neuron_alpha).__name__}")
        elif not 0 <= self.neuron_alpha <= 1:
            errors.append(f"neuron_alpha must be between 0 and 1, got {self.neuron_alpha}")
            
        if not isinstance(self.synapse_alpha, (int, float)):
            errors.append(f"synapse_alpha must be a number, got {type(self.synapse_alpha).__name__}")
        elif not 0 <= self.synapse_alpha <= 1:
            errors.append(f"synapse_alpha must be between 0 and 1, got {self.synapse_alpha}")
            
        # Optional floats (can be None)
        if self.skeleton_mesh_simplification is not None:
            if not isinstance(self.skeleton_mesh_simplification, (int, float)):
                errors.append(f"skeleton_mesh_simplification must be a number or None, got {type(self.skeleton_mesh_simplification).__name__}")
            elif not 0 <= self.skeleton_mesh_simplification <= 1:
                errors.append(f"skeleton_mesh_simplification must be between 0 and 1, got {self.skeleton_mesh_simplification}")
                
        if self.soma_mesh_simplification is not None:
            if not isinstance(self.soma_mesh_simplification, (int, float)):
                errors.append(f"soma_mesh_simplification must be a number or None, got {type(self.soma_mesh_simplification).__name__}")
            elif not 0 <= self.soma_mesh_simplification <= 1:
                errors.append(f"soma_mesh_simplification must be between 0 and 1, got {self.soma_mesh_simplification}")
                
        if self.soma_radius_cap is not None:
            if not isinstance(self.soma_radius_cap, (int, float)):
                errors.append(f"soma_radius_cap must be a number or None, got {type(self.soma_radius_cap).__name__}")
            elif self.soma_radius_cap <= 0:
                errors.append(f"soma_radius_cap must be positive, got {self.soma_radius_cap}")
                
        if self.webdriver_render_wait is not None:
            if not isinstance(self.webdriver_render_wait, (int, float)):
                errors.append(f"webdriver_render_wait must be a number or None, got {type(self.webdriver_render_wait).__name__}")
            elif self.webdriver_render_wait < 0:
                errors.append(f"webdriver_render_wait must be non-negative, got {self.webdriver_render_wait}")
        
        # === Synapse size validation ===
        # Accepts a pixel value (1-12, default 3); for backward compatibility the
        # parser tolerates 'real' and legacy 'Nx real' fold notation, resolving
        # them to their numeric fold (kept for mesh sizing). Unparseable strings
        # are rejected.
        if isinstance(self.synapse_size, str):
            parsed = self._parse_synapse_size(self.synapse_size)
            if parsed is None:
                errors.append(
                    "synapse_size must be a number (or 'real' for the default), "
                    f"got '{self.synapse_size}'"
                )
            else:
                self.synapse_size = parsed
        if not isinstance(self.synapse_size, (int, float, str)):
            errors.append(f"synapse_size must be a number or 'real', got {type(self.synapse_size).__name__}")
        elif isinstance(self.synapse_size, (int, float)) and self.synapse_size < 0:
            errors.append(f"synapse_size must be non-negative, got {self.synapse_size}")
            
        # === Mode validations (done in detail later, basic type check here) ===
        if not isinstance(self.skeleton_mode, str):
            errors.append(f"skeleton_mode must be a string, got {type(self.skeleton_mode).__name__}")

        if not isinstance(self.neuprint_skeleton_pipeline, str):
            errors.append(
                'neuprint_skeleton_pipeline must be a string, '
                f'got {type(self.neuprint_skeleton_pipeline).__name__}'
            )
        elif self.neuprint_skeleton_pipeline.strip().lower() not in (
                'fast', 'fine', 'artistic', 'direct', 'fine_opt',
                'fine_opt1', 'fafb'):
            errors.append(
                'neuprint_skeleton_pipeline must be "fast", "fine", '
                'or "artistic", '
                f'got "{self.neuprint_skeleton_pipeline}"'
            )
            
        if not isinstance(self.synapse_mode, str):
            errors.append(f"synapse_mode must be a string, got {type(self.synapse_mode).__name__}")
            
        if not isinstance(self.brain_mesh, str):
            errors.append(f"brain_mesh must be a string, got {type(self.brain_mesh).__name__}")
            
        if not isinstance(self.legend_mode, str):
            errors.append(f"legend_mode must be a string, got {type(self.legend_mode).__name__}")

        if not isinstance(self.color_mode, str):
            errors.append(f"color_mode must be a string, got {type(self.color_mode).__name__}")
            
        if not isinstance(self.expand_colors, str):
            errors.append(f"expand_colors must be a string, got {type(self.expand_colors).__name__}")
            
        if not isinstance(self.export_method, str):
            errors.append(f"export_method must be a string, got {type(self.export_method).__name__}")
        elif self.export_method not in ('webdriver', 'webdriver-fast', 'kaleido'):
            errors.append(f"export_method must be 'webdriver' or 'kaleido', got '{self.export_method}'")
            
        # === export_views validation ===
        valid_views = {'front', 'back', 'top', 'bottom', 'left', 'right', 'lateral', 'all'}
        if isinstance(self.export_views, bool):
            pass  # Valid
        elif isinstance(self.export_views, list):
            for view in self.export_views:
                if not isinstance(view, str):
                    errors.append(f"export_views list items must be strings, got {type(view).__name__}")
                elif view.lower() not in valid_views:
                    errors.append(f"Invalid view in export_views: '{view}'. Valid options: {sorted(valid_views)}")
        else:
            errors.append(f"export_views must be a boolean or list, got {type(self.export_views).__name__}")
            
        # === verbose validation ===
        if self.verbose not in (True, False, 'full', 'simple'):
            errors.append(f"verbose must be True, False, 'full', or 'simple', got {repr(self.verbose)}")
            
        # === Color validations (now more flexible - accepts multiple formats) ===
        # Empty/invalid values are coerced to defaults in _normalize_color_inputs();
        # anything still malformed here warns and falls back instead of failing.
        # neuron_colors: can be tuple, list, or single color string
        if isinstance(self.neuron_colors, str):
            # Single color string - will be wrapped in a list later
            pass  # Valid
        elif isinstance(self.neuron_colors, (tuple, list)):
            if len(self.neuron_colors) == 0:
                self._vprint("\033[33m⚠️  Warning: neuron_colors is empty; falling back to default palette.\033[0m", level='simple')
                self.neuron_colors = None
        else:
            self._vprint(f"\033[33m⚠️  Warning: neuron_colors must be a tuple, list, or color string, got {type(self.neuron_colors).__name__}; falling back to default palette.\033[0m", level='simple')
            self.neuron_colors = None
            
        # synapse_colors: same flexibility
        if isinstance(self.synapse_colors, str):
            pass  # Valid - single color string
        elif isinstance(self.synapse_colors, (tuple, list)):
            if len(self.synapse_colors) == 0:
                self._vprint("\033[33m⚠️  Warning: synapse_colors is empty; falling back to default palette.\033[0m", level='simple')
                self.synapse_colors = list(bokeh.palettes.Category10[10])
        else:
            self._vprint(f"\033[33m⚠️  Warning: synapse_colors must be a tuple, list, or color string, got {type(self.synapse_colors).__name__}; falling back to default palette.\033[0m", level='simple')
            self.synapse_colors = list(bokeh.palettes.Category10[10])
            
        # === mesh_roi validation ===
        if self.mesh_roi is not None and not isinstance(self.mesh_roi, (list, str)):
            errors.append(f"mesh_roi must be a list, string, or None, got {type(self.mesh_roi).__name__}")
            
        # === html_size_cap validation ===
        if self.html_size_cap is not None:
            if not isinstance(self.html_size_cap, int):
                errors.append(f"html_size_cap must be an integer or None, got {type(self.html_size_cap).__name__}")
            elif self.html_size_cap < 1:
                errors.append(f"html_size_cap must be at least 1 MB, got {self.html_size_cap}")
                
        # === export_simplified_png validation ===
        if not isinstance(self.export_simplified_png, (bool, int)):
            errors.append(f"export_simplified_png must be a boolean or integer, got {type(self.export_simplified_png).__name__}")
        elif isinstance(self.export_simplified_png, int) and self.export_simplified_png < 0:
            errors.append(f"export_simplified_png threshold must be non-negative, got {self.export_simplified_png}")
        
        # === Additional boolean validations ===
        additional_bools = [
            ('force_API_fetching', self.force_API_fetching),
            ('vnc_mesh', self.vnc_mesh),
            ('mirror_on_contralateral', self.mirror_on_contralateral),
            ('show_soma', self.show_soma),
            ('show_fig', self.show_fig),
            ('show_connectors', self.show_connectors),
            ('FAFB_template_correction', self.FAFB_template_correction),
            ('uniform_synapse_size', self.uniform_synapse_size),
            ('pre_post_scatter', self.pre_post_scatter),
        ]
        for name, value in additional_bools:
            if not isinstance(value, bool):
                errors.append(f"{name} must be a boolean, got {type(value).__name__}")
        
        # === Additional float validations ===
        if self.soma_region_radius is not None:
            if not isinstance(self.soma_region_radius, (int, float)):
                errors.append(f"soma_region_radius must be a number, got {type(self.soma_region_radius).__name__}")
            elif self.soma_region_radius <= 0:
                errors.append(f"soma_region_radius must be positive, got {self.soma_region_radius}")
                
        if self.roi_mesh_simplification is not None:
            if not isinstance(self.roi_mesh_simplification, (int, float)):
                errors.append(f"roi_mesh_simplification must be a number, got {type(self.roi_mesh_simplification).__name__}")
            elif not 0 <= self.roi_mesh_simplification <= 1:
                errors.append(f"roi_mesh_simplification must be between 0 and 1, got {self.roi_mesh_simplification}")
        
        # === String path validations ===
        if not isinstance(self.transforms_dir, str):
            errors.append(f"transforms_dir must be a string, got {type(self.transforms_dir).__name__}")
            
        # === Color validations for brain_mesh_color and vnc_mesh_color (flexible formats) ===
        # These now accept: 'auto', named colors, hex, rgba strings, tuples.
        # Malformed values warn and fall back to 'auto' instead of failing.
        def _validate_mesh_color(color, name):
            """Helper to validate flexible color format."""
            if isinstance(color, str):
                return  # Strings are validated later when standardizing
            elif isinstance(color, (tuple, list)):
                if len(color) >= 3:
                    return  # Valid tuple format
                self._vprint(f"\033[33m⚠️  Warning: {name} tuple must have at least 3 values (RGB), got {len(color)}; falling back to 'auto'.\033[0m", level='simple')
                setattr(self, name, 'auto')
            else:
                self._vprint(f"\033[33m⚠️  Warning: {name} must be a string, tuple, or list, got {type(color).__name__}; falling back to 'auto'.\033[0m", level='simple')
                setattr(self, name, 'auto')
        
        _validate_mesh_color(self.brain_mesh_color, 'brain_mesh_color')
        _validate_mesh_color(self.vnc_mesh_color, 'vnc_mesh_color')
            
        # === mesh_color validation (flexible formats now) ===
        if self.mesh_color is None:
            # None means "not specified": use the default gray tuple
            self.mesh_color = (100, 100, 100)
        if isinstance(self.mesh_color, str):
            pass  # Will be standardized later
        elif isinstance(self.mesh_color, (tuple, list)):
            # Check if it's a single color or list of colors
            if len(self.mesh_color) >= 3:
                # Could be (R, G, B, alpha) or [(color1), (color2), ...]
                if all(isinstance(x, (int, float)) for x in self.mesh_color):
                    # Single color tuple
                    if len(self.mesh_color) >= 4:
                        r, g, b, a = self.mesh_color[0], self.mesh_color[1], self.mesh_color[2], self.mesh_color[3]
                        # Numeric alpha accepts normalized 0-1 and byte
                        # 0-255 notation, just like the shared parser.
                        if not isinstance(a, (int, float, np.number)) or not 0 <= a <= 255:
                            self._vprint(f"\033[33m⚠️  Warning: mesh_color alpha must be 0-1 or 0-255, got {a}; using mesh_alpha instead.\033[0m", level='simple')
                            self.mesh_color = (r, g, b)
                # else: list of colors, validated later when standardizing
        else:
            self._vprint(f"\033[33m⚠️  Warning: mesh_color must be a tuple, list, or string, got {type(self.mesh_color).__name__}; falling back to default gray.\033[0m", level='simple')
            self.mesh_color = (100, 100, 100)
        
        # === Raise all errors together ===
        if errors:
            error_msg = "VisualizeSkeleton input validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)

    @staticmethod
    def _parse_synapse_size(value: str):
        """Parse a synapse_size string into 'real' or a numeric value.

        The UI sends a bare pixel size (1-12, default 3). For backward
        compatibility the parser also tolerates 'real' and legacy 'Nx real'
        fold notation ("2", "2.5x", "2 x real"), resolving them to their
        numeric fold. Returns None for anything unparseable.
        """
        text = value.strip().lower()
        if text == 'real':
            return 'real'
        match = re.fullmatch(r'([0-9]*\.?[0-9]+)\s*(?:x|\u00d7)?(?:\s*real)?', text)
        if match:
            return float(match.group(1))
        return None

    def _is_valid_color(self, color) -> bool:
        """Return True if `color` parses to a usable color (any supported format)."""
        try:
            standardize_color(color)
            return True
        except Exception:
            return False

    @staticmethod
    def _unwrap_palette_value(value):
        """Unwrap a serialized or list-compatible continuous palette value."""
        if isinstance(value, dict) and 'colors' in value:
            return value.get('colors'), bool(value.get('continuous', False))
        return value, bool(getattr(value, 'is_continuous_palette', False))

    def _normalize_color_inputs(self):
        """
        Coerce empty or invalid color inputs to their defaults before validation.

        Empty lists (e.g., the UI sends synapse_colors=[] for a single layer)
        and unparseable values never hard-fail: each falls back to the
        parameter's default palette with a warning, regardless of skip_synapse.
        """
        def warn(name, detail, fallback_label):
            self._vprint(
                f"\033[33m⚠️  Warning: {name} {detail}; falling back to default ({fallback_label}).\033[0m",
                level='simple'
            )

        def normalize_sequence(value, name, fallback, fallback_label, allow_none=False, silent_empty=False):
            """Normalize a parameter that accepts a single color or a list of colors."""
            if hasattr(value, 'tolist') and not isinstance(value, (str, tuple, list)):
                value = value.tolist()
            if value is None:
                return None if allow_none else fallback
            # Single color string
            if isinstance(value, str):
                if self._is_valid_color(value):
                    return value
                warn(name, f"contains an invalid color '{value}'", fallback_label)
                return fallback
            if isinstance(value, (tuple, list)):
                # A valid 3/4-channel sequence is one color. A palette of
                # names/strings fails this check and continues below.
                if len(value) in (3, 4) and self._is_valid_color(value):
                    return value
                if len(value) == 0:
                    if not silent_empty:
                        warn(name, "is empty", fallback_label)
                    return fallback
                # List of colors: drop unparseable entries, keep the rest
                valid = [c for c in value if self._is_valid_color(c)]
                dropped = len(value) - len(valid)
                if dropped and valid:
                    self._vprint(
                        f"\033[33m⚠️  Warning: {name} contains {dropped} invalid color(s) that were dropped.\033[0m",
                        level='simple'
                    )
                elif dropped:
                    warn(name, f"contains {dropped} invalid color(s)", fallback_label)
                return valid if valid else fallback
            warn(name, f"has unsupported type {type(value).__name__}", fallback_label)
            return fallback

        def normalize_single(value, name, fallback='auto'):
            """Normalize a parameter that accepts exactly one color (or 'auto')."""
            if hasattr(value, 'tolist') and not isinstance(value, (str, tuple, list)):
                value = value.tolist()
            if isinstance(value, str):
                if value.strip().lower() == 'auto' or self._is_valid_color(value):
                    return value
                warn(name, f"contains an invalid color '{value}'", f"'{fallback}'")
                return fallback
            if isinstance(value, (tuple, list)):
                if len(value) in (3, 4) and self._is_valid_color(value):
                    return value
                warn(name, f"contains an invalid color {value!r}", f"'{fallback}'")
                return fallback
            warn(name, f"has unsupported type {type(value).__name__}", f"'{fallback}'")
            return fallback

        # Preserve metadata from the UI's continuous-preset selection while
        # normalizing its list values. Shared analysis tabs serialize this as
        # {"colors": [...], "continuous": True} before launching a worker;
        # the direct 3D tab may pass the list-compatible UI value instead.
        self.neuron_colors, self._neuron_colors_continuous = self._unwrap_palette_value(
            self.neuron_colors
        )
        self.synapse_colors, self._synapse_colors_continuous = self._unwrap_palette_value(
            self.synapse_colors
        )
        self.mesh_color, self._mesh_colors_continuous = self._unwrap_palette_value(
            self.mesh_color
        )

        # neuron_colors: None triggers the background-dependent default palette below
        self.neuron_colors = normalize_sequence(
            self.neuron_colors, 'neuron_colors', None, 'default palette', allow_none=True
        )
        # synapse_colors: fall back to the field default palette. An empty
        # list is the expected input when there is only one neuron layer
        # (no inter-layer connections), so fall back silently in that case.
        n_layers = len(self.neuron_layers) if isinstance(self.neuron_layers, list) else 1
        self.synapse_colors = normalize_sequence(
            self.synapse_colors, 'synapse_colors',
            list(bokeh.palettes.Category10[10]), 'Category10 palette',
            silent_empty=(n_layers <= 1)
        )
        # mesh_color: fall back to the default gray (None is resolved by validation)
        self.mesh_color = normalize_sequence(
            self.mesh_color, 'mesh_color', (100, 100, 100), '(100, 100, 100)', allow_none=True
        )
        # brain/VNC mesh colors: 'auto' or a single valid color
        self.brain_mesh_color = normalize_single(self.brain_mesh_color, 'brain_mesh_color')
        self.vnc_mesh_color = normalize_single(self.vnc_mesh_color, 'vnc_mesh_color')
        # Normalize background colors to the same canonical CSS RGBA form used
        # by the renderers. This accepts names, hex/RGBA, RGB(A) tuples, CSS
        # rgb()/hsl() strings, percentages, and matplotlib-compatible names.
        try:
            if isinstance(self.background_color, str) and self.background_color.strip().lower() == 'auto':
                raise ValueError("'auto' is not a background color")
            self.background_color = standardize_color(self.background_color, default_alpha=1.0)
        except (TypeError, ValueError):
            warn('background_color', f"value {self.background_color!r} is not a valid color", "'white'")
            self.background_color = standardize_color('white')

    def _overlay_source_metadata(self, dataset, bids):
        """Neuron-table metadata rows for cross-dataset overlay layers.

        Overlay (``custom_neurons``) layers would otherwise export with only
        their bodyId: the per-layer metadata lookup runs against the SCENE
        dataset's table, which does not know the source dataset's bodyIds.
        Reads the source dataset's neuron index / allneurons table (local
        files, no network) and returns the rows for *bids*, or ``None``.
        """
        from pathlib import Path

        if not dataset or not bids:
            return None
        safe_name = str(dataset).replace(':', '_').replace('.', '_')
        root = Path(getattr(self, 'script_path', '') or '')
        candidates = [
            root / 'neuron_indexes' / safe_name / 'neuron_index.parquet',
            root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.parquet',
            root / 'datasets' / safe_name / f'{safe_name}_allneurons_neuron_df.csv',
        ]
        wanted = {str(int(b)) for b in bids}
        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix == '.parquet':
                    frame = pd.read_parquet(path)
                else:
                    frame = pd.read_csv(path, low_memory=False)
            except Exception:
                continue
            if 'bodyId' not in frame.columns:
                continue
            keys = frame['bodyId'].map(
                lambda v: str(int(v)) if pd.notna(v) and str(v).replace(
                    '.0', '').isdigit() else str(v))
            rows = frame[keys.isin(wanted)]
            if rows.empty:
                continue
            return rows.reset_index(drop=True)
        return None

    def _inject_custom_neurons(self):
        """Prepend custom_neurons overlay layers as the scene's first layer.

        Runs in ``__post_init__`` so every downstream consumer — the render
        loop, ``_save_neuron_info_csv``, and ``_save_viz_layer_info_csv`` —
        sees the overlay as a real layer (not only in the rendered traces).
        The overlay is inserted at index 0 so ``query_transformed_{name}``
        sits as layer 1, before the rank-1 homolog. Called once per instance;
        a second call is a no-op.
        """
        self._custom_neurons_by_layer = {}
        if getattr(self, '_custom_neurons_injected', False):
            return
        self._custom_neurons_injected = True
        if not getattr(self, 'custom_neurons', None):
            return
        # Hermetic test objects may be built without __post_init__ field
        # defaults; ensure the arrays we append to exist.
        if not hasattr(self, 'neuron_dfs') or self.neuron_dfs is None:
            self.neuron_dfs = []
        if not hasattr(self, 'neuron_layers') or self.neuron_layers is None:
            self.neuron_layers = []
        if not hasattr(self, 'layer_criteria') or self.layer_criteria is None:
            self.layer_criteria = []
        if not hasattr(self, 'roi_dfs') or self.roi_dfs is None:
            self.roi_dfs = []
        if not hasattr(self, 'layer_names') or self.layer_names is None:
            self.layer_names = []
        if not hasattr(self, '_viz_input_neuron_layers') or \
                self._viz_input_neuron_layers is None:
            self._viz_input_neuron_layers = []
        if not hasattr(self, '_viz_input_custom_layer_names') or \
                self._viz_input_custom_layer_names is None:
            self._viz_input_custom_layer_names = []
        if self._viz_input_custom_layer_names == '':
            self._viz_input_custom_layer_names = []

        import navis
        insert_pos = 0
        for entry in self.custom_neurons:
            try:
                layer_nm, neurons = entry
            except (TypeError, ValueError):
                self._vprint('  ⚠️ custom_neurons entries must be '
                             '(layer_name, neurons); entry skipped',
                             level='simple')
                continue
            try:
                neurons = navis.NeuronList(
                    neurons if isinstance(neurons, (list, navis.NeuronList))
                    else [neurons])
            except Exception as e:
                self._vprint(f'  ⚠️ custom_neurons layer "{layer_nm}" '
                             f'could not be wrapped: {e}', level='simple')
                continue
            if neurons.empty:
                continue
            bids = []
            for n in neurons:
                try:
                    bids.append(int(getattr(n, 'id', None)
                                     or getattr(n, 'bodyId', None) or 0))
                except (TypeError, ValueError):
                    bids.append(0)
            layer_nm = str(layer_nm)
            # Overlay neurons belong to the SOURCE dataset (stamped by the
            # homolog overlay builder); enrich their export rows from the
            # source dataset's neuron table so neuron_info.csv is not blank.
            source_dataset = None
            for n in neurons:
                ds = getattr(n, '_drocat_source_dataset', None)
                if ds:
                    source_dataset = str(ds)
                    break
            meta_frame = None
            if source_dataset:
                meta_frame = self._overlay_source_metadata(source_dataset, bids)
            if meta_frame is not None:
                self._vprint(
                    f'  ℹ️ Overlay layer "{layer_nm}" metadata resolved from '
                    f'{source_dataset} ({len(meta_frame)} rows)',
                    level='full',
                )
            else:
                meta_frame = pd.DataFrame({'bodyId': bids})
            # Insert at the front so the overlay is layer 1 (before the
            # rank-1 homolog). Multiple overlays keep their order at the
            # front.
            self.neuron_layers.insert(insert_pos, bids)
            self.neuron_dfs.insert(insert_pos, meta_frame)
            self.layer_criteria.insert(insert_pos, None)
            self.roi_dfs.insert(insert_pos, None)
            self.layer_names.insert(insert_pos, layer_nm)
            # Track in the input arrays the export writers read so the
            # overlay appears as an entry in viz_layer_info.csv /
            # neuron_info.csv.
            self._viz_input_neuron_layers.insert(insert_pos, bids)
            if isinstance(self._viz_input_custom_layer_names, list):
                self._viz_input_custom_layer_names.insert(insert_pos, layer_nm)
            elif self._viz_input_custom_layer_names is None:
                self._viz_input_custom_layer_names = [layer_nm]
            self._custom_neurons_by_layer[layer_nm] = neurons
            insert_pos += 1
        if self._custom_neurons_by_layer:
            n_layers = len(self.neuron_layers)
            self.neuron_colors = self._expand_color_sequence(
                self._base_neuron_colors, n_layers,
                target_label='layers', tip_parameter='neuron_colors',
                warn=False,
                continuous=self._neuron_colors_continuous)
            # One shared color for every overlay layer: the per-neuron
            # legend entries read as one group (query_transformed_*).
            self.neuron_colors = list(self.neuron_colors)
            for _i in range(len(self._custom_neurons_by_layer)):
                self.neuron_colors[_i] = QUERY_OVERLAY_COLOR
            self._vprint(
                '  ➕ Overlay layer(s) injected (pre-transformed, no '
                f'fetch/coordinate transform): '
                f'{", ".join(self._custom_neurons_by_layer)}',
                level='simple')

    def __post_init__(self):
        # Keep the local binding available before auto-generated output names
        # are sanitized below (a later compatibility branch also imports re).
        import re
        import copy

        # Keep the editor-facing inputs for the reusable layer-map export.
        # Resolution below intentionally converts names to body IDs for data
        # access, but the exported ``viz_layer_info.csv`` should remain a
        # readable representation of what the user entered.
        self._viz_input_neuron_layers = copy.deepcopy(self.neuron_layers)
        self._viz_input_custom_layer_names = copy.deepcopy(self.custom_layer_names)
        self._viz_input_neuron_colors = copy.deepcopy(self.neuron_colors)
        self._viz_input_synapse_colors = copy.deepcopy(self.synapse_colors)

        # Empty or invalid color inputs fall back to their default palettes
        # (with warnings) instead of failing validation.
        self._normalize_color_inputs()

        # Default neuron palette follows the background color: bokeh
        # Category10 on white, bokeh Set3 on black (resolved before the
        # color validation below runs).
        if self.neuron_colors is None:
            if is_dark_color(self.background_color):
                self.neuron_colors = bokeh.palettes.Set3[12]
            else:
                self.neuron_colors = bokeh.palettes.Category10[10]

        # Apply dataset-specific defaults BEFORE validation
        if self.dataset and 'manc' in self.dataset.lower():
            # For MANC, enable VNC mesh by default if not explicitly disabled
            # We assume if the user passed False explicitly, they know what they are doing.
            # But dataclasses don't track "default vs explicit" easily.
            # So we'll just set it to True if it's currently False, assuming default was False.
            # The default in field definition is False.
            # EXCEPTION: If brain_mesh='none', we assume user wants no context meshes at all,
            # so we do NOT enable VNC mesh by default.
            if self.vnc_mesh is False and self.brain_mesh != 'none':
                # We can't distinguish "User set False" vs "Default False".
                # To follow user request "set vnc_mesh=True by default":
                self.vnc_mesh = True
        
        # Check for elastix dependency if MANC 'whole' mode is requested
        if 'manc' in self.dataset.lower() and self.brain_mesh == 'whole':
            import shutil
            if shutil.which('elastix') is None:
                print('⚠️  Elastix not found: Cannot transform MANC to Male-CNS space.')
                print('   automatically changing brain_mesh from "whole" to "template".')
                print('   Tip: Install elastix or use male-cns dataset for full CNS context.')
                self.brain_mesh = 'template'
        
        # === INPUT VALIDATION ===
        # Validate all input parameters before processing
        self._validate_inputs()
        
        # Normalize verbose parameter
        if self.verbose is True:
            self.verbose = 'full'
        elif self.verbose is False:
            self.verbose = False

        # Record FlyWire readiness without rejecting cache-only/configuration
        # callers.  The actual FAFB readiness check runs at plot_skeleton(),
        # immediately before any skeleton query.  BANC is rejected here so
        # its converter cannot be mistaken for skeleton support.
        flywire_access = flywire_skeleton_readiness(
            self.dataset,
            project_root=self.script_path,
        )
        self._flywire_skeleton_access = flywire_access
        if flywire_access.get('is_banc'):
            require_flywire_skeleton_access(
                self.dataset,
                project_root=self.script_path,
                log=self._vprint,
            )
        
        # Silence navis INFO messages (like "Use the `.show()` method to plot the figure.")
        # These are not useful for automated visualization and clutter output
        try:
            # Set to ERROR to suppress "radii are missing" warnings which are common/harmless
            # when generating meshes from skeletons without radius info
            navis.set_loggers('ERROR') 
        except Exception:
            pass  # Ignore if function not available in older versions
            
        # Ensure cleanup of logging handlers if needed
        logging.getLogger('navis').setLevel(logging.ERROR)

        # Silence navis and other libraries' debug output if verbose is not full
        if self.verbose != 'full':
            logging.getLogger('navis').setLevel(logging.ERROR)
            logging.getLogger('trimesh').setLevel(logging.ERROR)
        
        # Initialize output_dir if not set
        if self.output_dir is None:
            self.output_dir = self.data_folder

        # Initialize list to store meshes for export
        self.exportable_meshes = []
        
        # Auto-detect client_type from dataset if not explicitly set to flywire
        if self.client_type == 'neuprint' and ('flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()):
            self.client_type = 'flywire'
            self._vprint(f"Auto-detected client_type='flywire' from dataset '{self.dataset}'", level='full')

        # For FlyWire/FAFB: enable the representation-specific mesh cache for
        # morphology.  Synapse caching remains enabled when requested: the
        # local master connection table is the dataset's canonical synapse
        # cache and is also the source for pre/post connector sites.
        if self.client_type == 'flywire' or 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            # Raw skeleton pkl caching is disabled (files too large and need transformation anyway)
            if self.cache_neurons:
                self._vprint("  ℹ️  FlyWire/FAFB: Using mesh cache (simplified) instead of raw skeletons", level='full')
        
        # Set the default mesh level based on the selected pipeline if it was
        # not specified. Fast/direct renders use 90% removal; fine/artistic
        # renders use 95%, for both NeuPrint and FlyWire/FAFB tube renders.
        if self.skeleton_mesh_simplification is None:
            if ('flywire' in self.dataset.lower()
                    or 'fafb' in self.dataset.lower()):
                pipeline = self._resolved_fafb_pipeline()
            else:
                pipeline = self._resolved_neuprint_skeleton_pipeline()
            self.skeleton_mesh_simplification = (
                0.90 if pipeline in {'fast', 'direct'} else 0.95
            )
            self._vprint(
                f"  ℹ️  Using default skeleton_mesh_simplification="
                f"{self.skeleton_mesh_simplification:.2f} for {self.dataset} "
                f"({self.neuprint_skeleton_pipeline} pipeline)",
                level='full',
            )

        # Auto-detect version from dataset if not provided
        if self.client_type == 'flywire' and self.version is None:
            import re
            # Look for v783 or version 783
            match = re.search(r'v(\d+)', self.dataset)
            if match:
                self.version = int(match.group(1))
                self._vprint(f"Auto-detected version={self.version} from dataset '{self.dataset}'", level='full')

        # Initialize client if needed
        if self.client_type == 'neuprint':
            import neuprint
            
            # Use provided client if available
            if self.client is not None:
                self._vprint(f'Using provided client for {self.dataset}', level='full')
            else:
                # Check if global client exists AND matches our dataset
                # This is critical: if a default client exists for a DIFFERENT dataset,
                # we MUST create a new client for the correct dataset, otherwise
                # fetch_skeletons will fail with "No neurons matching the given criteria found!"
                use_existing_client = False
                try:
                    existing_client = neuprint.default_client()
                    if existing_client is not None:
                        # Check if datasets match (normalize for comparison)
                        existing_ds = existing_client.dataset.lower().replace('_', ':').replace('.', '.')
                        target_ds = self.dataset.lower().replace('_', ':').replace('.', '.')
                        if existing_ds == target_ds:
                            # Keep the matching client on the visualizer.
                            # Merely leaving it as the process-wide default is
                            # racy: another NeuPrint operation can create a
                            # client before the aggregated render fetch runs.
                            self.client = existing_client
                            use_existing_client = True
                            self._vprint(f'Using existing default client for {self.dataset}', level='full')
                        else:
                            self._vprint(f'Default client is for {existing_client.dataset}, need new client for {self.dataset}', level='full')
                except RuntimeError:
                    pass

                if not use_existing_client:
                    # Use TokenManager to get token
                    try:
                        from utils.token_manager import token_manager
                        self.token = token_manager.get_token('NEUPRINT_TOKEN', self.token)
                        env_token = token_manager.get_neuprint_token()
                    except ImportError:
                        env_token = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')

                    def _token_rejected(exc) -> bool:
                        text = str(exc)
                        return '401' in text or 'Unauthorized' in text

                    if self.token:
                        try:
                            self.client = Client(self.server, dataset=self.dataset, token=self.token)
                            self.client.fetch_version()
                        except Exception as exc:
                            if not _token_rejected(exc):
                                raise
                            self.client = None
                            self._warn_neuprint_token_rejected()
                        else:
                            # Set as default to avoid "multiple clients" error
                            neuprint.set_default_client(self.client)
                            self._vprint(f'Client initialized for {self.dataset} (set as default)', level='full')
                    elif env_token:
                        # Auto-detect from env/config files
                        try:
                            self.client = Client(self.server, dataset=self.dataset)
                            self.client.fetch_version()
                        except Exception as exc:
                            if not _token_rejected(exc):
                                raise
                            self.client = None
                            self._warn_neuprint_token_rejected(from_env=True)
                        else:
                            # Set as default to avoid "multiple clients" error
                            neuprint.set_default_client(self.client)
                            self._vprint(f'Client initialized from env/config for {self.dataset} (set as default)', level='full')
                    else:
                        # Only warn if we are not using local cache/files exclusively
                        # But we don't know that yet.
                        pass
        
        # Initialize FlyWire client if needed
        if self.client_type == 'flywire' and self.client_flywire is None:
            # FlyWire API fetching removed
            pass

        # Check FlyWire visualization files
        if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            # Ensure data is prepared using the converter
            dataset_dir = os.path.join(self.script_path, 'datasets', self.dataset)
            
            # Use the converter module to ensure data is ready
            if 'BANC' in self.dataset:
                success = BANC_file_converter.ensure_banc_data(self.dataset, dataset_dir)
            else:
                success = FAFB_file_converter.ensure_flywire_data(self.dataset, dataset_dir)

            if not success:
                print("\\n\033[31mCRITICAL ERROR: FlyWire/BANC data preparation failed.\033[0m")
                print("Please follow the instructions above to download the required files.")
                sys.exit(1)
            

        if self.synapse_mode not in ['scatter', 'sphere', 'cone', 'tetrahedron', 'pre_post']:
            raise ValueError('synapse_mode can only be "scatter", "sphere", "cone", "tetrahedron", or "pre_post"')
        
        # Validate legend_mode
        if self.legend_mode not in LEGEND_MODES:
            raise ValueError(
                'legend_mode must be one of: ' + ', '.join(LEGEND_MODES)
            )

        if self.color_mode not in ['per_layer', 'per_neuron']:
            raise ValueError('color_mode must be "per_layer" or "per_neuron"')
        
        # Validate expand_colors
        if self.expand_colors not in ['interpolation', 'darken', 'cycle']:
            raise ValueError('expand_colors must be "interpolation", "darken", or "cycle"')
        
        if self.skeleton_mode not in ['line','tube']:
            raise ValueError('skeleton_mode can only be "line" or "tube"')
        if self._resolved_skeleton_radius_style() not in ['fafb', 'source', 'constant']:
            raise ValueError(
                'skeleton_radius_style must be "auto", "fafb", "source", or "constant"')
        if self.brain_mesh not in ['none', 'whole', 'template']:
            raise ValueError('brain_mesh must be "none", "template", or "whole"')
        if self.backend not in ['plotly', 'k3d']:
            raise ValueError('backend must be "plotly" or "k3d"')
        
        # Early transform check - advise user on transformation-free options
        self._check_transform_requirements_early()
        
        # Parse layer_map_csv if provided (overrides neuron_layers and custom_layer_names)
        if self.layer_map_csv is not None:
            self._parse_layer_map_csv()
        
        # convert neuron_layers str to list, if is str
        if type(self.neuron_layers) is str:
            if self.neuron_layers.strip():
                # Non-empty string: parse by '->' separator
                self.neuron_layers = self.neuron_layers.replace(' ','').split('->')
                for i,layer in enumerate(self.neuron_layers):
                    if layer.isnumeric():
                        self.neuron_layers[i] = (
                            normalize_flywire_body_id(layer)
                            if is_flywire_dataset(self.dataset)
                            else int(layer)
                        )
            else:
                # Empty string: convert to empty list for mesh-only mode
                self.neuron_layers = []
        
        if self.synapse_mode == 'scatter' and self.synapse_size == 0:
            self.synapse_size = 2
        if self.synapse_mode == 'scatter' and self.synapse_size == 'real':
            # Scatter markers use a fixed pixel size; 'real' only has a
            # distance to scale in sphere/cone/tetrahedron modes.
            self._vprint("\033[33m⚠️  Warning: synapse_size='real' is not supported in scatter mode; using pixel size 1.\033[0m", level='simple')
            self.synapse_size = 1
        # elif self.synapse_mode in ['sphere', 'cone', 'tetrahedron']:
        #     # Only check size limit if synapse_size is a number (not 'real')
        #     # For these modes, synapse_size is a multiplier, so small values (e.g. 1.0) are valid
        #     if isinstance(self.synapse_size, (int, float)) and self.synapse_size < 20 and self.brain_mesh != 'whole':
        #         self.synapse_size = 20
        #         self._vprint('\033[33mSynapse size is too small (< 20) for sphere, cone, or tetrahedron mode, automatically reset to 20\033[0m', level='full')
            
        if self.mesh_roi == None:
            self.mesh_roi = []
        
        # Convert single string to list for mesh_roi
        if isinstance(self.mesh_roi, str):
            self.mesh_roi = [self.mesh_roi]
        
        # Initialize nested ROI groups (will be populated if nested lists are used)
        self._nested_roi_groups = {}
        
        # Handle nested lists in mesh_roi for color grouping
        # e.g., ['AME', ['aL', 'bL', 'gL'], 'EB'] -> flattened list with same colors for grouped ROIs
        if self.mesh_roi:
            self.mesh_roi, self.mesh_color, self._nested_roi_groups = self._flatten_nested_roi_groups(
                self.mesh_roi, self.mesh_color
            )
        
        # Expand special keywords and regex patterns in mesh_roi
        # 'primary' -> all primary ROIs, 'all' -> all available ROIs
        # 'ME.*' -> all ROIs matching regex pattern
        if self.mesh_roi:
            self.mesh_roi = self._expand_mesh_roi_patterns(self.mesh_roi)
        
        # Expand ROI names to include bilateral (L/R) variants
        # e.g., 'LH' -> ['LH(L)', 'LH(R)']
        # Also expand colors to match the expanded ROIs
        if self.mesh_roi:
            original_rois = list(self.mesh_roi)
            original_colors = self.mesh_color if isinstance(self.mesh_color, list) else self.mesh_color
            
            # Use the new function that expands both ROIs and colors together
            expanded_rois, expanded_colors = self._expand_roi_names_with_colors(
                self.mesh_roi, 
                self.mesh_color
            )
            
            self.mesh_roi = expanded_rois
            # Only update mesh_color if it was a list (preserving single color behavior)
            if isinstance(original_colors, list):
                self.mesh_color = expanded_colors
            elif len(set(expanded_colors)) == 1:
                # If all expanded colors are the same, keep as single color
                self.mesh_color = expanded_colors[0]
            else:
                self.mesh_color = expanded_colors
            
            if self.mesh_roi != original_rois:
                self._vprint(f"   🔄 ROI expansion: {original_rois} → {self.mesh_roi}", level='simple')
                if isinstance(original_colors, list) and len(original_colors) != len(expanded_colors):
                    self._vprint(f"   🔄 Color expansion: {len(original_colors)} colors → {len(expanded_colors)} colors", level='full')
        
        # === Standardize color inputs ===
        # Standardize neuron_colors to a list of rgba strings
        # Check if neuron_colors have explicit alpha values that override neuron_alpha
        self._neuron_colors_have_explicit_alpha = self._colors_have_explicit_alpha(self.neuron_colors)
        if self._neuron_colors_have_explicit_alpha:
            self._vprint(
                f"\033[33m⚠️  Warning: neuron_colors contains explicit alpha values. "
                f"Those layer colors override neuron_alpha={self.neuron_alpha}; "
                f"colors without alpha still use the global opacity.\033[0m",
                level='simple'
            )
        # Always pass the global alpha as the parser default. Each input color
        # keeps its own explicit alpha, while colors without one inherit this
        # layer-set fallback. This also supports mixed-alpha palettes.
        self.neuron_colors = self._standardize_color_input(
            self.neuron_colors, 'neuron_colors', default_alpha=self.neuron_alpha
        )
        self._base_neuron_colors = tuple(self.neuron_colors)
        
        # Standardize synapse_colors
        # Check if synapse_colors have explicit alpha values that override synapse_alpha
        self._synapse_colors_have_explicit_alpha = self._colors_have_explicit_alpha(self.synapse_colors)
        if self._synapse_colors_have_explicit_alpha:
            self._vprint(
                f"\033[33m⚠️  Warning: synapse_colors contains explicit alpha values. "
                f"Those layer colors override synapse_alpha={self.synapse_alpha}; "
                f"colors without alpha still use the global opacity.\033[0m",
                level='simple'
            )
        self.synapse_colors = self._standardize_color_input(
            self.synapse_colors, 'synapse_colors', default_alpha=self.synapse_alpha
        )
        
        # Standardize mesh_color (for ROI meshes)
        # Store original for detecting custom colors
        self._original_mesh_color = self.mesh_color
        
        # Check if mesh_color has explicit alpha values that override mesh_alpha
        self._mesh_colors_have_explicit_alpha = self._colors_have_explicit_alpha(self.mesh_color)
        if self._mesh_colors_have_explicit_alpha:
            self._vprint(
                f"\033[33m⚠️  Warning: mesh_color contains explicit alpha values. "
                f"Those ROI colors override mesh_alpha={self.mesh_alpha}; "
                f"colors without alpha still use the global opacity.\033[0m",
                level='simple'
            )
        self.mesh_color = self._standardize_mesh_color_input(
            self.mesh_color,
            default_alpha=self.mesh_alpha,
            continuous=self._mesh_colors_continuous,
        )
        
        # Warn if custom mesh colors are provided that brain/VNC mesh colors are separate
        if self._is_custom_mesh_color_specified() and self.mesh_roi:
            self._vprint(
                f"\033[34mℹ️  Note: mesh_color applies to ROI meshes only. "
                f"Brain mesh and VNC mesh colors are controlled by brain_mesh_color and vnc_mesh_color.\033[0m",
                level='simple'
            )
        
        # Standardize brain_mesh_color and vnc_mesh_color if not 'auto'.
        # Their default mesh opacity is 0.1; an explicit alpha in any
        # supported color format overrides it.
        if str(self.brain_mesh_color).strip().lower() != 'auto':
            self.brain_mesh_color = standardize_color(
                self.brain_mesh_color, default_alpha=0.1
            )
        if str(self.vnc_mesh_color).strip().lower() != 'auto':
            self.vnc_mesh_color = standardize_color(
                self.vnc_mesh_color, default_alpha=0.1
            )
        
        # Ensure enough colors for all layers by expanding if needed
        n_layers = len(self.neuron_layers)
        self.neuron_colors = self._expand_color_sequence(
            self._base_neuron_colors,
            n_layers,
            target_label='layers',
            tip_parameter='neuron_colors',
            warn=self.color_mode == 'per_layer',
            continuous=self._neuron_colors_continuous,
        )
        
        # Same for synapse colors (one fewer than neuron layers for connections between layers)
        n_synapse_needed = max(0, n_layers - 1)
        self.synapse_colors = self._expand_color_sequence(
            tuple(self.synapse_colors),
            n_synapse_needed,
            target_label='synapse layers',
            warn=not self.skip_synapse,
            continuous=self._synapse_colors_continuous,
        )
        
        if self.skeleton_mode == 'line':
            self.show_skeleton_radius = False
            # neuron_alpha is now supported for line mode via opacity
        elif self.skeleton_mode == 'tube':
            self.show_skeleton_radius = True
        
        # fetch neurons and automatically generate layer names
        self.neuron_dfs = []
        self.roi_dfs = []
        self.layer_criteria = []
        self.layer_names = []
        
        n_layers = len(self.neuron_layers)
        self._vprint(f'\n📊 Fetching neuron info for {n_layers} layer(s)...')
        
        # Use tqdm for progress bar
        from tqdm import tqdm
        # Layer metadata loading is part of setup; keep it out of the
        # skeleton-render progress stream so rendering has one visible row.
        layer_iter = tqdm(
            range(n_layers),
            desc="Loading layers",
            disable=True,
        )
        
        total_neurons = 0
        for i in layer_iter:
            layer_input = self.neuron_layers[i]
            if not isinstance(layer_input, list):
                layer_input = [layer_input]
            
            # Update progress bar description
            layer_desc = str(layer_input[0])[:20] if layer_input else f"layer_{i}"
            layer_iter.set_description(f"Layer {i}: {layer_desc}")
            
            ndf, rdf, auto_name, cri = sv.getNeurons(
                layer_input,
                dataset=self.dataset,
                client=self.client,
                verbose=False,
                search_columns=self.search_columns,
            )
            if self.hemisphere != 'both' and ndf is not None and not ndf.empty:
                ndf, rdf = self._filter_neuron_df_by_hemisphere(ndf, rdf)
            self.neuron_dfs.append(ndf)
            self.roi_dfs.append(rdf)
            self.layer_criteria.append(cri)
            self.layer_names.append(auto_name)
            
            n_neurons = len(ndf) if ndf is not None else 0
            total_neurons += n_neurons
            
            # Update postfix with neuron count
            layer_iter.set_postfix(neurons=n_neurons, total=total_neurons)

        # A custom layer CSV may identify a neuron by type or instance while
        # the renderer later asks for its bodyId (or the reverse). Expand the
        # parsed color maps with all identifiers present in the resolved
        # neuron frames before any skeleton/synapse rendering starts.
        self._link_layer_map_color_overrides_to_neuron_dfs()
        
        # Print summary
        self._vprint(f'✓ Loaded {total_neurons:,} neurons across {n_layers} layers')
        
        # Show detailed breakdown if full verbose
        if self.verbose == 'full':
            self._vprint('\n  Layer summary:')
            for i, (ndf, name) in enumerate(zip(self.neuron_dfs, self.layer_names)):
                n = len(ndf) if ndf is not None else 0
                if n > 0 and 'type' in ndf.columns:
                    types = ndf['type'].dropna().unique()
                    n_types = len(types)
                    type_preview = ', '.join(str(t) for t in types[:3])
                    if n_types > 3:
                        type_preview += f' (+{n_types-3} more)'
                    self._vprint(f'    [{i}] {name}: {n} neurons, {n_types} types ({type_preview})')
                else:
                    self._vprint(f'    [{i}] {name}: {n} neurons')

        # Generate smart layer names based on types (if not using custom names)
        if (not self.custom_layer_names
                or getattr(self, '_layer_map_numeric_layers', False)):
            self.layer_names = self._generate_smart_layer_names()
        else:
            # Support partial custom_layer_names - merge with auto-generated names
            auto_names = self._generate_smart_layer_names()
            n_layers = len(self.neuron_layers)
            n_custom = len(self.custom_layer_names)
            
            merged_names = []
            for i in range(n_layers):
                if i < n_custom and self.custom_layer_names[i]:
                    # Use custom name if provided and non-empty
                    merged_names.append(self.custom_layer_names[i])
                elif i < len(auto_names):
                    # Use auto-generated name
                    merged_names.append(auto_names[i])
                else:
                    # Fallback
                    merged_names.append(f"layer_{i}")
            
            self.layer_names = merged_names

        self._per_neuron_colors = {}
        if self.color_mode == 'per_neuron':
            self._per_neuron_colors = self._build_per_neuron_color_map()
            
        if self.saveas is None:
            # Generate saveas from layer names
            # Use exact names for ≤3 layers, first 2 + etc{remaining} for >3 layers
            n_layers = len(self.layer_names) if self.layer_names else 0
            if n_layers == 0:
                # Mesh-only mode without neurons
                if self.mesh_roi:
                    n_rois = len(self.mesh_roi)
                    if n_rois == 0:
                        self.saveas = "brain_mesh"
                    elif n_rois <= 3:
                        # Use all ROI names for ≤3 ROIs
                        roi_names = []
                        for roi in self.mesh_roi[:3]:
                            # Clean up ROI name for filename (remove parentheses)
                            roi_clean = str(roi).replace('(', '').replace(')', '')
                            roi_names.append(roi_clean)
                        self.saveas = '_'.join(roi_names) + '_mesh'
                    else:
                        # Use first 2 ROI names + count for >3 ROIs
                        roi_names = []
                        for roi in self.mesh_roi[:2]:
                            roi_clean = str(roi).replace('(', '').replace(')', '')
                            roi_names.append(roi_clean)
                        self.saveas = '_'.join(roi_names) + f'_etc{n_rois - 2}_mesh'
                else:
                    self.saveas = "brain_mesh"
            elif n_layers <= 3:
                # Use all layer names for ≤3 layers
                self.saveas = '_'.join(self.layer_names)
            else:
                # Use first 2 names + count of remaining layers for >3 layers
                first_two = '_'.join(self.layer_names[:2])
                remaining = n_layers - 2
                self.saveas = f"{first_two}_etc{remaining}"
        
        # Ensure saveas doesn't exceed reasonable length (max 80 chars)
        if len(self.saveas) > 80:
            # Truncate and add hash for uniqueness
            import hashlib
            hash_suffix = hashlib.md5('_'.join(self.layer_names).encode()).hexdigest()[:6]
            self.saveas = self.saveas[:70] + f"_{hash_suffix}"
        
        # Get dataset abbreviation for folder naming
        dataset_abbrev = self._get_dataset_abbreviation()
        
        # Create output subfolder (with or without timestamp based on include_timestamp)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_stem = re.sub(
            r'[^A-Za-z0-9._-]+',
            '-',
            str(self.saveas).split('.')[0],
        ).strip('._-') or 'skeleton'
        base_folder_name = f'plot-3d_{dataset_abbrev}_{folder_stem}'
        if self.include_timestamp:
            self.save_folder = os.path.join(self.output_dir, base_folder_name + '_' + timestamp)
        else:
            self.save_folder = os.path.join(self.output_dir, base_folder_name)
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        
        # Save parameters to text file with comprehensive formatting
        param_file = os.path.join(self.save_folder, 'parameters.txt')
        with open(param_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("VisualizeSkeleton Parameters - Complete Configuration\n")
            f.write("=" * 70 + "\n\n")
            
            # Basic Info
            f.write("[Basic Info]\n")
            f.write(f"  Dataset:          {self.dataset}\n")
            f.write(f"  Timestamp:        {timestamp}\n")
            if self.version:
                f.write(f"  Version:          {self.version}\n")
            f.write(f"  Client Type:      {self.client_type}\n")
            f.write(f"  Server:           {self.server}\n")
            f.write(f"  Output Directory: {self.output_dir}\n")
            f.write(f"  Save Filename:    {self.saveas}\n")
            f.write(f"  Include Timestamp:{self.include_timestamp}\n")
            f.write(f"  Verbose:          {self.verbose}\n")
            f.write("\n")
            
            # Layer Info
            f.write("[Layers]\n")
            for i, (layer, name) in enumerate(zip(self.neuron_layers, self.layer_names)):
                n_neurons = len(self.neuron_dfs[i]) if i < len(self.neuron_dfs) and self.neuron_dfs[i] is not None else 0
                f.write(f"  Layer {i}: {name} ({n_neurons} neurons)\n")
                # Show first few neuron IDs if available
                if n_neurons > 0 and 'bodyId' in self.neuron_dfs[i].columns:
                    body_ids = self.neuron_dfs[i]['bodyId'].tolist()[:5]
                    ids_str = ', '.join(str(bid) for bid in body_ids)
                    if n_neurons > 5:
                        ids_str += f", ... (+{n_neurons - 5} more)"
                    f.write(f"           IDs: {ids_str}\n")
            f.write(f"  Total Layers:     {len(self.neuron_layers)}\n")
            f.write("\n")
            
            # Color Settings
            f.write("[Color Settings]\n")
            if self.neuron_colors:
                if isinstance(self.neuron_colors, list):
                    colors_preview = self.neuron_colors[:5]
                    f.write(f"  Neuron Colors:    {colors_preview}")
                    if len(self.neuron_colors) > 5:
                        f.write(f" ... (+{len(self.neuron_colors) - 5} more)")
                    f.write("\n")
                else:
                    f.write(f"  Neuron Colors:    {self.neuron_colors}\n")
            else:
                f.write(f"  Neuron Colors:    (auto-assigned)\n")
            f.write(f"  Neuron Alpha:     {self.neuron_alpha}\n")
            f.write(f"  Synapse Alpha:    {self.synapse_alpha}\n")
            f.write(f"  Brain Mesh Color: {self.brain_mesh_color}\n")
            f.write(f"  VNC Mesh Color:   {self.vnc_mesh_color}\n")
            if self.mesh_color:
                f.write(f"  Mesh ROI Color:   {self.mesh_color}\n")
            f.write("\n")
            
            # Visualization Settings
            f.write("[Visualization]\n")
            f.write(f"  Skeleton Mode:    {self.skeleton_mode}\n")
            f.write(f"  Skeleton Mesh Simplification: {self.skeleton_mesh_simplification}\n")
            f.write(f"  Backend:          {self.backend}\n")
            f.write(f"  Brain Mesh:       {self.brain_mesh}\n")
            f.write(f"  VNC Mesh:         {self.vnc_mesh}\n")
            if self.mesh_roi:
                f.write(f"  Mesh ROI:         {self.mesh_roi}\n")
            f.write(f"  Legend Mode:      {self.legend_mode}\n")
            f.write(f"  Expand Colors:    {self.expand_colors}\n")
            f.write(f"  Show Soma:        {self.show_soma}\n")
            f.write("\n")
            
            # Synapse Settings
            f.write("[Synapse Settings]\n")
            f.write(f"  Synapse Mode:     {self.synapse_mode}\n")
            f.write(f"  Synapse Size:     {self.synapse_size}\n")
            f.write(f"  Uniform Size:     {self.uniform_synapse_size}\n")
            f.write(f"  Min Synapse Num:  {self.min_synapse_num}\n")
            f.write(f"  Synapse Alpha:    {self.synapse_alpha}\n")
            f.write("\n")
            
            f.write("=" * 70 + "\n")

        self._write_user_warning_notes()
        if self.backend == 'plotly':
            self.fig_3d = go.Figure()
        elif self.backend == 'k3d':
            try:
                import k3d
                self.fig_3d = k3d.plot()
            except ImportError:
                self._vprint("⚠️  k3d not installed. Please install it with `pip install k3d`")
                self._vprint("   Falling back to plotly backend")
                self.backend = 'plotly'
                self.fig_3d = go.Figure()

        # Plotly sliders are assembled after the synapse/site traces exist and
        # are restored by save_figure() for the interactive HTML export.
        self._plotly_sliders = []
        
        # Save one compact neuron-info table.  ROI counts are fetched for
        # internal query/filtering needs, but are not part of this export.
        self._inject_custom_neurons()
        self._save_neuron_info_csv()
        # Always leave a small, directly reusable layer-map beside the
        # visualization. It preserves the readable source layer/neuron/color
        # entries when they are available, while remaining uploadable for a
        # later rerun.
        self._save_viz_layer_info_csv()

    def _save_neuron_info_csv(self):
        """Save all resolved neuron metadata in one CSV file.

        Each source frame represents one visualization layer.  The numeric
        ``viz_layer`` column records the layer index on every row so the
        merged table can be used without relying on per-layer Excel sheets
        or positional indices.  It is placed directly before ``bodyId``.

        Returns
        -------
        str or None
            The CSV path, or ``None`` when no neuron layers were resolved.
        """
        if not self.neuron_dfs:
            return None

        frames = []
        for i, neuron_df in enumerate(self.neuron_dfs):
            if neuron_df is None:
                continue

            frame = neuron_df.copy()
            # Drop serialized-index columns (unnamed / 'column_1' style)
            # carried over from upstream data sources.
            frame = sv.drop_leading_index_columns(frame)
            # Ensure the export owns this column even if an upstream data
            # source happens to provide a column with the same name.
            frame = frame.drop(columns=['viz_layer'], errors='ignore')
            frame.insert(
                frame.columns.get_loc('bodyId') if 'bodyId' in frame.columns else 0,
                'viz_layer',
                i,
            )
            frames.append(frame)

        if frames:
            neuron_info = pd.concat(frames, ignore_index=True, sort=False)
        else:
            neuron_info = pd.DataFrame(columns=['viz_layer', 'bodyId'])

        file_path = os.path.join(
            self.save_folder, self.saveas + '_neuron_info.csv'
        )
        neuron_info.to_csv(file_path, index=False)
        return file_path

    def _save_viz_layer_info_csv(self):
        """Write a reusable ``viz_layer_info.csv`` for this visualization.

        The file intentionally follows the ``layer_map_csv`` contract rather
        than the wider neuron metadata export.  When the visualization came
        from a layer-map CSV, retain those source layer/neuron/color entries
        instead of replacing readable labels with resolved body IDs.  Direct
        layer inputs use the same original entries as a fallback.

        Synapse colors and pre/post-site colors are mutually exclusive in this
        artifact: connector mode exports ``synapse_color`` only, pre/post mode
        exports the two site columns only, and skipped synapses export none of
        them.  This prevents a later CSV upload from carrying contradictory
        styling instructions.

        A header is written even for a mesh-only/empty run so every Skeleton
        export has the same reusable artifact.
        """
        columns = [
            "layer",
            "neuron",
            "color",
            "synapse_color",
            "pre_synaptic_color",
            "post_synaptic_color",
        ]

        def _csv_text(value):
            if value is None:
                return ""
            try:
                if pd.isna(value):
                    return ""
            except (TypeError, ValueError):
                pass
            return str(value).strip()

        mode = str(getattr(self, "synapse_mode", "") or "").strip().lower()
        skip_synapses = bool(getattr(self, "skip_synapse", False))

        def _apply_mode_exclusivity(row):
            """Blank the color family that is not active for this render."""
            if skip_synapses:
                row["synapse_color"] = ""
                row["pre_synaptic_color"] = ""
                row["post_synaptic_color"] = ""
            elif mode == "pre_post":
                row["synapse_color"] = ""
            else:
                row["pre_synaptic_color"] = ""
                row["post_synaptic_color"] = ""
            return row

        def _raw_palette_value(values, index):
            """Return one readable value from a direct palette input."""
            if values is None:
                return None
            if isinstance(values, str):
                return values
            if isinstance(values, (list, tuple)):
                if not values:
                    return None
                # A numeric RGB(A) tuple is one color, not a palette of three
                # scalar entries.
                if len(values) in (3, 4) and all(
                    isinstance(item, (int, float)) for item in values
                ):
                    return values
                return values[index] if index < len(values) else values[-1]
            return values

        def _export_color_value(value, fallback, default_alpha):
            if value is None:
                return fallback
            if isinstance(value, str):
                return value
            try:
                return standardize_color(value, default_alpha=default_alpha)
            except (TypeError, ValueError):
                return fallback

        rows = []
        source_rows = getattr(self, "_viz_layer_info_input_rows", None)
        if source_rows is not None:
            # A CSV (including an empty CSV) is authoritative.  Do not fall
            # back to resolved neuron_dfs: doing so would lose the user's
            # labels and could add rows that were not in the editor input.
            for source in source_rows:
                row = {
                    column: _csv_text(source.get(column, ""))
                    for column in columns
                }
                rows.append(_apply_mode_exclusivity(row))
        else:
            # Direct constructor input has no source table to round-trip.
            # Preserve its layer entries and use the effective resolved colors
            # for the corresponding input identifier.
            raw_layers = getattr(self, "_viz_input_neuron_layers", None)
            if isinstance(raw_layers, str):
                raw_layers = (
                    raw_layers.replace(' ', '').split('->')
                    if raw_layers.strip() else []
                )
            elif raw_layers is None:
                raw_layers = []
            else:
                raw_layers = list(raw_layers)

            raw_names = getattr(self, "_viz_input_custom_layer_names", None) or []
            raw_neuron_colors = getattr(self, "_viz_input_neuron_colors", None)
            raw_synapse_colors = getattr(self, "_viz_input_synapse_colors", None)
            entry_index = 0
            for layer_index, layer_input in enumerate(raw_layers):
                if isinstance(layer_input, (list, tuple, set)):
                    identifiers = list(layer_input)
                else:
                    identifiers = [layer_input]
                layer_label = (
                    raw_names[layer_index]
                    if layer_index < len(raw_names)
                    and _csv_text(raw_names[layer_index])
                    else layer_index + 1
                )
                for identifier in identifiers:
                    identifier_text = _csv_text(identifier)
                    if not identifier_text:
                        continue
                    neuron_color_input = _raw_palette_value(
                        raw_neuron_colors,
                        entry_index if getattr(self, "color_mode", "per_layer") == "per_neuron"
                        else layer_index,
                    )
                    synapse_color_input = _raw_palette_value(
                        raw_synapse_colors, layer_index,
                    )
                    row = {
                        "layer": _csv_text(layer_label),
                        "neuron": identifier_text,
                        "color": _export_color_value(
                            neuron_color_input,
                            self._resolve_neuron_color(identifier, layer_index),
                            getattr(self, "neuron_alpha", 1.0),
                        ),
                        "synapse_color": _export_color_value(
                            synapse_color_input,
                            self._resolve_synapse_color(identifier, layer_index),
                            getattr(self, "synapse_alpha", 1.0),
                        ),
                        "pre_synaptic_color": self._resolve_pre_color(
                            identifier, layer_index,
                        ),
                        "post_synaptic_color": self._resolve_post_color(
                            identifier, layer_index,
                        ),
                    }
                    rows.append(_apply_mode_exclusivity(row))
                    entry_index += 1

        file_path = os.path.join(self.save_folder, "viz_layer_info.csv")
        pd.DataFrame(rows, columns=columns).to_csv(file_path, index=False)
        return file_path

    def _save_synapse_data(self, synapse_frames):
        """Save all non-empty inter-layer synapse frames as one merged file.

        Each frame must already contain a ``viz_layer`` column identifying its
        source and target layer, for example ``0->1``.
        """
        if not synapse_frames:
            return None

        merged_synapses = pd.concat(synapse_frames, ignore_index=True, sort=False)
        # Keep the layer identifier as the first column so the merged table
        # is immediately grouped by its source layer.
        if 'viz_layer' in merged_synapses.columns:
            merged_synapses.insert(
                0, 'viz_layer', merged_synapses.pop('viz_layer')
            )
        output_format = str(getattr(self, 'output_format', 'csv')).lower()
        if output_format not in ('xlsx', 'csv'):
            raise ValueError(f"output_format must be 'xlsx' or 'csv', got '{output_format}'")

        extension = 'csv' if output_format == 'csv' else 'xlsx'
        file_path = os.path.join(self.save_folder, self.saveas + f'_synapses.{extension}')

        if output_format == 'csv':
            merged_synapses.to_csv(file_path, index=False)
        else:
            with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
                merged_synapses.to_excel(writer, sheet_name='synapses', index=False)

        self._vprint(f'  ✓ Saved merged synapse data to: {file_path}', level='full')
        return file_path

    def _write_user_warning_notes(self):
        """Record the effective mesh simplification for this render.

        Skeleton simplification changes the rendered tube mesh only; it does
        not trim analysis graphs or alter the queried neuron set. Keep this
        note beside every standalone and analysis-generated visualization so
        the output remains self-describing.
        """

        value = self.skeleton_mesh_simplification
        if value is None:
            value_text = "not set"
        else:
            value_text = f"{float(value):.2f}"
        skeleton_mode = getattr(self, "skeleton_mode", "tube")
        requested_pipeline = str(
            getattr(self, "neuprint_skeleton_pipeline", "fast") or "fast"
        ).strip().lower()
        pipeline = {
            "direct": "fast",
            "fine_opt": "fine",
            "fine_opt1": "artistic",
            "fafb": "fine",
        }.get(requested_pipeline, requested_pipeline)
        is_fafb_dataset = is_flywire_dataset(self.dataset)
        if skeleton_mode == "tube":
            scope = "applied to the rendered tube mesh"
        else:
            scope = "configured but not applied because skeleton_mode='line'"
        if skeleton_mode == "line":
            warning_note = (
                "in-page export-pipeline warning raised for line mode "
                "(simplification warning disabled)"
            )
        elif is_fafb_dataset:
            warning_note = "in-page warning threshold >0.95"
        elif pipeline in {"fine", "artistic"}:
            warning_note = "in-page warning threshold >0.95"
        else:
            warning_note = "in-page warning threshold >0.90"
        if is_fafb_dataset:
            pipeline_note = f"neuprint_skeleton_pipeline={pipeline}"
        else:
            pipeline_note = f"neuprint_skeleton_pipeline={pipeline}"
        synapse_mode = str(getattr(self, "synapse_mode", "") or "").strip().lower()
        pre_post_note = ""
        if synapse_mode == "pre_post":
            pre_post_note = (
                f"- [pre/post-synaptic site mode] dataset={self.dataset}; "
                "shows the pre- and post-synaptic SITES of the queried neurons "
                "(post/input sites as spheres, pre/output sites as cones), not the "
                "paired inter-layer synapses; there is a separate legend for pre and "
                "post sites per layer.\n"
            )
        # Per-layer truncation details for capped type layers.  The in-page
        # banner carries only a one-row summary pointing here, so this file
        # must list every affected layer.
        sample_notes = list(getattr(self, "layer_sample_notes", None) or [])
        truncation_note = ""
        if sample_notes:
            count = len(sample_notes)
            layer_word = "layer" if count == 1 else "layers"
            verb = "shows" if count == 1 else "show"
            details = "".join(f"  - {note}\n" for note in sample_notes)
            truncation_note = (
                f"- [truncated type layers] {count} type {layer_word} {verb} a "
                "capped sample of their type's members (fewer neurons than the "
                f"type has); per-layer details:\n{details}"
            )
        note_path = os.path.join(self.save_folder, "user_warning_notes.txt")
        text = (
            "DROCAT user warning notes\n"
            + "=" * 60
            + "\n"
            + "The following visualization settings affect rendering only; they "
              "do not trim analysis graphs or change query results.\n\n"
            + f"- [skeleton mesh simplification] dataset={self.dataset}; "
              f"{pipeline_note}; skeleton_mode={skeleton_mode}; "
              f"skeleton_mesh_simplification={value_text} ({scope}; "
              f"{warning_note}).\n"
            + pre_post_note
            + truncation_note
        )
        try:
            with open(note_path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as exc:
            self._vprint(
                f"  Warning: could not write user_warning_notes.txt: {exc}",
                level="full",
            )

    def _record_large_html_warning(self, html_path):
        """Append a render warning when a visualization HTML exceeds 50 MB.

        The warning is recorded after the final HTML has been written, so its
        size includes the embedded Plotly runtime and any in-page banners.
        Temporary HTML used only by the WebDriver export session is excluded;
        the main visualization HTML remains the user-facing artifact.
        Keep this separate from ``_write_user_warning_notes`` because that
        method runs during initialization, before the HTML exists.
        """

        if os.path.basename(html_path) == "_temp_export.html":
            return False

        threshold_mb = 50
        try:
            size_mb = os.path.getsize(html_path) / (1024 * 1024)
        except OSError:
            return False

        if size_mb <= threshold_mb:
            return False

        save_folder = getattr(self, "save_folder", "") or os.path.dirname(
            os.path.abspath(html_path)
        )
        note_path = os.path.join(save_folder, "user_warning_notes.txt")
        html_name = os.path.basename(html_path)
        marker = f"- [render warning] visualization HTML is too large: {html_name}"
        warning = (
            f"{marker} ({size_mb:.1f} MB > {threshold_mb} MB). "
            "Browser rendering and exporting may fail. Try increasing the "
            "simplification value in Advanced settings (for example, from "
            "0.90 to 0.98 or 0.99), or change the skeleton mode to 'line'.\n"
        )

        try:
            os.makedirs(save_folder, exist_ok=True)
            if os.path.exists(note_path):
                with open(note_path, "r", encoding="utf-8") as handle:
                    existing = handle.read()
            else:
                existing = ""

            # Avoid duplicate notes if the same HTML is rewritten by a retry
            # or reused by another export path.
            if marker in existing:
                return True

            with open(note_path, "a", encoding="utf-8") as handle:
                if not existing:
                    handle.write(
                        "DROCAT user warning notes\n"
                        + "=" * 60
                        + "\n"
                    )
                elif not existing.endswith("\n"):
                    handle.write("\n")
                handle.write("\n" + warning)
        except OSError as exc:
            self._vprint(
                f"  Warning: could not write large HTML render warning: {exc}",
                level="full",
            )
            return False

        return True

    def _get_cache_path(self, cache_type):
        """Get the cache directory for skeletons or synapses
        
        Uses project cache/ folder for organized storage:
        cache/{dataset}/skeletons/ - for raw skeleton cache namespaces
        cache/{dataset}/synapses/ - for synapse cache files
        
        For datasets folder resources:
        datasets/{dataset}/*_synapse_table.parquet - synapse table
        
        Example:
        - cache/hemibrain_v1_2_1/skeletons/raw_skeletons/{bodyId}.swc.zst
        - datasets/flywire_FAFB_v783/flywire_FAFB_v783_synapse_table.parquet
        """
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        cache_dir = os.path.join(self.script_path, 'cache', dataset_normalized, cache_type)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _get_dataset_abbreviation(self) -> str:
        """Get a short abbreviation for the dataset name for use in folder naming.
        
        Maps common dataset names to short abbreviations:
        - hemibrain -> HEMI
        - male-cns -> MCNS
        - flywire_FAFB -> FAFB
        - flywire_BANC -> BANC
        - optic-lobe -> OL
        - manc -> MANC
        
        Returns
        -------
        str
            Short abbreviation for the dataset, or first 4 chars if not recognized
        """
        # Dataset abbreviation mapping
        abbrev_map = {
            'hemibrain': 'HEMI',
            'male-cns': 'MCNS',
            'male_cns': 'MCNS',
            'flywire_fafb': 'FAFB',
            'fafb': 'FAFB',
            'flywire_banc': 'BANC',
            'banc': 'BANC',
            'optic-lobe': 'OL',
            'optic_lobe': 'OL',
            'manc': 'MANC',
        }
        
        if self.dataset is None:
            return 'UNKN'
        
        # Try exact match first (lowercase)
        dataset_lower = self.dataset.lower()
        for key, abbrev in abbrev_map.items():
            if key in dataset_lower:
                return abbrev
        
        # If no match, use first 4 chars uppercase
        return self.dataset[:4].upper().replace('-', '').replace('_', '')
    
    def _generate_smart_layer_names(self) -> List[str]:
        """Generate smart layer names based on neuron types.
        
        For each layer, generates a name in format:
        - {type} if all neurons in layer are the same type (even if multiple neurons)
        - {type}_etc if multiple neurons with different types (uses most common type)
        - {bodyId} if single untyped neuron
        - {bodyId}_etc if multiple untyped neurons with different IDs
        
        This method looks at the neuron_dfs to determine types and counts.
        
        Returns:
            List[str]: Smart layer names for each layer
        """
        smart_names = []
        
        for i, ndf in enumerate(self.neuron_dfs):
            if ndf is None or len(ndf) == 0:
                # Fallback to original auto-generated name
                smart_names.append(self.layer_names[i] if i < len(self.layer_names) else f"layer_{i}")
                continue
            
            n_neurons = len(ndf)
            
            # Get type column (different datasets may use different column names)
            type_col = None
            for col in ['type', 'cell_type', 'neuronType']:
                if col in ndf.columns:
                    type_col = col
                    break
            
            # Get types from the dataframe
            if type_col and type_col in ndf.columns:
                types = ndf[type_col].dropna().unique().tolist()
                # Filter out empty strings and None
                types = [t for t in types if t and str(t).strip()]
            else:
                types = []
            
            if types:
                # Count unique types
                n_unique_types = len(types)
                
                # Use the most common type as the representative
                if type_col in ndf.columns:
                    type_counts = ndf[type_col].value_counts()
                    primary_type = type_counts.index[0] if len(type_counts) > 0 else types[0]
                else:
                    primary_type = types[0]
                
                # Only add _etc if there are multiple different types
                if n_unique_types > 1:
                    smart_names.append(f"{primary_type}_etc")
                else:
                    # All neurons are the same type
                    smart_names.append(str(primary_type))
            else:
                # No type info - use bodyId
                body_ids = ndf['bodyId'].tolist() if 'bodyId' in ndf.columns else []
                if body_ids:
                    first_id = body_ids[0]
                    # Multiple untyped neurons with different IDs -> _etc
                    if n_neurons > 1:
                        smart_names.append(f"{first_id}_etc")
                    else:
                        smart_names.append(str(first_id))
                else:
                    # Ultimate fallback
                    smart_names.append(self.layer_names[i] if i < len(self.layer_names) else f"layer_{i}")
        
        return smart_names

    def _get_synapse_table_path(self):
        """Get path to synapse table in datasets folder.
        
        Returns the path to the synapse table parquet file.
        For FlyWire/FAFB: datasets/flywire_FAFB_v783/flywire_FAFB_v783_synapse_table.parquet
        For NeuPrint: datasets/{dataset}/{dataset}_synapse_table.parquet
        
        Returns:
            str: Path to synapse table, or None if not found
        """
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        if is_flywire_dataset(self.dataset):
            # FlyWire-family datasets are independent sources. In particular,
            # a BANC query must never read FAFB synapses merely because the
            # BANC table is absent.
            dataset_dir = resolve_flywire_dataset_dir(
                self.script_path, self.dataset
            )
            if dataset_dir is None:
                return None
            names = [dataset_dir.name, dataset_normalized]
            candidates = []
            for name in names:
                path = dataset_dir / f"{name}_synapse_table.parquet"
                if path not in candidates:
                    candidates.append(path)
            return next((str(path) for path in candidates if path.exists()), None)

        datasets_dir = os.path.join(self.script_path, 'datasets', dataset_normalized)
        parquet_file = os.path.join(
            datasets_dir, f"{dataset_normalized}_synapse_table.parquet"
        )
        return parquet_file if os.path.exists(parquet_file) else None
    
    def _preload_fafb_skeletons(self, body_ids_filter=None):
        """Pre-load FAFB skeletons from ZIP file in a single batch.
        
        This is much faster than opening the ZIP file for each layer.
        
        Parameters
        ----------
        body_ids_filter : set, optional
            If provided, only load these bodyIds from the ZIP.
            If None, load all bodyIds from self.neuron_dfs.
        
        Returns:
            dict: bodyId -> TreeNeuron mapping
        """
        from tqdm import tqdm
        import sys
        
        # Collect all body IDs needed. Internal FlyWire cache/file keys stay
        # canonical strings; navis object IDs are converted to integers only
        # when assigning the third-party object attribute below.
        if is_flywire_dataset(self.dataset):
            if body_ids_filter is not None:
                all_body_ids = set(normalize_flywire_body_ids(body_ids_filter))
            else:
                all_body_ids = set()
                for df in self.neuron_dfs:
                    if df is not None:
                        all_body_ids.update(
                            normalize_flywire_body_ids(df['bodyId'].tolist())
                        )
        elif body_ids_filter is not None:
            all_body_ids = {int(body_id) for body_id in body_ids_filter}
        else:
            all_body_ids = set()
            for df in self.neuron_dfs:
                if df is not None:
                    all_body_ids.update(
                        int(body_id) for body_id in df['bodyId'].tolist()
                    )
        
        if not all_body_ids:
            return {}
        
        skeleton_cache = {}
        
        try:
            import fafb_utils
            project_root = os.path.dirname(os.path.dirname(__file__))
            
            # Try to find dataset directory by name
            data_dir = resolve_flywire_dataset_dir(project_root, self.dataset)
            if data_dir is None:
                return {}

            # The healed bundle reads .zst first and falls back to the ZIP
            # with lazy per-skeleton conversion on every ZIP-served id.
            bundle = fafb_utils.get_fafb_skeleton_bundle(data_dir)
            if bundle is not None:
                import io

                self._vprint(
                    f'  📦 Loading {len(all_body_ids)} skeletons from '
                    'the healed bundle...')
                for bid in all_body_ids:
                    try:
                        content = bundle.get(int(bid))
                        if content is None:
                            continue
                        n = navis.read_swc(io.StringIO(content))
                        n.units = 'nm'
                        n.id = (
                            body_id_to_api_int(bid)
                            if is_flywire_dataset(self.dataset)
                            else int(bid)
                        )
                        n.name = str(bid)
                        skeleton_cache[
                            normalize_flywire_body_id(bid)
                            if is_flywire_dataset(self.dataset)
                            else int(bid)
                        ] = n
                    except Exception:
                        pass  # Skip errors silently
                try:
                    bundle.close()
                except Exception:
                    pass
                self._vprint(
                    f'  ✓ Loaded {len(skeleton_cache)}/{len(all_body_ids)} '
                    'skeletons from the healed bundle')
                return skeleton_cache

            zip_path = fafb_utils.get_fafb_skeleton_zip(data_dir)
            
            if zip_path:
                import zipfile
                import io
                
                self._vprint(f'  📦 Loading {len(all_body_ids)} skeletons from ZIP...')
                
                with zipfile.ZipFile(zip_path, 'r') as z:
                    zip_files = set(z.namelist())
                    
                    # Progress bar for skeleton loading
                    # The render-wide skeleton bar is the only progress row.
                    pbar = all_body_ids
                    
                    for bid in pbar:
                        filename = f"{bid}.swc"
                        try:
                            if filename in zip_files:
                                with z.open(filename) as f:
                                    content = f.read().decode('utf-8')
                                    n = navis.read_swc(io.StringIO(content))
                                    n.units = 'nm'
                                    n.id = (
                                        body_id_to_api_int(bid)
                                        if is_flywire_dataset(self.dataset)
                                        else int(bid)
                                    )
                                    n.name = str(bid)
                                    skeleton_cache[
                                        normalize_flywire_body_id(bid)
                                        if is_flywire_dataset(self.dataset)
                                        else int(bid)
                                    ] = n
                        except Exception:
                            pass  # Skip errors silently
                
                self._vprint(f'  ✓ Loaded {len(skeleton_cache)}/{len(all_body_ids)} skeletons from ZIP')
        except ImportError:
            pass
        except Exception as e:
            self._vprint(f'  ⚠️  Error pre-loading FAFB skeletons: {e}')
        
        return skeleton_cache

    def _load_extrusion_check_cache(self):
        """Load cached extrusion check results from parquet.
        
        Returns
        -------
        dict
            Dictionary of bodyId -> bool (True if has extrusion)
        """
        import fafb_utils

        project_root = getattr(
            self, 'script_path', os.path.dirname(os.path.dirname(__file__))
        )
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        return fafb_utils.load_extrusion_check_cache(
            project_root, dataset_safe)
    
    def _save_extrusion_check_cache(self, results_dict):
        """Save extrusion check results to parquet cache.
        
        Parameters
        ----------
        results_dict : dict
            Dictionary of bodyId -> bool (True if has extrusion)
        """
        import fafb_utils

        project_root = getattr(
            self, 'script_path', os.path.dirname(os.path.dirname(__file__))
        )
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
        fafb_utils.save_extrusion_check_cache(
            project_root, dataset_safe, results_dict)

    def _detect_extrusions_in_skeletons(self, skeletons_dict, simplification=0.95,
                                        use_cache=True):
        """Detect extrusion artifacts in a batch of skeletons.

        Delegates to the shared ``fafb_utils.flag_extrusions`` pipeline
        (cached per-neuron results, vectorized edge analysis, parallel
        batch with a serial fallback) so every FAFB consumer runs the same
        check. Returns the body IDs that have moderate or severe extrusions
        and should be replaced via API.

        ``use_cache=False`` runs the check fully in memory and leaves
        ``extrusion_check_results.parquet`` untouched.
        """
        from fafb_utils import flag_extrusions

        project_root = os.path.dirname(os.path.dirname(__file__))
        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')

        def _vprint_simple(msg):
            self._vprint(msg, level='simple')

        self._vprint(
            f'  🔍 Checking {len(skeletons_dict)} skeletons for extrusions...',
            level='simple',
        )
        return flag_extrusions(
            project_root, dataset_safe, skeletons_dict,
            # plot_skeleton owns the single render-wide progress row.  The
            # extrusion checker must not open its own tqdm row; the summary
            # callback still reports flagged neurons.
            verbose=False,
            log=_vprint_simple,
            simplification=simplification,
            use_cache=use_cache,
        )
    
    def _skeleton_cache_is_simplified(self) -> bool:
        """Compatibility shim for the removed simplified skeleton cache.

        Skeleton files are now always raw compressed SWC. Simplification is
        selected per visualization run and is never represented by a folder
        marker, so this method intentionally always returns ``False``.
        """
        return False

    def _effective_render_simplification(
            self, is_fafb: bool, using_simplified_cache: bool | None = None
    ) -> float:
        """Effective render-time mesh decimation fraction for tube mode.

        The source skeleton is always raw, so the requested render target is
        always applied directly to the in-memory tube mesh. There is no
        cache-relative adjustment and no second simplification level encoded
        on disk.

        Returns the fraction of faces to remove (0.0 = keep all).

        ``using_simplified_cache`` remains an accepted compatibility argument;
        it has no effect because no simplified skeleton cache is used.
        """
        return self.skeleton_mesh_simplification

    @staticmethod
    def _cached_skeleton_level(neuron) -> int:
        """Read the simplification level carried by a cached TreeNeuron."""
        try:
            return max(0, int(getattr(neuron, '_drocat_simplification', 0) or 0))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _relevel_cached_skeleton(neuron, target_level: int):
        """Return a cached skeleton at *target_level* without mutating it."""
        stored = VisualizeSkeleton._cached_skeleton_level(neuron)
        target = max(0, int(target_level))
        if target <= stored or not hasattr(neuron, 'nodes'):
            return neuron
        try:
            from morphology import _relevel_for_target
            result = _relevel_for_target(neuron, stored, target)
            try:
                result._drocat_simplification = target
            except Exception:
                pass
            return result
        except Exception:
            return neuron

    def _resolve_neuprint_render_cache(self, raw_cache, body_ids):
        """Resolve a common source level for NeuPrint render skeletons.

        Older visualization runs persisted the shared ``raw_skeletons`` files
        at simplification level 90, while other files are truly raw. Mixing
        those files in one scene makes a neuron look as if it used a different
        simplification setting. Raw level-0 files are preferred. Coarser legacy
        files are marked stale and are refreshed online when possible; if the
        refresh is unavailable, the fallback re-levels every available cached
        source to the same coarsest stored level instead of mixing resolutions.

        Returns ``(preferred, stale, missing)`` mappings/lists. The caller can
        add ``stale`` to its online fetch set and use ``fallback`` after a
        failed fetch through :meth:`_normalize_neuprint_cache_fallback`.
        """
        preferred = {}
        stale = {}
        missing = []
        for body_id in body_ids:
            try:
                neuron = raw_cache.load_skeleton(body_id)
            except Exception:
                neuron = None
            if neuron is None:
                missing.append(body_id)
                continue
            level = self._cached_skeleton_level(neuron)
            if level == self.NEUPRINT_RENDER_CACHE_SIMPLIFICATION:
                preferred[body_id] = neuron
            else:
                stale[body_id] = (neuron, level)
        return preferred, stale, missing

    def _normalize_neuprint_cache_fallback(self, preferred, stale, fetched):
        """Merge an offline legacy-cache fallback at one common level."""
        if not stale:
            return dict(fetched)
        fallback_level = max(level for _neuron, level in stale.values())
        available = dict(preferred)
        available.update(fetched)
        for body_id, (neuron, _level) in stale.items():
            available.setdefault(body_id, neuron)
        normalized = {
            body_id: self._relevel_cached_skeleton(neuron, fallback_level)
            for body_id, neuron in available.items()
        }
        self._vprint(
            f'  ⚠️ Using legacy NeuPrint skeleton cache fallback at '
            f'simplification level {fallback_level}; all available neurons '
            'were normalized to that level',
            level='simple',
        )
        return normalized

    def _persist_neuprint_render_skeletons(self, raw_cache, raw_items):
        """Persist visualization sources at raw level 0.

        A small compatibility fallback keeps older cache adapters usable when
        they do not expose the newer ``simplification`` keyword.
        """
        try:
            return raw_cache.persist_skeletons(
                raw_items,
                simplification=self.NEUPRINT_RENDER_CACHE_SIMPLIFICATION,
            )
        except TypeError as exc:
            if 'simplification' not in str(exc):
                raise
            return raw_cache.persist_skeletons(raw_items)

    def _load_cached_neurons(self, neuron_df, transformed_target=None,
                             ignore_cache=False):
        """Load NeuPrint skeletons from the shared SWC cache.

        The shared cache is the single source for every visualization
        simplification mode; visualization writes use raw level 0 (recorded
        in the header) so render-time mesh simplification is applied exactly
        once. Legacy simp90 files are treated as stale and are only an
        normalized offline fallback. Legacy ``skeletons/{bodyId}.pkl`` files
        are not read because their representation level is ambiguous.
        """
        body_ids = [int(bid) for bid in neuron_df['bodyId'].tolist()]
        if ignore_cache:
            return None, body_ids
        if (self.client_type == 'flywire'
                or 'flywire' in self.dataset.lower()
                or 'fafb' in self.dataset.lower()
                or 'banc' in self.dataset.lower()):
            return None, body_ids

        try:
            from morphology import find_similar_raw_cache
            raw_cache = find_similar_raw_cache(
                self.dataset, project_root=self.script_path, verbose=False,
            )
        except Exception as exc:
            self._vprint(
                f'  ⚠ Raw skeleton cache unavailable: {exc}', level='full')
            return None, body_ids

        preferred, stale, missing_ids = self._resolve_neuprint_render_cache(
            raw_cache, body_ids,
        )
        # Legacy non-aggregate callers may still fetch their own misses. Do
        # not silently render a mixture of raw and simp90 files here.
        missing_ids.extend(stale)
        neurons = list(preferred.values())

        if neurons:
            self._vprint(
                f'  ✓ Loaded {len(neurons)} raw skeletons from SWC cache',
                level='full',
            )
            if missing_ids:
                self._vprint(
                    f'  ℹ  {len(missing_ids)} raw skeletons not in cache, will fetch',
                    level='full',
                )
            return navis.NeuronList(neurons), missing_ids
        return None, body_ids
    
    def _save_cached_neurons(self, neuron_df, neuron_vols):
        """Persist NeuPrint skeletons in the shared SWC cache.

        The method name is retained for compatibility with older callers,
        but it no longer writes visualization pickles or a simplification
        marker. Persistence uses raw level-0 ``.swc.zst`` files; render-time
        ``fast``/``fine`` decimation always starts from the same source level.
        """
        is_neuprint = not (self.client_type == 'flywire'
                           or 'flywire' in self.dataset.lower()
                           or 'fafb' in self.dataset.lower()
                           or 'banc' in self.dataset.lower())
        if not is_neuprint or neuron_vols is None:
            return

        raw_items = {}
        for neuron in neuron_vols:
            bid = getattr(neuron, 'id', None) or getattr(neuron, 'bodyId', None)
            if bid is None:
                continue
            try:
                raw_items[int(bid)] = neuron
            except (TypeError, ValueError):
                continue
        if not raw_items:
            return

        try:
            from morphology import find_similar_raw_cache
            raw_cache = find_similar_raw_cache(
                self.dataset, project_root=self.script_path, verbose=False,
            )
            saved_count = self._persist_neuprint_render_skeletons(
                raw_cache, raw_items,
            )
            self._vprint(
                f'  💾 Saved {saved_count}/{len(raw_items)} '
                f'.swc.zst skeletons (raw level 0)',
                level='full',
            )
        except Exception as exc:
            self._vprint(
                f'  ⚠ Failed to save raw skeleton cache: {exc}', level='full')

    def _downsample_neuprint_skeleton_for_cache(self, neuron):
        """Deprecated compatibility helper; do not use for cache writes.

        NeuPrint skeleton persistence is raw compressed SWC. Fast/direct
        simplification belongs to the transient render mesh, so this helper
        remains only for older external callers that imported it directly.
        """
        if not hasattr(neuron, 'nodes'):
            return neuron
        source = neuron
        try:
            soma = getattr(neuron, 'soma', None)
            if soma is not None and hasattr(soma, '__len__') and len(soma) > 1:
                source = neuron.copy()
                source.soma = None
            simplified = navis.downsample_neuron(
                source,
                downsampling_factor=self.NEUPRINT_SKELETON_DOWNSAMPLE,
            )
            if getattr(simplified, 'id', None) is None:
                simplified.id = getattr(neuron, 'id', None)
            return simplified
        except Exception:
            return neuron
    
    # Cache stores meshes simplified with soma-aware parameters
    # Skeleton: 0.95 simplification (keep 5% of faces) 
    # Soma: 0.8 simplification (keep 20% of faces) to prevent extrusion artifacts
    FAFB_MESH_CACHE_SIMPLIFICATION = FLYWIRE_MESH_CACHE_SIMPLIFICATION
    FAFB_MESH_CACHE_SOMA_SIMPLIFICATION = FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION

    # Compatibility threshold retained as the direct/fast default render
    # target. It is not an on-disk skeleton-cache level. New visualization
    # writes explicitly use level 0 (raw) so the render decimator is applied
    # exactly once and every neuron starts from the same source resolution.
    NEUPRINT_SKELETON_CACHE_LEVEL = 0.9
    NEUPRINT_RENDER_CACHE_SIMPLIFICATION = 0
    NEUPRINT_SKELETON_DOWNSAMPLE = 10  # deprecated helper compatibility
    FAFB_MESH_CACHE_SOMA_RADIUS = FLYWIRE_MESH_CACHE_SOMA_RADIUS
    
    # Number of radial segments used when converting TreeNeurons to tube
    # meshes. This controls the tube cross-section only; it is deliberately
    # independent of skeleton_mesh_simplification.
    SKELETON_TUBE_POINTS = 6

    # ---- NeuPrint "FAFB-format" transform + transient mesh ----
    # This remains the default fine-render target level. The resulting tube
    # mesh is deliberately not written to disk.
    NEUPRINT_MESH_CACHE_SIMPLIFICATION = 0.95

    # Path smoothing for the voxel-grid staircase tracing (rolling window).
    NEUPRINT_PATH_SMOOTH_WINDOW = 9
    # FAFB ring density target: median neuprint edge length / this factor
    # (~25 native units on 8nm-voxel datasets = ~200nm, FAFB median 187nm).
    NEUPRINT_RESAMPLE_DIVISOR = 3.6
    # FAFB-style radius profile (native units; NeuPrint radii are a near-
    # constant placeholder, FAFB SWCs carry the real EM thickness profile).
    NEUPRINT_RADIUS_BASE_FACTOR = 2.0   # base radius multiplier
    NEUPRINT_RADIUS_TIP_TAPER = 0.30    # terminal tips taper to 30% of base
    NEUPRINT_RADIUS_TAPER_TAU = 500.0   # taper length constant
    NEUPRINT_RADIUS_SOMA_CAP = 590.0    # soma radius cap (~4.7um like FAFB)
    # Experimental fine-path alternative: one constant radius matching the
    # NeuPrint baseline radius used by male-cns skeletons.
    NEUPRINT_DEFAULT_RADIUS = 32.0

    # artistic vertex-clustering calibration. The voxel factor provides the
    # initial surface-area estimate; the bounded search then tunes it to the
    # requested face count for the current mesh.
    NEUPRINT_FINE_OPT1_CLUSTER_VOXEL_FACTOR = 1.65
    NEUPRINT_FINE_OPT1_CLUSTER_SEARCH_STEPS = 6

    # Shared NeuPrint online fetch tuning. A single request can contain several
    # visualization layers, but very large body-ID lists make the metadata
    # query and response unnecessarily expensive. Each bounded batch still
    # uses navis' own parallel per-skeleton fetcher. Three workers was the
    # fastest stable setting in the live six-neuron aMe12 probe; keeping this
    # explicit also avoids relying on a dependency default that may change.
    NEUPRINT_FETCH_BATCH_SIZE = 64
    NEUPRINT_FETCH_MAX_THREADS = 3

    def _get_fafb_mesh_cache_key(self):
        """Generate a cache key based on coordinate space and simplification settings.
        
        Returns a subfolder name like 'FLYWIRE_simp95_soma80_r20' for caching purposes.
        Cache stores meshes with soma-aware simplification (gentler on cell body).
        Note: FAFB cache stores UN-TRANSFORMED, UN-ROTATED meshes (native FLYWIRE).
        """
        return flywire_mesh_cache_key(
            self.FAFB_MESH_CACHE_SIMPLIFICATION,
            self.FAFB_MESH_CACHE_SOMA_SIMPLIFICATION,
            self.FAFB_MESH_CACHE_SOMA_RADIUS,
        )
    
    def _load_cached_fafb_meshes(self, body_ids):
        """Load simplified and meshed FAFB neurons from cache.
        
        Cache contains meshes with soma-aware simplification:
        - Skeleton: 0.95 simplification (keep 5% of faces)
        - Soma region (within 20µm): 0.8 simplification (keep 20% of faces)
        
        No coordinate transformation is needed since FAFB uses native FLYWIRE coordinates.
        Only used when skeleton_mesh_simplification >= 0.95.
        If simplification > 0.95, additional simplification is applied after loading.
        
        Parameters
        ----------
        body_ids : list
            List of bodyIds to load
            
        Returns
        -------
        tuple: (dict of bodyId -> MeshNeuron, list of missing bodyIds)
        """
        if not self.cache_neurons:
            return {}, body_ids
        
        # Only use cache when simplification >= cache level (0.95)
        if self.skeleton_mesh_simplification < self.FAFB_MESH_CACHE_SIMPLIFICATION:
            return {}, body_ids
        
        # Check for flywire/fafb dataset
        if not is_flywire_dataset(self.dataset):
            return {}, body_ids
        
        mesh_cache = FlyWireMeshCache(
            self.dataset,
            project_root=self.script_path,
            simplification=self.FAFB_MESH_CACHE_SIMPLIFICATION,
            soma_simplification=self.FAFB_MESH_CACHE_SOMA_SIMPLIFICATION,
            soma_radius=self.FAFB_MESH_CACHE_SOMA_RADIUS,
        )
        
        loaded = {}
        missing = []
        
        for bid in body_ids:
            bid = normalize_flywire_body_id(bid)
            mesh_neuron = mesh_cache.load(bid)
            if mesh_neuron is not None:
                loaded[bid] = mesh_neuron
            else:
                missing.append(bid)
        
        if loaded:
            self._vprint(f'  ✓ Loaded {len(loaded)} neurons from mesh cache (skel={self.FAFB_MESH_CACHE_SIMPLIFICATION}, soma={self.FAFB_MESH_CACHE_SOMA_SIMPLIFICATION})', level='full')
        
        return loaded, missing
    
    def _save_cached_fafb_meshes(self, mesh_neurons_dict):
        """Save transformed and meshed FAFB neurons to cache.
        
        Parameters
        ----------
        mesh_neurons_dict : dict
            Dictionary of bodyId -> MeshNeuron to save
        """
        if not self.cache_neurons:
            return
        
        # Check for flywire/fafb dataset
        if not is_flywire_dataset(self.dataset):
            return
        
        mesh_cache = FlyWireMeshCache(
            self.dataset,
            project_root=self.script_path,
            simplification=self.FAFB_MESH_CACHE_SIMPLIFICATION,
            soma_simplification=self.FAFB_MESH_CACHE_SOMA_SIMPLIFICATION,
            soma_radius=self.FAFB_MESH_CACHE_SOMA_RADIUS,
        )
        normalized = {
            normalize_flywire_body_id(bid): mesh
            for bid, mesh in mesh_neurons_dict.items()
            if isinstance(mesh, navis.MeshNeuron)
        }
        saved_count = mesh_cache.save(normalized)
        
        if saved_count > 0:
            self._vprint(f'  💾 Saved {saved_count} new meshes to cache', level='full')

    # ---- NeuPrint "FAFB-format" transform helpers ----

    def _resolved_neuprint_skeleton_pipeline(self) -> str:
        """Return the internal NeuPrint tube-mode pipeline name.

        ``fast``, ``fine`` and ``artistic`` are the active public names.
        ``direct``, ``fine_opt``, ``fine_opt1`` and ``fafb`` remain accepted
        as compatibility aliases. Line mode always uses the direct path
        because simplification is not applied to line rendering.
        """
        name = (self.neuprint_skeleton_pipeline or 'fast').strip().lower()
        if getattr(self, 'skeleton_mode', 'tube') == 'line':
            return 'direct'
        return {
            'fast': 'direct',
            'fine': 'fine',
            'artistic': 'artistic',
            'direct': 'direct',
            'fine_opt': 'fine',
            'fine_opt1': 'artistic',
            'fafb': 'fine',
        }.get(name, name)

    def _resolved_fafb_pipeline(self) -> str:
        """Return the FAFB tube-mode pipeline name.

        FAFB shares the ``neuprint_skeleton_pipeline`` public names. The
        class default ``fast`` applies unless ``fine`` or ``artistic`` is
        explicitly selected; compatibility aliases resolve the same way as
        the NeuPrint resolver.
        """
        name = (self.neuprint_skeleton_pipeline or 'fast').strip().lower()
        return {
            'fast': 'fast',
            'fine': 'fine',
            'artistic': 'artistic',
            'direct': 'fast',
            'fine_opt': 'fine',
            'fine_opt1': 'artistic',
            'fafb': 'fine',
        }.get(name, 'fast')

    def _fetch_neuprint_skeletons_batched(
            self, body_ids, fetch_kwargs=None, persist=True):
        """Fetch raw NeuPrint skeletons in bounded parallel batches.

        ``navis.interfaces.neuprint.fetch_skeletons`` already parallelizes
        the individual SWC requests, but the visualization used to call it
        separately for every layer. The shared path now collects missing IDs
        across layers, splits them into bounded DataFrame requests, and keeps
        that per-batch parallelism explicit for fast, fine, artistic, and
        line-mode NeuPrint renders. This removes repeated metadata queries
        without creating nested thread pools.

        Parameters
        ----------
        body_ids : iterable or pandas.DataFrame
            Body IDs to fetch. A DataFrame must contain ``bodyId``.
        fetch_kwargs : dict, optional
            Common arguments for ``neu.fetch_skeletons``. The fetch remains
            parallel and the worker count is controlled by the shared fetch
            constants above.
        persist : bool, default True
            When False (strict ``cache_neurons=False``), the vector cache
            and raw compressed-SWC persistence phases are skipped; online
            fetches stay in memory only.

        Returns
        -------
        navis.NeuronList
            Results in the same order as the requested IDs when available.
            NeuPrint's ``missing_swc='warn'`` behavior is preserved for IDs
            that do not have an SWC.
        """
        import navis

        if isinstance(body_ids, pd.DataFrame):
            if 'bodyId' not in body_ids.columns:
                raise ValueError('DataFrame must contain a bodyId column.')
            request_df = body_ids[['bodyId']].copy()
        else:
            request_df = pd.DataFrame({
                'bodyId': [] if body_ids is None else list(body_ids),
            })

        if request_df.empty:
            return navis.NeuronList([])

        # Deduplicate without changing the caller's order. Duplicate IDs can
        # occur when the same neuron is selected in more than one layer.
        seen = set()
        keep_rows = []
        requested_ids = []
        for row_index, body_id in enumerate(request_df['bodyId'].tolist()):
            key = str(body_id)
            if key in seen:
                continue
            seen.add(key)
            keep_rows.append(row_index)
            requested_ids.append(body_id)
        request_df = request_df.iloc[keep_rows].reset_index(drop=True)

        common_kwargs = dict(fetch_kwargs or {})
        if self.client is not None:
            common_kwargs.setdefault('client', self.client)
        common_kwargs['parallel'] = True
        fetched = []
        batch_size = max(1, int(self.NEUPRINT_FETCH_BATCH_SIZE))
        total_batches = (len(request_df) + batch_size - 1) // batch_size
        batch_pbar = tqdm(
            total=len(request_df),
            desc="Fetching NeuPrint skeletons",
            unit="neuron",
            # Do not create a second row while plot_skeleton owns the shared
            # render bar.  The object is still used as a lightweight counter
            # by the raw-cache persistence callback.
            disable=True,
            leave=False,
            file=sys.stdout,
        )
        try:
            for start in range(0, len(request_df), batch_size):
                batch = request_df.iloc[start:start + batch_size].copy()
                batch_kwargs = dict(common_kwargs)
                batch_kwargs['max_threads'] = max(
                    1,
                    min(int(self.NEUPRINT_FETCH_MAX_THREADS), len(batch)),
                )
                batch_input = batch
                fetch_client = batch_kwargs.get('client')
                if fetch_client is not None:
                    # navis 1.5.0 creates ``NeuronCriteria(bodyId=...)``
                    # internally when given a DataFrame.  That constructor asks
                    # neuprint-python for a global default client before navis
                    # can forward its explicit ``client`` argument.  With
                    # clients for multiple datasets in one process, this raises
                    # "more than one Client exists".  Bind the criteria here so
                    # the explicit dataset client survives that compatibility
                    # seam without mutating global client state.
                    from neuprint import NeuronCriteria
                    batch_input = NeuronCriteria(
                        bodyId=batch['bodyId'].to_numpy(),
                        client=fetch_client,
                    )
                # Keep navis' progress output behind the visualizer's shared
                # progress row, just like the legacy per-layer path.
                with self._suppress_output():
                    result = neu.fetch_skeletons(batch_input, **batch_kwargs)
                if result is not None:
                    fetched.extend(list(result))
                batch_pbar.update(len(batch))
                batch_pbar.set_postfix_str(
                    f"batch {start // batch_size + 1}/{total_batches}"
                )
        except Exception:
            batch_pbar.close()
            raise

        # The online batch is not complete until the raw vector cache append
        # and raw-skeleton persistence have finished (kept synchronous and
        # visible before the neurons are ordered/returned so a
        # visualization-only fetch becomes a reusable Find Similar cache).
        # ``persist=False`` (strict cache_neurons=False) skips both cache
        # phases: online fetches stay in memory only.
        if persist:
            self._persist_fetched_neuprint_skeletons(batch_pbar, fetched)
        batch_pbar.close()

        # navis normally preserves order, but the worker completion order is
        # intentionally unspecified. Re-order by bodyId so each layer gets a
        # deterministic result and cache writes remain stable.
        by_id = {}
        for neuron in fetched:
            neuron_id = getattr(neuron, 'id', None)
            if neuron_id is None:
                continue
            by_id[str(neuron_id)] = neuron
            try:
                by_id[str(int(neuron_id))] = neuron
            except (TypeError, ValueError):
                pass

        ordered = [
            by_id[str(body_id)] for body_id in requested_ids
            if str(body_id) in by_id
        ]
        return navis.NeuronList(ordered)

    def _persist_fetched_neuprint_skeletons(self, batch_pbar, fetched):
        """Persist a fresh online NeuPrint fetch to the shared raw caches.

        The online batch is not complete until the raw vector cache append
        and raw-skeleton persistence have finished.  Both phases are kept
        synchronous and visible before the neurons are ordered/returned, so
        a visualization-only fetch becomes a reusable Find Similar cache.
        Skipped entirely under the strict ``cache_neurons=False`` policy.
        """
        try:
            from morphology import (
                cache_fetched_skeleton_vectors, find_similar_raw_cache,
            )

            raw_cache = find_similar_raw_cache(
                self.dataset,
                project_root=self.script_path,
                n_workers=self.NEUPRINT_FETCH_MAX_THREADS,
                verbose=False,
            )

            # Reuse the fetch bar for the synchronous vector-cache phase.
            # Resetting its total keeps the displayed count meaningful while
            # retaining one physical progress row for the whole fetch.
            vector_total = len(fetched)
            if hasattr(batch_pbar, 'reset'):
                batch_pbar.reset(total=vector_total)
            else:
                # Keep lightweight progress doubles used by callers and tests
                # compatible with the shared-bar path.
                batch_pbar.total = vector_total
                batch_pbar.n = 0
            if hasattr(batch_pbar, 'set_description'):
                batch_pbar.set_description('Vectorizing fetched skeletons')

            vector_done = 0

            def _vector_progress(done, total, message):
                nonlocal vector_done
                current = max(0, min(int(done), vector_total))
                delta = current - vector_done
                if delta > 0:
                    batch_pbar.update(delta)
                    vector_done = current

                if message.startswith('Vector cache complete'):
                    description = 'Vector cache complete'
                    postfix = message[len('Vector cache complete'):].strip()
                else:
                    description = 'Vectorizing fetched skeletons'
                    # tqdm already renders the current/total count; keep the
                    # bar as compact as the fetch bar instead of duplicating
                    # that count in a postfix.
                    postfix = ''
                if hasattr(batch_pbar, 'set_description'):
                    batch_pbar.set_description(description)
                batch_pbar.set_postfix_str(postfix.strip('()'))

            vector_stats = cache_fetched_skeleton_vectors(
                self.dataset,
                fetched,
                project_root=self.script_path,
                vector_cache=raw_cache,
                progress_callback=_vector_progress,
                verbose=bool(self.verbose),
            )
            self._vprint(
                f'  ✓ Vector cache phase complete: '
                f'{vector_stats["cached"]}/{vector_stats["vectorized"]} '
                f'raw vectors in {vector_stats["elapsed"]:.2f}s',
                level='simple',
                use_tqdm=True,
            )
            raw_items = {}
            for neuron in fetched:
                neuron_id = getattr(neuron, 'id', None)
                if neuron_id is None:
                    continue
                try:
                    raw_items[int(neuron_id)] = neuron
                except (TypeError, ValueError):
                    continue
            if raw_items:
                self._vprint(
                    f'  Writing raw skeleton cache ({len(raw_items)} neurons)',
                    level='simple',
                    use_tqdm=True,
                )
                written = self._persist_neuprint_render_skeletons(
                    raw_cache, raw_items,
                )
                self._vprint(
                    f'  ✓ Raw skeleton cache complete: {written}/'
                    f'{len(raw_items)} .swc.zst files (raw level 0)',
                    level='simple',
                    use_tqdm=True,
                )
        except Exception as exc:
            # A visualization should still render when a local cache is
            # read-only, but the failure must be explicit rather than hidden.
            self._vprint(
                f'  ⚠️ Vector cache phase failed after skeleton fetch: {exc}',
                level='simple',
                use_tqdm=True,
            )

    def _get_neuprint_online_fetch_ids(
            self, use_neuprint_fine_pipeline,
            use_neuprint_mesh_cache, neuprint_mesh_cache):
        """Return unique NeuPrint body IDs that need an online fetch.

        Every visualization pipeline uses the same raw compressed-SWC source.
        Render-time ``fast``/``fine`` simplification never changes this
        fetch decision. Mesh-cache hits can still avoid a render fetch when
        that separate mesh cache is explicitly enabled.
        """
        if ('flywire' in self.dataset.lower()
                or 'fafb' in self.dataset.lower()
                or self.client_type == 'flywire'):
            return []

        all_fetch_ids = []
        seen_ids = set()
        cached_mesh_ids = {
            str(body_id) for body_id in neuprint_mesh_cache.keys()
        }
        for df in self.neuron_dfs:
            if df is None or 'bodyId' not in df.columns:
                continue

            _, layer_fetch_ids = self._load_cached_neurons(
                df, ignore_cache=False)
            if use_neuprint_mesh_cache:
                layer_fetch_ids = [
                    body_id for body_id in layer_fetch_ids
                    if str(body_id) not in cached_mesh_ids
                ]

            for body_id in layer_fetch_ids:
                key = str(body_id)
                if key not in seen_ids:
                    seen_ids.add(key)
                    all_fetch_ids.append(body_id)
        return all_fetch_ids

    def _aggregate_neuprint_body_ids(self):
        """Return ordered, unique body IDs across every visualization layer."""
        body_ids = []
        seen = set()
        for df in self.neuron_dfs:
            if df is None or 'bodyId' not in df.columns:
                continue
            for body_id in df['bodyId'].tolist():
                try:
                    body_id = int(body_id)
                except (TypeError, ValueError):
                    continue
                if body_id not in seen:
                    seen.add(body_id)
                    body_ids.append(body_id)
        return body_ids

    def _prepare_neuprint_skeletons_for_render(
            self, body_ids, use_neuprint_fine_pipeline,
            use_neuprint_mesh_cache, neuprint_mesh_cache):
        """Prepare one complete NeuPrint render before entering the layer loop.

        This is the four-stage NeuPrint pipeline boundary:

        1. aggregate all body IDs from all layers;
        2. resolve raw cache hits and the raw IDs still needed;
        3. fetch those misses in bounded parallel batches, run the active
           FAFB-format/mesh simplification once, and persist only reusable
           raw skeleton/vector artifacts;
        4. let each layer select already-prepared objects by body ID.

        The returned mapping is intentionally independent of layer grouping.
        That makes repeated body IDs and many small layers use exactly the
        same online-fetch and preprocessing work.
        """
        started = time.perf_counter()
        body_ids = [int(bid) for bid in body_ids]
        if not body_ids:
            return {}, False

        all_df = pd.DataFrame({'bodyId': body_ids})
        fine = bool(use_neuprint_fine_pipeline)
        target_simp = float(self.skeleton_mesh_simplification or 0.0)
        prepared = {}
        raw_cached_skeletons = {}
        legacy_cached_skeletons = {}

        def _id_key(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        # The shared raw cache is a valid source for every visualization
        # pipeline, but the strict ``cache_neurons=False`` policy disables
        # all local reads: every body is fetched online and nothing is
        # persisted. Legacy root skeleton pickles are intentionally not
        # considered raw sources. Older shared SWC files can be simp90 even
        # though the directory is named raw_skeletons; keep them out of the
        # preferred source mapping so a render never mixes source levels.
        if self.cache_neurons:
            try:
                from morphology import find_similar_raw_cache
                raw_cache = find_similar_raw_cache(
                    self.dataset, project_root=self.script_path,
                    n_workers=self.NEUPRINT_FETCH_MAX_THREADS, verbose=False,
                )
                (raw_cached_skeletons, legacy_cached_skeletons,
                 missing_cached_ids) = self._resolve_neuprint_render_cache(
                    raw_cache, body_ids,
                )
            except Exception as exc:
                self._vprint(
                    f'  ⚠️ Shared raw-cache check failed: {exc}',
                    level='full',
                )
                missing_cached_ids = list(body_ids)

        # Stage 2: every visualization mode consumes the same raw source.
        # ``fast`` only changes the in-memory mesh decimation below; it never
        # selects a separate skeleton cache.
        fetch_ids = [bid for bid in body_ids
                     if bid not in raw_cached_skeletons]

        self._vprint(
            f'  🗃️ NeuPrint cache check complete: {len(body_ids)} unique '
            f'{len(raw_cached_skeletons)} shared raw-cache hits, '
            f'{len(fetch_ids)} online fetches required',
            level='simple',
        )

        # Stage 3a: one online fetch phase. The retry policy is kept here,
        # outside the layer loop, so a network hiccup cannot cause one retry
        # per visualization layer.
        # The downstream preparation code consumes a single bodyId-keyed raw
        # source mapping. Seed it from the shared raw cache, then overlay the
        # online results below; diagnostics keep the two counts separate.
        fetched = dict(raw_cached_skeletons)
        online_fetched = {}
        fetch_started = time.perf_counter()
        if fetch_ids:
            fetch_df = pd.DataFrame({'bodyId': fetch_ids})
            fetch_kwargs = {
                'with_synapses': self.show_connectors,
                'missing_swc': 'warn',
            }
            if self.client:
                fetch_kwargs['client'] = self.client

            for attempt in range(5):
                try:
                    self._vprint(
                        f'  🌐 {self._resolved_neuprint_skeleton_pipeline()} '
                        f'preprocessing fetch: {len(fetch_ids)} skeletons '
                        f'(batch={self.NEUPRINT_FETCH_BATCH_SIZE}, '
                        f'workers={self.NEUPRINT_FETCH_MAX_THREADS})',
                        level='full')
                    fetched_list = self._fetch_neuprint_skeletons_batched(
                        fetch_df, fetch_kwargs,
                        persist=self.cache_neurons)
                    for neuron in fetched_list:
                        key = _id_key(getattr(neuron, 'id', None))
                        if key is not None:
                            online_fetched[key] = neuron
                            fetched[key] = neuron
                    break
                except Exception as exc:
                    message = str(exc).lower()
                    response = getattr(exc, 'response', None)
                    status_code = getattr(response, 'status_code', None)
                    retryable = any(token in message for token in (
                        'timeout', 'connection', 'network', 'refused',
                        'reset', 'temporary'))
                    # neuprint-python's ``verbose_errors`` wrapper reduces a
                    # connection-level requests error to just
                    # ``Error accessing POST <url>``.  That message is a
                    # transient transport failure, but it does not contain
                    # the word "connection", so the old classifier stopped
                    # after the first failed aggregate fetch.  HTTP 408,
                    # 429, and 5xx responses are retryable too; 4xx auth or
                    # query errors are deliberately surfaced immediately.
                    if (
                        'error accessing post ' in message
                        and 'returned error' not in message
                    ):
                        retryable = True
                    if status_code in {408, 429} or (
                            isinstance(status_code, int)
                            and status_code >= 500):
                        retryable = True
                    if retryable and attempt < 4:
                        wait_time = attempt + 1
                        tqdm.write(
                            f'  ⚠️  NeuPrint preprocessing fetch failed; '
                            f'retrying in {wait_time}s '
                            f'(attempt {attempt + 1}/5): {exc}')
                        time.sleep(wait_time)
                        continue
                    tqdm.write(
                        f'  ⚠️  NeuPrint preprocessing fetch failed: {exc}')
                    break

            self._vprint(
                f'  ✓ Skeleton fetch phase complete: {len(online_fetched)}/'
                f'{len(fetch_ids)} skeletons in '
                f'{time.perf_counter() - fetch_started:.2f}s',
                level='simple',
                use_tqdm=True,
            )
        else:
            self._vprint(
                '  ✓ Skeleton fetch phase complete: all requested raw sources '
                f'came from cache ({len(raw_cached_skeletons)} Find Similar hits) '
                f'in {time.perf_counter() - fetch_started:.2f}s',
                level='simple',
                use_tqdm=True,
            )

        # MANC's NeuPrint coordinates are voxel units. Apply the same guarded
        # scale once to the in-memory render source. Raw cache files remain
        # in the source coordinate system and are never rewritten as scaled
        # render products.
        def _scale_manc(neuron):
            if 'manc' not in self.dataset.lower():
                return neuron
            try:
                bbox = neuron.bbox if hasattr(neuron, 'bbox') else None
                if bbox is not None and np.max(bbox) < 150000:
                    return neuron * 8
            except Exception:
                pass
            return neuron

        for key in list(fetched):
            fetched[key] = _scale_manc(fetched[key])
        # If an old simp90 file could not be refreshed, use it only as a
        # consistent fallback. Re-leveling every available source to the same
        # legacy level prevents one neuron from silently using raw geometry
        # while another uses a pre-simplified cache file.
        unresolved_legacy = {
            body_id: entry for body_id, entry in legacy_cached_skeletons.items()
            if body_id not in online_fetched
        }
        if unresolved_legacy:
            fetched = self._normalize_neuprint_cache_fallback(
                raw_cached_skeletons,
                unresolved_legacy,
                fetched,
            )

        # Online fetches are persisted by the shared batched fetcher. Keep
        # this phase render-only; no simplified skeleton pickle is written.

        # Stage 3b/4: transform and simplify in the same pre-processing phase
        # for all fetched neurons, then combine them with cache hits. The
        # layer loop will only select from this mapping.
        def _as_mesh(neuron):
            if isinstance(neuron, navis.MeshNeuron):
                return neuron
            if not isinstance(neuron, navis.TreeNeuron):
                return neuron
            try:
                source = neuron.copy()
                if 'radius' in source.nodes.columns:
                    invalid = (source.nodes['radius'] <= 0
                               ) | source.nodes['radius'].isna()
                    if invalid.any():
                        source.nodes.loc[invalid, 'radius'] = 1
                else:
                    source.nodes['radius'] = 1
                mesh = navis.conversion.tree2meshneuron(
                    source, tube_points=self.SKELETON_TUBE_POINTS)
                mesh.id = getattr(neuron, 'id', None)
                if hasattr(neuron, 'name'):
                    mesh.name = neuron.name
                return mesh
            except Exception:
                return neuron

        def _decimate(neuron, simplification):
            if simplification <= 0 or not hasattr(neuron, 'trimesh'):
                return neuron
            try:
                n_faces = len(neuron.trimesh.faces)
                target_faces = max(100, int(n_faces * (1 - simplification)))
                if target_faces >= n_faces:
                    return neuron
                simplified = self._simplify_mesh_for_neuprint_pipeline(
                    neuron.trimesh, target_faces)
                result = navis.MeshNeuron(simplified)
                result.id = getattr(neuron, 'id', None)
                if hasattr(neuron, 'name'):
                    result.name = neuron.name
                return result
            except Exception:
                return neuron

        self._vprint(
            f'  🧰 Starting NeuPrint preprocessing: transform, tube mesh, '
            f'and {self._resolved_neuprint_skeleton_pipeline()} simplification '
            f'for {len(body_ids)} neurons',
            level='simple',
            use_tqdm=True,
        )
        preprocess_pbar = tqdm(
            total=len(body_ids),
            desc="Preprocessing NeuPrint skeletons",
            unit="neuron",
            # This aggregate phase owns one transient tqdm row.  It closes
            # before the layered-render progress bar starts, so preprocessing
            # reports a compact n/N count without nesting progress displays.
            disable=not self.verbose,
            leave=False,
            file=sys.stdout,
        )

        def _set_preprocess_stage(body_id, stage):
            """Show the active per-neuron stage on the shared bar."""
            preprocess_pbar.set_postfix_str(f"{stage}: {body_id}")

        def _mark_preprocessed(body_id, stage):
            preprocess_pbar.update(1)
            _set_preprocess_stage(body_id, stage)

        if fine:
            optimized = self._uses_neuprint_fine_optimized_pipeline()
            for body_id in body_ids:
                base = fetched.get(body_id)
                if base is None:
                    _mark_preprocessed(body_id, 'missing')
                    continue

                _set_preprocess_stage(body_id, 'transform')
                transformed = self._fafb_style_neuprint_skeleton(
                    base, optimized=optimized)
                if target_simp <= 0 or self.show_connectors:
                    # Connector-rich fine renders retain the skeleton object;
                    # the common layer stage can attach connector traces.
                    prepared[body_id] = transformed
                    _mark_preprocessed(body_id, 'transform')
                    continue

                _set_preprocess_stage(body_id, 'tube mesh')
                mesh = _as_mesh(transformed)
                _set_preprocess_stage(body_id, 'simplify')
                prepared[body_id] = _decimate(mesh, target_simp)
                _mark_preprocessed(body_id, 'fine')
        elif self.skeleton_mode == 'tube' and target_simp > 0 \
                and not self.show_connectors:
            # Fast/direct tube mode converts the raw skeleton to a mesh and
            # applies the requested simplification directly in memory.
            for body_id in body_ids:
                base = fetched.get(body_id)
                if base is None:
                    _mark_preprocessed(body_id, 'missing')
                    continue
                _set_preprocess_stage(body_id, 'tube mesh')
                mesh = _as_mesh(base)
                _set_preprocess_stage(body_id, 'simplify')
                prepared[body_id] = _decimate(mesh, target_simp)
                _mark_preprocessed(body_id, 'fast')
        else:
            # Line renders and unsimplified tube/fine renders consume the
            # prepared skeleton directly.  Line mode applies the dataset
            # line-reduction stage in memory on a copy (NeuPrint 50%); the
            # raw source is never modified and no reduced skeleton is
            # persisted.
            line_reduction = (
                NEUPRINT_LINE_NODE_REDUCTION
                if self.skeleton_mode == 'line' else None)
            for body_id in body_ids:
                base = fetched.get(body_id)
                if base is None:
                    _mark_preprocessed(body_id, 'missing')
                    continue
                if (line_reduction is not None
                        and isinstance(base, navis.TreeNeuron)):
                    _set_preprocess_stage(body_id, 'simplify lines')
                    try:
                        work, stats = simplify_skeleton_nodes(
                            base, line_reduction)
                        self._vprint(
                            f'  🔽 simplify lines {body_id} '
                            f'{stats["raw_nodes"]}→'
                            f'{stats["achieved_nodes"]}',
                            level='full', use_tqdm=True)
                        prepared[body_id] = work
                        _mark_preprocessed(body_id, 'line')
                    except Exception as e:
                        self._vprint(
                            f'  ⚠️ line simplification failed for '
                            f'{body_id}: {e}', level='full',
                            use_tqdm=True)
                        prepared[body_id] = base
                        _mark_preprocessed(body_id, 'ready')
                else:
                    prepared[body_id] = base
                    _mark_preprocessed(body_id, 'ready')

        preprocess_pbar.close()
        prepared_mesh = bool(prepared) and all(
            isinstance(neuron, navis.MeshNeuron)
            for neuron in prepared.values())
        elapsed = time.perf_counter() - started
        self._vprint(
            f'  ✓ NeuPrint preprocessing ready: {len(prepared)}/'
            f'{len(body_ids)} neurons in {elapsed:.2f}s '
            f'(cache={len(raw_cached_skeletons)}, '
            f'fetched={len(fetched)}, mesh={prepared_mesh})',
            level='full')
        self._vprint(
            f'  ✓ NeuPrint preprocessing phase complete in {elapsed:.2f}s; '
            'entering layered rendering',
            level='simple',
            use_tqdm=True,
        )
        return prepared, prepared_mesh

    def _uses_neuprint_fine_pipeline(self) -> bool:
        """Whether the current NeuPrint tube render uses a fine pipeline."""
        return self._resolved_neuprint_skeleton_pipeline() in {
            'fine', 'artistic'}

    def _uses_neuprint_fine_optimized_pipeline(self) -> bool:
        """Whether the current fine render uses optimized calculations."""
        return self._resolved_neuprint_skeleton_pipeline() in {
            'fine', 'artistic'}

    def _resolved_skeleton_radius_style(self) -> str:
        """Resolve ``skeleton_radius_style`` ('auto' -> 'fafb' for NeuPrint).

        ``constant`` is an experimental style that keeps the FAFB path
        smoothing/resampling but skips the graph-based radius profile and
        assigns ``NEUPRINT_DEFAULT_RADIUS`` to every node.
        """
        style = (self.skeleton_radius_style or 'auto').strip().lower()
        if style == 'auto':
            is_flywire = ('flywire' in self.dataset.lower()
                          or 'fafb' in self.dataset.lower()
                          or 'banc' in self.dataset.lower())
            return 'source' if is_flywire else 'fafb'
        return {'default': 'constant'}.get(style, style)

    def _neuron_edge_lengths(self, neuron) -> np.ndarray:
        """Parent->child edge lengths of a TreeNeuron (native units)."""
        nodes = neuron.nodes.set_index('node_id')
        lengths = []
        for nid, parent in nodes['parent_id'].items():
            if parent is None or (isinstance(parent, float) and np.isnan(parent)) or parent < 0:
                continue
            if parent not in nodes.index:
                continue
            p1 = nodes.loc[nid, ['x', 'y', 'z']].astype(float).values
            p2 = nodes.loc[parent, ['x', 'y', 'z']].astype(float).values
            lengths.append(np.linalg.norm(p1 - p2))
        return np.asarray(lengths, dtype=float)

    def _neuron_edge_lengths_fast(self, neuron) -> np.ndarray:
        """Vectorized parent->child edge lengths for the optimized pipeline."""
        nodes = neuron.nodes
        node_ids = nodes['node_id'].to_numpy()
        parent_ids = nodes['parent_id'].to_numpy()
        parent_indices = pd.Index(node_ids).get_indexer(parent_ids)
        parent_numbers = pd.to_numeric(
            pd.Series(parent_ids), errors='coerce'
        ).to_numpy(dtype=float)
        valid = (
            (parent_indices >= 0)
            & np.isfinite(parent_numbers)
            & (parent_numbers >= 0)
        )
        if not np.any(valid):
            return np.asarray([], dtype=float)

        coordinates = nodes[['x', 'y', 'z']].to_numpy(dtype=float)
        child_indices = np.flatnonzero(valid)
        deltas = (
            coordinates[child_indices]
            - coordinates[parent_indices[valid]]
        )
        return np.linalg.norm(deltas, axis=1)

    def _neuron_dist_to_tip(self, neuron) -> dict:
        """Graph distance from every node to the nearest terminal tip node."""
        import heapq
        nodes = neuron.nodes.set_index('node_id')
        adj = {int(i): [] for i in nodes.index}
        for nid, row in nodes.iterrows():
            p = row['parent_id']
            if p is not None and p >= 0 and p in nodes.index:
                length = float(np.linalg.norm(
                    row[['x', 'y', 'z']].astype(float).values
                    - nodes.loc[p, ['x', 'y', 'z']].astype(float).values))
                adj[int(nid)].append((int(p), length))
                adj[int(p)].append((int(nid), length))
        # tips: leaf nodes (degree 1) other than the root itself
        roots = [int(i) for i in nodes.index
                 if nodes.loc[i, 'parent_id'] is None
                 or (isinstance(nodes.loc[i, 'parent_id'], float)
                     and np.isnan(nodes.loc[i, 'parent_id']))
                 or nodes.loc[i, 'parent_id'] < 0]
        tips = [i for i in adj if len(adj[i]) == 1 and i not in roots]
        if not tips:
            return {int(i): 0.0 for i in nodes.index}
        dist = {t: 0.0 for t in tips}
        pq = [(0.0, t) for t in tips]
        heapq.heapify(pq)
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return {int(i): dist.get(int(i), 0.0) for i in nodes.index}

    def _neuron_dist_to_tip_fast(self, neuron) -> dict:
        """Array-backed equivalent of :meth:`_neuron_dist_to_tip`.

        The original implementation repeatedly indexes a pandas frame while
        constructing the graph.  This version keeps the same multi-source
        shortest-path calculation, but maps node IDs to array positions once
        and performs coordinate arithmetic with NumPy.
        """
        import heapq

        nodes = neuron.nodes
        node_ids = nodes['node_id'].to_numpy()
        parent_ids = nodes['parent_id'].to_numpy()
        coordinates = nodes[['x', 'y', 'z']].to_numpy(dtype=float)
        parent_indices = pd.Index(node_ids).get_indexer(parent_ids)
        parent_numbers = pd.to_numeric(
            pd.Series(parent_ids), errors='coerce'
        ).to_numpy(dtype=float)
        valid = (
            (parent_indices >= 0)
            & np.isfinite(parent_numbers)
            & (parent_numbers >= 0)
        )
        child_indices = np.flatnonzero(valid)

        adjacency = [[] for _ in range(len(node_ids))]
        if len(child_indices):
            parent_indices_valid = parent_indices[valid]
            edge_lengths = np.linalg.norm(
                coordinates[child_indices]
                - coordinates[parent_indices_valid],
                axis=1,
            )
            for child, parent, length in zip(
                    child_indices, parent_indices_valid, edge_lengths):
                weight = float(length)
                child = int(child)
                parent = int(parent)
                adjacency[child].append((parent, weight))
                adjacency[parent].append((child, weight))

        # Match the source implementation's root rule for normal NeuPrint
        # skeletons: nodes with no valid non-negative parent are roots.
        roots = ~valid
        degrees = np.fromiter(
            (len(neighbors) for neighbors in adjacency),
            dtype=np.int64,
            count=len(adjacency),
        )
        tips = np.flatnonzero((degrees == 1) & ~roots)
        if not len(tips):
            return {int(node_id): 0.0 for node_id in node_ids}

        distances = np.full(len(node_ids), np.inf, dtype=float)
        priority_queue = []
        for tip in tips:
            tip = int(tip)
            distances[tip] = 0.0
            priority_queue.append((0.0, tip))
        heapq.heapify(priority_queue)

        while priority_queue:
            distance, node_index = heapq.heappop(priority_queue)
            if distance > distances[node_index]:
                continue
            for neighbor, weight in adjacency[node_index]:
                candidate = distance + weight
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    heapq.heappush(
                        priority_queue, (candidate, neighbor)
                    )

        return {
            int(node_id): float(distance)
            for node_id, distance in zip(node_ids, distances)
        }

    def _fafb_style_radius(
            self, neuron, optimized: bool = False) -> 'navis.TreeNeuron':
        """Synthesize a FAFB-style radius profile on a NeuPrint skeleton.

        NeuPrint radius columns are a near-constant placeholder (the "sketch"
        look); FAFB SWCs carry the real EM thickness profile (thin tips,
        thicker branchpoints, big soma). This is a visualization convention:
        the base radius is multiplied by ``NEUPRINT_RADIUS_BASE_FACTOR``, tips
        taper to ``NEUPRINT_RADIUS_TIP_TAPER`` over ``NEUPRINT_RADIUS_TAPER_TAU``
        (distance to the nearest terminal), and the soma is capped at
        ``NEUPRINT_RADIUS_SOMA_CAP``. Existing branchpoint/soma bumps are
        preserved by the multiplication.
        """
        nn = neuron.copy()
        dist = (
            self._neuron_dist_to_tip_fast(nn)
            if optimized else self._neuron_dist_to_tip(nn)
        )
        tau = self.NEUPRINT_RADIUS_TAPER_TAU
        cap = self.NEUPRINT_RADIUS_SOMA_CAP
        base = self.NEUPRINT_RADIUS_BASE_FACTOR
        tip = self.NEUPRINT_RADIUS_TIP_TAPER
        radii = nn.nodes['radius'].to_numpy(dtype=float)
        distances = np.fromiter(
            (dist.get(int(nid), 0.0) for nid in nn.nodes['node_id'].values),
            dtype=float,
            count=len(nn.nodes),
        )
        taper = tip + (1.0 - tip) * (1.0 - np.exp(-distances / tau))
        updated_radii = np.minimum(radii * base * taper, cap)
        if optimized:
            nn.nodes['radius'] = updated_radii
        else:
            for i, radius in enumerate(updated_radii):
                nn.nodes.loc[i, 'radius'] = radius
        return nn

    def _constant_neuprint_radius(self, neuron) -> 'navis.TreeNeuron':
        """Use one NeuPrint baseline radius for every node.

        This is intentionally separate from ``_fafb_style_radius`` so the
        expensive distance-to-tip graph calculation is skipped entirely in
        the experimental constant-radius fine variant.
        """
        nn = neuron.copy()
        nn.nodes['radius'] = float(self.NEUPRINT_DEFAULT_RADIUS)
        return nn

    def _fafb_style_neuprint_skeleton(
            self, neuron, radius_style=None,
            optimized: bool = False) -> 'navis.TreeNeuron':
        """Transform a raw NeuPrint skeleton into FAFB-format:

        1. smooth the voxel-grid staircase path (rolling window)
        2. resample to FAFB-like ring density (median edge / divisor)
        3. apply the selected radius style

        All steps operate in the skeleton's native units. ``radius_style``
        defaults to the visualizer's resolved style and can be set to
        ``constant`` for a no-radius-calculation comparison. ``optimized``
        selects the array-backed edge and graph calculations used by
        ``fine`` and ``artistic``.
        """
        import navis
        nn = neuron
        try:
            # navis prints its own per-segment "Smoothing" bar; suppress it
            # so the shared preprocessing bar stays the single progress row.
            with self._suppress_navis_progress(), self._suppress_output():
                nn = navis.smooth_skeleton(
                    nn, window=self.NEUPRINT_PATH_SMOOTH_WINDOW)
        except Exception:
            pass  # keep the raw path if smoothing is unavailable
        edges = (
            self._neuron_edge_lengths_fast(nn)
            if optimized else self._neuron_edge_lengths(nn)
        )
        if len(edges):
            target = max(1.0, float(np.median(edges))
                         / self.NEUPRINT_RESAMPLE_DIVISOR)
            try:
                with self._suppress_navis_progress(), self._suppress_output():
                    nn = navis.resample_skeleton(nn, resample_to=target)
            except Exception:
                pass
        style = radius_style or self._resolved_skeleton_radius_style()
        if style == 'fafb':
            return self._fafb_style_radius(nn, optimized=optimized)
        if style == 'constant':
            return self._constant_neuprint_radius(nn)
        return nn

    def _get_neuprint_mesh_cache_key(self):
        """Return the legacy NeuPrint fine-mesh cache key.

        Kept only for compatibility with old reports/tests. New rendering
        never reads or writes this namespace.
        """
        simp = int(self.NEUPRINT_MESH_CACHE_SIMPLIFICATION * 100)
        radius_style = self._resolved_skeleton_radius_style()
        suffix = '' if radius_style == 'fafb' else f'_radius{radius_style}'
        if self._resolved_neuprint_skeleton_pipeline() == 'artistic':
            suffix += '_vertexcluster'
        return f"NEUPRINT_simp{simp}{suffix}"

    def _load_cached_neuprint_meshes(self, body_ids):
        """Ignore historical NeuPrint fine-mesh directories.

        This compatibility seam intentionally returns no mesh hits. Fine and
        artistic rendering must load the standalone raw level-0 ``.swc.zst``
        source and rebuild the tube mesh for the current visualization
        settings.
        """
        return {}, list(body_ids)

    def _save_cached_neuprint_meshes(self, mesh_neurons_dict):
        """Ignore historical NeuPrint fine-mesh save requests.

        Raw ``.swc.zst`` persistence is handled by the batched fetch phase;
        mesh objects are render-time products and must not become a second
        cache representation.
        """
        if mesh_neurons_dict:
            self._vprint(
                '  ℹ️ NeuPrint fine mesh caching is disabled; using the '
                'simplified .swc.zst source',
                level='full',
            )
        return 0

    def _load_api_cached_skeletons(self, body_ids: list) -> tuple:
        """Load raw skeletons from the shared compressed-SWC cache.
        
        These are skeletons previously fetched via CAVE API and cached locally.
        The shared raw cache takes priority over ZIP files as it contains more
        up-to-date data. The old API-cache pickle namespace is read only as a
        migration fallback.
        
        Parameters
        ----------
        body_ids : list
            List of body IDs to look for in cache
            
        Returns
        -------
        tuple
            (dict of loaded skeletons {bodyId: TreeNeuron}, list of missing bodyIds)
        """
        requested_ids = (
            normalize_flywire_body_ids(body_ids)
            if is_flywire_dataset(self.dataset)
            else list(body_ids)
        )
        cached = {}
        missing = []

        try:
            from morphology import find_similar_raw_cache
            raw_cache = find_similar_raw_cache(
                self.dataset, project_root=self.script_path, verbose=False,
            )
        except Exception:
            raw_cache = None

        for bid in requested_ids:
            neuron = raw_cache.load_skeleton(bid) if raw_cache is not None else None
            if neuron is not None:
                # The shared raw directory historically also contains files
                # written by the old visualization path at simp90.  They are
                # not valid raw render inputs: leave them for the refresh or
                # prepared-mesh fallback instead of mixing them with level-0
                # sources in the same scene.
                if self._cached_skeleton_level(neuron) != (
                        self.NEUPRINT_RENDER_CACHE_SIMPLIFICATION):
                    missing.append(bid)
                    continue
                key = (
                    normalize_flywire_body_id(bid)
                    if is_flywire_dataset(self.dataset)
                    else bid
                )
                cached[key] = neuron
                continue

            # Compatibility read for API_cache/{bodyId}.pkl produced by older
            # CAVE versions. New writes never target this path.
            dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
            legacy_file = os.path.join(
                self.script_path, 'cache', dataset_safe, 'API_cache',
                'skeletons', f'{bid}.pkl')
            if os.path.exists(legacy_file):
                try:
                    import pickle
                    with open(legacy_file, 'rb') as f:
                        neuron = pickle.load(f)
                    if raw_cache is not None:
                        self._persist_neuprint_render_skeletons(
                            raw_cache, {bid: neuron},
                        )
                    key = (
                        normalize_flywire_body_id(bid)
                        if is_flywire_dataset(self.dataset)
                        else bid
                    )
                    cached[key] = neuron
                    continue
                except Exception:
                    pass
            missing.append(bid)
        
        return cached, missing

    def _fafb_soma_positions(self, body_ids: list) -> dict:
        """Read numeric FAFB soma positions from the local neuron table."""
        requested = set(normalize_flywire_body_ids(body_ids))
        positions = {}
        for frame in self.neuron_dfs:
            if frame is None or 'bodyId' not in frame.columns:
                continue
            position_col = next(
                (name for name in (
                    'position', 'soma_position', 'soma_pos', 'somaLocation')
                 if name in frame.columns),
                None,
            )
            if position_col is None:
                continue
            for body_id, value in zip(
                    frame['bodyId'].tolist(), frame[position_col].tolist()):
                key = normalize_flywire_body_id(body_id)
                if key not in requested or key in positions:
                    continue
                soma_pos = parse_soma_position(value)
                if soma_pos is not None:
                    positions[key] = soma_pos
        return positions

    def _resolve_fafb_sources(self, body_ids, allow_mesh_cache=True,
                              api_only=False):
        """Resolve per-body FAFB sources for rendering.

        Priority per body:

        Tube mode (``allow_mesh_cache`` — caching enabled at/above the
        prepared level):
        1. prepared CAVE mesh cache (``FlyWireMeshCache``) — CAVE-derived,
           extrusion-free, and already at the render preparation level,
        2. shared raw SWC cache (+ legacy API-pickle migration),
        3. healed-ZIP SWC (canonical raw bundle),
        4. CAVE API (online, no skeletonization, no SWC writes).

        Line mode (mesh cache disabled): the order is SWC-first —
        1. shared raw SWC cache, 2. healed-ZIP SWC, 3. CAVE API — because
        mesh sources would have to be skeletonized in memory at the render
        boundary and would render with different geometry than the trees
        used in tube mode.

        Bodies are classified as ``zip``, ``raw_cache``, ``mesh_cache``,
        ``local_repaired`` or ``cave``.  Cache sources (2/3) are skipped when
        the ``cache_neurons`` policy disables them; the healed ZIP is the
        canonical raw source and stays eligible.  ``api_only``
        (``force_API_fetching``) routes every body straight to CAVE.

        Extrusion repair runs on TreeNeuron sources only and honors the
        same ``use_cache`` policy for the parquet check results.

        Returns
        -------
        tuple
            (sources, skeleton_cache, mesh_cache):
            sources: canonical bodyId -> one of ``zip``, ``raw_cache``,
                     ``local_repaired``, ``mesh_cache`` or ``cave``
            skeleton_cache: canonical bodyId -> TreeNeuron
            mesh_cache: canonical bodyId -> MeshNeuron (mesh_cache/cave)
        """
        requested = normalize_flywire_body_ids(body_ids)
        sources = {}
        skeleton_cache = {}
        mesh_cache = {}

        if api_only:
            fetched = self._fetch_fafb_skeletons_via_api(
                requested, cache_prepared=allow_mesh_cache)
            for key, neuron in fetched.items():
                sources[normalize_flywire_body_id(key)] = 'cave'
            mesh_cache.update(fetched)
            return sources, skeleton_cache, mesh_cache

        remaining = list(requested)

        def _take_bundle():
            """Healed-ZIP SWC (canonical raw bundle)."""
            nonlocal remaining
            if not remaining:
                return
            zip_skeletons = self._preload_fafb_skeletons(
                body_ids_filter=remaining)
            if zip_skeletons:
                zip_neurons = {
                    normalize_flywire_body_id(key): neuron
                    for key, neuron in zip_skeletons.items()
                }
                for canonical in zip_neurons:
                    sources[canonical] = 'zip'
                remaining = [b for b in remaining
                             if b not in zip_neurons]
                skeleton_cache.update(zip_neurons)

        def _take_raw_cache():
            """Shared raw SWC cache (skipped under use_cache=False)."""
            nonlocal remaining
            if not remaining or not self.cache_neurons:
                return
            raw_skeletons, missing = self._load_api_cached_skeletons(
                remaining)
            if raw_skeletons:
                raw_neurons = {
                    normalize_flywire_body_id(key): neuron
                    for key, neuron in raw_skeletons.items()
                }
                for canonical in raw_neurons:
                    sources[canonical] = 'raw_cache'
                remaining = [b for b in missing
                             if b not in raw_neurons]
                skeleton_cache.update(raw_neurons)

        def _take_mesh_cache():
            """Prepared CAVE mesh cache (tube mode only)."""
            nonlocal remaining
            if not remaining:
                return
            loaded, missing = self._load_cached_fafb_meshes(remaining)
            if loaded:
                mesh_neurons = {
                    normalize_flywire_body_id(key): neuron
                    for key, neuron in loaded.items()
                }
                for canonical in mesh_neurons:
                    sources[canonical] = 'mesh_cache'
                remaining = [b for b in missing
                             if b not in mesh_neurons]
                mesh_cache.update(mesh_neurons)

        if allow_mesh_cache:
            # Tube mode: the prepared mesh cache wins. Its entries are
            # CAVE-derived (extrusion-free) and already at the render
            # preparation level, so they skip the extrusion check/repair
            # cycle that tree sources go through below.
            _take_mesh_cache()
            _take_raw_cache()
            _take_bundle()
        else:
            # Line mode: SWC-first (raw cache before the canonical bundle,
            # matching the pipeline priority); mesh sources would be
            # skeletonized in memory at the render boundary with different
            # geometry.
            _take_raw_cache()
            _take_bundle()

        # Extrusion repair on TreeNeuron sources only.
        tree_bodies = [
            body_id for body_id, source in sources.items()
            if source in {'zip', 'raw_cache'}
        ]
        if tree_bodies and self.auto_fix_extrusions:
            extrusion_ids = self._detect_extrusions_in_skeletons(
                {body_id: skeleton_cache[body_id]
                 for body_id in tree_bodies if body_id in skeleton_cache},
                use_cache=self.cache_neurons,
            )
            if extrusion_ids:
                self._vprint(
                    f'  🔍 Detected {len(extrusion_ids)} meshes with '
                    f'extrusion artifacts, fetching fresh from API...',
                    level='simple',
                )
                api_fixed = self._fetch_fafb_skeletons_via_api(
                    extrusion_ids,
                    cache_prepared=allow_mesh_cache,
                    force_refresh=True,
                )
                fixed_ids = set()
                repair_statuses = {}
                if api_fixed:
                    for key, neuron in api_fixed.items():
                        canonical = normalize_flywire_body_id(key)
                        fixed_ids.add(canonical)
                        sources[canonical] = 'cave'
                        skeleton_cache.pop(canonical, None)
                        mesh_cache[canonical] = neuron
                        if self.cache_neurons:
                            repair_statuses[canonical] = 'api_repaired'
                    self._vprint(
                        f'  ✓ Replaced {len(api_fixed)} extrusion-affected '
                        f'meshes', level='simple')

                # A CAVE outage, missing token, or a per-body API miss must
                # not put the known-bad ZIP tree back into the render
                # silently.  Keep the source local and prune only the
                # diagnosed extrusion branch when the cut is safe.
                failed_ids = []
                seen_failed = set()
                for body_id in extrusion_ids:
                    canonical = normalize_flywire_body_id(body_id)
                    if canonical not in fixed_ids and canonical not in seen_failed:
                        failed_ids.append(canonical)
                        seen_failed.add(canonical)
                if failed_ids:
                    from fafb_utils import repair_extruded_skeleton

                    locally_repaired = 0
                    for canonical in failed_ids:
                        source_neuron = skeleton_cache.get(canonical)
                        if source_neuron is None:
                            if self.cache_neurons:
                                repair_statuses[canonical] = 'api_failed'
                            continue
                        repaired, repair_stats = repair_extruded_skeleton(
                            source_neuron)
                        if repair_stats.get('repaired'):
                            skeleton_cache[canonical] = repaired
                            sources[canonical] = 'local_repaired'
                            locally_repaired += 1
                            if self.cache_neurons:
                                # Keep the detection flag. A local fallback
                                # remains retryable when CAVE is available on
                                # a later run.
                                repair_statuses[canonical] = 'local_fallback'
                            self._vprint(
                                f'  🩹 CAVE fetch failed for {canonical}; '
                                f'pruned {repair_stats["removed_nodes"]} '
                                f'extrusion node(s) locally',
                                level='simple',
                            )
                    if locally_repaired:
                        self._vprint(
                            f'  🩹 Locally repaired {locally_repaired}/'
                            f'{len(failed_ids)} extrusion-affected '
                            f'skeleton(s)', level='simple')
                    if locally_repaired < len(failed_ids):
                        self._vprint(
                            f'  ⚠️  CAVE fetch failed for '
                            f'{len(failed_ids) - locally_repaired} '
                            f'extrusion-affected skeleton(s); no safe local '
                            f'branch cut was available',
                            level='simple',
                        )
                    for canonical in failed_ids:
                        repair_statuses.setdefault(canonical, 'api_failed')

                if self.cache_neurons and repair_statuses:
                    try:
                        from fafb_utils import set_extrusion_repair_status

                        dataset_safe = self.dataset.replace(':', '_').replace('.', '_')
                        set_extrusion_repair_status(
                            self.script_path,
                            dataset_safe,
                            repair_statuses,
                        )
                    except Exception as exc:
                        self._vprint(
                            f'  ⚠️ Failed to save extrusion repair status: '
                            f'{exc}', level='full')

        # 4. CAVE API for anything still missing.
        if remaining:
            fetched = self._fetch_fafb_skeletons_via_api(
                remaining, cache_prepared=allow_mesh_cache)
            for key, neuron in fetched.items():
                sources[normalize_flywire_body_id(key)] = 'cave'
            mesh_cache.update(fetched)

        counts = {}
        for source in sources.values():
            counts[source] = counts.get(source, 0) + 1
        summary = ', '.join(
            f'{counts[source]} {source}' for source in
            ('zip', 'raw_cache', 'local_repaired', 'mesh_cache', 'cave')
            if source in counts)
        self._vprint(f'  🗂️  FAFB sources resolved: {summary}', level='simple')
        return sources, skeleton_cache, mesh_cache

    def _fetch_fafb_skeletons_via_api(
            self, body_ids: list, cache_prepared: bool = True,
            soma_positions: dict | None = None,
            force_refresh: bool = False) -> dict:
        """Fetch FAFB meshes through CAVE without skeletonizing.

        With ``cache_prepared=True`` this uses the shared FlyWire mesh cache
        and applies the visualization cache policy (95% branch reduction,
        80% soma reduction within 20 µm).  When that cache is intentionally
        bypassed, the online path returns raw ``MeshNeuron`` objects for a
        lower render simplification.  ``force_refresh=True`` bypasses the
        prepared-mesh cache read for repair requests, fetches a fresh CAVE
        mesh, and rewrites the prepared mesh cache when caching is enabled.
        Neither path writes NeuPrint SWC.
        """
        access = getattr(self, '_flywire_skeleton_access', None)
        if access is not None and not access.get('cave_token'):
            self._vprint(
                '  ⚠️  CAVE_TOKEN is not configured; skipping FAFB API fallback '
                'and using local skeleton data only.',
                level='simple',
            )
            return {}

        try:
            from cave_data_fetcher import CAVEDataFetcher
        except ImportError:
            try:
                from .cave_data_fetcher import CAVEDataFetcher
            except ImportError:
                self._vprint("  ⚠️  cave_data_fetcher module not found, falling back to ZIP", use_tqdm=True)
                return {}
        
        requested_ids = (
            normalize_flywire_body_ids(body_ids)
            if is_flywire_dataset(self.dataset)
            else list(body_ids)
        )

        # Split the progress line so prepared-cache hits are reported
        # separately from the online CAVE fetches that actually hit the
        # network.  ``force_refresh`` bypasses prepared-cache reads, so no
        # hit report is emitted for repair requests.
        prepared_ids = []
        online_ids = list(requested_ids)
        if cache_prepared and self.cache_neurons and not force_refresh:
            try:
                mesh_cache = FlyWireMeshCache(
                    self.dataset,
                    project_root=self.script_path,
                    simplification=self.FAFB_MESH_CACHE_SIMPLIFICATION,
                    soma_simplification=self.FAFB_MESH_CACHE_SOMA_SIMPLIFICATION,
                    soma_radius=self.FAFB_MESH_CACHE_SOMA_RADIUS,
                )
                existing = mesh_cache.existing_ids()
                prepared_ids = [
                    bid for bid in online_ids if str(bid) in existing
                ]
                online_ids = [
                    bid for bid in online_ids if str(bid) not in existing
                ]
            except Exception:
                prepared_ids, online_ids = [], list(requested_ids)
        if prepared_ids:
            self._vprint(
                f'  🗃️  {len(prepared_ids)} prepared mesh-cache hit(s) '
                f'via CAVE path', use_tqdm=True)
        if online_ids:
            self._vprint(
                f'  🌐 Fetching {len(online_ids)} meshes via CAVE API...',
                use_tqdm=True)
        if not prepared_ids and not online_ids:
            self._vprint(
                f'  🌐 Fetching {len(requested_ids)} meshes via CAVE API...',
                use_tqdm=True)

        # Convert to integers only at the CAVE boundary.
        int_body_ids = [
            body_id_to_api_int(bid) if is_flywire_dataset(self.dataset)
            else int(bid)
            for bid in requested_ids
        ]
        
        fetcher = CAVEDataFetcher(
            dataset=self.dataset,
            project_root=self.script_path,
            # Keep CAVE's per-mesh tqdm out of the render-wide bar.
            verbose=False,
        )
        
        positions = soma_positions or self._fafb_soma_positions(requested_ids)
        mesh_cache = {}
        if cache_prepared and self.cache_neurons:
            neurons = fetcher.fetch_fafb_meshes(
                int_body_ids,
                use_cache=True,
                force_refresh=force_refresh,
                simplify_mesh=self.FAFB_MESH_CACHE_SIMPLIFICATION,
                soma_simplification=self.FAFB_MESH_CACHE_SOMA_SIMPLIFICATION,
                soma_radius=self.FAFB_MESH_CACHE_SOMA_RADIUS,
                soma_positions=positions,
            )
        else:
            # This is the detail-preserving render path.  It intentionally
            # fetches online-only raw meshes and never calls fetch_skeleton.
            neurons = []
            for body_id in int_body_ids:
                mesh = fetcher.fetch_mesh(body_id, use_cache=False)
                if mesh is None:
                    continue
                key = normalize_flywire_body_id(body_id)
                soma_pos = positions.get(key, positions.get(body_id))
                if soma_pos is not None:
                    try:
                        mesh.soma_pos = soma_pos
                    except Exception:
                        pass
                neurons.append(mesh)
        
        for n in neurons:
            if isinstance(n, navis.MeshNeuron) and hasattr(n, 'id'):
                key = (
                    normalize_flywire_body_id(n.id)
                    if is_flywire_dataset(self.dataset)
                    else n.id
                )
                mesh_cache[key] = n

        self._vprint(f'  ✓ Fetched {len(mesh_cache)}/{len(requested_ids)} meshes via API', use_tqdm=True)
        return mesh_cache

    @staticmethod
    def detect_mesh_extrusions(mesh, soma_pos=None, soma_radius=20000, 
                               distance_threshold_factor=3.0, verbose=False) -> dict:
        """Detect extrusion artifacts in a mesh (spikes/protrusions from simplification).
        
        Extrusions typically manifest as:
        1. Vertices with very long edge connections (spiky protrusions)
        2. Isolated vertex clusters far from the main mesh body
        3. Non-manifold geometry issues
        
        Parameters
        ----------
        mesh : trimesh.Trimesh or navis.MeshNeuron
            The mesh to analyze for extrusions.
        soma_pos : array-like, optional
            [x, y, z] position of the soma center. If provided, analysis focuses
            on the soma region where extrusions are most common.
        soma_radius : float
            Radius (in nm) around soma_pos to focus analysis (default 20000nm = 20µm).
        distance_threshold_factor : float
            Vertices with average edge length > (median_edge_length * this_factor) 
            are flagged as potential extrusions (default 3.0).
        verbose : bool
            Whether to print detailed analysis info.
            
        Returns
        -------
        dict
            Analysis results with keys:
            - 'has_extrusions': bool - Whether extrusions were detected
            - 'severity': str - 'none', 'mild', 'moderate', 'severe'
            - 'extrusion_count': int - Number of extrusion vertices detected
            - 'extrusion_vertices': array - Indices of extrusion vertices
            - 'max_edge_length': float - Maximum edge length in mesh
            - 'median_edge_length': float - Median edge length
            - 'edge_length_ratio': float - Ratio of max to median edge length
            - 'soma_region_issues': bool - Whether issues are near soma
            - 'recommendation': str - Suggested action
            
        Example
        -------
        >>> from coana import VisualizeSkeleton
        >>> import trimesh
        >>> # Analyze a mesh
        >>> result = VisualizeSkeleton.detect_mesh_extrusions(mesh, soma_pos=[100, 200, 300])
        >>> if result['has_extrusions']:
        ...     print(f"Extrusions detected! Severity: {result['severity']}")
        ...     print(result['recommendation'])
        """
        import numpy as np
        
        # Extract trimesh object if MeshNeuron
        if hasattr(mesh, 'trimesh'):
            trimesh_obj = mesh.trimesh
        elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            trimesh_obj = mesh
        else:
            return {
                'has_extrusions': False,
                'severity': 'unknown',
                'extrusion_count': 0,
                'extrusion_vertices': np.array([]),
                'max_edge_length': 0,
                'median_edge_length': 0,
                'edge_length_ratio': 0,
                'soma_region_issues': False,
                'recommendation': 'Unable to analyze - invalid mesh format'
            }
        
        vertices = trimesh_obj.vertices
        faces = trimesh_obj.faces
        
        # Calculate edge lengths
        edges = []
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edges.append((min(v1, v2), max(v1, v2)))
        
        edges = list(set(edges))  # Unique edges
        edge_lengths = []
        edge_to_length = {}
        for e in edges:
            length = np.linalg.norm(vertices[e[0]] - vertices[e[1]])
            edge_lengths.append(length)
            edge_to_length[e] = length
        
        edge_lengths = np.array(edge_lengths)
        median_edge = np.median(edge_lengths)
        max_edge = np.max(edge_lengths)
        edge_ratio = max_edge / median_edge if median_edge > 0 else 0
        
        # Calculate per-vertex average edge length
        vertex_edge_lengths = {i: [] for i in range(len(vertices))}
        for e, length in edge_to_length.items():
            vertex_edge_lengths[e[0]].append(length)
            vertex_edge_lengths[e[1]].append(length)
        
        vertex_avg_edge = np.array([
            np.mean(vertex_edge_lengths[i]) if vertex_edge_lengths[i] else 0 
            for i in range(len(vertices))
        ])
        
        # Detect extrusion vertices (abnormally long edges)
        threshold = median_edge * distance_threshold_factor
        extrusion_mask = vertex_avg_edge > threshold
        extrusion_vertices = np.where(extrusion_mask)[0]
        
        # Check if extrusions are near soma region
        soma_region_issues = False
        if soma_pos is not None and len(extrusion_vertices) > 0:
            soma_pos = np.array(soma_pos).flatten()[:3]
            extrusion_positions = vertices[extrusion_vertices]
            distances_to_soma = np.linalg.norm(extrusion_positions - soma_pos, axis=1)
            soma_region_issues = np.any(distances_to_soma <= soma_radius)
        
        # Determine severity
        n_extrusions = len(extrusion_vertices)
        total_vertices = len(vertices)
        extrusion_ratio = n_extrusions / total_vertices if total_vertices > 0 else 0
        
        if n_extrusions == 0:
            severity = 'none'
        elif extrusion_ratio < 0.001 or n_extrusions < 5:
            severity = 'mild'
        elif extrusion_ratio < 0.01 or n_extrusions < 50:
            severity = 'moderate'
        else:
            severity = 'severe'
        
        has_extrusions = n_extrusions > 0 and (edge_ratio > distance_threshold_factor)
        
        # Generate recommendation
        if not has_extrusions:
            recommendation = 'No extrusions detected. Mesh appears clean.'
        elif severity == 'mild':
            recommendation = (
                'Mild extrusions detected. Consider using soma-aware simplification '
                '(soma_mesh_simplification=0.8) or fetch fresh skeleton via CAVE API.'
            )
        elif severity == 'moderate':
            if soma_region_issues:
                recommendation = (
                    'Moderate extrusions near soma region. Strongly recommend using '
                    'VisualizeSkeleton.fix_fafb_extrusions([bodyId]) to fetch fresh '
                    'skeleton from CAVE API.'
                )
            else:
                recommendation = (
                    'Moderate extrusions detected. Use soma-aware simplification or '
                    'VisualizeSkeleton.fix_fafb_extrusions([bodyId]) to fix.'
                )
        else:  # severe
            recommendation = (
                'Severe extrusions detected! This skeleton should be replaced. Use '
                'VisualizeSkeleton.fix_fafb_extrusions([bodyId]) to fetch fresh '
                'skeleton from CAVE API.'
            )
        
        if verbose:
            print(f"Mesh Analysis Results:")
            print(f"  Total vertices: {total_vertices:,}")
            print(f"  Total faces: {len(faces):,}")
            print(f"  Median edge length: {median_edge:.1f}nm")
            print(f"  Max edge length: {max_edge:.1f}nm")
            print(f"  Edge ratio (max/median): {edge_ratio:.1f}x")
            print(f"  Extrusion vertices: {n_extrusions} ({extrusion_ratio*100:.2f}%)")
            print(f"  Severity: {severity}")
            print(f"  Near soma: {soma_region_issues}")
            print(f"  Recommendation: {recommendation}")
        
        return {
            'has_extrusions': has_extrusions,
            'severity': severity,
            'extrusion_count': n_extrusions,
            'extrusion_vertices': extrusion_vertices,
            'max_edge_length': max_edge,
            'median_edge_length': median_edge,
            'edge_length_ratio': edge_ratio,
            'soma_region_issues': soma_region_issues,
            'recommendation': recommendation
        }

    @staticmethod
    def fix_fafb_extrusions(body_ids: list, dataset: str = 'flywire_FAFB_v783', 
                            verbose: bool = True) -> dict:
        """Fix FAFB mesh extrusion issues with a fresh CAVE mesh fetch.
        
        The downloaded FAFB skeleton ZIP (sk_lod1_783_healed.zip) may contain neurons
        with extrusion artifacts (mesh errors appearing as spikes/protrusions). This
        method fetches fresh mesh data from CAVE, applies the visualization
        95% branch / 80% soma preparation, and caches the resulting
        ``MeshNeuron`` objects in the dedicated FlyWire mesh cache. It does
        not skeletonize the CAVE mesh.
        
        Once cached, VisualizeSkeleton will automatically prefer the prepared
        mesh versions over the problematic ZIP versions.
        
        Parameters
        ----------
        body_ids : list
            List of body IDs (as integers or strings) with extrusion issues
        dataset : str
            Dataset name, default 'flywire_FAFB_v783'
        verbose : bool
            Whether to print progress messages
            
        Returns
        -------
        dict
            Dictionary of fixed meshes {bodyId: MeshNeuron}
            
        Example
        -------
        >>> # Fix specific neurons with extrusion issues
        >>> from coana import VisualizeSkeleton
        >>> fixed = VisualizeSkeleton.fix_fafb_extrusions([720575940596125868, 720575940597856265])
        >>> print(f"Fixed {len(fixed)} neurons")
        
        >>> # Now use them in visualization (automatically uses fixed versions)
        >>> vs = VisualizeSkeleton(
        ...     dataset='flywire_FAFB_v783',
        ...     neuron_layers=['l-LNv'],
        ...     force_API_fetching=False,  # API cache still takes priority
        ... )
        >>> vs.plot_neurons()
        """
        require_flywire_skeleton_access(
            dataset,
            project_root=os.path.dirname(os.path.dirname(__file__)),
            log=print if verbose else (lambda _message: None),
        )

        try:
            from cave_data_fetcher import CAVEDataFetcher
        except ImportError:
            try:
                from .cave_data_fetcher import CAVEDataFetcher
            except ImportError:
                raise ImportError(
                    "CAVE data fetcher not available. Install with: "
                    "pip install caveclient cloud-volume"
                )
        
        if verbose:
            print(f"🔧 Fixing FAFB mesh extrusions for {len(body_ids)} neurons...")
        
        requested_ids = (
            normalize_flywire_body_ids(body_ids)
            if is_flywire_dataset(dataset)
            else list(body_ids)
        )
        api_body_ids = [
            body_id_to_api_int(bid) if is_flywire_dataset(dataset)
            else int(bid)
            for bid in requested_ids
        ]
        
        fetcher = CAVEDataFetcher(
            dataset=dataset,
            project_root=os.path.dirname(os.path.dirname(__file__)),
            verbose=verbose,
        )

        # FAFB/CAVE is mesh-native.  Repair fetches the prepared mesh at the
        # same 95% branch / 80% soma policy as visualization and writes only
        # the dedicated FlyWire mesh cache; it never skeletonizes the mesh.
        neurons = fetcher.fetch_fafb_meshes(
            api_body_ids,
            use_cache=True,
            force_refresh=True,
            simplify_mesh=FLYWIRE_MESH_CACHE_SIMPLIFICATION,
            soma_simplification=FLYWIRE_MESH_CACHE_SOMA_SIMPLIFICATION,
            soma_radius=FLYWIRE_MESH_CACHE_SOMA_RADIUS,
        )
        
        result = {}
        for n in neurons:
            if hasattr(n, 'id'):
                key = (
                    normalize_flywire_body_id(n.id)
                    if is_flywire_dataset(dataset)
                    else n.id
                )
                result[key] = n
        
        if verbose:
            print(f"✓ Fixed and cached {len(neurons)}/{len(body_ids)} FAFB meshes")
            print(
                "  Cache location: "
                f"cache/{dataset.replace(':', '_').replace('.', '_')}"
                "/meshes/FLYWIRE_simp95_soma80_r20/"
            )
            print("  These meshes will automatically be preferred over ZIP data")
        
        return result

    @staticmethod
    def check_fafb_skeleton_for_extrusions(body_id, dataset: str = 'flywire_FAFB_v783',
                                           simplification: float = 0.95,
                                           verbose: bool = True,
                                           auto_fix: bool = False) -> dict:
        """Load a FAFB skeleton from ZIP and check for extrusion artifacts.
        
        This is a convenience method that combines loading a skeleton from the local
        ZIP file, applying simplification, and running extrusion detection.
        
        Parameters
        ----------
        body_id : int or str
            The body ID to check.
        dataset : str
            Dataset name, default 'flywire_FAFB_v783'.
        simplification : float
            Simplification level to apply before checking (default 0.95).
        verbose : bool
            Whether to print detailed analysis results.
        auto_fix : bool
            If True and extrusions are detected, automatically fetch a fresh
            prepared mesh from CAVE API (default False).
            
        Returns
        -------
        dict
            Analysis results with additional keys:
            - All keys from detect_mesh_extrusions()
            - 'body_id': The body ID that was checked
            - 'auto_fixed': bool - Whether auto-fix was applied
            - 'skeleton': The loaded render mesh (MeshNeuron)
            
        Example
        -------
        >>> from coana import VisualizeSkeleton
        >>> # Check a specific neuron
        >>> result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
        ...     720575940624086675, 
        ...     simplification=0.95,
        ...     verbose=True
        ... )
        >>> if result['has_extrusions']:
        ...     # Auto-fix it
        ...     VisualizeSkeleton.fix_fafb_extrusions([720575940624086675])
        
        >>> # Or use auto_fix=True
        >>> result = VisualizeSkeleton.check_fafb_skeleton_for_extrusions(
        ...     720575940624086675,
        ...     auto_fix=True  # Automatically fetch from API if extrusions found
        ... )
        """
        import navis
        from pathlib import Path
        import zipfile
        import re
        import numpy as np
        
        body_id_str = str(body_id)
        body_id_int = int(body_id)
        
        if verbose:
            print(f"🔍 Checking FAFB skeleton {body_id} for extrusions...")
        
        # The healed bundle reads .zst first; the ZIP is the fallback
        # with lazy per-skeleton conversion.
        dataset_clean = dataset.replace(':', '_').replace('.', '_')
        skeleton = None
        soma_pos = None
        try:
            import fafb_utils
            _bundle = fafb_utils.get_fafb_skeleton_bundle(
                Path('datasets') / dataset_clean)
        except Exception:
            _bundle = None
        if _bundle is not None:
            try:
                _content = _bundle.get(body_id_int)
            finally:
                try:
                    _bundle.close()
                except Exception:
                    pass
            if _content is not None:
                from io import StringIO
                skeleton = navis.read_swc(StringIO(_content))
                if isinstance(skeleton, navis.NeuronList):
                    skeleton = skeleton[0]
                skeleton.id = body_id_int
                # Get soma position if available
                if hasattr(skeleton, 'soma') and skeleton.soma is not None:
                    soma_idx = skeleton.soma
                    if isinstance(soma_idx, (list, np.ndarray)) and len(soma_idx) > 0:
                        soma_idx = soma_idx[0]
                    if soma_idx is not None:
                        soma_node = skeleton.nodes[skeleton.nodes['node_id'] == soma_idx]
                        if len(soma_node) > 0:
                            soma_pos = soma_node[['x', 'y', 'z']].values[0]
                if verbose:
                    print(f"  ✓ Loaded skeleton with {len(skeleton.nodes):,} nodes")

        zip_path = Path('datasets') / dataset_clean / 'sk_lod1_783_healed.zip'

        if not zip_path.exists() and skeleton is None:
            return {
                'has_extrusions': False,
                'severity': 'unknown',
                'extrusion_count': 0,
                'extrusion_vertices': np.array([]),
                'max_edge_length': 0,
                'median_edge_length': 0,
                'edge_length_ratio': 0,
                'soma_region_issues': False,
                'recommendation': f'ZIP file not found: {zip_path}',
                'body_id': body_id_int,
                'auto_fixed': False,
                'skeleton': None
            }

        # Find matching file in ZIP (fallback: bundle had no entry)
        if skeleton is None:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Find file matching this body ID
                    pattern = re.compile(rf'^{body_id_str}\..*\.swc$', re.IGNORECASE)
                    matching_files = [f for f in zf.namelist() if pattern.match(f)]
                
                    if not matching_files:
                        # Try without extension pattern
                        matching_files = [f for f in zf.namelist() if f.startswith(f'{body_id_str}.')]
                
                    if not matching_files:
                        return {
                            'has_extrusions': False,
                            'severity': 'unknown',
                            'extrusion_count': 0,
                            'extrusion_vertices': np.array([]),
                            'max_edge_length': 0,
                            'median_edge_length': 0,
                            'edge_length_ratio': 0,
                            'soma_region_issues': False,
                            'recommendation': f'Body ID {body_id} not found in ZIP',
                            'body_id': body_id_int,
                            'auto_fixed': False,
                            'skeleton': None
                        }
                
                    swc_filename = matching_files[0]
                
                    # Read SWC content
                    with zf.open(swc_filename) as swc_file:
                        swc_content = swc_file.read().decode('utf-8')
                
                    # Parse with navis
                    from io import StringIO
                    skeleton = navis.read_swc(StringIO(swc_content))
                
                    if isinstance(skeleton, navis.NeuronList):
                        skeleton = skeleton[0]
                
                    skeleton.id = body_id_int
                
                    # Get soma position if available
                    if hasattr(skeleton, 'soma') and skeleton.soma is not None:
                        soma_idx = skeleton.soma
                        if isinstance(soma_idx, (list, np.ndarray)) and len(soma_idx) > 0:
                            soma_idx = soma_idx[0]
                        if soma_idx is not None:
                            soma_node = skeleton.nodes[skeleton.nodes['node_id'] == soma_idx]
                            if len(soma_node) > 0:
                                soma_pos = soma_node[['x', 'y', 'z']].values[0]
                
                    if verbose:
                        print(f"  ✓ Loaded skeleton with {len(skeleton.nodes):,} nodes")
                        if soma_pos is not None:
                            print(f"  ✓ Soma position: [{soma_pos[0]:.0f}, {soma_pos[1]:.0f}, {soma_pos[2]:.0f}]")
            except Exception as e:
                return {
                    'has_extrusions': False,
                    'severity': 'unknown',
                    'extrusion_count': 0,
                    'extrusion_vertices': np.array([]),
                    'max_edge_length': 0,
                    'median_edge_length': 0,
                    'edge_length_ratio': 0,
                    'soma_region_issues': False,
                    'recommendation': f'Error loading skeleton: {e}',
                    'body_id': body_id_int,
                    'auto_fixed': False,
                    'skeleton': None
                }

        # Convert to mesh with simplification
        if verbose:
            print(f"  ⚙️  Converting to mesh (simplification={simplification})...")
        
        try:
            mesh_neuron = navis.conversion.tree2meshneuron(
                skeleton,
                tube_points=VisualizeSkeleton.SKELETON_TUBE_POINTS,
                use_normals=True
            )
            
            if hasattr(mesh_neuron, 'trimesh') and simplification > 0:
                import trimesh
                original_faces = len(mesh_neuron.trimesh.faces)
                target_faces = max(100, int(original_faces * (1 - simplification)))
                
                # Simplify using quadric decimation
                try:
                    import open3d as o3d
                    o3d_mesh = o3d.geometry.TriangleMesh()
                    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_neuron.trimesh.vertices)
                    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_neuron.trimesh.faces)
                    simplified = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
                    simplified_trimesh = trimesh.Trimesh(
                        vertices=np.asarray(simplified.vertices),
                        faces=np.asarray(simplified.triangles)
                    )
                    mesh_neuron._trimesh = simplified_trimesh
                except ImportError:
                    # Fallback to trimesh simplification
                    mesh_neuron._trimesh = mesh_neuron.trimesh.simplify_quadric_decimation(target_faces)
                
                if verbose:
                    print(f"  ✓ Simplified: {original_faces:,} → {len(mesh_neuron.trimesh.faces):,} faces")
        
        except Exception as e:
            return {
                'has_extrusions': False,
                'severity': 'unknown',
                'extrusion_count': 0,
                'extrusion_vertices': np.array([]),
                'max_edge_length': 0,
                'median_edge_length': 0,
                'edge_length_ratio': 0,
                'soma_region_issues': False,
                'recommendation': f'Error converting to mesh: {e}',
                'body_id': body_id_int,
                'auto_fixed': False,
                'skeleton': skeleton
            }
        
        # Run extrusion detection
        result = VisualizeSkeleton.detect_mesh_extrusions(
            mesh_neuron,
            soma_pos=soma_pos,
            soma_radius=20000,
            verbose=verbose
        )
        
        result['body_id'] = body_id_int
        result['auto_fixed'] = False
        result['skeleton'] = mesh_neuron
        
        # Auto-fix if requested
        if auto_fix and result['has_extrusions']:
            if verbose:
                print(f"\n  🔧 Auto-fixing: Fetching fresh mesh from CAVE API...")
            try:
                fixed = VisualizeSkeleton.fix_fafb_extrusions([body_id_int], dataset=dataset, verbose=verbose)
                if body_id_int in fixed or body_id_str in fixed:
                    result['auto_fixed'] = True
                    result['skeleton'] = fixed.get(body_id_int) or fixed.get(body_id_str)
                    result['recommendation'] = 'Extrusions fixed! Fresh mesh fetched and cached from CAVE API.'
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  Auto-fix failed: {e}")
                result['recommendation'] += f' Auto-fix failed: {e}'
        
        return result

    def _process_fafb_layer(self, neuron_vols, cached_mesh_neurons,
                            fafb_pipeline, use_fafb_cache,
                            render_mesh_cache=None,
                            progress_callback=None):
        """Pipeline-driven FAFB processing for one render layer.

        TreeNeuron sources never write the prepared mesh cache (only CAVE
        fetches populate it):
        - fast: TreeNeuron sources run the node-reduction stage (25%
          retention), then tube meshing + fine surface decimation;
          MeshNeuron sources (prepared cache / CAVE) skip the node stage
          and receive surface decimation only.  CAVE meshes are never
          skeletonized for tube rendering.
        - fine: soma-aware decimation as before, no node stage.
        - artistic: vertex clustering, no node stage.
        - line: TreeNeuron sources are node-reduced 90%; MeshNeuron
          sources are skeletonized in memory at the render boundary and
          never written to any cache.

        Returns ``(processed_neuron_vols, already_simplified)`` so the
        caller can skip the generic tube simplification block.

        ``progress_callback`` is an optional callable receiving
        ``(neuron_id, stage, done)``.  It lets the caller put the per-source
        stage and completion count on its shared render progress bar without
        creating a nested tqdm instance.
        """
        if render_mesh_cache is None:
            render_mesh_cache = {}

        def report(neuron_id, stage, done=False):
            if progress_callback is not None:
                progress_callback(neuron_id, stage, done)

        def stage_log(message):
            """Keep stage details in the bar when a shared bar is active."""
            if progress_callback is None:
                self._vprint(message, level='full', use_tqdm=True)

        if self.skeleton_mode == 'tube':
            all_mesh_neurons = []
            fafb_target_simp = self.skeleton_mesh_simplification
            fafb_soma_simp = self.soma_mesh_simplification
            use_soma_aware = (
                fafb_soma_simp is not None
                and fafb_soma_simp != fafb_target_simp
            )
            decimator = (
                self._simplify_mesh_vertex_clustering
                if fafb_pipeline == 'artistic'
                else self._simplify_mesh_fafb_fine
            )
            # Only the fast pipeline runs the node-reduction stage;
            # artistic never combines with it.
            node_stage = (fafb_pipeline == 'fast')

            # --- MeshNeuron sources: surface decimation only.
            for mesh_n in cached_mesh_neurons:
                neuron_id = getattr(mesh_n, 'id', None)
                try:
                    render_key = normalize_flywire_body_id(neuron_id)
                except (TypeError, ValueError):
                    render_key = str(neuron_id)
                if render_key in render_mesh_cache:
                    report(neuron_id, 'cached', done=True)
                    all_mesh_neurons.append(render_mesh_cache[render_key])
                    continue
                report(neuron_id, 'mesh source')
                stage_log(f'  🧱 mesh source (no node stage) {neuron_id}')
                if getattr(mesh_n, 'trimesh', None) is not None:
                    n_faces = len(mesh_n.trimesh.faces)
                    if use_fafb_cache:
                        # Prepared-level meshes (cache hits and
                        # prepared CAVE fetches) need only the
                        # additional relative step to the target.
                        cache_simp = self.FAFB_MESH_CACHE_SIMPLIFICATION
                        keep = max(
                            0.0,
                            (1 - fafb_target_simp)
                            / max(1 - cache_simp, 1e-6),
                        )
                        target_faces = max(100, int(n_faces * keep))
                    else:
                        # Raw CAVE meshes decimate to the absolute
                        # render target.
                        target_faces = max(
                            100, int(n_faces * (1 - fafb_target_simp)))
                    if target_faces < n_faces:
                        try:
                            report(neuron_id, 'decimate')
                            simplified = decimator(
                                mesh_n.trimesh, target_faces)
                            stage_log(f'  ⚡ decimate {neuron_id}')
                            new_mesh = navis.MeshNeuron(simplified)
                            new_mesh.id = (
                                mesh_n.id
                                if hasattr(mesh_n, 'id') else None)
                            if hasattr(mesh_n, 'name'):
                                new_mesh.name = mesh_n.name
                            render_mesh_cache[render_key] = new_mesh
                            all_mesh_neurons.append(new_mesh)
                            report(neuron_id, 'ready', done=True)
                            continue
                        except Exception as e:
                            self._vprint(
                                f'  ⚠️ decimation failed for '
                                f'{neuron_id}: {e}', level='full',
                                use_tqdm=True)
                render_mesh_cache[render_key] = mesh_n
                all_mesh_neurons.append(mesh_n)
                report(neuron_id, 'ready', done=True)

            # --- TreeNeuron sources (healed ZIP / raw SWC cache).
            if neuron_vols is not None and len(neuron_vols) > 0:
                neurons_list = (
                    list(neuron_vols)
                    if isinstance(neuron_vols, navis.NeuronList)
                    else [neuron_vols])
                for n in neurons_list:
                    neuron_id = getattr(n, 'id', None)
                    try:
                        render_key = normalize_flywire_body_id(neuron_id)
                    except (TypeError, ValueError):
                        render_key = str(neuron_id)
                    if render_key in render_mesh_cache:
                        report(neuron_id, 'cached', done=True)
                        all_mesh_neurons.append(
                            render_mesh_cache[render_key])
                        continue

                    soma_pos = None
                    if (hasattr(n, 'soma_pos')
                            and n.soma_pos is not None):
                        soma_pos = n.soma_pos

                    if isinstance(n, navis.TreeNeuron):
                        work = n
                        if node_stage:
                            # Node reduction operates on a copy; the
                            # raw SWC source stays canonical.  The
                            # constant is the retention (25%), the
                            # module parameter is the reduction.
                            try:
                                report(neuron_id, 'reduce nodes')
                                work, stats = simplify_skeleton_nodes(
                                    n,
                                    1.0 - FAFB_FAST_NODE_RETENTION)
                                stage_log(
                                    f'  🔽 reduce nodes {neuron_id} '
                                    f'{stats["raw_nodes"]}→'
                                    f'{stats["achieved_nodes"]}')
                            except Exception as e:
                                self._vprint(
                                    f'  ⚠️ node reduction failed for '
                                    f'{neuron_id}: {e}', level='full',
                                    use_tqdm=True)
                        # Fix radii on the working copy, never on the
                        # canonical source.
                        if (hasattr(work, 'nodes')
                                and 'radius' in work.nodes.columns):
                            invalid_mask = (
                                (work.nodes['radius'] <= 0)
                                | (work.nodes['radius'].isna()))
                            if invalid_mask.any():
                                work.nodes.loc[
                                    invalid_mask, 'radius'] = 1
                        elif hasattr(work, 'nodes'):
                            work.nodes['radius'] = 1
                        if (hasattr(navis, 'conversion')
                                and hasattr(navis.conversion,
                                            'tree2meshneuron')):
                            report(neuron_id, 'tube mesh')
                            mesh_n = navis.conversion.tree2meshneuron(
                                work,
                                tube_points=self.SKELETON_TUBE_POINTS,
                            )
                            stage_log(f'  🧪 tube mesh {neuron_id}')
                        else:
                            report(neuron_id, 'ready', done=True)
                            all_mesh_neurons.append(n)
                            continue
                    elif isinstance(n, navis.MeshNeuron):
                        report(neuron_id, 'mesh source')
                        mesh_n = n
                    else:
                        report(neuron_id, 'ready', done=True)
                        all_mesh_neurons.append(n)
                        continue

                    if getattr(mesh_n, 'trimesh', None) is not None:
                        n_faces = len(mesh_n.trimesh.faces)
                        try:
                            if (fafb_pipeline in {'fine', 'artistic'}
                                    and use_soma_aware
                                    and soma_pos is not None):
                                report(neuron_id, 'decimate')
                                simplified = self._simplify_mesh_with_soma_awareness(
                                    mesh_n.trimesh,
                                    skeleton_simp=fafb_target_simp,
                                    soma_simp=fafb_soma_simp,
                                    soma_pos=soma_pos,
                                    soma_radius=self.soma_region_radius,
                                    decimator=decimator,
                                )
                            else:
                                target_faces = max(
                                    100,
                                    int(n_faces * (1 - fafb_target_simp)))
                                report(neuron_id, 'decimate')
                                simplified = decimator(
                                    mesh_n.trimesh, target_faces)
                            stage_log(f'  ⚡ decimate {neuron_id}')
                        except Exception as e:
                            self._vprint(
                                f'  ⚠️ decimation failed for '
                                f'{neuron_id}: {e}', level='full',
                                use_tqdm=True)
                            simplified = mesh_n.trimesh
                        mesh_n = navis.MeshNeuron(simplified)
                        mesh_n.id = n.id if hasattr(n, 'id') else None
                        if hasattr(n, 'name'):
                            mesh_n.name = n.name
                    render_mesh_cache[render_key] = mesh_n
                    all_mesh_neurons.append(mesh_n)
                    report(neuron_id, 'ready', done=True)

            if all_mesh_neurons:
                neuron_vols = navis.NeuronList(all_mesh_neurons)
            # Mark FAFB as already simplified to skip generic
            # simplification below.
            return neuron_vols, True

        if self.skeleton_mode == 'line':
            # FAFB line mode: no prepared mesh cache, no tube/mesh
            # stages.  TreeNeuron sources are node-reduced 90%;
            # MeshNeuron sources are skeletonized in memory at the
            # render boundary and never written to any cache.
            prepared_lines = []
            if neuron_vols is not None and len(neuron_vols) > 0:
                neurons_list = (
                    list(neuron_vols)
                    if isinstance(neuron_vols, navis.NeuronList)
                    else [neuron_vols])
                for n in neurons_list:
                    neuron_id = getattr(n, 'id', None)
                    if isinstance(n, navis.TreeNeuron):
                        try:
                            report(neuron_id, 'reduce nodes')
                            work, stats = simplify_skeleton_nodes(
                                n, FAFB_LINE_NODE_REDUCTION)
                            stage_log(
                                f'  🔽 reduce nodes {neuron_id} '
                                f'{stats["raw_nodes"]}→'
                                f'{stats["achieved_nodes"]}')
                            prepared_lines.append(work)
                        except Exception as e:
                            self._vprint(
                                f'  ⚠️ line node reduction failed for '
                                f'{neuron_id}: {e}', level='full',
                                use_tqdm=True)
                            prepared_lines.append(n)
                        report(neuron_id, 'ready', done=True)
                    else:
                        prepared_lines.append(n)
                        report(neuron_id, 'ready', done=True)
            for mesh_n in cached_mesh_neurons:
                neuron_id = getattr(mesh_n, 'id', None)
                report(neuron_id, 'skeletonize')
                stage_log(
                    f'  🦴 skeletonize mesh source {neuron_id} '
                    f'(line mode, in-memory)')
                try:
                    sk = navis.skeletonize(mesh_n)
                    work, stats = simplify_skeleton_nodes(
                        sk, FAFB_LINE_NODE_REDUCTION)
                    stage_log(
                        f'  🔽 reduce nodes {neuron_id} '
                        f'{stats["raw_nodes"]}→'
                        f'{stats["achieved_nodes"]}')
                    prepared_lines.append(work)
                except Exception as e:
                    self._vprint(
                        f'  ⚠️ skeletonize failed for {neuron_id}: {e}',
                        level='full', use_tqdm=True)
                report(neuron_id, 'ready', done=True)
            if prepared_lines:
                neuron_vols = navis.NeuronList(prepared_lines)
            return neuron_vols, False

        return neuron_vols, False

    def plot_skeleton(self):
        from tqdm import tqdm
        import sys

        # Validate access at the script's skeleton-query boundary.  This
        # blocks BANC and gives FAFB users the local-preparation/token
        # instructions before any CAVE or ZIP fetch is attempted.
        flywire_access = require_flywire_skeleton_access(
            self.dataset,
            project_root=self.script_path,
            log=self._vprint,
        )
        self._flywire_skeleton_access = flywire_access
        if flywire_access.get('is_fafb') and flywire_access.get('cave_token'):
            self._vprint(
                '  🌐 FAFB CAVE fallback is enabled; local cache/ZIP data is '
                'preferred, and missing or extrusion-affected skeletons will '
                'be requested from CAVE.',
                level='simple',
            )
        
        n_layers = len(self.neuron_layers)
        total_skeletons = sum(len(df) if df is not None else 0 for df in self.neuron_dfs)
        self._vprint(f'\n🔬 Fetching skeletons for {n_layers} layers ({total_skeletons:,} neurons total)...')
        
        # For FAFB: Check mesh cache first (transformed + meshed neurons)
        # Cache stores pre-simplified meshes at FAFB_MESH_CACHE_SIMPLIFICATION
        # (0.95 = keep 5% of faces).
        # 
        # Cache usage decision:
        # - If user wants simplification >= 0.95 (keep ≤5% faces): use cache, apply additional simplification if needed
        # - If user wants simplification < 0.95 (keep >5% faces): bypass cache, load from ZIP and apply user's simplification
        #
        # Example scenarios:
        # - simplification=0.98 (keep 2%): load from cache (5%), simplify to 2% → additional_keep = 0.02/0.05 = 40%
        # - simplification=0.95 (keep 5%): load from cache (5%), no additional simplification needed
        # - simplification=0.5 (keep 50%): cannot use cache (only has 10%), load from ZIP and apply 0.5 simplification
        is_fafb = is_flywire_dataset(self.dataset)
        neuprint_pipeline = self._resolved_neuprint_skeleton_pipeline()
        use_neuprint_fine_pipeline = (
            not is_fafb
            and self.skeleton_mode == 'tube'
            and neuprint_pipeline in {'fine', 'artistic'}
        )
        use_neuprint_fine_optimized_pipeline = (
            use_neuprint_fine_pipeline
            and neuprint_pipeline in {'fine', 'artistic'}
        )
        fafb_pipeline = self._resolved_fafb_pipeline() if is_fafb else None

        # FAFB prepared mesh cache eligibility: caching enabled, tube mode,
        # and the render target at/above the prepared cache level (0.95).
        # Line mode disables the prepared mesh cache entirely.
        use_fafb_cache = (
            is_fafb and self.skeleton_mode == 'tube' and self.cache_neurons
            and self.skeleton_mesh_simplification
            >= self.FAFB_MESH_CACHE_SIMPLIFICATION
        )

        # Check for force_API_fetching - bypasses ZIP loading for FAFB
        use_api_fetching = is_fafb and self.force_API_fetching

        # FAFB source resolution: SWC-first for every render mode.
        # skeleton_cache holds TreeNeuron sources (ZIP / raw SWC cache),
        # mesh_cache holds MeshNeuron sources (prepared cache / CAVE online).
        fafb_sources = {}           # canonical bodyId -> source name
        fafb_skeleton_cache = {}    # canonical bodyId -> TreeNeuron
        fafb_mesh_cache = {}        # canonical bodyId -> MeshNeuron
        # Render-time mesh products are reused when a body appears in more
        # than one layer.  This cache is intentionally in-memory only and is
        # scoped to one visualization run.
        fafb_render_mesh_cache = {}  # canonical bodyId -> MeshNeuron
        if is_fafb:
            # Collect all body IDs first
            all_fafb_body_ids = []
            for df in self.neuron_dfs:
                if df is not None and 'bodyId' in df.columns:
                    all_fafb_body_ids.extend(
                        normalize_flywire_body_ids(df['bodyId'].tolist())
                        if is_flywire_dataset(self.dataset)
                        else [int(body_id) for body_id in df['bodyId'].tolist()]
                    )
            all_fafb_body_ids = list(set(all_fafb_body_ids))

            if use_api_fetching:
                self._vprint(f'  ℹ️  force_API_fetching=True: Using CAVE API instead of ZIP', level='simple')
            elif use_fafb_cache:
                self._vprint(f'  ℹ️  FAFB prepared mesh cache eligible (simplification={self.skeleton_mesh_simplification} >= cache level {self.FAFB_MESH_CACHE_SIMPLIFICATION})', level='full')
            else:
                self._vprint(f'  ℹ️  FAFB prepared mesh cache bypassed (SWC-first sources only)', level='full')

            fafb_sources, fafb_skeleton_cache, fafb_mesh_cache = (
                self._resolve_fafb_sources(
                    all_fafb_body_ids,
                    allow_mesh_cache=use_fafb_cache,
                    api_only=use_api_fetching,
                )
            )

        
        # NeuPrint fine/artistic rendering never loads or writes a transformed
        # mesh cache. It starts from the shared raw ``.swc.zst`` source when
        # available and applies the FAFB transform, tube conversion, and
        # requested simplification in the aggregate preprocessing phase below.
        neuprint_mesh_cache = {}
        use_neuprint_mesh_cache = False

        # NeuPrint has one preprocessing boundary for the complete render.
        # The layer loop below only selects from this bodyId-keyed mapping;
        # it never issues a layer-scoped online NeuPrint fetch.
        neuprint_prepared_skeletons = {}
        neuprint_prepared_mesh = False
        neuprint_preprocessing_active = (
            not is_fafb and self.client_type != 'flywire')
        if neuprint_preprocessing_active:
            neuprint_prepared_skeletons, neuprint_prepared_mesh = (
                self._prepare_neuprint_skeletons_for_render(
                    self._aggregate_neuprint_body_ids(),
                    use_neuprint_fine_pipeline,
                    use_neuprint_mesh_cache,
                    neuprint_mesh_cache,
                )
            )
            # The complete render cache has already been resolved above.
            # Disable the old per-layer mesh-cache branch so it cannot append
            # the prepared mesh a second time or decimate it twice.
            use_neuprint_mesh_cache = False
            neuprint_mesh_cache = {}

        
        
        # Note: For legend_mode='type', neurons keep their layer colors but get separate legend entries
        # Per-neuron colors from CSV are stored in self._neuron_color_overrides (bodyId -> color)
        self._type_color_map = {}  # Not used anymore - types keep layer colors
        # Pre/post site traces are appended after morphology traces.  These
        # maps let their legend labels and ranks mirror the exact skeleton
        # grouping chosen above instead of relying on connector-row order.
        self._neuron_legend_labels_by_layer = {}
        self._neuron_legend_ranks = {}
        
        # One render-wide bar is more useful than a layer counter: a single
        # layer can contain hundreds of neurons and otherwise remains at 0%
        # while its mesh stages run.  FAFB processing reports each source
        # through this bar; other datasets complete their layer's count once
        # the already-aggregated sources have been plotted.
        layer_pbar = tqdm(
            total=total_skeletons,
            desc="Processing skeletons",
            unit="neuron",
            disable=not self.verbose,
            leave=True,
            file=sys.stdout,
        )
        for i in range(n_layers):
            layer_name = self.layer_names[i] if i < len(self.layer_names) else f"layer_{i}"
            n_in_layer = len(self.neuron_dfs[i]) if self.neuron_dfs[i] is not None else 0
            layer_pbar.set_postfix_str(f"{layer_name} (0/{n_in_layer})")
            processed_this_layer = 0

            def report_fafb_progress(neuron_id, stage, done=False):
                """Refresh the single render bar for one FAFB source.

                Stage names remain available to callers/tests, but they are
                intentionally not printed.  The render bar is the only
                progress row and shows the global ``n/N`` plus the current
                layer-local ``n/N`` in its postfix.
                """
                nonlocal processed_this_layer
                if done and processed_this_layer < n_in_layer:
                    layer_pbar.update(1)
                    processed_this_layer += 1
                layer_pbar.set_postfix_str(
                    f"{layer_name} ({processed_this_layer}/{n_in_layer})"
                )
            
            # Determine if we need transformation
            # Use _needs_skeleton_transform() which checks for skip_transform flag (FAFB uses native coords)
            needs_transform = self._needs_skeleton_transform()
            template_info = None
            if self.brain_mesh in ['whole', 'template']:
                template_info = self._get_template_info()

            # Overlay layers (custom_neurons) are already in this scene's
            # coordinate space: render as-is, skip every cache/fetch branch.
            # Guard with getattr so hermetic test objects built without
            # __post_init__ (no overlay map) fall through to the normal path.
            custom_layer_neurons = getattr(
                self, '_custom_neurons_by_layer', {}).get(layer_name)
            is_custom_layer = custom_layer_neurons is not None
            if is_custom_layer:
                needs_transform = False

            neuron_vols = None
            
            # For FAFB: split the layer into MeshNeuron sources (prepared
            # cache / CAVE online, resolved up front) and the remaining IDs
            # that resolve to TreeNeuron sources.
            layer_body_ids = self.neuron_dfs[i]['bodyId'].tolist() if self.neuron_dfs[i] is not None else []
            layer_body_ids = (
                normalize_flywire_body_ids(layer_body_ids)
                if is_flywire_dataset(self.dataset)
                else [int(body_id) for body_id in layer_body_ids]
            )
            neuprint_layer_prepared = (
                neuprint_preprocessing_active
                and not is_custom_layer
                and all(body_id in neuprint_prepared_skeletons
                        for body_id in layer_body_ids)
            )
            cached_mesh_neurons = []  # MeshNeurons loaded from cache
            mesh_missing_ids = layer_body_ids  # IDs that need processing
            
            if is_fafb and fafb_mesh_cache and not is_custom_layer:
                # Separate mesh sources vs TreeNeuron sources with
                # type-robust matching
                cached_mesh_neurons = []
                mesh_missing_ids = []
                for bid in layer_body_ids:
                    # Check both int and str versions of the ID
                    if bid in fafb_mesh_cache:
                        cached_mesh_neurons.append(fafb_mesh_cache[bid])
                    elif str(bid) in fafb_mesh_cache:
                        cached_mesh_neurons.append(fafb_mesh_cache[str(bid)])
                    elif isinstance(bid, str) and bid.isdigit() and int(bid) in fafb_mesh_cache:
                        cached_mesh_neurons.append(fafb_mesh_cache[int(bid)])
                    else:
                        mesh_missing_ids.append(bid)
                
                if cached_mesh_neurons:
                    self._vprint(f'    ✓ {len(cached_mesh_neurons)}/{len(layer_body_ids)} from mesh cache', level='full', use_tqdm=True)
            
            # NeuPrint tube mode: split cached transformed meshes vs missing
            if use_neuprint_mesh_cache and neuprint_mesh_cache and not is_custom_layer:
                cached_mesh_neurons = []
                mesh_missing_ids = []
                for bid in layer_body_ids:
                    if bid in neuprint_mesh_cache:
                        cached_mesh_neurons.append(neuprint_mesh_cache[bid])
                    elif str(bid) in neuprint_mesh_cache:
                        cached_mesh_neurons.append(neuprint_mesh_cache[str(bid)])
                    elif isinstance(bid, str) and bid.isdigit() and int(bid) in neuprint_mesh_cache:
                        cached_mesh_neurons.append(neuprint_mesh_cache[int(bid)])
                    else:
                        mesh_missing_ids.append(bid)
                if cached_mesh_neurons:
                    self._vprint(f'    ✓ {len(cached_mesh_neurons)}/{len(layer_body_ids)} from NeuPrint mesh cache', level='full', use_tqdm=True)
            
            # Every NeuPrint render starts from the same raw SWC cache. The
            # selected pipeline and mesh simplification are render-time
            # choices, never cache-level selectors.
            ignore_skeleton_cache = False
            if neuprint_preprocessing_active or is_custom_layer:
                # The aggregate preprocessing phase owns all NeuPrint cache
                # resolution. Do not re-open the fixed skeleton cache here.
                # Overlay layers bring their own neurons; nothing to resolve.
                cached_neurons, missing_ids = None, []
            else:
                cached_neurons, missing_ids = self._load_cached_neurons(
                    self.neuron_dfs[i], ignore_cache=ignore_skeleton_cache)

            # Check and fix MANC scaling for cached neurons (compatibility for
            # non-aggregate callers; aggregate NeuPrint rendering scales the
            # in-memory raw source once in _prepare_neuprint_skeletons_for_render).
            if cached_neurons is not None and 'manc' in self.dataset.lower():
                try:
                    # Check first neuron
                    bbox = cached_neurons[0].bbox if hasattr(cached_neurons[0], 'bbox') else None
                    if bbox is not None and np.max(bbox) < 150000:
                        cached_neurons = cached_neurons * 8
                        self._vprint(f'  ℹ️  Applied 8x scaling to cached MANC skeletons', level='full')
                except Exception as e:
                    self._vprint(f'  ⚠️  Failed to check/apply scaling to cache: {e}', level='full')
            
            raw_neuron_vols = None
            if neuprint_layer_prepared:
                raw_neuron_vols = navis.NeuronList(
                    [neuprint_prepared_skeletons[body_id]
                     for body_id in layer_body_ids])

            # Overlay layer: inject the caller-provided neurons directly.
            if is_custom_layer:
                raw_neuron_vols = navis.NeuronList(custom_layer_neurons)

            # Fetch missing neurons (only those not in the mesh cache when
            # the FAFB or NeuPrint mesh cache is in use)
            fetch_ids = [] if (neuprint_preprocessing_active or is_custom_layer) else (
                mesh_missing_ids if (is_fafb or use_neuprint_mesh_cache)
                else missing_ids)
            remaining_fetch_ids = list(fetch_ids)
            if fetch_ids:
                # Special handling for FAFB local data - use pre-loaded cache
                if fafb_skeleton_cache:
                    neurons = []
                    for bid in fetch_ids:
                        # Handle both int and string types for body ID lookup
                        if bid in fafb_skeleton_cache:
                            neurons.append(fafb_skeleton_cache[bid])
                        elif str(bid) in fafb_skeleton_cache:
                            neurons.append(fafb_skeleton_cache[str(bid)])
                        elif isinstance(bid, str) and bid.isdigit() and int(bid) in fafb_skeleton_cache:
                            neurons.append(fafb_skeleton_cache[int(bid)])
                    if neurons:
                        raw_neuron_vols = navis.NeuronList(neurons)

                # Fetch from API if not loaded locally
                if (raw_neuron_vols is None and remaining_fetch_ids
                        and not neuprint_preprocessing_active):
                    if self.client_type == 'flywire' and self.client_flywire:
                        missing_df = self.neuron_dfs[i][self.neuron_dfs[i]['bodyId'].isin(remaining_fetch_ids)]
                        # Retry logic for network errors
                        max_retries = 5
                        for attempt in range(max_retries):
                            try:
                                layer_pbar.set_postfix_str(
                                    f"{layer_name} (fetching {len(remaining_fetch_ids)} skeletons...)")
                                # navis prints its own "Fetching" bar;
                                # suppress it and keep the shared layer bar
                                # as the single progress row.
                                with self._suppress_output():
                                    raw_neuron_vols = self.client_flywire.fetch_skeletons(self.layer_criteria[i], with_synapses=self.show_connectors)
                                break  # Success
                            except Exception as e:
                                error_msg = str(e)
                                is_network_error = any(x in error_msg.lower() for x in 
                                    ['timeout', 'connection', 'network', 'refused', 'reset', 'temporary'])
                                
                                if is_network_error and attempt < max_retries - 1:
                                    import time
                                    wait_time = (attempt + 1) * 2
                                    tqdm.write(f'  ⚠️  Network error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries}): {e}')
                                    time.sleep(wait_time)
                                else:
                                    tqdm.write(f'  ⚠️  FlyWire fetch failed for layer {layer_name}: {e}')
                                    raw_neuron_vols = None
                                    break
                    else:
                        # NeuPrint requests are owned by the aggregate
                        # preprocessing stage above. Keeping a layer-scoped
                        # online fallback here would reintroduce duplicate
                        # fetches and make cache state depend on layer order.
                        self._vprint(
                            f'  ⚠️  No prepared skeletons for layer '
                            f'{layer_name}; skipping missing IDs',
                            level='full')

                # NeuPrint tube-mode fine/artistic transform for rendering.
                # Applied to COPIES in native units BEFORE any dataset
                        # scaling; the raw skeletons used by the shared cache and
                        # line mode are left untouched. Radius style is selected
                # inside the transform, so the constant-radius experiment
                # still receives the fine/artistic path smoothing/resampling.
                render_neuron_vols = raw_neuron_vols
                if (use_neuprint_fine_pipeline
                        and raw_neuron_vols is not None
                        and not neuprint_layer_prepared):
                    transformed = []
                    total = len(raw_neuron_vols)
                    for k, n in enumerate(raw_neuron_vols):
                        # Report per-neuron progress through the shared layer
                        # bar; navis' internal smoothing bar is suppressed
                        # inside the helper.
                        layer_pbar.set_postfix_str(
                            f"{layer_name} (FAFB-format {k}/{total}...)")
                        try:
                            transformed.append(
                                self._fafb_style_neuprint_skeleton(
                                    n,
                                    optimized=use_neuprint_fine_optimized_pipeline,
                                )
                            )
                        except Exception as e:
                            self._vprint(
                                f'  ⚠️ FAFB-format transform failed for '
                                f'{getattr(n, "id", n)}: {e}', level='full')
                            transformed.append(n)  # keep the raw skeleton
                    layer_pbar.set_postfix_str(f"{layer_name} (FAFB-format done)")
                    render_neuron_vols = navis.NeuronList(transformed)

                # Apply scaling for MANC datasets (Raw -> NM)
                # MANC skeletons from NeuPrint are in 8nm voxels, but meshes are in nm
                if (raw_neuron_vols is not None
                        and 'manc' in self.dataset.lower()
                        and not neuprint_layer_prepared):
                    # Check if scaling is needed (max coord < 150000 indicates raw units)
                    # Typical MANC nm extent is ~300k-500k
                    try:
                        bbox = raw_neuron_vols[0].bbox if hasattr(raw_neuron_vols[0], 'bbox') else None
                        if bbox is not None and np.max(bbox) < 150000:
                            raw_neuron_vols = raw_neuron_vols * 8
                            render_neuron_vols = render_neuron_vols * 8
                            self._vprint(f'  ℹ️  Applied 8x scaling to fetched MANC skeletons', level='full')
                    except Exception as e:
                        self._vprint(f'  ⚠️  Failed to check/apply scaling: {e}', level='full')

                # The aggregate NeuPrint fetcher already persists raw
                # compressed-SWC files. Keep rendering on the selected
                # in-memory representation; never persist a transformed or
                # simplified skeleton here.
                if (raw_neuron_vols is not None and not is_fafb
                        and not neuprint_layer_prepared):
                    self._save_cached_neurons(self.neuron_dfs[i], raw_neuron_vols)
                    raw_neuron_vols = render_neuron_vols
            
            # Combine cached and newly fetched neurons
            if cached_neurons is not None and raw_neuron_vols is not None:
                all_neurons = list(cached_neurons) + list(raw_neuron_vols)
                neuron_vols = navis.NeuronList(all_neurons)
            elif cached_neurons is not None:
                neuron_vols = cached_neurons
            elif raw_neuron_vols is not None:
                neuron_vols = raw_neuron_vols
            else:
                neuron_vols = None

            # Normalize to NeuronList so downstream len()/iteration works for single TreeNeuron
            if neuron_vols is not None and not isinstance(neuron_vols, (list, navis.NeuronList)):
                neuron_vols = navis.NeuronList([neuron_vols])
            
            # For FAFB with all meshes cached, we can skip skeleton processing
            if is_fafb and cached_mesh_neurons and (neuron_vols is None or len(neuron_vols) == 0):
                # All neurons loaded from mesh cache - neuron_vols stays None/empty
                # The combine block below will handle adding cached_mesh_neurons with simplification
                pass
            elif neuron_vols is None or len(neuron_vols) == 0:
                if cached_mesh_neurons:
                    # Partial cache hit - neuron_vols stays None/empty
                    # The combine block below will handle cached_mesh_neurons
                    pass
                else:
                    tqdm.write(f'  ⚠️  Failed to fetch skeletons for layer {i}: {layer_name}')
                    layer_pbar.update(n_in_layer)
                    continue

            # Apply soma radius capping to prevent extrusion artifacts
            if self.soma_radius_cap is not None and self.skeleton_mode == 'tube':
                self._apply_soma_radius_cap(neuron_vols)

            # Transformation logic moved to after caching to ensure cache stores raw FlyWire coordinates

            # FAFB processing is pipeline-driven from the resolved per-body
            # sources (see _process_fafb_layer for the per-pipeline rules).
            fafb_already_simplified = False
            if is_fafb:
                neuron_vols, fafb_already_simplified = (
                    self._process_fafb_layer(
                        neuron_vols,
                        cached_mesh_neurons,
                        fafb_pipeline,
                        use_fafb_cache,
                        render_mesh_cache=fafb_render_mesh_cache,
                        progress_callback=report_fafb_progress,
                    )
                )

            # Legacy non-aggregate paths (client_flywire / direct layer
            # fetches) bypass the aggregate preprocessing above, so the
            # dataset-aware line simplification is applied here before
            # plotting.  FAFB line renders were handled by their own branch.
            if (self.skeleton_mode == 'line'
                    and neuron_vols is not None
                    and not is_fafb
                    and not neuprint_preprocessing_active):
                prepared_lines = []
                neurons_list = (
                    list(neuron_vols)
                    if isinstance(neuron_vols, navis.NeuronList)
                    else [neuron_vols])
                for n in neurons_list:
                    neuron_id = getattr(n, 'id', None)
                    if isinstance(n, navis.TreeNeuron):
                        try:
                            work, stats = simplify_skeleton_nodes(
                                n, NEUPRINT_LINE_NODE_REDUCTION)
                            self._vprint(
                                f'  🔽 simplify lines {neuron_id} '
                                f'{stats["raw_nodes"]}→'
                                f'{stats["achieved_nodes"]}',
                                level='full', use_tqdm=True)
                            prepared_lines.append(work)
                        except Exception as e:
                            self._vprint(
                                f'  ⚠️ line simplification failed for '
                                f'{neuron_id}: {e}', level='full',
                                use_tqdm=True)
                            prepared_lines.append(n)
                    else:
                        prepared_lines.append(n)
                if prepared_lines:
                    neuron_vols = navis.NeuronList(prepared_lines)

            # NeuPrint tube mode: build transformed tube meshes, decimate ONCE
            # to the cache level (absolute), cache them, then combine cached
            # + new meshes. The FAFB-format transform already ran at fetch
            # time (smooth + resample + radius profile).
            # Aggregate preprocessing already performed the active mesh
            # simplification for this layer. Keep the later generic branch
            # from decimating the same prepared mesh a second time.
            neuprint_already_simplified = (
                neuprint_layer_prepared and neuprint_prepared_mesh)
            if use_neuprint_mesh_cache and self.skeleton_mode == 'tube':
                try:
                    cache_level = self.NEUPRINT_MESH_CACHE_SIMPLIFICATION
                    meshes_to_cache = {}
                    mesh_neurons_list = []

                    mesh_missing_ids_set = set()
                    for mid in mesh_missing_ids:
                        mesh_missing_ids_set.add(mid)
                        if isinstance(mid, (int, np.integer)):
                            mesh_missing_ids_set.add(str(mid))
                        elif isinstance(mid, str) and mid.isdigit():
                            mesh_missing_ids_set.add(int(mid))

                    iterator = neuron_vols if neuron_vols is not None else []
                    # Keep the render's shared bar as the only tqdm row.
                    # Mesh transformation progress is reflected when the
                    # containing layer completes below.

                    for n in iterator:
                        if hasattr(n, 'id') and n.id in mesh_missing_ids_set:
                            # Convert TreeNeuron to tube MeshNeuron
                            if isinstance(n, navis.TreeNeuron):
                                if hasattr(n, 'nodes') and 'radius' in n.nodes.columns:
                                    invalid_mask = (n.nodes['radius'] <= 0) | (n.nodes['radius'].isna())
                                    if invalid_mask.any():
                                        n.nodes.loc[invalid_mask, 'radius'] = 1
                                elif hasattr(n, 'nodes'):
                                    n.nodes['radius'] = 1
                                try:
                                    mesh_n = navis.conversion.tree2meshneuron(
                                        n, tube_points=self.SKELETON_TUBE_POINTS)
                                except Exception:
                                    mesh_neurons_list.append(n)
                                    continue
                            elif isinstance(n, navis.MeshNeuron):
                                mesh_n = n
                            else:
                                mesh_neurons_list.append(n)
                                continue

                            # ONE absolute uniform decimation to the cache level
                            if hasattr(mesh_n, 'trimesh'):
                                n_faces = len(mesh_n.trimesh.faces)
                                target_faces = max(100, int(n_faces * (1 - cache_level)))
                                if target_faces < n_faces:
                                    try:
                                        simplified_trimesh = self._simplify_mesh_for_neuprint_pipeline(
                                            mesh_n.trimesh, target_faces)
                                        mesh_n = navis.MeshNeuron(simplified_trimesh)
                                        mesh_n.id = n.id
                                        if hasattr(n, 'name'):
                                            mesh_n.name = n.name
                                    except Exception as e:
                                        self._vprint(f'      ⚠️ Simplification failed for {n.id}: {e}', level='full', use_tqdm=True)

                            meshes_to_cache[n.id] = mesh_n
                            mesh_neurons_list.append(mesh_n)
                        else:
                            mesh_neurons_list.append(n)

                    if meshes_to_cache:
                        self._save_cached_neuprint_meshes(meshes_to_cache)
                        self._vprint(f'    ✓ Cached {len(meshes_to_cache)} NeuPrint meshes (simp={cache_level})', level='full', use_tqdm=True)

                    # Additional relative decimation when user level > cache level
                    target_simp = self.skeleton_mesh_simplification
                    if target_simp > cache_level and mesh_neurons_list:
                        remaining_after_cache = 1 - cache_level
                        remaining_target = 1 - target_simp
                        additional_keep_factor = remaining_target / remaining_after_cache
                        self._vprint(f'    ⚡ Applying additional simplification: {target_simp} (keep {additional_keep_factor:.1%})', level='full', use_tqdm=True)
                        further_simplified = []
                        for mesh_n in mesh_neurons_list:
                            if hasattr(mesh_n, 'trimesh'):
                                n_faces = len(mesh_n.trimesh.faces)
                                target_faces = max(100, int(n_faces * additional_keep_factor))
                                if target_faces < n_faces:
                                    try:
                                        simplified_trimesh = self._simplify_mesh_for_neuprint_pipeline(
                                            mesh_n.trimesh, target_faces)
                                        new_mesh = navis.MeshNeuron(simplified_trimesh)
                                        new_mesh.id = mesh_n.id if hasattr(mesh_n, 'id') else None
                                        if hasattr(mesh_n, 'name'):
                                            new_mesh.name = mesh_n.name
                                        further_simplified.append(new_mesh)
                                        continue
                                    except Exception:
                                        pass
                            further_simplified.append(mesh_n)
                        mesh_neurons_list = further_simplified

                    if mesh_neurons_list:
                        neuron_vols = navis.NeuronList(mesh_neurons_list)
                except Exception as e:
                    self._vprint(f'    ⚠️ NeuPrint mesh caching failed: {e}', level='full')

            # Combine NeuPrint cached meshes + newly built meshes, applying
            # the additional relative decimation to cached ones when the user
            # level is above the cache level.
            if use_neuprint_mesh_cache and self.skeleton_mode == 'tube':
                cache_level = self.NEUPRINT_MESH_CACHE_SIMPLIFICATION
                all_mesh_neurons = []
                if cached_mesh_neurons:
                    target_simp = self.skeleton_mesh_simplification
                    if target_simp > cache_level:
                        remaining_after_cache = 1 - cache_level
                        remaining_target = 1 - target_simp
                        additional_keep_factor = remaining_target / remaining_after_cache
                        simplified_cached = []
                        for mesh_n in cached_mesh_neurons:
                            if hasattr(mesh_n, 'trimesh'):
                                n_faces = len(mesh_n.trimesh.faces)
                                target_faces = max(100, int(n_faces * additional_keep_factor))
                                if target_faces < n_faces:
                                    try:
                                        simplified_trimesh = self._simplify_mesh_for_neuprint_pipeline(
                                            mesh_n.trimesh, target_faces)
                                        new_mesh = navis.MeshNeuron(simplified_trimesh)
                                        new_mesh.id = mesh_n.id if hasattr(mesh_n, 'id') else None
                                        if hasattr(mesh_n, 'name'):
                                            new_mesh.name = mesh_n.name
                                        simplified_cached.append(new_mesh)
                                        continue
                                    except Exception:
                                        pass
                            simplified_cached.append(mesh_n)
                        cached_mesh_neurons = simplified_cached
                    all_mesh_neurons.extend(cached_mesh_neurons)
                if neuron_vols is not None and len(neuron_vols) > 0:
                    all_mesh_neurons.extend(
                        list(neuron_vols) if isinstance(neuron_vols, navis.NeuronList) else [neuron_vols])
                if all_mesh_neurons:
                    neuron_vols = navis.NeuronList(all_mesh_neurons)
                neuprint_already_simplified = True

            # Transform if needed (Now applies to ALL neurons: cached + new)
            needs_actual_transform = needs_transform
            if needs_actual_transform and neuron_vols is not None:
                layer_pbar.set_postfix_str(f"{layer_name} (transforming {len(neuron_vols)}...)")
                try:
                    # Ensure float64 coordinates to avoid dtype warnings in navis
                    if isinstance(neuron_vols, (list, navis.NeuronList)):
                        for n in neuron_vols:
                            if hasattr(n, 'nodes') and isinstance(n.nodes, pd.DataFrame):
                                for col in ['x', 'y', 'z']:
                                    if col in n.nodes.columns:
                                        n.nodes[col] = n.nodes[col].astype('float64')
                            # Also handle MeshNeuron vertices
                            if hasattr(n, 'vertices') and n.vertices is not None:
                                n.vertices = n.vertices.astype('float64')
                    elif hasattr(neuron_vols, 'nodes') and isinstance(neuron_vols.nodes, pd.DataFrame):
                         for col in ['x', 'y', 'z']:
                            if col in neuron_vols.nodes.columns:
                                neuron_vols.nodes[col] = neuron_vols.nodes[col].astype('float64')

                    # Per-neuron transform with explicit progress: a single
                    # navis.xform_brain() call gives no visibility into WHICH
                    # neuron it is on, so a stall looked like a silent hang.
                    neuron_vols = self._xform_neurons_safe(
                        neuron_vols,
                        source=template_info['source'],
                        target=template_info['target'],
                        layer_label=f"Layer {i} ({layer_name})",
                        progress_bar=layer_pbar,
                        compact_progress=True,
                    )
                except Exception as e:
                    tqdm.write(f'  ⚠️  Layer {i} transform failed: {e}')
                    if self.verbose:
                        import traceback
                        frame = traceback.extract_tb(e.__traceback__)[-1]
                        tqdm.write(
                            f'  🐞 raised in {frame.filename}:{frame.lineno} '
                            f'({frame.name})')
                    if self._dataset_needs_transform() and not self._check_and_download_transforms():
                        self.brain_mesh = 'none'
                    else:
                        # Retry transformation after download
                        try:
                            neuron_vols = self._xform_neurons_safe(
                                neuron_vols,
                                source=template_info['source'],
                                target=template_info['target'],
                                layer_label=f"Layer {i} ({layer_name}, retry)",
                                progress_bar=layer_pbar,
                                compact_progress=True,
                            )
                        except Exception as retry_e:
                            tqdm.write(f'  ⚠️  Transformation still failed, setting brain_mesh to "none"')
                            if self.verbose:
                                import traceback
                                frame = traceback.extract_tb(
                                    retry_e.__traceback__)[-1]
                                tqdm.write(
                                    f'  🐞 raised in {frame.filename}:'
                                    f'{frame.lineno} ({frame.name})')
                            self.brain_mesh = 'none'
            
            # Apply FAFB tilt correction if using template mode
            # This corrects the left-right tilt in the FLYWIRE template mesh
            if is_fafb and self.brain_mesh == 'template' and neuron_vols is not None:
                # Per-neuron correction reports through the shared layer bar;
                # a whole-list navis.xform would spawn navis' own "Xforming"
                # bar that prints on alternate rows instead of refreshing this one.
                neuron_vols = self._apply_fafb_tilt_correction(
                    neuron_vols,
                    progress_bar=layer_pbar,
                    layer_label=f"Layer {i} ({layer_name})",
                )
            
            # Ensure iterable after potential transforms (navis may return TreeNeuron)
            if neuron_vols is not None and not isinstance(neuron_vols, (list, navis.NeuronList)):
                neuron_vols = navis.NeuronList([neuron_vols])

            # Mirror neurons if requested
            if self.mirror_on_contralateral:
                try:
                    template = None
                    if self.brain_mesh == 'whole':
                        template_info = self._get_template_info()
                        template = template_info['target']
                    elif self.brain_mesh == 'template':
                         if 'hemibrain' in self.dataset or 'optic-lobe' in self.dataset:
                             template = 'JRCFIB2018F'
                         elif 'male-cns' in self.dataset:
                             template = 'JRCFIB2022M'
                    
                    if template:
                        layer_pbar.set_postfix_str(f"{layer_name} (mirroring...)")
                        # navis.mirror_brain prints its own progress bar; keep
                        # the shared layer bar as the single progress row.
                        with self._suppress_output():
                            mirrored = navis.mirror_brain(neuron_vols, template, mirror_axis='x')
                        if isinstance(neuron_vols, navis.NeuronList):
                            neuron_vols = neuron_vols + mirrored
                        else:
                            neuron_vols = navis.NeuronList([neuron_vols, mirrored])
                except Exception as e:
                    tqdm.write(f'  ⚠️ Mirror failed for layer {i}: {e}')

            # Simplify individual neurons if requested (and not merging).
            # Fine NeuPrint meshes have already been built from raw skeletons
            # in the aggregate preprocessing phase; this block handles only
            # the remaining legacy/direct paths and line mode is skipped.
            render_simplification = self._effective_render_simplification(
                is_fafb,
            )
            if (render_simplification > 0 and self.skeleton_mode == 'tube'
                    and not fafb_already_simplified
                    and not neuprint_already_simplified):
                try:
                    import trimesh
                    simplified_neurons = []
                    total_original_faces = 0
                    total_simplified_faces = 0
                    # Ensure iterable
                    neurons_to_simplify = neuron_vols if isinstance(neuron_vols, navis.NeuronList) else [neuron_vols]
                    
                    for n in neurons_to_simplify:
                        try:
                            # Convert to mesh if needed (TreeNeuron -> MeshNeuron)
                            mesh_n = None
                            if isinstance(n, navis.TreeNeuron):
                                # Fix radii if needed
                                if hasattr(n, 'nodes') and 'radius' in n.nodes.columns:
                                    invalid_mask = (n.nodes['radius'] <= 0) | (n.nodes['radius'].isna())
                                    if invalid_mask.any():
                                        n.nodes.loc[invalid_mask, 'radius'] = 1
                                elif hasattr(n, 'nodes'):
                                    n.nodes['radius'] = 1
                                
                                # Convert
                                if hasattr(navis, 'conversion') and hasattr(navis.conversion, 'tree2meshneuron'):
                                    mesh_n = navis.conversion.tree2meshneuron(
                                        n,
                                        tube_points=self.SKELETON_TUBE_POINTS,
                                    )
                            elif isinstance(n, navis.MeshNeuron):
                                mesh_n = n
                                
                            # Simplify if we have a mesh neuron
                            # MeshNeuron.trimesh is read-only, so we must create a new MeshNeuron
                            if mesh_n and hasattr(mesh_n, 'trimesh'):
                                n_faces = len(mesh_n.trimesh.faces)
                                total_original_faces += n_faces
                                target_faces = max(100, int(n_faces * (1 - render_simplification)))  # Keep at least 100 faces
                                if target_faces < n_faces:
                                    # Use open3d for accurate simplification (trimesh 4.x fast_simplification only achieves ~60%)
                                    # Create NEW MeshNeuron from simplified trimesh (can't just assign to .trimesh)
                                    try:
                                        simplified_tm = self._simplify_mesh_for_neuprint_pipeline(
                                            mesh_n.trimesh, target_faces)
                                    except Exception as simp_err:
                                        # Fallback to original if simplification fails
                                        simplified_tm = mesh_n.trimesh
                                    
                                    new_mesh_n = navis.MeshNeuron(simplified_tm)
                                    # Copy over attributes
                                    new_mesh_n.id = mesh_n.id if hasattr(mesh_n, 'id') else n.id
                                    if hasattr(mesh_n, 'name'):
                                        new_mesh_n.name = mesh_n.name
                                    total_simplified_faces += len(new_mesh_n.trimesh.faces)
                                    simplified_neurons.append(new_mesh_n)
                                else:
                                    total_simplified_faces += n_faces
                                    simplified_neurons.append(mesh_n)
                            else:
                                # Keep original if conversion failed or not applicable
                                simplified_neurons.append(n)
                        except Exception as e:
                            # Keep original if failed
                            simplified_neurons.append(n if mesh_n is None else mesh_n)
                    
                    neuron_vols = navis.NeuronList(simplified_neurons)
                    
                    # Log simplification results
                    if total_original_faces > 0:
                        reduction = (1 - total_simplified_faces / total_original_faces) * 100
                        self._vprint(f'    ✓ Simplified: {total_original_faces:,} → {total_simplified_faces:,} faces ({reduction:.1f}% reduction)', level='full', use_tqdm=True)
                except Exception as e:
                    self._vprint(f'    ⚠️ Simplification failed: {e}', level='full', use_tqdm=True)
                    pass  # Keep original neurons if simplification fails

            # Update status and plot
            layer_pbar.set_postfix_str(f"{layer_name} (plotting...)")
            
            # Determine soma rendering
            show_soma_here = self.show_soma if not isinstance(neuron_vols, navis.Volume) else False
            
            # Colors are standardized with neuron_alpha as the fallback, so
            # extracting the alpha here handles explicit and inherited values
            # uniformly, including mixed-alpha palettes.
            layer_neuron_alpha = self._extract_alpha_from_color(self.neuron_colors[i])

            if self.skeleton_mode == 'tube' and neuron_vols is not None:
                neurons_for_export = neuron_vols if isinstance(neuron_vols, (list, navis.NeuronList)) else [neuron_vols]
                for unit_index, neuron in enumerate(neurons_for_export):
                    neuron_id = str(getattr(neuron, 'id', f'neuron_{unit_index}'))
                    neuron_color = self._resolve_neuron_color(neuron_id, i)
                    neuron_alpha = self._extract_alpha_from_color(neuron_color)
                    self._append_exportable_mesh(
                        neuron,
                        color=neuron_color,
                        alpha=neuron_alpha,
                        name=getattr(neuron, 'name', neuron_id),
                        role='neuron',
                    )
            
            if self.backend == 'plotly':
                trace_entries = []
                # Custom layer-map colors are per-neuron even when the global
                # color mode is ``per_layer``. Plot those neurons separately so
                # navis cannot collapse differently colored rows into one
                # pre-colored layer trace before the override is applied.
                render_per_neuron = (
                    self.color_mode == 'per_neuron'
                    or bool(getattr(self, '_neuron_color_overrides', None))
                ) and not isinstance(neuron_vols, navis.Volume)
                if render_per_neuron and neuron_vols is not None:
                    for unit_index, neuron in enumerate(neuron_vols):
                        neuron_id = str(getattr(neuron, 'id', f'neuron_{unit_index}'))
                        neuron_color = self._resolve_neuron_color(neuron_id, i)
                        neuron_alpha = self._extract_alpha_from_color(neuron_color)
                        neuron_color_hex = self._rgba_to_hex(neuron_color)

                        with self._suppress_output():
                            fig_layer = navis.plot3d(
                                neuron,
                                backend='plotly',
                                color=neuron_color_hex,
                                alpha=neuron_alpha,
                                soma=show_soma_here,
                                radius=self.show_skeleton_radius,
                                connectors=self.show_connectors,
                            )

                        for trace in fig_layer.data:
                            trace_entries.append((trace, neuron_id, unit_index, neuron_color))
                else:
                    # Convert rgba color to hex for navis compatibility
                    layer_color_hex = self._rgba_to_hex(self.neuron_colors[i])

                    with self._suppress_output():
                        fig_layer = navis.plot3d(
                            neuron_vols,
                            backend='plotly',
                            color=layer_color_hex,
                            alpha=layer_neuron_alpha,
                            soma=show_soma_here,
                            # fig=self.fig_3d,
                            radius=self.show_skeleton_radius,
                            connectors=self.show_connectors if not isinstance(neuron_vols, navis.Volume) else False,
                        )
                    fig_traces = fig_layer.data
                    # Companion traces without their own name (soma meshes)
                    # must resolve to their owning neuron, not to whichever
                    # bodyId happens to sit at the trace's list position.
                    identities = self._resolve_plotly_trace_identities(
                        neuron_vols, fig_traces,
                    )
                    for (trace, (neuron_id, source_index)) in zip(
                            fig_traces, identities):
                        neuron_color = self._resolve_neuron_color(neuron_id, i)
                        trace_entries.append(
                            (trace, neuron_id, source_index, neuron_color)
                        )

                # Build a mapping of neuron ID to type for 'type' legend mode
                neuron_type_map = {}
                if self.legend_mode in ('type', 'tree') and self.neuron_dfs[i] is not None:
                    ndf = self.neuron_dfs[i]
                    type_col = None
                    for col in ['type', 'cell_type', 'neuronType']:
                        if col in ndf.columns:
                            type_col = col
                            break
                    if type_col and 'bodyId' in ndf.columns:
                        for _, row in ndf.iterrows():
                            body_id = str(row['bodyId'])
                            neuron_type = self._clean_pre_post_type_label(row[type_col])
                            if neuron_type:
                                neuron_type_map[body_id] = neuron_type
                                # Also map neuron name/instance to type if available (helpful when trace name != bodyId)
                                for name_col in ['name', 'instance', 'roi']:
                                    if name_col in row and pd.notna(row[name_col]):
                                        neuron_type_map[str(row[name_col])] = neuron_type

                # Track which legend groups we've shown (for type/layer modes)
                shown_legend_groups = set()
                # Track legend info for fixing opacity later
                legend_color_map = {}  # legend_group -> (color, should_show)

                for trace, neuron_id, source_index, neuron_color in trace_entries:
                    # Enforce opacity for lines if not already set or if we want to override
                    if self.skeleton_mode == 'line':
                        trace.opacity = self._extract_alpha_from_color(neuron_color)

                    # When custom colors forced one-neuron-at-a-time plotting,
                    # ``neuron_color`` is already the bodyId-resolved value
                    # passed to navis. Re-resolving from a generic trace name
                    # can otherwise fall back to the layer color and overwrite
                    # the correctly colored trace.
                    if not render_per_neuron:
                        neuron_color = self._resolve_neuron_color(neuron_id, i)
                    if neuron_color != self.neuron_colors[i]:
                        self._apply_plotly_trace_color(trace, neuron_color)

                    if self.legend_mode == 'layer':
                        # Group all neurons in layer under one legend entry
                        legend_group = self.layer_names[i]
                        legend_rank = self._neuron_legend_ranks.setdefault(
                            legend_group, len(self._neuron_legend_ranks) * 100,
                        )
                        trace.name = legend_group
                        trace.legendgroup = legend_group
                        trace.legendrank = legend_rank
                        should_show = legend_group not in shown_legend_groups
                        trace.showlegend = should_show
                        if should_show:
                            legend_color_map[legend_group] = (self.neuron_colors[i], True)
                        shown_legend_groups.add(legend_group)
                        trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                        trace.hoverinfo = 'name'
                        self.fig_3d.add_trace(trace)

                    elif self.legend_mode in ('type', 'tree'):
                        # Group by neuron type - each type gets separate legend but keeps layer color
                        neuron_type = neuron_type_map.get(neuron_id, None)

                        # Fallback: try different ID strategies if type not found
                        if not neuron_type and source_index < len(neuron_vols):
                            # Try using the ID from the source neuron object
                            try:
                                vid = str(neuron_vols[source_index].id)
                                neuron_type = neuron_type_map.get(vid, None)
                            except:
                                pass

                            # If still not found, try using the name from source neuron
                            if not neuron_type and hasattr(neuron_vols[source_index], 'name'):
                                try:
                                    vname = str(neuron_vols[source_index].name)
                                    neuron_type = neuron_type_map.get(vname, None)
                                except:
                                    pass

                        if neuron_type:
                            legend_group = f"{neuron_type}"
                        else:
                            # Fallback to layer name if type unknown
                            legend_group = self.layer_names[i]
                        legend_rank = self._neuron_legend_ranks.setdefault(
                            legend_group, len(self._neuron_legend_ranks) * 100,
                        )
                        trace.name = legend_group
                        trace.legendgroup = legend_group  # Same type shares legend group
                        trace.legendrank = legend_rank
                        should_show = legend_group not in shown_legend_groups
                        trace.showlegend = should_show
                        if should_show:
                            legend_color_map[legend_group] = (neuron_color, True)
                        shown_legend_groups.add(legend_group)
                        trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                        trace.hoverinfo = 'name'
                        if self.legend_mode == 'tree':
                            # The exported HTML's collapsible legend panel
                            # groups traces by 'group' and labels each
                            # expandable row by 'item'. With custom groups the
                            # hierarchy is group > type > bodyId: a type row
                            # appears only when a group holds 2+ neurons of
                            # that type; singletons and untyped neurons
                            # become direct bodyId/instance leaves.
                            tree_label = str(neuron_id)
                            if self.neuron_dfs[i] is not None and source_index < len(self.neuron_dfs[i]):
                                source_row = self.neuron_dfs[i].iloc[source_index]
                                for label_col in ('bodyId', 'instance'):
                                    if label_col in source_row.index and pd.notna(source_row.get(label_col)):
                                        tree_label = str(source_row[label_col])
                                        break
                            tree_meta = dict(getattr(trace, 'meta', None) or {})
                            if self._tree_uses_custom_groups():
                                tree_meta['drocatLegend'] = {
                                    'kind': 'neuron',
                                    'group': self.layer_names[i],
                                    'type': neuron_type or None,
                                    'item': tree_label,
                                }
                            else:
                                tree_meta['drocatLegend'] = {
                                    'kind': 'neuron',
                                    'group': legend_group,
                                    'item': tree_label,
                                }
                            trace.meta = tree_meta
                        self.fig_3d.add_trace(trace)

                    elif self.legend_mode == 'single':
                        # Each neuron gets its own legend entry with layer color
                        new_trace_name = f"{neuron_id}_{self.layer_names[i]}"
                        legend_group = new_trace_name
                        legend_rank = self._neuron_legend_ranks.setdefault(
                            legend_group, len(self._neuron_legend_ranks) * 100,
                        )
                        trace.name = new_trace_name
                        trace.legendgroup = legend_group
                        trace.legendrank = legend_rank
                        trace.showlegend = True
                        legend_color_map[new_trace_name] = (neuron_color, True)
                        trace.hoverinfo = 'name'
                        trace.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                        self.fig_3d.add_trace(trace)
                    else:
                        raise ValueError(f'legend_mode {self.legend_mode} not supported')

                    # Connector site rows normally identify the owner by
                    # bodyId while navis may identify the skeleton trace by
                    # name/type.  Register every identifier available on the
                    # resolved row so pre/post sites can recover this exact
                    # skeleton legend label later.
                    aliases = list(self._normalize_neuron_lookup_keys(neuron_id))
                    if (
                        self.neuron_dfs[i] is not None
                        and source_index < len(self.neuron_dfs[i])
                    ):
                        source_row = self.neuron_dfs[i].iloc[source_index]
                        for column in (
                            'bodyId', 'name', 'type', 'instance', 'roi',
                            'flywireType', 'hemibrainType', 'mancType',
                        ):
                            if column in source_row.index:
                                aliases.extend(
                                    self._normalize_neuron_lookup_keys(
                                        source_row[column]
                                    )
                                )
                    label_map = self._neuron_legend_labels_by_layer.setdefault(i, {})
                    for alias in aliases:
                        label_map[alias] = str(legend_group)
                
                # Fix legend opacity: Add invisible marker traces with full opacity for legend display
                # Needed when alpha < 1.0 (either from neuron_alpha or explicit alpha in colors)
                # to ensure legend color patches are clearly visible
                needs_legend_fix = (layer_neuron_alpha < 1.0 or self._neuron_colors_have_explicit_alpha) and legend_color_map
                if needs_legend_fix:
                    import plotly.graph_objects as go
                    for legend_group, (color, _) in legend_color_map.items():
                        # Get opaque version of color for legend display
                        opaque_color = self._get_opaque_color(color)
                        # Add an invisible scatter point (no coords = not rendered) for legend only
                        legend_trace = go.Scatter3d(
                            x=[None], y=[None], z=[None],  # No coordinates = invisible in plot
                            mode='markers',
                            marker=dict(size=10, color=opaque_color, opacity=1.0),  # Full opacity for legend
                            name=legend_group,
                            legendgroup=legend_group,
                            showlegend=True,
                            hoverinfo='skip',
                        )
                        legend_trace.legendrank = self._neuron_legend_ranks.get(
                            legend_group, 1000,
                        )
                        # Hide original traces from legend (but keep them visible in plot)
                        for existing_trace in self.fig_3d.data:
                            if getattr(existing_trace, 'legendgroup', None) == legend_group:
                                existing_trace.showlegend = False
                        # Add the legend-only trace
                        self.fig_3d.add_trace(legend_trace)
            
            elif self.backend == 'k3d':
                try:
                    render_per_neuron = (
                        self.color_mode == 'per_neuron'
                        or bool(getattr(self, '_neuron_color_overrides', None))
                    ) and not isinstance(neuron_vols, navis.Volume)
                    if render_per_neuron and neuron_vols is not None:
                        for unit_index, neuron in enumerate(neuron_vols):
                            neuron_id = str(getattr(neuron, 'id', f'neuron_{unit_index}'))
                            neuron_color = self._resolve_neuron_color(neuron_id, i)
                            neuron_alpha = self._extract_alpha_from_color(neuron_color)
                            neuron_color_hex = self._rgba_to_hex(neuron_color)

                            with self._suppress_output():
                                temp_plot = navis.plot3d(
                                    neuron,
                                    backend='k3d',
                                    color=neuron_color_hex,
                                    alpha=neuron_alpha,
                                    soma=show_soma_here,
                                    radius=self.show_skeleton_radius,
                                    connectors=self.show_connectors,
                                    inline=False
                                )

                            for obj in temp_plot.objects:
                                self._apply_k3d_object_color(obj, neuron_color)
                                if hasattr(obj, 'name'):
                                    obj.name = self.layer_names[i]
                                self.fig_3d += obj
                    else:
                        # navis.plot3d with k3d backend returns a k3d.Plot object
                        # Convert rgba color to hex for navis compatibility
                        layer_color_hex = self._rgba_to_hex(self.neuron_colors[i])
                        
                        with self._suppress_output():
                            temp_plot = navis.plot3d(
                                neuron_vols,
                                backend='k3d',
                                color=layer_color_hex,
                                alpha=layer_neuron_alpha,  # Use layer-specific alpha (from color or neuron_alpha)
                                soma=show_soma_here,
                                radius=self.show_skeleton_radius,
                                connectors=self.show_connectors if not isinstance(neuron_vols, navis.Volume) else False,
                                inline=False
                            )
                        
                        for j, obj in enumerate(temp_plot.objects):
                            neuron_id = getattr(obj, 'name', None)
                            if not neuron_id and j < len(neuron_vols):
                                neuron_id = str(neuron_vols[j].id)
                            neuron_color = self._resolve_neuron_color(neuron_id, i)
                            self._apply_k3d_object_color(obj, neuron_color)
                            if hasattr(obj, 'name'):
                                obj.name = self.layer_names[i]
                            self.fig_3d += obj
                except Exception as e:
                    self._vprint(f'⚠️  k3d plotting failed: {e}', level='full')

            # FAFB source processing normally accounts for every body above.
            # Fill any gap caused by an unavailable source or a layer with no
            # bodyId column so the render-wide bar still reaches its total.
            remaining = (
                n_in_layer - processed_this_layer
                if is_fafb else n_in_layer
            )
            if remaining > 0:
                layer_pbar.update(remaining)

        layer_pbar.close()
        return 0
    
    def _get_synapse_cache_path(self, pre_id, post_id):
        """Get cache file path for synapses between a specific pre/post neuron pair.
        
        Cache structure: cache/{dataset}/synapses/{pre_id}_{post_id}.parquet
        
        This caches by neuron pair rather than by layer, because:
        1. The same synapse data is reusable across different queries
        2. Layer indices are arbitrary and session-specific
        3. Avoids duplicate storage of the same synaptic connections
        """
        return str(self._get_synapse_cache().pair_path(pre_id, post_id))

    def _get_synapse_cache(self):
        """Return the run-scoped dataset synapse cache backend.

        The backend is deliberately independent from morphology/SWC caches.
        It owns pair files, broad connector-query files, and their metadata;
        all three are reusable synapse artifacts.
        """
        root = getattr(self, 'script_path', os.path.dirname(os.path.dirname(__file__)))
        enabled = bool(getattr(self, 'cache_synapses', False))
        cache = getattr(self, '_synapse_cache_backend', None)
        if (
            cache is None
            or cache.dataset != str(self.dataset)
            or str(cache.project_root) != str(root)
            or cache.enabled != enabled
        ):
            cache = SynapseCache(str(self.dataset), str(root), enabled=enabled)
            self._synapse_cache_backend = cache
        return cache

    def _synapse_query_spec(self, source_criteria=None, target_criteria=None,
                            source_ids=None, target_ids=None,
                            source_kind='neuprint', source_signature=None,
                            min_total_weight=None):
        """Build the complete cache contract for one synapse query."""
        threshold = (
            int(getattr(self, 'min_synapse_num', 0) or 0)
            if min_total_weight is None else int(min_total_weight or 0)
        )
        return {
            'dataset': str(self.dataset),
            'source_kind': str(source_kind),
            'source_endpoint': getattr(self, 'server', None),
            'source_version': getattr(self, 'version', None),
            'source_criteria': source_criteria,
            'target_criteria': target_criteria,
            'source_ids': sorted({str(value) for value in (source_ids or [])}),
            'target_ids': sorted({str(value) for value in (target_ids or [])}),
            'min_total_weight': threshold,
            'synapse_criteria': getattr(self, 'synapse_criteria', None),
            'source_signature': source_signature,
        }

    def _get_or_fetch_synapse_query(
            self, source_criteria=None, target_criteria=None,
            source_ids=None, target_ids=None, source_kind='neuprint',
            source_signature=None, fetcher=None, min_total_weight=None):
        """Load one broad synapse query from memory/disk or fetch it once.

        This is the connector/site path.  It intentionally does not consult
        neuron skeleton caches.  A single visualization run also gets an
        in-memory query memo, so size estimation and site rendering do not
        issue the same upstream/downstream request twice.
        """
        cache = self._get_synapse_cache()
        threshold = (
            int(getattr(self, 'min_synapse_num', 0) or 0)
            if min_total_weight is None else int(min_total_weight or 0)
        )
        spec = self._synapse_query_spec(
            source_criteria=source_criteria,
            target_criteria=target_criteria,
            source_ids=source_ids,
            target_ids=target_ids,
            source_kind=source_kind,
            source_signature=source_signature,
            min_total_weight=threshold,
        )
        key = cache.query_key(spec)
        memory = getattr(self, '_synapse_query_memory', None)
        if memory is None:
            memory = {}
            self._synapse_query_memory = memory
        if key in memory:
            return memory[key].copy()

        if cache.enabled:
            cached = cache.load_query(spec)
            if cached is not None:
                memory[key] = cached.copy()
                self._vprint(
                    f'  ✓ Loaded synapse query {key[:8]} from cache '
                    f'({len(cached)} rows)', level='full')
                return cached.copy()

        if fetcher is None:
            fetcher = lambda: fetch_synapse_connections(
                source_criteria=source_criteria,
                target_criteria=target_criteria,
                min_total_weight=threshold,
                synapse_criteria=self.synapse_criteria,
                client=self.client,
            )
        try:
            frame = fetcher()
        except Exception as exc:
            self._vprint(
                f'  ⚠️ Synapse query failed ({source_kind}): {exc}',
                level='full')
            return None
        if frame is None:
            return None
        if not isinstance(frame, pd.DataFrame):
            try:
                frame = pd.DataFrame(frame)
            except Exception:
                return None
        frame = frame.copy()
        # A criteria query can be broader than the selected table rows.  The
        # body-ID sets are part of the cache key, so enforce them on the rows
        # before memoizing/persisting; this keeps exact layer/site queries from
        # leaking connectors belonging to another selection.
        source_set = {str(value) for value in (source_ids or [])}
        target_set = {str(value) for value in (target_ids or [])}
        if source_set and 'bodyId_pre' in frame.columns:
            frame = frame[frame['bodyId_pre'].astype(str).isin(source_set)]
        if target_set and 'bodyId_post' in frame.columns:
            frame = frame[frame['bodyId_post'].astype(str).isin(target_set)]
        frame = frame.reset_index(drop=True)
        memory[key] = frame.copy()
        if cache.enabled:
            cache.save_query(spec, frame, source=source_kind)
        return frame

    def _load_cached_synapses(self, source_ids, target_ids):
        """Load cached synapse connections for given source/target neuron pairs.
        
        For FlyWire/FAFB datasets, loads from the master synapse table at:
            datasets/{dataset}/{dataset}_synapse_table.parquet
        and filters by source_ids and target_ids.
        
        For other datasets, loads individual cache files per neuron pair from:
            cache/{dataset}/synapses/{pre_id}_{post_id}.parquet
        
        Args:
            source_ids: Set/list of source (presynaptic) body IDs
            target_ids: Set/list of target (postsynaptic) body IDs
            
        Returns:
            Tuple of (cached_df, missing_pairs) where:
            - cached_df: DataFrame of cached synapses (may be None if nothing cached)
            - missing_pairs: List of (pre_id, post_id) tuples not found in cache
        """
        if not self.cache_synapses:
            # Return all pairs as missing
            all_pairs = [(s, t) for s in source_ids for t in target_ids]
            return None, all_pairs
        
        source_ids = set(str(s) for s in source_ids)
        target_ids = set(str(t) for t in target_ids)
        
        # FlyWire/FAFB have one dataset-native paired synapse table.  It is
        # the synapse cache source, not a skeleton/mesh cache, and the reader
        # applies the same column/coordinate normalization used everywhere.
        if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            frame = self._read_flywire_connection_frame(
                source_ids=source_ids, target_ids=target_ids)
            return (
                frame,
                [] if frame is not None else all_pairs,
            )

        cached_df, missing_pairs = self._get_synapse_cache().load_pairs(
            source_ids, target_ids)
        if cached_df is not None:
            self._vprint(
                f'  ✓ Loaded {len(cached_df)} synapses from cache '
                f'({len(missing_pairs)} pairs missing)', level='full')
        return cached_df, missing_pairs
    
    def _save_cached_synapses(self, conn_df, attempted_pairs=None,
                              persist_pairs=True):
        """Save synapse connections to cache, organized by pre/post neuron pairs.
        
        Each unique (pre_id, post_id) pair gets its own cache file at:
            cache/{dataset}/synapses/{pre_id}_{post_id}.parquet
            
        This approach ensures:
        1. Synapses are cached by their actual content (neuron pairs + positions)
        2. Same synapse data is reusable across different queries/layers
        3. Incremental caching - only fetch what's not already cached
        
        Args:
            conn_df: DataFrame containing synapse data
            attempted_pairs: Optional list of (pre_id, post_id) tuples that were queried.
                             Used to cache empty results for pairs with no synapses.
            persist_pairs: Keep the legacy pair index/files for callers that
                           explicitly need pair-level resume.  Broad
                           visualization queries are already persisted once in
                           the query cache and set this to ``False`` to avoid
                           duplicating every row across a query file and one
                           file per neuron pair.
        """
        if not self.cache_synapses:
            return
        if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
            # The dataset-native master table is already the canonical
            # FlyWire synapse cache; do not duplicate it into pair files.
            return
        if not persist_pairs:
            # _get_or_fetch_synapse_query has already stored this broad query
            # (when caching is enabled).  Keeping only that file prevents a
            # layer query from exploding into a cartesian set of pair files.
            return
        backend = self._get_synapse_cache()
        before = len(backend._read_manifest().get('pairs', {}))
        backend.save_pairs(conn_df, attempted_pairs)
        after = len(backend._read_manifest().get('pairs', {}))
        self._vprint(
            f'  💾 Synapse pair cache updated ({max(0, after - before)} new index entries)',
            level='full')
    
    def plot_synapses(self):
        if self.backend == 'plotly':
            self._plotly_sliders = []
        if self.skip_synapse:
            self._vprint('Skipping synapse plotting as requested.', level='full')
            return

        # Pre/post-site mode renders every queried neuron's input/output sites
        # rather than the paired inter-layer synapses; it bypasses the adjacent
        # layer loop and returns its own export.
        if self.synapse_mode == 'pre_post':
            return self._plot_pre_post_sites()

        synapse_frames = []
        for i in range(len(self.neuron_layers) - 1):
            source_criteria = self.layer_criteria[i]
            target_criteria = self.layer_criteria[i + 1]
            # Collect all inter-layer frames for one merged export alongside
            # neuron_info.csv.
            conn_df = None

            # --- Begin FlyWire/NeuPrint synapse loading logic ---
            if self.client_type == 'flywire':
                source_ids = set(self.neuron_dfs[i]['bodyId'].astype(str))
                target_ids = set(self.neuron_dfs[i + 1]['bodyId'].astype(str))
                # All FlyWire/FAFB connector consumers use this one
                # coordinate-normalizing synapse-table reader.  In addition to
                # keeping rendering, site mode, and size estimation identical,
                # it applies Parquet predicate pushdown and a run-scoped memo.
                conn_df = self._read_flywire_connection_frame(
                    source_ids=source_ids, target_ids=target_ids)
                if conn_df is None:
                    dataset_dir = os.path.join(
                        self.script_path, 'datasets',
                        self.dataset.replace(':', '_').replace('.', '_'))
                    self._vprint(
                        f"  ⚠️ Synapse table not found or unreadable for "
                        f"dataset '{self.dataset}'.",
                        level='full')
                    if 'fafb' in self.dataset.lower():
                        self._vprint(
                            "  Please download the synapse table from: "
                            "https://codex.flywire.ai/api/download?dataset=fafb",
                            level='full')
                    self._vprint(f"  Save the file to: {dataset_dir}", level='full')
                    self._vprint(
                        "  Skipping synapse plotting for this layer.",
                        level='full')
                    continue
                self._vprint(
                    f'  ✓ Found {len(conn_df)} synapses in the shared '
                    'synapse cache/table', level='full')
            else:
                # Fetch from NeuPrint - use new caching strategy
                source_ids = set(self.neuron_dfs[i]['bodyId'].astype(str))
                target_ids = set(self.neuron_dfs[i+1]['bodyId'].astype(str))
                
                # Try to load from cache first
                cached_df, missing_pairs = self._load_cached_synapses(source_ids, target_ids)
                
                if not missing_pairs:
                    # All data cached
                    conn_df = cached_df
                elif cached_df is not None and len(missing_pairs) < len(source_ids) * len(target_ids):
                    # Partial cache - fetch missing and combine
                    self._vprint(f'  Fetching {len(missing_pairs)} missing neuron pairs from NeuPrint...')
                    fetched_df = self._get_or_fetch_synapse_query(
                        source_criteria=source_criteria,
                        target_criteria=target_criteria,
                        source_ids=source_ids,
                        target_ids=target_ids,
                        source_kind='neuprint',
                    )
                    if fetched_df is not None and not fetched_df.empty:
                        conn_df = pd.concat([cached_df, fetched_df], ignore_index=True)
                        # The broad query may include rows already recovered
                        # from pair files.  Deduplicate before plotting and
                        # exporting so cache granularity never changes the
                        # visible connector count.
                        conn_df = conn_df.drop_duplicates().reset_index(drop=True)
                        # Save newly fetched data to cache
                        self._save_cached_synapses(
                            fetched_df, attempted_pairs=missing_pairs,
                            persist_pairs=False)
                    else:
                        conn_df = cached_df
                        # Also save empty results for missing pairs
                        self._save_cached_synapses(
                            None, attempted_pairs=missing_pairs,
                            persist_pairs=False)
                else:
                    # No cache - fetch all
                    conn_df = self._get_or_fetch_synapse_query(
                        source_criteria=source_criteria,
                        target_criteria=target_criteria,
                        source_ids=source_ids,
                        target_ids=target_ids,
                        source_kind='neuprint',
                    )
                    # Save to cache
                    all_pairs = [(s, t) for s in source_ids for t in target_ids]
                    self._save_cached_synapses(
                        conn_df, attempted_pairs=all_pairs,
                        persist_pairs=False)
        
            if conn_df is None or conn_df.empty:
                self._vprint('  No synapses found.', level='full')
                continue

            export_df = conn_df.copy()
            export_df['viz_layer'] = f'{i}->{i+1}'
            synapse_frames.append(export_df)
            
            self._vprint('plotting...', end='', level='full')
            
            if self.synapse_mode == 'scatter' or self.backend == 'k3d':
                X = (conn_df['x_pre']+conn_df['x_post'])/2
                Y = (conn_df['y_pre']+conn_df['y_post'])/2
                Z = (conn_df['z_pre']+conn_df['z_post'])/2
                xyz_df = pd.DataFrame({'x':X, 'y':Y, 'z':Z})
                
                # Ensure coordinates are float to avoid dtype warnings during transform
                xyz_df = xyz_df.astype(float)

                # Attach colors to dataframe to preserve order during transform.
                # When per-neuron synapse-color overrides exist (custom CSV), each
                # point takes the set color of its pre neuron; otherwise the layer's
                # connection color is used (possibly as a per-point array when the
                # palette already varies per connection).
                c_val = self.synapse_colors[i]
                is_color_array = False
                per_neuron_synapse_colors = getattr(self, '_neuron_synapse_color_overrides', None)
                if per_neuron_synapse_colors and 'bodyId_pre' in conn_df.columns:
                    xyz_df['__color'] = [
                        self._resolve_synapse_color(pre_id, i)
                        for pre_id in conn_df['bodyId_pre']
                    ]
                    is_color_array = True
                elif isinstance(c_val, (list, np.ndarray)) and len(c_val) == len(xyz_df):
                     # Check if it's not just a single RGB tuple
                     if len(xyz_df) != 3 or (isinstance(c_val[0], (str, list, tuple, np.ndarray))):
                         xyz_df['__color'] = c_val
                         is_color_array = True
                
                # Transform synapses only if needed (skip for FAFB native coords)
                if self._needs_skeleton_transform():
                    template_info = self._get_template_info()
                    self._vprint(f'Transforming synapses of layer {i} -> {i+1}...', end='', level='full')
                    with self._suppress_output():
                        xyz_df = navis.xform_brain(xyz_df, source=template_info['source'], target=template_info['target'])
                
                # Apply FAFB tilt correction if using template mode
                # This corrects the left-right tilt in the FLYWIRE template mesh
                is_fafb = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()
                if is_fafb and self.brain_mesh == 'template':
                    xyz_df = self._apply_fafb_tilt_correction(xyz_df)
                
                # Retrieve colors
                if is_color_array and '__color' in xyz_df.columns:
                    plot_colors = xyz_df['__color'].tolist()
                else:
                    plot_colors = self.synapse_colors[i]

                # Plotly applies marker opacity in addition to any alpha in
                # an rgba() color. Strip the embedded alpha here so the
                # layer's resolved alpha is applied exactly once below.
                if isinstance(plot_colors, str):
                    plot_colors = self._get_opaque_color(plot_colors)
                elif isinstance(plot_colors, (list, tuple, np.ndarray)):
                    plot_colors = [self._get_opaque_color(color) for color in plot_colors]
                
                if self.backend == 'plotly':

                    # A single scatter marker per synapse layer with the layer's
                    # alpha applied uniformly. Scatter markers are screen-space
                    # pixels — they have no physical radius — so rendering a
                    # soft "center-surround" gradient would need several stacked
                    # traces (inflating the HTML) while also producing a
                    # non-uniform center/edge alpha. A plain marker keeps the
                    # HTML small and applies the alpha once.
                    base_alpha = self._extract_alpha_from_color(self.synapse_colors[i])
                    current_size = self._scatter_synapse_marker_size()

                    sp = go.Scatter3d(
                        x = xyz_df['x'],
                        y = xyz_df['y'],
                        z = xyz_df['z'],
                        mode = 'markers',
                        name = f'synapses {i} -> {i+1} ({len(conn_df)})',
                        hoverinfo = 'name',
                        hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z}<br>name: %{fullData.name}<extra></extra>',
                        legendgroup = f'synapses {i} -> {i+1} ({len(conn_df)})',
                        showlegend = True,
                        marker = dict(
                            size = current_size,
                            color = plot_colors,
                            symbol = 'circle',
                            opacity = base_alpha,
                        ),
                        # Private metadata lets the figure-level Plotly
                        # slider target only synapse scatter traces, not
                        # morphology or legend markers.
                        meta = {
                            'drocat_scatter_size_role': 'synapse',
                            'drocat_scatter_size_factor': 1.0,
                        },
                    )
                    self.fig_3d.add_trace(sp)
                elif self.backend == 'k3d':
                    try:
                        import k3d
                        # import numpy as np # Removed to avoid UnboundLocalError
                        
                        # Color conversion helper
                        def to_int_color(c):
                            try:
                                if isinstance(c, (list, tuple, np.ndarray)) and len(c) == 1:
                                    return to_int_color(c[0])
                                r, g, b, _ = extract_rgba_tuple(
                                    c, default_alpha=self.synapse_alpha
                                )
                                return (int(r) << 16) + (int(g) << 8) + int(b)
                            except (TypeError, ValueError, IndexError):
                                return 0xff0000  # Fallback red

                        # Determine if we have per-point colors or single color
                        c_val = plot_colors
                        colors_to_pass = None
                        
                        # Check if c_val is a list/array of colors matching the number of points
                        # Note: A single RGB tuple (r,g,b) has len 3, but we shouldn't treat it as 3 points if len(xyz_df) != 3
                        is_array_of_colors = False
                        if isinstance(c_val, (list, np.ndarray)):
                            if len(c_val) == len(xyz_df) and len(xyz_df) > 0:
                                # It matches length, but is it a list of colors or a single RGB tuple?
                                # If len(xyz_df) == 3, it's ambiguous. Assume RGB tuple if elements are numbers.
                                first_elem = c_val[0]
                                if isinstance(first_elem, (str, list, tuple, np.ndarray)):
                                    is_array_of_colors = True
                                elif len(xyz_df) != 3: # If not 3 points, it must be array of colors
                                    is_array_of_colors = True
                                # If len is 3 and elements are numbers, assume single RGB color (default behavior)

                        if is_array_of_colors:
                            # Convert each color to int
                            colors_to_pass = [to_int_color(c) for c in c_val]
                            # k3d expects uint32 array for per-point colors
                            colors_to_pass = np.array(colors_to_pass, dtype=np.uint32)
                        else:
                            # Single color
                            colors_to_pass = to_int_color(c_val)

                        # Standardized colors carry either their explicit
                        # alpha or the global synapse_alpha fallback.
                        k3d_synapse_opacity = self._extract_alpha_from_color(
                            self.synapse_colors[i]
                        )

                        pts = k3d.points(
                            positions=xyz_df[['x', 'y', 'z']].values.astype(np.float32),
                            point_size=(
                                self._scatter_synapse_marker_size()
                                if self.synapse_mode == 'scatter'
                                else float(self.synapse_size) / 10.0
                            ),
                            color=colors_to_pass,
                            opacity=k3d_synapse_opacity,
                            name=f'synapses {i} -> {i+1} ({len(conn_df)})'
                        )
                        self.fig_3d += pts
                    except Exception as e:
                        self._vprint(f'⚠️  k3d synapse plotting failed: {e}', level='full')
            
            elif self.synapse_mode in ['sphere', 'cone', 'tetrahedron'] and self.backend == 'plotly':
                pre_coords = conn_df[['x_pre', 'y_pre', 'z_pre']].rename(columns={'x_pre':'x', 'y_pre':'y', 'z_pre':'z'})
                post_coords = conn_df[['x_post', 'y_post', 'z_post']].rename(columns={'x_post':'x', 'y_post':'y', 'z_post':'z'})
                
                # Transform synapses only if needed (skip for FAFB native coords)
                if self._needs_skeleton_transform():
                    template_info = self._get_template_info()
                    self._vprint(f'Transforming synapses of layer {i} -> {i+1}...', end='', level='full')
                    # Keep synapse transforms off the progress row: navis may
                    # print its own internal bars otherwise.
                    with self._suppress_output():
                        pre_coords = navis.xform_brain(pre_coords, source=template_info['source'], target=template_info['target'])
                        post_coords = navis.xform_brain(post_coords, source=template_info['source'], target=template_info['target'])
                
                # Apply FAFB tilt correction if using template mode
                # This corrects the left-right tilt in the FLYWIRE template mesh
                is_fafb = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()
                if is_fafb and self.brain_mesh == 'template':
                    pre_coords = self._apply_fafb_tilt_correction(pre_coords)
                    post_coords = self._apply_fafb_tilt_correction(post_coords)
                
                # Calculate sizes
                # Calculate Euclidean distance
                diff = pre_coords[['x', 'y', 'z']].values - post_coords[['x', 'y', 'z']].values
                dists = np.linalg.norm(diff, axis=1)
                
                if self.uniform_synapse_size and dists.size:
                    # One shared marker size: the median distance of this
                    # layer's connections replaces each individual distance.
                    dists = np.full_like(dists, float(np.median(dists)))
                
                if self.synapse_size == 'real':
                    multiplier = 1.0
                else:
                    try:
                        multiplier = float(self.synapse_size)
                    except ValueError:
                        multiplier = 1.0 # Default fallback
                
                # Adjust for template geometry in statvis.py to ensure size=1.0 matches real distance
                # Cone: template height 2 (-1 to 1) -> divide by 2
                # Sphere: template diameter 2 (radius 1) -> divide by 2
                # Tetrahedron: template height 3 (-1.5 to 1.5) -> divide by 3
                if self.synapse_mode == 'tetrahedron':
                    current_size = (dists * multiplier) / 3.0
                else:
                    # Cone and Sphere (diameter)
                    current_size = (dists * multiplier) / 2.0
                
                # Standardized colors carry either their explicit alpha or the
                # global synapse_alpha fallback.
                # A custom layer map may assign connector colors per pre
                # neuron. Mesh3d accepts one color per trace, so split the
                # connections into color groups instead of silently using the
                # fallback layer color for every solid synapse.
                resolved_synapse_colors = [
                    self._resolve_synapse_color(pre_id, i)
                    for pre_id in conn_df['bodyId_pre']
                ] if (
                    'bodyId_pre' in conn_df.columns
                    and getattr(self, '_neuron_synapse_color_overrides', None)
                ) else [self.synapse_colors[i]] * len(conn_df)
                color_groups = {}
                for position, color in enumerate(resolved_synapse_colors):
                    color_groups.setdefault(str(color), []).append(position)

                synapse_name = f'synapses {i} -> {i+1} ({len(conn_df)})'
                legend_group = f'synapses {i} -> {i+1}'
                for color_key, positions in color_groups.items():
                    group_positions = np.asarray(positions, dtype=int)
                    group_pre = pre_coords.iloc[group_positions]
                    group_post = post_coords.iloc[group_positions]
                    group_size = current_size[group_positions]
                    group_opacity = self._extract_alpha_from_color(color_key)

                    mesh = sv.build_synapse_mesh(
                        group_pre,
                        group_post,
                        mode=self.synapse_mode,
                        size=group_size,
                        # Mesh3d also combines color alpha with opacity; keep
                        # the color opaque and apply the resolved alpha once.
                        color=self._get_opaque_color(color_key),
                        opacity=group_opacity,
                        name=synapse_name,
                    )
                    mesh.hoverinfo = 'name'
                    mesh.legendgroup = legend_group
                    mesh.hovertemplate = '<b>%{fullData.name}</b><extra></extra>'
                    mesh.showlegend = False
                    self.fig_3d.add_trace(mesh)
                    self._append_exportable_mesh(
                        mesh,
                        color=color_key,
                        alpha=group_opacity,
                        name=synapse_name,
                        role='synapse',
                    )

                # Add dummy scatter trace for legend with opaque color for visibility
                opaque_synapse_color = self._get_opaque_color(
                    resolved_synapse_colors[0]
                )
                dummy_legend = go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    name=synapse_name,
                    legendgroup=legend_group,
                    showlegend=True,
                    marker=dict(
                        size=10,
                        color=opaque_synapse_color,  # Use opaque color for legend visibility
                        symbol='circle'
                    )
                )
                self.fig_3d.add_trace(dummy_legend)
            self._vprint('Done', level='full')

        self._add_plotly_synapse_size_slider()
        self._save_synapse_data(synapse_frames)
        return 0

    # ------------------------------------------------------------------ pre/post sites
    def _scatter_synapse_marker_size(self):
        """Return the visible Plotly/K3D scatter size in screen-space pixels."""
        return max(1.0, self._synapse_size_px())

    def _synapse_size_px(self):
        """Return the requested synapse size as a pixel value (1-12, default 1).

        The Synapse Size control is now expressed directly in pixels (the UI
        offers 1-12, default 1), so scatter markers use this value as their
        Plotly marker size.
        """
        parsed = self._parse_synapse_size(str(self.synapse_size))
        if parsed in (None, 'real'):
            return 1.0
        return max(1.0, min(12.0, float(parsed)))

    def _synapse_size_fold(self):
        """Return the requested size as a non-negative real-size multiplier.

        Mesh (sphere/cone/tetrahedron / solid pre-post) markers are sized in
        world units from the real pre->post distance; the numeric Synapse Size
        value is used directly as that multiplier (3 = 3x the real distance), so
        mesh sizing is unchanged by the px-based scatter control.
        """
        parsed = self._parse_synapse_size(str(self.synapse_size))
        if parsed in (None, 'real'):
            return 1.0
        return max(0.0, float(parsed))

    def _add_plotly_synapse_size_slider(self):
        """Add a Plotly slider for screen-space scatter synapse/site sizes.

        Solid cone/sphere/tetrahedron markers are sized from the real
        pre-to-post distance and remain intentionally static.  Scatter
        markers have no physical radius in Plotly, so the slider changes their
        visible pixel size in the exported interactive figure (1-12 px, step 1,
        default 3).  In pre/post mode the same control adjusts both the diamond
        pre-sites and circular post-sites.
        """
        self._plotly_sliders = []
        if self.backend != 'plotly' or not hasattr(self.fig_3d, 'data'):
            return
        if not (
            self.synapse_mode == 'scatter'
            or (
                self.synapse_mode == 'pre_post'
                and getattr(self, 'pre_post_scatter', False)
            )
        ):
            return

        trace_specs = []
        for trace_index, trace in enumerate(self.fig_3d.data):
            meta = getattr(trace, 'meta', None)
            if not isinstance(meta, dict):
                continue
            role = meta.get('drocat_scatter_size_role')
            if role not in {'synapse', 'pre_post_site'}:
                continue
            trace_specs.append(
                (
                    trace_index,
                    role,
                    float(meta.get('drocat_scatter_size_factor', 1.0)),
                )
            )
        if not trace_specs:
            return

        # Slider is a 1-12 px integer ruler, defaulting to the current size.
        current_px = self._synapse_size_px()
        px_values = [float(value) for value in range(1, 13)]
        trace_indices = [spec[0] for spec in trace_specs]
        steps = []
        for px in px_values:
            sizes = []
            for _trace_index, role, size_factor in trace_specs:
                if role == 'synapse':
                    size = px * size_factor
                else:
                    # Scatter has no world-space radius, so apply the same
                    # pixel size uniformly to every site.
                    size = px
                sizes.append(round(size, 4))
            steps.append({
                'label': f"{px:g} px",
                'method': 'restyle',
                'args': [{'marker.size': sizes}, trace_indices],
            })

        active = max(0, min(len(px_values) - 1, int(round(current_px)) - 1))
        self._plotly_sliders = [{
            'active': active,
            'currentvalue': {'prefix': 'Synapse size: '},
            'pad': {'t': 24},
            'x': 0.12,
            'len': 0.76,
            'steps': steps,
        }]
        self.fig_3d.update_layout(sliders=self._plotly_sliders)

    def _pre_post_site_scale(self):
        """Return one uniform pre/post site size for this visualization.

        Paired synapse markers are centered between their pre/post endpoints,
        so their mesh scale is normalized differently.  A pre/post-site marker
        is centered directly on its own pre- or post-synaptic coordinate; use
        the full mean real connector distance as its visual size and apply the
        requested ``synapse_size`` fold uniformly to every site.
        """
        mean_distance = self._estimate_pre_post_real_synapse_distance()
        return mean_distance * self._synapse_size_fold()

    @staticmethod
    def _sample_size_estimation_frame(frame, max_rows=10000):
        """Keep size estimation bounded while retaining a deterministic sample."""
        import pandas as pd
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        required = {
            'bodyId_pre', 'bodyId_post',
            'x_pre', 'y_pre', 'z_pre',
            'x_post', 'y_post', 'z_post',
        }
        if not required.issubset(frame.columns):
            return None
        sampled = frame.dropna(subset=list(required)).copy()
        if sampled.empty:
            return None
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=0)
        return sampled

    def _connection_distance_values(self, frame):
        """Return transformed real connector distances from a paired frame."""
        import numpy as np
        sampled = self._sample_size_estimation_frame(frame)
        if sampled is None:
            return np.array([], dtype=float)
        pre = sampled[['x_pre', 'y_pre', 'z_pre']].rename(
            columns={'x_pre': 'x', 'y_pre': 'y', 'z_pre': 'z'}
        )
        post = sampled[['x_post', 'y_post', 'z_post']].rename(
            columns={'x_post': 'x', 'y_post': 'y', 'z_post': 'z'}
        )
        # Match plot_synapses: estimate size in the same transformed coordinate
        # space in which the final markers are rendered.
        pre = self._transform_site_df(pre)
        post = self._transform_site_df(post)
        distances = np.linalg.norm(
            pre[['x', 'y', 'z']].to_numpy(dtype=float)
            - post[['x', 'y', 'z']].to_numpy(dtype=float),
            axis=1,
        )
        return distances[np.isfinite(distances) & (distances > 0)]

    def _read_flywire_connection_frame(
            self, source_ids=None, target_ids=None, touching_ids=None,
            min_synapse_num=None):
        """Read a standardized paired FlyWire synapse-cache frame.

        ``source_ids``/``target_ids``/``touching_ids`` are pushed into the
        Parquet reader when supported, then applied again in pandas as a
        correctness guard.  This keeps pre/post size estimation and site
        rendering from loading the entire connection table for every layer.
        ``min_synapse_num`` is an explicit cache/query threshold override used
        by the pre/post size estimator so its real-size baseline does not
        change with the display cutoff.
        """
        import pandas as pd
        threshold = (
            int(getattr(self, 'min_synapse_num', 0) or 0)
            if min_synapse_num is None else int(min_synapse_num or 0)
        )
        parquet_file = self._get_synapse_table_path()
        if parquet_file is None:
            dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
            dataset_dir = os.path.join(self.script_path, 'datasets', dataset_normalized)
            candidate = os.path.join(
                dataset_dir, f"{dataset_normalized}_synapse_table.parquet")
            parquet_file = candidate if os.path.exists(candidate) else None
        if parquet_file is None or not os.path.exists(parquet_file):
            return None
        import pyarrow.parquet as pq
        try:
            source_set = {str(value) for value in (source_ids or [])}
            target_set = {str(value) for value in (target_ids or [])}
            touching_set = {str(value) for value in (touching_ids or [])}
            stat = os.stat(parquet_file)
            memory_key = (
                str(parquet_file), stat.st_size, stat.st_mtime_ns,
                tuple(sorted(source_set)), tuple(sorted(target_set)),
                tuple(sorted(touching_set)), threshold,
            )
            memory = getattr(self, '_flywire_synapse_frame_memory', None)
            if memory is None:
                memory = {}
                self._flywire_synapse_frame_memory = memory
            if memory_key in memory:
                return memory[memory_key].copy()

            schema = pq.read_schema(parquet_file)
            pre_col = next((c for c in schema.names
                            if c.startswith(('pre_root_id', 'pre_pt_root_id'))), None)
            post_col = next((c for c in schema.names
                             if c.startswith(('post_root_id', 'post_pt_root_id'))), None)
            if not (pre_col and post_col):
                return None
            coord_cols = ['x_pre', 'y_pre', 'z_pre', 'x_post', 'y_post', 'z_post']
            alt_map = {
                'x_pre': ['pre_x', 'pre_pt_x'], 'y_pre': ['pre_y', 'pre_pt_y'],
                'z_pre': ['pre_z', 'pre_pt_z'], 'x_post': ['post_x', 'post_pt_x'],
                'y_post': ['post_y', 'post_pt_y'], 'z_post': ['post_z', 'post_pt_z'],
            }
            use = {}
            for target in coord_cols:
                if target in schema.names:
                    use[target] = target
                else:
                    for alt in alt_map[target]:
                        if alt in schema.names:
                            use[target] = alt
                            break
            if len(use) != len(coord_cols):
                return None
            # Infer voxel-vs-nanometre units from row-group statistics before
            # filtering.  Looking only at the selected rows can misclassify a
            # perfectly valid nanometre subset whose z range happens to be
            # below the historical 10,000 threshold.
            coordinate_scale = None
            try:
                parquet = pq.ParquetFile(parquet_file)
                z_index = schema.names.index(use['z_pre'])
                maxima = []
                for row_index in range(parquet.metadata.num_row_groups):
                    stats = parquet.metadata.row_group(row_index).column(
                        z_index).statistics
                    has_max = bool(
                        getattr(stats, 'has_max',
                                getattr(stats, 'has_min_max', False))
                    ) if stats is not None else False
                    if has_max:
                        maxima.append(float(stats.max))
                if maxima:
                    coordinate_scale = (
                        (1.0, 1.0, 1.0)
                        if max(maxima) > 10000
                        else (4.0, 4.0, 40.0)
                    )
            except Exception:
                # Older writers may omit statistics; use the selected frame
                # as a conservative fallback after the filtered read.
                coordinate_scale = None
            weight_col = next(
                (c for c in schema.names if c in ('weight', 'syn_count', 'synapses')),
                None,
            )
            read_cols = list(use.values()) + [pre_col, post_col]
            if weight_col is not None:
                read_cols.append(weight_col)
            # Match filter literal types to the Parquet schema.  Passing
            # stringified FlyWire root IDs to an integer column makes pandas
            # abandon predicate pushdown and scan the complete table.
            try:
                import pyarrow as pa

                def _filter_ids(column, values):
                    arrow_type = schema.field(column).type
                    if pa.types.is_integer(arrow_type):
                        try:
                            return [int(value) for value in values]
                        except (TypeError, ValueError):
                            # Let the read fallback handle an unusual mixed
                            # identifier column rather than dropping all site
                            # data because one filter literal is non-numeric.
                            return list(values)
                    return list(values)
            except Exception:
                def _filter_ids(column, values):
                    return list(values)

            filters = None
            if source_set and target_set:
                filters = [[
                    (pre_col, 'in', _filter_ids(pre_col, source_set)),
                    (post_col, 'in', _filter_ids(post_col, target_set)),
                ]]
            elif source_set:
                filters = [[(pre_col, 'in', _filter_ids(pre_col, source_set))]]
            elif target_set:
                filters = [[(post_col, 'in', _filter_ids(post_col, target_set))]]
            elif touching_set:
                filters = [
                    [(pre_col, 'in', _filter_ids(pre_col, touching_set))],
                    [(post_col, 'in', _filter_ids(post_col, touching_set))],
                ]
            if weight_col is not None and threshold > 0:
                weight_filter = (weight_col, '>=', threshold)
                if filters is None:
                    filters = [[weight_filter]]
                else:
                    for clause in filters:
                        clause.append(weight_filter)
            try:
                frame = pd.read_parquet(
                    parquet_file, columns=read_cols, filters=filters)
            except Exception:
                # Some Parquet writers store root IDs as uint64 while the
                # caller's IDs are strings; retry without predicate pushdown
                # and apply the exact string filter below.
                frame = pd.read_parquet(parquet_file, columns=read_cols)
            frame = frame.rename(columns={value: key for key, value in use.items()})
            if weight_col is not None:
                frame = frame[frame[weight_col] >= threshold]
            frame[pre_col] = frame[pre_col].astype(str)
            frame[post_col] = frame[post_col].astype(str)
            if coordinate_scale is None and not frame.empty:
                coordinate_scale = (
                    (1.0, 1.0, 1.0)
                    if float(frame['z_pre'].max()) > 10000
                    else (4.0, 4.0, 40.0)
                )
            if coordinate_scale is not None and coordinate_scale != (1.0, 1.0, 1.0):
                for column in coord_cols:
                    frame[column] = frame[column] * (
                        coordinate_scale[2]
                        if column.startswith('z_') else coordinate_scale[0]
                    )
            frame = frame.rename(columns={pre_col: 'bodyId_pre', post_col: 'bodyId_post'})
            if source_set:
                frame = frame[frame['bodyId_pre'].isin(source_set)]
            if target_set:
                frame = frame[frame['bodyId_post'].isin(target_set)]
            if touching_set:
                frame = frame[
                    frame['bodyId_pre'].isin(touching_set)
                    | frame['bodyId_post'].isin(touching_set)
                ]
            memory[memory_key] = frame.copy()
            return frame
        except Exception as exc:
            self._vprint(
                f'  ⚠️ Failed to read FlyWire synapse table for size estimation: {exc}',
                level='full',
            )
            return None

    def _fetch_pre_post_size_frame(
            self, source_criteria, target_criteria,
            source_ids=None, target_ids=None, min_total_weight=None):
        """Fetch a paired frame for sizing without making pre/post rendering fragile."""
        if source_criteria is None and target_criteria is None:
            return None
        return self._get_or_fetch_synapse_query(
            source_criteria=source_criteria,
            target_criteria=target_criteria,
            source_ids=source_ids,
            target_ids=target_ids,
            source_kind='neuprint',
            min_total_weight=min_total_weight,
        )

    def _pre_post_size_criteria(self, layer_index):
        """Return the layer query or an exact body-id fallback for sizing."""
        if layer_index < len(self.layer_criteria):
            criteria = self.layer_criteria[layer_index]
            if criteria is not None:
                return criteria
        if layer_index >= len(self.neuron_dfs):
            return None
        frame = self.neuron_dfs[layer_index]
        if frame is None or 'bodyId' not in frame.columns or frame.empty:
            return None
        body_ids = [value for value in frame['bodyId'].tolist() if value is not None]
        if not body_ids:
            return None
        try:
            from neuprint import NeuronCriteria
            return NeuronCriteria(bodyId=body_ids)
        except Exception as exc:
            self._vprint(
                f'  ⚠️ Could not build body-id criteria for pre/post sizing: {exc}',
                level='full',
            )
            return None

    def _estimate_pre_post_real_synapse_distance(self):
        """Estimate the universal real synapse distance for pre/post sites.

        With adjacent layers, use their actual inter-layer connector pairs.
        With one layer, query both downstream and upstream connectors so the
        estimate still comes from real pair distances rather than a hard-coded
        glyph size.  Only a bounded deterministic sample is used per query.
        """
        import numpy as np
        cached = getattr(self, '_pre_post_real_synapse_distance', None)
        if cached is not None:
            return cached

        frames = []
        layer_count = len(self.neuron_layers)
        if self.client_type == 'flywire':
            if layer_count > 1:
                for index in range(layer_count - 1):
                    source_ids = {
                        str(value) for value in self.neuron_dfs[index]['bodyId'].tolist()
                    }
                    target_ids = {
                        str(value) for value in self.neuron_dfs[index + 1]['bodyId'].tolist()
                    }
                    frame = self._read_flywire_connection_frame(
                        source_ids=source_ids, target_ids=target_ids,
                        min_synapse_num=0,
                    )
                    if frame is not None:
                        frames.append(frame)
            elif layer_count == 1:
                layer_ids = {
                    str(value) for value in self.neuron_dfs[0]['bodyId'].tolist()
                }
                # Query both directions; pre/post mode deliberately shows
                # every site owned by the queried neurons, including partners
                # outside the displayed layer.
                for kwargs in (
                    {'source_ids': layer_ids},
                    {'target_ids': layer_ids},
                ):
                    frame = self._read_flywire_connection_frame(
                        **kwargs, min_synapse_num=0,
                    )
                    if frame is not None:
                        frames.append(frame)
        else:
            if layer_count > 1:
                for index in range(layer_count - 1):
                    source_criteria = self._pre_post_size_criteria(index)
                    target_criteria = self._pre_post_size_criteria(index + 1)
                    source_frame = (
                        self.neuron_dfs[index]
                        if index < len(self.neuron_dfs) else None
                    )
                    target_frame = (
                        self.neuron_dfs[index + 1]
                        if index + 1 < len(self.neuron_dfs) else None
                    )
                    source_ids = (
                        source_frame['bodyId'].tolist()
                        if source_frame is not None
                        and 'bodyId' in source_frame.columns else None
                    )
                    target_ids = (
                        target_frame['bodyId'].tolist()
                        if target_frame is not None
                        and 'bodyId' in target_frame.columns else None
                    )
                    frame = self._fetch_pre_post_size_frame(
                        source_criteria, target_criteria,
                        source_ids=source_ids, target_ids=target_ids,
                        min_total_weight=0,
                    )
                    if frame is not None:
                        frames.append(frame)
            elif layer_count == 1:
                layer_criteria = self._pre_post_size_criteria(0)
                if layer_criteria is not None:
                    layer_ids = self.neuron_dfs[0]['bodyId'].tolist()
                    for source_criteria, target_criteria in (
                        (layer_criteria, None),
                        (None, layer_criteria),
                    ):
                        source_ids = layer_ids if target_criteria is None else None
                        target_ids = layer_ids if source_criteria is None else None
                        frame = self._fetch_pre_post_size_frame(
                            source_criteria, target_criteria,
                            source_ids=source_ids, target_ids=target_ids,
                            min_total_weight=0,
                        )
                        if frame is not None:
                            frames.append(frame)

        if frames:
            import pandas as pd
            required_columns = {
                'bodyId_pre', 'bodyId_post',
                'x_pre', 'y_pre', 'z_pre',
                'x_post', 'y_post', 'z_post',
            }
            frames = [
                frame for frame in frames
                if isinstance(frame, pd.DataFrame)
                and required_columns.issubset(frame.columns)
            ]
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.drop_duplicates(subset=[
                'bodyId_pre', 'bodyId_post',
                'x_pre', 'y_pre', 'z_pre',
                'x_post', 'y_post', 'z_post',
            ])
            distances = self._connection_distance_values(combined)
        else:
            distances = np.array([], dtype=float)
        if distances.size:
            mean_distance = float(np.mean(distances))
            if np.isfinite(mean_distance) and mean_distance > 0:
                self._pre_post_real_synapse_distance = mean_distance
                self._pre_post_real_synapse_distance_sample_count = len(distances)
                self._vprint(
                    f'  Pre/post site size baseline: mean real synapse distance '
                    f'{mean_distance:.3f} from {len(distances)} sampled pairs',
                    level='simple',
                )
                return mean_distance

        # Preserve the previous nominal baseline size when no connector source
        # is available (e.g. an offline empty dataset), but make the fallback
        # explicit so normal runs always use real pair distances.
        fallback_distance = 40.0
        self._pre_post_real_synapse_distance = fallback_distance
        self._pre_post_real_synapse_distance_sample_count = 0
        self._vprint(
            '  ⚠️ No real connector pairs available for pre/post site sizing; '
            f'using fallback distance {fallback_distance:.1f}',
            level='simple',
        )
        return fallback_distance

    def _collect_layer_sites(self, layer_idx):
        """Return (pre_sites, post_sites) long-form DataFrames for a layer.

        Each frame has columns ``x``/``y``/``z`` (the site coordinate) and
        ``neuron_id`` (the owning neuron). Returns ``(None, None)`` when the
        layer has no resolvable neurons or no available synapse source.
        """
        ndf = self.neuron_dfs[layer_idx] if layer_idx < len(self.neuron_dfs) else None
        if ndf is None or 'bodyId' not in ndf.columns or ndf.empty:
            return None, None
        layer_ids = set(str(b) for b in ndf['bodyId'].tolist())

        if self.client_type == 'flywire':
            site_df = self._read_flywire_site_frame(layer_ids)
        else:
            site_df = self._fetch_neuprint_site_frame(layer_idx, layer_ids)
        if site_df is None or site_df.empty:
            return None, None

        pre = site_df[site_df['role'] == 'pre'][['x', 'y', 'z', 'neuron_id']]
        post = site_df[site_df['role'] == 'post'][['x', 'y', 'z', 'neuron_id']]
        return pre, post

    def _read_flywire_site_frame(self, layer_ids):
        """Read all synapses touching *layer_ids* from the FlyWire parquet table.

        Returns a long-form frame (role in {pre, post}), or ``None``.
        """
        import pandas as pd
        standardized = self._read_flywire_connection_frame(
            touching_ids=layer_ids)
        if standardized is not None:
            pre_mask = standardized['bodyId_pre'].isin(layer_ids)
            post_mask = standardized['bodyId_post'].isin(layer_ids)
            long_pre = standardized[pre_mask][
                ['x_pre', 'y_pre', 'z_pre', 'bodyId_pre']
            ].rename(columns={
                'x_pre': 'x', 'y_pre': 'y', 'z_pre': 'z',
                'bodyId_pre': 'neuron_id',
            })
            long_pre['role'] = 'pre'
            long_post = standardized[post_mask][
                ['x_post', 'y_post', 'z_post', 'bodyId_post']
            ].rename(columns={
                'x_post': 'x', 'y_post': 'y', 'z_post': 'z',
                'bodyId_post': 'neuron_id',
            })
            long_post['role'] = 'post'
            return pd.concat([long_pre, long_post], ignore_index=True)
        # The standardized paired reader is the only FlyWire/FAFB source of
        # site rows.  Keeping a second schema/scale implementation here would
        # allow plotting and pre/post mode to disagree and would defeat its
        # filtered-read memoization.
        return None

    def _fetch_neuprint_site_frame(self, layer_idx, layer_ids):
        """Fetch all synapses where a layer neuron is pre or post (NeuPrint).

        Returns a long-form frame (role in {pre, post}), or ``None``.
        """
        import pandas as pd
        source_criteria = self.layer_criteria[layer_idx]
        rows = []
        try:
            pre_df = self._get_or_fetch_synapse_query(
                source_criteria=source_criteria,
                target_criteria=None,
                source_ids=layer_ids,
                source_kind='neuprint',
            )
            if pre_df is not None and not pre_df.empty and 'bodyId_pre' in pre_df.columns:
                pre_pdf = pre_df[['x_pre', 'y_pre', 'z_pre', 'bodyId_pre']].rename(
                    columns={'x_pre': 'x', 'y_pre': 'y', 'z_pre': 'z', 'bodyId_pre': 'neuron_id'}
                )
                pre_pdf['role'] = 'pre'
                rows.append(pre_pdf)
            post_df = self._get_or_fetch_synapse_query(
                source_criteria=None,
                target_criteria=source_criteria,
                target_ids=layer_ids,
                source_kind='neuprint',
            )
            if post_df is not None and not post_df.empty and 'bodyId_post' in post_df.columns:
                post_pdf = post_df[['x_post', 'y_post', 'z_post', 'bodyId_post']].rename(
                    columns={'x_post': 'x', 'y_post': 'y', 'z_post': 'z', 'bodyId_post': 'neuron_id'}
                )
                post_pdf['role'] = 'post'
                rows.append(post_pdf)
        except Exception as exc:
            self._vprint(f'  ⚠️ Failed to fetch pre/post sites for layer {layer_idx}: {exc}', level='full')
            return None
        if not rows:
            return None
        return pd.concat(rows, ignore_index=True)

    def _transform_site_df(self, site_df):
        """Apply the same transform + FAFB tilt correction used for synapse data."""
        if site_df is None or site_df.empty:
            return site_df
        coords = site_df[['x', 'y', 'z']].astype(float)
        if self._needs_skeleton_transform():
            template_info = self._get_template_info()
            with self._suppress_output():
                coords = navis.xform_brain(coords, source=template_info['source'], target=template_info['target'])
        is_fafb = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()
        if is_fafb and self.brain_mesh == 'template':
            coords = self._apply_fafb_tilt_correction(coords)
        return site_df.assign(x=coords['x'], y=coords['y'], z=coords['z'])

    def _plot_pre_post_sites(self):
        """Render pre- and post-synaptic SITES of every queried neuron.

        Unlike connector (inter-layer) mode, this plots the actual input/output
        sites of all queried neurons across every layer: post-synaptic sites
        (inputs) render as spheres (circles in scatter) and pre-synaptic sites
        (outputs) as cones (squares in scatter). Each site is colored by its
        neuron's layer color (or a per-neuron override) and follows the same
        ``legend_mode`` contract as morphology traces. It is not a
        paired-synapse view: the partner is not restricted to the adjacent
        layer.
        """
        import pandas as pd
        if self.backend == 'plotly':
            self._plotly_sliders = []
            self._pre_post_seen_legend_groups = set()
            self._pre_post_dynamic_legend_ranks = {}
        site_frames = []
        layer_site_groups = []
        site_scale = self._pre_post_site_scale()

        for i in range(len(self.neuron_layers)):
            layer_name = self.layer_names[i] if i < len(self.layer_names) else f'layer_{i}'
            pre_df, post_df = self._collect_layer_sites(i)
            if pre_df is None and post_df is None:
                self._vprint(f'  No connector data for layer {i} ({layer_name})', level='full')
                continue
            pre_df = self._transform_site_df(pre_df)
            post_df = self._transform_site_df(post_df)

            if pre_df is not None and not pre_df.empty:
                export = pre_df.copy()
                export['viz_layer'] = f'{i}:pre'
                site_frames.append(export)
            if post_df is not None and not post_df.empty:
                export = post_df.copy()
                export['viz_layer'] = f'{i}:post'
                site_frames.append(export)

            layer_site_groups.append((i, layer_name, pre_df, post_df))

        # ``plot_skeleton`` has already emitted the morphology traces.  In
        # single-neuron mode, keep each owner's pre/post pair adjacent in the
        # legend (and in trace insertion order for standalone callers) instead
        # of emitting every pre site followed by every post site.  Layer/type
        # modes intentionally stay merged at their requested legend level.
        if str(getattr(self, 'legend_mode', 'layer') or 'layer').lower() == 'single':
            for layer_idx, layer_name, pre_df, post_df in layer_site_groups:
                owner_values = []
                frame = (
                    self.neuron_dfs[layer_idx]
                    if layer_idx < len(getattr(self, 'neuron_dfs', []))
                    else None
                )
                if frame is not None and 'bodyId' in frame.columns:
                    owner_values.extend(frame['bodyId'].tolist())
                # Include any connector owner that was not present in the
                # metadata frame as a deterministic tail rather than dropping
                # its sites.
                for site_df in (pre_df, post_df):
                    if site_df is None or site_df.empty:
                        continue
                    for value in site_df['neuron_id'].tolist():
                        if not any(
                            set(self._normalize_neuron_lookup_keys(value))
                            & set(self._normalize_neuron_lookup_keys(existing))
                            for existing in owner_values
                        ):
                            owner_values.append(value)
                for owner in owner_values:
                    for site_df, site_type in (
                        (pre_df, 'pre'), (post_df, 'post')
                    ):
                        owned = self._site_rows_for_owner(site_df, owner)
                        if owned is None or owned.empty:
                            continue
                        self._plot_site_group(
                            owned, site_type, layer_idx, layer_name,
                            site_scale=site_scale,
                        )
        else:
            for layer_idx, layer_name, pre_df, post_df in layer_site_groups:
                for site_df, site_type in (
                    (pre_df, 'pre'), (post_df, 'post')
                ):
                    if site_df is None or site_df.empty:
                        continue
                    self._plot_site_group(
                        site_df, site_type, layer_idx, layer_name,
                        site_scale=site_scale,
                    )

        self._add_plotly_synapse_size_slider()
        self._save_synapse_data(site_frames)
        return 0

    @staticmethod
    def _clean_pre_post_type_label(value):
        """Normalize metadata to the same type-level label used by morphology."""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        label = str(value or "").strip()
        if not label or label.lower() in {"nan", "none"}:
            return ""
        # Some NeuPrint/MCNS metadata paths expose a type-like value with the
        # instance/body suffix attached (for example ``s-LNv_15832.0``).  The
        # morphology legend is type-level, so remove only that terminal numeric
        # suffix and preserve meaningful digits inside the type name.
        return re.sub(r"_[0-9]+(?:\.0+)?$", "", label)

    def _pre_post_site_metadata_row(self, neuron_id, layer_idx):
        """Find one site's owner row using numeric-safe body-ID matching."""
        frame = (
            self.neuron_dfs[layer_idx]
            if layer_idx < len(getattr(self, 'neuron_dfs', []))
            else None
        )
        if frame is None or frame.empty or 'bodyId' not in frame.columns:
            return None

        target_keys = {
            str(key).strip()
            for key in self._normalize_neuron_lookup_keys(neuron_id)
            if str(key).strip()
        }
        try:
            numeric_value = float(str(neuron_id).strip())
            if numeric_value.is_integer():
                target_keys.add(str(int(numeric_value)))
        except (TypeError, ValueError):
            pass

        for _, row in frame.iterrows():
            row_keys = {
                str(key).strip()
                for key in self._normalize_neuron_lookup_keys(row.get('bodyId'))
                if str(key).strip()
            }
            if target_keys.intersection(row_keys):
                return row
        return None

    def _pre_post_site_owner_label(self, neuron_id, layer_idx):
        """Return the best stable type/name label for a site owner."""
        row = self._pre_post_site_metadata_row(neuron_id, layer_idx)
        if row is None:
            return str(neuron_id)
        for column in ('type', 'cell_type', 'neuronType', 'instance', 'name'):
            if column not in row.index:
                continue
            value = self._clean_pre_post_type_label(row[column])
            if value:
                return value
        return str(neuron_id)

    def _pre_post_site_type_label(self, neuron_id, layer_idx):
        """Resolve the exact type-level owner label used by skeleton legends."""
        row = self._pre_post_site_metadata_row(neuron_id, layer_idx)
        if row is not None:
            for column in ('type', 'cell_type', 'neuronType'):
                if column not in row.index:
                    continue
                value = self._clean_pre_post_type_label(row[column])
                if value:
                    return value

        # If the source table has no explicit type column, retain a useful
        # type-like fallback without leaking a body/instance numeric suffix into
        # a type-level legend entry.
        return self._clean_pre_post_type_label(
            self._pre_post_site_owner_label(neuron_id, layer_idx)
        ) or str(neuron_id)

    def _pre_post_site_owner_identity(self, neuron_id, layer_idx, layer_name):
        """Return the morphology legend identity for one site owner.

        Site data is keyed by connector body IDs, while a skeleton trace may
        be named by a type/instance.  ``plot_skeleton`` records aliases for
        the actual legend group; using that map keeps site labels synchronized
        with the visible skeleton legend at every level.
        """
        mode = str(getattr(self, 'legend_mode', 'layer') or 'layer').lower()
        layer = str(layer_name).strip() or f'layer_{layer_idx}'
        label_map = getattr(self, '_neuron_legend_labels_by_layer', {}).get(
            layer_idx, {}
        )
        aliases = list(self._normalize_neuron_lookup_keys(neuron_id))
        row = self._pre_post_site_metadata_row(neuron_id, layer_idx)
        if row is not None:
            for column in (
                'bodyId', 'name', 'type', 'instance', 'roi',
                'flywireType', 'hemibrainType', 'mancType',
            ):
                if column in row.index:
                    aliases.extend(
                        self._normalize_neuron_lookup_keys(row[column])
                    )
        for alias in aliases:
            if alias in label_map:
                return str(label_map[alias])

        # Standalone site rendering (or a caller that did not run
        # plot_skeleton first) still follows the documented legend levels.
        if mode == 'layer':
            return layer
        if mode in ('type', 'tree'):
            return self._pre_post_site_type_label(neuron_id, layer_idx)
        owner = self._pre_post_site_owner_label(neuron_id, layer_idx)
        return f'{owner}_{layer}'

    def _pre_post_site_legend_rank(self, owner_identity, site_type):
        """Place an owner's pre/post entries immediately after its skeleton."""
        skeleton_ranks = getattr(self, '_neuron_legend_ranks', {}) or {}
        base_rank = skeleton_ranks.get(owner_identity)
        if base_rank is None:
            dynamic = getattr(self, '_pre_post_dynamic_legend_ranks', {})
            if owner_identity not in dynamic:
                dynamic[owner_identity] = (
                    len(skeleton_ranks) + len(dynamic)
                ) * 100
                self._pre_post_dynamic_legend_ranks = dynamic
            base_rank = dynamic[owner_identity]
        return base_rank + (1 if str(site_type).lower() == 'pre' else 2)

    def _site_rows_for_owner(self, site_df, owner):
        """Filter a long-form site frame to one owner, preserving row order."""
        if site_df is None or site_df.empty or 'neuron_id' not in site_df.columns:
            return None
        owner_keys = set(self._normalize_neuron_lookup_keys(owner))
        if not owner_keys:
            return site_df.iloc[0:0].copy()
        mask = site_df['neuron_id'].map(
            lambda value: bool(
                owner_keys.intersection(
                    self._normalize_neuron_lookup_keys(value)
                )
            )
        )
        return site_df[mask].copy()

    def _pre_post_site_legend(self, site_type, layer_idx, layer_name,
                              neuron_id):
        """Return a legend group and display name for one site owner.

        The site label is the matching skeleton legend label plus ``_pre`` or
        ``_post``. Layer mode therefore produces ``{layer_name}_{role}``, type
        mode merges by the same type identity as morphology, and single mode
        keeps each skeleton entry distinct. The role is part of every key so
        pre and post sites can never collapse into one legend entry.
        """
        role = str(site_type).strip().lower()
        owner_identity = self._pre_post_site_owner_identity(
            neuron_id, layer_idx, layer_name,
        )
        label = f'{owner_identity}_{role}'
        group = f'pre_post:{role}:{owner_identity}'
        rank = self._pre_post_site_legend_rank(owner_identity, role)
        return group, label, rank

    def _plot_site_group(self, site_df, site_type, layer_idx, layer_name,
                         site_scale):
        """Render one layer's pre (or post) site group as trace(s)."""
        import numpy as np
        # Group by resolved color and legend identity so each Mesh3d (single
        # color) stays valid while all legend levels remain distinct.
        color_groups = {}
        for _, row in site_df.iterrows():
            neuron_id = row['neuron_id']
            if site_type == 'pre':
                color = self._resolve_pre_color(neuron_id, layer_idx)
            else:
                color = self._resolve_post_color(neuron_id, layer_idx)
            legend_group, legend_name, legend_rank = self._pre_post_site_legend(
                site_type, layer_idx, layer_name, neuron_id,
            )
            color_groups.setdefault(
                (str(color), legend_group, legend_name, legend_rank), []
            ).append((row['x'], row['y'], row['z']))

        for (
            color_key, legend_group, legend_name, legend_rank
        ), pts in color_groups.items():
            coords = np.array(pts, dtype=float)
            base_color = color_key
            base_alpha = self._extract_alpha_from_color(base_color)
            opaque_color = self._get_opaque_color(base_color)
            opacity = min(1.0, max(0.0, base_alpha))
            hover = f'<b>{site_type} site</b><extra></extra>'
            show_legend = legend_group not in getattr(
                self, '_pre_post_seen_legend_groups', set()
            )
            if show_legend:
                self._pre_post_seen_legend_groups.add(legend_group)

            # In tree mode the collapsible HTML legend groups a site with its
            # owner so one toggle covers the whole neuron (legend_name is
            # '<owner_identity>_<role>'). With custom groups the site rides
            # under its owner's custom group and type.
            tree_meta = None
            if self.legend_mode == 'tree':
                owner_identity = legend_name[:-(len(site_type) + 1)]
                if self._tree_uses_custom_groups():
                    tree_meta = {
                        'kind': 'site',
                        'group': self.layer_names[layer_idx],
                        'type': owner_identity,
                        'item': legend_name,
                    }
                else:
                    tree_meta = {
                        'kind': 'site',
                        'group': owner_identity,
                        'item': legend_name,
                    }

            if self.backend == 'plotly':
                if getattr(self, 'pre_post_scatter', False):
                    # Scatter markers: post/input sites as circles, pre/output
                    # sites as diamonds (the scatter variant of pre_post mode).
                    # ``triangle`` is not available for Plotly Scatter3d, so a
                    # diamond is the closest distinct, pointy marker.
                    symbol = 'diamond' if site_type == 'pre' else 'circle'
                    scatter_size = self._scatter_synapse_marker_size()
                    site_meta = {'drocat_scatter_size_role': 'pre_post_site'}
                    if tree_meta is not None:
                        site_meta['drocatLegend'] = tree_meta
                    self.fig_3d.add_trace(go.Scatter3d(
                        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                        mode='markers',
                        name=legend_name,
                        legendgroup=legend_group,
                        showlegend=show_legend,
                        legendrank=legend_rank,
                        marker=dict(size=scatter_size, color=opaque_color, symbol=symbol, opacity=opacity),
                        hovertemplate=hover,
                        hoverinfo='name',
                        meta=site_meta,
                    ))
                    continue

                # Solid sphere (post) / cone (pre) site markers.
                shape = 'cone' if site_type == 'pre' else 'sphere'
                mesh = sv.build_site_mesh(
                    coords, mode=shape, size=site_scale, color=opaque_color, opacity=opacity,
                    name=legend_name,
                )
                mesh.legendgroup = legend_group
                mesh.showlegend = show_legend
                mesh.legendrank = legend_rank
                mesh.hovertemplate = hover
                mesh.hoverinfo = 'name'
                if tree_meta is not None:
                    mesh_meta = dict(getattr(mesh, 'meta', None) or {})
                    mesh_meta['drocatLegend'] = tree_meta
                    mesh.meta = mesh_meta
                self.fig_3d.add_trace(mesh)
                self._append_exportable_mesh(
                    mesh, color=base_color, alpha=opacity,
                    name=legend_name, role='synapse',
                )
            elif self.backend == 'k3d':
                try:
                    import k3d
                    color_bytes = extract_rgba_tuple(base_color, default_alpha=opacity)
                    int_color = (int(color_bytes[0]) << 16) + (int(color_bytes[1]) << 8) + int(color_bytes[2])
                    pts_obj = k3d.points(
                        positions=coords.astype(np.float32),
                        point_size=(
                            self._scatter_synapse_marker_size()
                            if getattr(self, 'pre_post_scatter', False)
                            else float(site_scale) / 10.0
                        ),
                        color=int_color, opacity=opacity,
                        name=legend_name,
                    )
                    self.fig_3d += pts_obj
                except Exception as exc:
                    self._vprint(f'⚠️  k3d pre/post site plotting failed: {exc}', level='full')

    def _get_dataset_mesh_dir(self):
        """Get dataset-specific mesh directory path.
        
        Uses cache/ folder for ROI meshes:
        - hemibrain:v1.2.1 -> cache/hemibrain_v1_2_1/meshes/
        - optic-lobe:v1.1 -> cache/optic-lobe_v1_1/meshes/
        
        References:
        - navis mesh handling: https://navis.readthedocs.io/en/latest/source/api.html#navis.Volume
        - mesh compression: use navis.Volume.to_json() with compression for storage optimization
        """
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        cache_mesh_dir = os.path.join(self.script_path, 'cache', dataset_normalized, 'meshes')
        os.makedirs(cache_mesh_dir, exist_ok=True)
        return cache_mesh_dir
    
    def _roi_to_filename(self, roi_name: str) -> str:
        """Convert ROI name to a case-safe filename for storage.
        
        macOS and Windows filesystems are case-insensitive by default, so 'aL(L)' and 
        'AL(L)' would be treated as the same file. This function creates unique filenames
        by encoding lowercase letters with an underscore prefix.
        
        Parameters
        ----------
        roi_name : str
            The ROI name (e.g., 'AL(L)', 'aL(L)', 'LH(R)')
            
        Returns
        -------
        str
            Filesystem-safe filename with .json extension (e.g., 'AL(L).json', '_a_L(L).json')
            
        Notes
        -----
        Encoding scheme:
        - Lowercase letters are prefixed with '_' (underscore)
        - Example: 'aL(L)' -> '_aL(L).json' (the 'a' gets underscore prefix)
        - Example: 'AL(L)' -> 'AL(L).json' (no lowercase, no encoding)
        - This preserves readability while ensuring unique filenames on case-insensitive systems
        """
        # Always encode lowercase letters to ensure unique filenames
        if any(c.islower() for c in roi_name):
            encoded_name = ''
            for char in roi_name:
                if char.islower():
                    encoded_name += f'_{char}'
                else:
                    encoded_name += char
            return encoded_name + '.json'
        else:
            return roi_name + '.json'
    
    def _get_mesh_file_path(self, mesh_dir: str, roi_name: str) -> str:
        """Get the mesh file path for an ROI, handling case-sensitivity.
        
        Tries to find the mesh file using case-safe encoding first, then falls back
        to the legacy direct naming for backward compatibility (only for all-uppercase ROI names).
        
        Parameters
        ----------
        mesh_dir : str
            Directory containing mesh files
        roi_name : str
            ROI name to look up
            
        Returns
        -------
        str
            Path to the mesh file (may or may not exist)
            
        Notes
        -----
        On case-insensitive filesystems (macOS, Windows), 'aL(L).json' and 'AL(L).json' 
        would be treated as the same file. To avoid this:
        - ROI names with lowercase letters ONLY use encoded filenames (e.g., '_aL(L).json')
        - ROI names that are all uppercase can use legacy fallback for backward compatibility
        """
        has_lowercase = any(c.islower() for c in roi_name)
        
        # Get encoded filename path
        encoded_file = os.path.join(mesh_dir, self._roi_to_filename(roi_name))
        if os.path.exists(encoded_file):
            return encoded_file
        
        # Only use legacy fallback for all-uppercase ROI names
        # This prevents case-insensitive filesystem issues where 'aL(L).json' would match 'AL(L).json'
        if not has_lowercase:
            legacy_file = os.path.join(mesh_dir, roi_name + '.json')
            if os.path.exists(legacy_file):
                return legacy_file
        
        # If neither exists, return the encoded path for new files
        return encoded_file

    def _colors_have_explicit_alpha(self, colors) -> bool:
        """Return whether any input color carries an alpha channel."""
        return color_has_explicit_alpha(colors)
    
    def _extract_alpha_from_color(self, color_str: str) -> float:
        """
        Extract the alpha value from a standardized rgba color string.
        
        Parameters
        ----------
        color_str : str
            Standardized color string in 'rgba(r, g, b, a)' format
            
        Returns
        -------
        float
            Alpha value between 0.0 and 1.0
            
        Examples
        --------
        >>> self._extract_alpha_from_color('rgba(255, 0, 0, 0.5)')
        0.5
        >>> self._extract_alpha_from_color('rgba(128, 128, 128, 1.0)')
        1.0
        """
        try:
            return float(extract_rgba_tuple(color_str)[3])
        except (TypeError, ValueError):
            return 1.0  # Default to fully opaque if parsing fails
    
    def _rgba_to_hex(self, color_str: str) -> str:
        """
        Convert a standardized rgba color string to hex format for navis compatibility.
        
        Parameters
        ----------
        color_str : str
            Standardized color string in 'rgba(r, g, b, a)' format
            
        Returns
        -------
        str
            Hex color string like '#ff0000'
            
        Examples
        --------
        >>> self._rgba_to_hex('rgba(255, 0, 0, 0.5)')
        '#ff0000'
        >>> self._rgba_to_hex('rgba(128, 128, 128, 1.0)')
        '#808080'
        """
        try:
            return color_to_hex(color_str)
        except (TypeError, ValueError):
            return '#808080'  # Default gray
    
    def _get_opaque_color(self, color_str: str) -> str:
        """
        Get opaque version of a color (alpha=1.0) for legend display.
        
        Used to ensure legend color patches are fully visible regardless
        of the actual alpha used for plotting.
        
        Parameters
        ----------
        color_str : str
            Standardized color string in 'rgba(r, g, b, a)' format
            
        Returns
        -------
        str
            RGBA color string with alpha=1.0
            
        Examples
        --------
        >>> self._get_opaque_color('rgba(255, 0, 0, 0.2)')
        'rgba(255, 0, 0, 1.0)'
        """
        try:
            r, g, b, _ = extract_rgba_tuple(color_str)
            return f'rgba({r}, {g}, {b}, 1.0)'
        except (TypeError, ValueError):
            return color_str
    
    def _standardize_color_input(self, colors, name='colors', default_alpha=1.0):
        """
        Standardize color input to a list of rgba strings.
        
        Accepts multiple input formats and converts them to a consistent
        list of 'rgba(r, g, b, a)' strings for internal use.
        
        Parameters
        ----------
        colors : str, tuple, list
            Color input in any supported format:
            - Single color string: 'red', '#ff0000', 'rgba(255, 0, 0, 0.5)'
            - Tuple/list of colors: ['red', 'blue'], [(255, 0, 0), (0, 0, 255)]
            - Bokeh palette: bokeh.palettes.Category10[10]
        name : str
            Name of the parameter (for error messages)
        default_alpha : float
            Default alpha value if not specified in colors
            
        Returns
        -------
        list
            List of standardized rgba color strings
        """
        # Handle single color string - wrap in list
        if hasattr(colors, 'tolist') and not isinstance(colors, (str, tuple, list)):
            colors = colors.tolist()

        if isinstance(colors, str):
            try:
                std_color = standardize_color(colors, default_alpha=default_alpha)
                return [std_color]
            except ValueError as e:
                self._vprint(f"⚠️  Warning: Could not parse {name} '{colors}': {e}", level='simple')
                return ['rgba(128, 128, 128, 1.0)']  # Fallback gray
        
        # A tuple/list/array of three/four channels is one RGB(A) color,
        # including RGBA tuples whose alpha is written as a percentage.
        if isinstance(colors, (tuple, list)) and len(colors) in [3, 4]:
            try:
                std_color = standardize_color(colors, default_alpha=default_alpha)
                return [std_color]
            except ValueError:
                pass

        if isinstance(colors, tuple):
            colors = list(colors)

        # Handle list of colors
        if isinstance(colors, list):
            result = []
            for i, c in enumerate(colors):
                try:
                    std_color = standardize_color(c, default_alpha=default_alpha)
                    result.append(std_color)
                except ValueError as e:
                    self._vprint(f"⚠️  Warning: Could not parse {name}[{i}] '{c}': {e}", level='simple')
                    result.append('rgba(128, 128, 128, 1.0)')  # Fallback gray
            return result if result else ['rgba(128, 128, 128, 1.0)']
        
        # Fallback
        return ['rgba(128, 128, 128, 1.0)']
    
    def _standardize_mesh_color_input(
        self, color, default_alpha=0.1, continuous=False
    ):
        """
        Standardize mesh_color input (can be single color or list of colors).
        
        Parameters
        ----------
        color : str, tuple, list
            Single color or list of colors
        default_alpha : float
            Default alpha value to use if not specified in color
            
        Returns
        -------
        str or list
            Standardized color(s) - single rgba string or list of rgba strings
        """
        # Handle string input
        if isinstance(color, str):
            try:
                return standardize_color(color, default_alpha=default_alpha)
            except ValueError:
                return f'rgba(100, 100, 100, {default_alpha})'  # Default gray
        
        if hasattr(color, 'tolist') and not isinstance(color, (str, tuple, list)):
            color = color.tolist()

        if continuous and isinstance(color, (tuple, list)):
            roi_count = (
                len(self.mesh_roi)
                if isinstance(getattr(self, 'mesh_roi', None), (tuple, list))
                and self.mesh_roi
                else len(color)
            )
            color = self._sample_continuous_color_sequence(color, roi_count)

        # A tuple/list of three/four channels is one RGB(A) color. Attempt
        # this before treating the value as a palette so percentage alpha and
        # normalized float channels are decoded exactly once.
        if isinstance(color, (tuple, list)) and len(color) in [3, 4]:
            try:
                return standardize_color(color, default_alpha=default_alpha)
            except ValueError:
                pass

        # Tuple/list of colors for multiple ROIs.
        if isinstance(color, (tuple, list)):
            return [self._standardize_mesh_color_input(c, default_alpha=default_alpha) for c in color]
        
        # Fallback
        return f'rgba(100, 100, 100, {default_alpha})'

    def _is_custom_mesh_color_specified(self) -> bool:
        """
        Check if the user specified a custom mesh_color (not the default).
        
        Returns
        -------
        bool
            True if mesh_color was customized from default, False otherwise
        """
        default_color = (100, 100, 100)  # Default RGB without alpha
        
        # Get the original mesh_color before standardization
        # We check if it differs from the default gray
        if hasattr(self, '_original_mesh_color'):
            original = self._original_mesh_color
        else:
            original = self.mesh_color
        
        # If it's the default tuple, it's not custom
        if isinstance(original, tuple) and len(original) in [3, 4]:
            if original[:3] == default_color:
                return False
        
        # If it's a standardized rgba string matching default, it's not custom
        if isinstance(original, str):
            if 'rgba(100, 100, 100,' in original or 'rgb(100, 100, 100)' in original:
                return False
        
        # Otherwise, it's custom
        return True

    def _darken_color(self, color, brightness):
        """Darken a color by the given brightness factor.
        
        Parameters
        ----------
        color : str or tuple
            Color in any supported format (now uses standardize_color internally):
            - Hex string: '#RRGGBB', '#RGB'
            - RGB tuple: (r, g, b)
            - RGBA tuple: (r, g, b, a)
            - CSS string: 'rgb(r, g, b)', 'rgba(r, g, b, a)'
            - Named color: 'red', 'blue', etc.
        brightness : float
            Brightness factor from 0.0 (black) to 1.0 (original color)
        
        Returns
        -------
        str
            Darkened color in rgba format
        """
        # Use the standardize_color utility to parse any format
        try:
            rgba = extract_rgba_tuple(color)
            r, g, b, a = rgba
        except ValueError:
            # Fallback to manual parsing for legacy support
            if isinstance(color, str):
                if color.startswith('#'):
                    hex_str = color.lstrip('#')
                    if len(hex_str) == 6:
                        r = int(hex_str[0:2], 16)
                        g = int(hex_str[2:4], 16)
                        b = int(hex_str[4:6], 16)
                        a = 1.0
                    else:
                        return 'rgba(128, 128, 128, 1.0)'
                else:
                    return 'rgba(128, 128, 128, 1.0)'
            elif isinstance(color, (tuple, list)) and len(color) >= 3:
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                a = color[3] if len(color) > 3 else 1.0
            else:
                return 'rgba(128, 128, 128, 1.0)'
        
        # Apply brightness (darken)
        r = int(r * brightness)
        g = int(g * brightness)
        b = int(b * brightness)
        
        # Clamp values
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        return f'rgba({r}, {g}, {b}, {a})'
    
    def _interpolate_colors(self, base_colors, n_needed):
        """Generate n_needed colors by keeping original colors first, then appending interpolated ones.
        
        Uses a round-based approach to generate new colors:
        - Round 1: Generate midpoints between consecutive pairs (0-1, 1-2, ..., 8-9, 9-0)
          e.g., 10 colors → 10 new midpoints → 20 total candidates
        - Round 2: If more needed, generate midpoints between all consecutive pairs in expanded set
        - Continue until enough candidates are generated
        
        Final result: [original colors] + [interpolated colors as needed]
        
        Parameters
        ----------
        base_colors : tuple or list
            Base color palette to preserve. Accepts any format supported by
            standardize_color: hex strings, RGB tuples, RGBA tuples, named colors, rgba strings
        n_needed : int
            Number of colors needed
        
        Returns
        -------
        list
            List of n_needed rgba color strings, with original colors first, then interpolated
        """
        import numpy as np
        
        # Convert base colors to normalized RGBA tuples. Keep the original
        # alpha for every supplied color; mixed palettes must not collapse to
        # one average opacity when colors are expanded.
        rgba_colors = []
        for c in base_colors:
            try:
                rgba_colors.append(extract_rgba_tuple(c, default_alpha=1.0))
            except ValueError:
                rgba_colors.append((128, 128, 128, 1.0))  # fallback gray

        n_base = len(rgba_colors)
        if n_base == 0:
            return ['rgba(128, 128, 128, 1.0)'] * n_needed

        def format_rgba(rgba):
            r, g, b, a = rgba
            return f'rgba({int(round(r))}, {int(round(g))}, {int(round(b))}, {float(a)})'
        
        # Handle single color case - just repeat the color n_needed times
        if n_base == 1:
            return [format_rgba(rgba_colors[0]) for _ in range(n_needed)]
        
        # If we need fewer or equal colors than base, just use the base colors
        if n_needed <= n_base:
            return [format_rgba(rgba_colors[i]) for i in range(n_needed)]
        
        # Start with original colors, then generate interpolated colors to append
        # Generate interpolated colors using round-based midpoint approach
        interpolated_pool = []  # Pool of interpolated colors to draw from
        current_set = [
            np.array([r / 255.0, g / 255.0, b / 255.0, a], dtype=float)
            for r, g, b, a in rgba_colors
        ]
        
        while len(rgba_colors) + len(interpolated_pool) < n_needed:
            # Generate midpoints between consecutive pairs (with wrap-around)
            new_midpoints = []
            n_current = len(current_set)
            for i in range(n_current):
                next_idx = (i + 1) % n_current
                midpoint = (current_set[i] + current_set[next_idx]) / 2
                new_midpoints.append(midpoint)
            
            # Add new midpoints to the pool
            interpolated_pool.extend(new_midpoints)
            
            # For next round, use interleaved set (original + midpoints) as the new current_set
            # This creates finer gradations in the next round
            interleaved = []
            for i in range(n_current):
                interleaved.append(current_set[i])
                interleaved.append(new_midpoints[i])
            current_set = interleaved
        
        # Build final result: original colors first, then interpolated as needed
        result = [format_rgba(rgba) for rgba in rgba_colors]
        
        # Add interpolated colors until we reach n_needed
        n_extra_needed = n_needed - n_base
        for i in range(n_extra_needed):
            c = interpolated_pool[i]
            result.append(format_rgba((c[0] * 255, c[1] * 255, c[2] * 255, c[3])))
        
        return result

    def _sample_continuous_color_sequence(self, colors, n_needed):
        """Evenly sample a selected continuous palette range."""
        colors = tuple(colors)
        if n_needed <= 0 or not colors:
            return []
        if n_needed == 1:
            return [colors[0]]
        if len(colors) == 1:
            return [colors[0]] * n_needed
        indices = [
            round(i * (len(colors) - 1) / (n_needed - 1))
            for i in range(n_needed)
        ]
        return [colors[index] for index in indices]

    def _expand_color_sequence(
        self,
        colors,
        n_needed,
        target_label='items',
        tip_parameter=None,
        warn=True,
        continuous=False,
    ):
        """Expand a color sequence to the requested size using the configured policy."""
        colors = tuple(colors)
        n_colors = len(colors)

        if n_needed <= 0:
            return tuple()

        if continuous:
            # A selected continuous range must be represented by its endpoints
            # and evenly spaced interior colors, not by the first N swatches.
            return tuple(self._sample_continuous_color_sequence(colors, n_needed))

        if n_needed <= n_colors:
            return tuple(colors[:n_needed])

        if self.expand_colors == 'interpolation':
            expanded = self._interpolate_colors(colors, n_needed)
            warning = f'\033[33m⚠️  Warning: {n_needed} {target_label} but only {n_colors} colors. Generated {n_needed} colors via interpolation.\033[0m'
        elif self.expand_colors == 'darken':
            n_cycles = (n_needed - 1) // n_colors
            expanded = []
            for i in range(n_needed):
                cycle_num = i // n_colors
                color_idx = i % n_colors
                base_color = colors[color_idx]
                brightness = 1.0 - (0.3 * cycle_num / n_cycles) if n_cycles > 0 else 1.0
                expanded.append(self._darken_color(base_color, brightness))
            warning = f'\033[33m⚠️  Warning: {n_needed} {target_label} but only {n_colors} colors. Recycling with darkening (100%→70%).\033[0m'
        else:
            expanded = [colors[i % n_colors] for i in range(n_needed)]
            warning = f'\033[33m⚠️  Warning: {n_needed} {target_label} but only {n_colors} colors. Cycling colors (repeating pattern).\033[0m'

        if warn:
            self._vprint(warning)
            if tip_parameter:
                self._vprint(f'\033[33m   💡 Tip: Use {tip_parameter} parameter with custom palette to specify more colors.\033[0m')

        return tuple(expanded)

    def _normalize_neuron_lookup_keys(self, value):
        """Normalize neuron identifiers for robust lookup by bodyId or name-like fields."""
        if value is None:
            return []

        if isinstance(value, (int, np.integer)):
            int_value = int(value)
            return [int_value, str(int_value)]

        try:
            if pd.isna(value):
                return []
        except TypeError:
            pass

        value_str = str(value).strip()
        if not value_str or value_str.lower() == 'nan':
            return []

        # Connector tables and CSV round-trips commonly turn integral body IDs
        # into strings such as ``"15832.0"``.  The old lookup only recognized
        # digit-only strings, so a pre/post site could miss the exact per-entry
        # color override even though the same body ID was present in the layer
        # map.  Normalize a *decimal-zero* suffix without converting through a
        # float: FlyWire IDs can exceed JavaScript/float integer precision and
        # must retain their original decimal digits.
        keys = [value_str]
        decimal_match = re.fullmatch(r'([+-]?\d+)(?:\.0+)?', value_str)
        if decimal_match:
            integer_text = decimal_match.group(1)
            try:
                integer_value = int(integer_text)
            except ValueError:
                integer_value = None
            if integer_value is not None:
                keys.insert(0, integer_value)
                canonical_text = str(integer_value)
                if canonical_text not in keys:
                    keys.append(canonical_text)
        return keys

    def _link_layer_map_color_overrides_to_neuron_dfs(self):
        """Make custom layer-map colors resolvable by every neuron identifier.

        ``layer_map_csv`` rows commonly use a type or instance name because
        that is what the UI displays.  Skeleton objects and connector tables
        generally expose ``bodyId`` instead.  Keep the original override
        values, then add aliases for each matching resolved neuron so all
        renderers (morphology, paired synapses, and pre/post sites) see the
        same color.
        """
        override_attrs = (
            "_neuron_color_overrides",
            "_neuron_synapse_color_overrides",
            "_neuron_pre_color_overrides",
            "_neuron_post_color_overrides",
        )
        identifier_columns = (
            "bodyId",
            "name",
            "type",
            "instance",
            "roi",
            "flywireType",
            "hemibrainType",
            "mancType",
        )

        for attr in override_attrs:
            overrides = getattr(self, attr, None)
            if not overrides:
                continue
            expanded = dict(overrides)
            layered_attr = f"{attr}_by_layer"
            layered_overrides = getattr(self, layered_attr, {}) or {}
            expanded_layered = {
                layer_index: dict(values)
                for layer_index, values in layered_overrides.items()
            }
            for layer_index, neuron_df in enumerate(
                getattr(self, "neuron_dfs", []) or []
            ):
                if neuron_df is None or neuron_df.empty:
                    continue
                for _, row in neuron_df.iterrows():
                    row_keys = []
                    for column in identifier_columns:
                        if column in row.index:
                            row_keys.extend(
                                self._normalize_neuron_lookup_keys(row[column])
                            )
                    if not row_keys:
                        continue
                    layer_values = layered_overrides.get(layer_index, {})
                    matched_color = next(
                        (
                            layer_values[key]
                            for key in row_keys
                            if key in layer_values
                        ),
                        None,
                    )
                    if matched_color is None:
                        matched_color = next(
                            (
                                overrides[key]
                                for key in row_keys
                                if key in overrides
                            ),
                            None,
                        )
                    if matched_color is None:
                        continue
                    for key in row_keys:
                        expanded[key] = matched_color
                        expanded_layered.setdefault(layer_index, {})[key] = matched_color
            setattr(self, attr, expanded)
            if layered_overrides:
                setattr(self, layered_attr, expanded_layered)

    def _build_per_neuron_color_map(self):
        """Build a lookup of neuron identifiers to per-neuron colors."""
        total_neurons = sum(len(ndf) for ndf in self.neuron_dfs if ndf is not None)
        if total_neurons == 0:
            return {}

        neuron_palette = self._expand_color_sequence(
            self._base_neuron_colors,
            total_neurons,
            target_label='neurons',
            tip_parameter='neuron_colors',
            continuous=self._neuron_colors_continuous,
        )

        neuron_color_map = {}
        color_index = 0
        for ndf in self.neuron_dfs:
            if ndf is None or ndf.empty:
                continue
            for _, row in ndf.iterrows():
                neuron_color = neuron_palette[color_index]
                for column in ['bodyId', 'name', 'instance', 'roi']:
                    if column in row.index:
                        for key in self._normalize_neuron_lookup_keys(row[column]):
                            neuron_color_map[key] = neuron_color
                color_index += 1

        return neuron_color_map

    def _resolve_neuron_override(self, attr, neuron_id, layer_index):
        """Resolve one custom-layer color, preferring the exact layer row.

        A layer-map CSV is a table of ``(layer, neuron)`` entries, not a
        global neuron-name dictionary.  Keep the old global lookup as a
        compatibility fallback, but consult the layer-aware aliases first so
        duplicate type/instance labels do not make one row inherit another
        row's pre/post color.
        """
        lookup_keys = self._normalize_neuron_lookup_keys(neuron_id)
        if not lookup_keys:
            return None

        layered = getattr(self, f"{attr}_by_layer", {}) or {}
        layer_candidates = [layer_index]
        try:
            numeric_layer = int(layer_index)
        except (TypeError, ValueError):
            numeric_layer = None
        if numeric_layer is not None and numeric_layer not in layer_candidates:
            layer_candidates.append(numeric_layer)
        for candidate in layer_candidates:
            layer_values = layered.get(candidate, {})
            for key in lookup_keys:
                if key in layer_values:
                    return layer_values[key]

        overrides = getattr(self, attr, {}) or {}
        for key in lookup_keys:
            if key in overrides:
                return overrides[key]
        return None

    def _resolve_neuron_color(self, neuron_id, layer_index):
        """Resolve the display color for a neuron trace, including overrides."""
        if self.neuron_colors:
            neuron_color = (
                self.neuron_colors[layer_index]
                if layer_index < len(self.neuron_colors)
                else self.neuron_colors[-1]
            )
        else:
            neuron_color = "rgba(0, 0, 0, 0.2)"
        lookup_keys = self._normalize_neuron_lookup_keys(neuron_id)

        if self.color_mode == 'per_neuron' and hasattr(self, '_per_neuron_colors'):
            for key in lookup_keys:
                if key in self._per_neuron_colors:
                    neuron_color = self._per_neuron_colors[key]
                    break

        override = self._resolve_neuron_override(
            '_neuron_color_overrides', neuron_id, layer_index,
        )
        if override is not None:
            return override

        return neuron_color

    def _resolve_synapse_color(self, neuron_id, layer_index):
        """Resolve the inter-layer (connector) synapse color for a pre neuron.

        Falls back to the layer's connection color (``synapse_colors[layer_index]``)
        when no per-neuron override exists.
        """
        if self.synapse_colors and layer_index < len(self.synapse_colors):
            color = self.synapse_colors[layer_index]
        elif self.synapse_colors:
            color = self.synapse_colors[0]
        else:
            color = "rgba(0, 0, 0, 0.6)"
        override = self._resolve_neuron_override(
            '_neuron_synapse_color_overrides', neuron_id, layer_index,
        )
        if override is not None:
            return override
        return color

    def _resolve_pre_color(self, neuron_id, layer_index):
        """Resolve the color for a neuron's pre-synaptic SITES (pre_post mode).

        Defaults to the neuron's display color; a per-neuron override wins.
        """
        color = self._resolve_neuron_color(neuron_id, layer_index)
        override = self._resolve_neuron_override(
            '_neuron_pre_color_overrides', neuron_id, layer_index,
        )
        if override is not None:
            return override
        return color

    def _resolve_post_color(self, neuron_id, layer_index):
        """Resolve the color for a neuron's post-synaptic SITES (pre_post mode).

        Defaults to the neuron's display color; a per-neuron override wins.
        """
        color = self._resolve_neuron_color(neuron_id, layer_index)
        override = self._resolve_neuron_override(
            '_neuron_post_color_overrides', neuron_id, layer_index,
        )
        if override is not None:
            return override
        return color

    def _apply_plotly_trace_color(self, trace, neuron_color):
        """Apply a resolved neuron color to a Plotly trace."""
        color_hex = self._rgba_to_hex(neuron_color)
        color_alpha = self._extract_alpha_from_color(neuron_color)

        if hasattr(trace, 'color'):
            try:
                trace.color = color_hex
            except Exception:
                pass
        if hasattr(trace, 'line') and trace.line is not None:
            trace.line.color = color_hex
        if hasattr(trace, 'marker') and trace.marker is not None:
            trace.marker.color = color_hex
        trace.opacity = color_alpha

    def _apply_k3d_object_color(self, obj, neuron_color):
        """Apply a resolved neuron color to a k3d object when supported."""
        try:
            r, g, b, a = extract_rgba_tuple(neuron_color)
            if hasattr(obj, 'color'):
                obj.color = (int(r) << 16) + (int(g) << 8) + int(b)
            if hasattr(obj, 'opacity'):
                obj.opacity = a
        except Exception:
            pass

    def _flatten_nested_roi_groups(self, roi_list, color_input):
        """Flatten nested ROI lists while assigning same color to grouped ROIs.
        
        Supports nested lists for grouping ROIs that should share the same color:
        e.g., ['AME', ['aL', 'bL', 'gL'], 'EB'] -> 
              flat list with same color for all ROIs in the nested list.
        
        Parameters
        ----------
        roi_list : list
            List of ROI names, potentially with nested lists for grouping.
            e.g., ['AME', ['aL', 'bL', 'gL', "a'L", "b'L"], 'EB', 'PB']
        color_input : str, list, or tuple
            Colors for each ROI group. Can be a single color (applies to all),
            or a list matching the number of top-level items in roi_list.
            
        Returns
        -------
        tuple
            (flattened_rois, expanded_colors, nested_groups)
            - flattened_rois: Flat list of all ROI names
            - expanded_colors: List of colors matching flattened_rois
            - nested_groups: Dict recording the source color group for each ROI
            
        Examples
        --------
        >>> _flatten_nested_roi_groups(['AME', ['aL', 'bL'], 'EB'], ['red', 'green', 'blue'])
        (['AME', 'aL', 'bL', 'EB'], ['red', 'green', 'green', 'blue'], 
         {'AME': 'AME', 'aL': 'MB_lobes', 'bL': 'MB_lobes', 'EB': 'EB'})
        """
        if not roi_list:
            return roi_list, color_input, {}
        
        # Normalize color_input to a list matching roi_list length
        if isinstance(color_input, str):
            colors = [color_input] * len(roi_list)
        elif isinstance(color_input, (tuple, list)):
            # Check if it's a single color tuple (RGB/RGBA with numeric values) or a list/tuple of colors
            # RGB/RGBA tuple: (255, 0, 0) or (1.0, 0.5, 0.0, 0.5)
            # List of colors: ['red', 'blue'] or ('#ff0000', '#0000ff') or [(255,0,0), (0,0,255)]
            is_single_rgb_color = (
                len(color_input) in [3, 4] and 
                all(isinstance(x, (int, float)) for x in color_input)
            )
            
            if is_single_rgb_color:
                # Single RGB/RGBA color tuple - apply to all ROIs
                colors = [color_input] * len(roi_list)
            else:
                # List/tuple of colors (strings, hex codes, or color tuples)
                colors = list(color_input)
                # Extend if needed
                while len(colors) < len(roi_list):
                    colors.append(colors[-1] if colors else 'gray')
        else:
            colors = ['gray'] * len(roi_list)
        
        flattened_rois = []
        expanded_colors = []
        nested_groups = {}
        
        for i, item in enumerate(roi_list):
            color = colors[i] if i < len(colors) else colors[-1]
            
            if isinstance(item, list):
                # Nested list - all items share the same color. Keep metadata
                # about the source group without coupling their legend entries.
                group_label = '+'.join(str(r) for r in item[:3])  # Use first 3 names
                if len(item) > 3:
                    group_label += f'+{len(item)-3}more'
                
                for roi in item:
                    flattened_rois.append(roi)
                    expanded_colors.append(color)
                    nested_groups[roi] = group_label
                    
                self._vprint(f"   🔗 Grouped [{', '.join(str(r) for r in item)}] → same color", level='simple')
            else:
                # Single ROI
                flattened_rois.append(item)
                expanded_colors.append(color)
                nested_groups[item] = str(item)
        
        return flattened_rois, expanded_colors, nested_groups
    
    def _expand_roi_names_with_colors(self, roi_list, color_list, available_rois=None):
        """Expand ROI names AND their corresponding colors to include bilateral (L/R) variants.
        
        When a user specifies 'LH' with color 'red', this function will automatically expand
        to ['LH(L)', 'LH(R)'] with colors ['red', 'red']. This ensures colors match expanded ROIs.
        
        Parameters
        ----------
        roi_list : list
            List of ROI names to expand
        color_list : list or single color
            Colors corresponding to each ROI. If a single color, applies to all.
            If a list shorter than roi_list, extra ROIs get the last color.
        available_rois : list, optional
            List of available ROI names. If None, will be fetched from cache/API.
            
        Returns
        -------
        tuple
            (expanded_rois, expanded_colors) - Both lists with matching lengths
            
        Examples
        --------
        >>> _expand_roi_names_with_colors(['LH', 'EB'], ['red', 'blue'])
        (['LH(L)', 'LH(R)', 'EB'], ['red', 'red', 'blue'])
        
        >>> _expand_roi_names_with_colors(['LH'], 'green')  # Single color
        (['LH(L)', 'LH(R)'], ['green', 'green'])
        """
        if not roi_list:
            return roi_list, color_list
        
        # Normalize color_list to a list
        if not isinstance(color_list, list):
            color_list = [color_list] * len(roi_list)
        
        # Ensure color_list matches roi_list length
        if len(color_list) < len(roi_list):
            # Extend with the last color
            last_color = color_list[-1] if color_list else None
            color_list = list(color_list) + [last_color] * (len(roi_list) - len(color_list))
        
        # Get available ROIs if not provided
        if available_rois is None:
            is_fafb = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()
            if is_fafb:
                malecns_cache = os.path.join(self.script_path, 'cache', 'male-cns_v0_9', 'available_rois.json')
                if os.path.exists(malecns_cache):
                    import json
                    with open(malecns_cache, 'r') as f:
                        available_rois = json.load(f)
                else:
                    available_rois = self._get_available_rois(use_cache=True, fetch_online=False)
            else:
                available_rois = self._get_available_rois(use_cache=True, fetch_online=False)
        
        available_set = set(available_rois) if available_rois else set()
        
        expanded_rois = []
        expanded_colors = []
        seen = set()
        
        for i, roi in enumerate(roi_list):
            color = color_list[i] if i < len(color_list) else color_list[-1]
            
            # Check if ROI already has (L) or (R) suffix
            if roi.endswith('(L)') or roi.endswith('(R)'):
                if roi not in seen:
                    expanded_rois.append(roi)
                    expanded_colors.append(color)
                    seen.add(roi)
                continue
            
            # Check if the ROI exists as-is (like 'EB' which is unpaired)
            if roi in available_set:
                if roi not in seen:
                    expanded_rois.append(roi)
                    expanded_colors.append(color)
                    seen.add(roi)
                continue
            
            # Try to expand to bilateral variants
            left_variant = f'{roi}(L)'
            right_variant = f'{roi}(R)'
            
            found_left = left_variant in available_set
            found_right = right_variant in available_set
            
            if found_left and found_right:
                # Both sides exist, expand to both with same color
                if left_variant not in seen:
                    expanded_rois.append(left_variant)
                    expanded_colors.append(color)
                    seen.add(left_variant)
                if right_variant not in seen:
                    expanded_rois.append(right_variant)
                    expanded_colors.append(color)
                    seen.add(right_variant)
            elif found_left:
                if left_variant not in seen:
                    expanded_rois.append(left_variant)
                    expanded_colors.append(color)
                    seen.add(left_variant)
            elif found_right:
                if right_variant not in seen:
                    expanded_rois.append(right_variant)
                    expanded_colors.append(color)
                    seen.add(right_variant)
            else:
                # No bilateral variants found, keep original
                if roi not in seen:
                    expanded_rois.append(roi)
                    expanded_colors.append(color)
                    seen.add(roi)
        
        return expanded_rois, expanded_colors

    def _get_metadata_primary_rois(self):
        """Load the dataset's authoritative NeuPrint primary ROI list.

        NeuPrint labels this set ``primaryRois`` in its metadata.  DROCAT's
        local dataset sidecars store the same values under
        ``roi_coverage.roi_list``.  Prefer the sidecar so the ``primary``
        keyword remains complete and deterministic in offline runs; FlyWire
        ROI meshes use the male-cns metadata as their backend does.
        """
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        folders = [dataset_normalized]
        dataset_lower = str(self.dataset).lower()
        if 'flywire' in dataset_lower or 'fafb' in dataset_lower:
            folders = ['male-cns_v0_9', 'male-cns_v1_0', *folders]

        for folder in folders:
            metadata_dir = os.path.join(self.script_path, 'datasets', folder)
            if not os.path.isdir(metadata_dir):
                continue
            metadata_files = []
            preferred = os.path.join(metadata_dir, f'{folder}_metadata.json')
            if os.path.isfile(preferred):
                metadata_files.append(preferred)
            metadata_files.extend(
                os.path.join(metadata_dir, name)
                for name in sorted(os.listdir(metadata_dir))
                if 'metadata' in name.lower() and name.endswith('.json')
                and os.path.join(metadata_dir, name) not in metadata_files
            )
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as handle:
                        metadata = json.load(handle)
                except (OSError, ValueError, TypeError):
                    continue
                if not isinstance(metadata, dict):
                    continue

                primary = metadata.get('primaryRois') or metadata.get('primary_rois')
                if not primary:
                    coverage = metadata.get('roi_coverage') or {}
                    if isinstance(coverage, dict):
                        primary = coverage.get('roi_list')
                if isinstance(primary, list):
                    values = []
                    seen = set()
                    for roi in primary:
                        roi = str(roi).strip()
                        if roi and roi != 'NotPrimary' and roi not in seen:
                            values.append(roi)
                            seen.add(roi)
                    if values:
                        return values
        return []

    def _expand_mesh_roi_patterns(self, roi_list):
        """Expand special keywords and regex patterns in mesh_roi list.
        
        Supports:
        - 'primary': All primary ROIs (major brain regions)
        - 'all': All available ROIs for the current dataset
        - Regex patterns: 'ME.*' matches all ROIs starting with 'ME'
        
        Parameters
        ----------
        roi_list : list
            List of ROI names, keywords, or regex patterns
            
        Returns
        -------
        list
            Expanded list of ROI names
            
        Examples
        --------
        >>> _expand_mesh_roi_patterns(['primary'])
        ['AL(L)', 'AL(R)', 'EB', 'FB', ...]  # All primary ROIs
        
        >>> _expand_mesh_roi_patterns(['ME.*'])
        ['ME(L)', 'ME(R)', 'ME_glomerulus(L)', ...]  # All ROIs matching ME.*
        
        >>> _expand_mesh_roi_patterns(['LH', 'ME.*', 'EB'])
        ['LH', 'ME(L)', 'ME(R)', ..., 'EB']  # Mixed: literal + regex
        """
        import re
        
        if not roi_list:
            return roi_list
        
        # Convert single string to list
        if isinstance(roi_list, str):
            roi_list = [roi_list]
        
        # Get available ROIs for pattern matching
        available_rois = self._get_available_rois(use_cache=True, fetch_online=True)
        available_set = set(available_rois) if available_rois else set()
        
        # Prefer the dataset's primaryRois metadata.  The hard-coded patterns
        # below remain only as a compatibility fallback for datasets that have
        # not been pulled locally yet.
        metadata_primary_rois = self._get_metadata_primary_rois()
        primary_roi_patterns = [
            # Central Complex
            'EB', 'FB', 'PB', 'NO', 'AB',
            # Mushroom Body  
            'MB.*', 'CA.*', 'PED.*',
            # Antennal Lobe
            'AL\\(.*\\)',
            # Lateral Horn
            'LH\\(.*\\)',
            # Optic Lobe
            'ME\\(.*\\)', 'LO\\(.*\\)', 'LOP\\(.*\\)', 'AME\\(.*\\)',
            # Subesophageal Zone
            'SEZ.*', 'GNG.*',
            # Other major regions
            'CRE.*', 'SCL.*', 'ICL.*', 'IB.*', 'ATL.*', 'AVLP.*', 'PVLP.*',
            'PLP.*', 'WED.*', 'SLP.*', 'SIP.*', 'SMP.*', 'CAN.*',
            'FLA.*', 'EPA.*', 'GOR.*', 'SPS.*', 'IPS.*',
        ]
        
        expanded = []
        seen = set()
        
        for item in roi_list:
            item_str = str(item).strip()
            
            # Handle special keywords
            if item_str.lower() == 'all':
                # Add all available ROIs
                for roi in sorted(available_rois):
                    if roi not in seen:
                        expanded.append(roi)
                        seen.add(roi)
                self._vprint(f"   🌐 'all': Added {len(available_rois)} ROIs", level='simple')
                continue
                
            elif item_str.lower() == 'primary':
                primary_count = 0
                if metadata_primary_rois:
                    # Keep metadata order, which is the order supplied by the
                    # dataset, and do not discard valid metadata names merely
                    # because an offline mesh inventory is incomplete.
                    for roi in metadata_primary_rois:
                        if roi not in seen:
                            expanded.append(roi)
                            seen.add(roi)
                            primary_count += 1
                else:
                    for pattern in primary_roi_patterns:
                        try:
                            regex = re.compile(f'^{pattern}$')
                            for roi in available_rois:
                                if regex.match(roi) and roi not in seen:
                                    expanded.append(roi)
                                    seen.add(roi)
                                    primary_count += 1
                        except re.error:
                            # If pattern is invalid regex, try exact match
                            if pattern in available_set and pattern not in seen:
                                expanded.append(pattern)
                                seen.add(pattern)
                                primary_count += 1
                self._vprint(f"   🏛️ 'primary': Added {primary_count} primary ROIs", level='simple')
                continue
            
            # Check if it's a regex pattern (contains regex special chars)
            # Common regex patterns: .* .+ [abc] ^ $ etc.
            is_regex = any(c in item_str for c in ['*', '+', '?', '[', ']', '^', '$', '|', '\\'])
            
            if is_regex:
                # Treat as regex pattern
                try:
                    # Anchor the pattern if not already anchored
                    pattern = item_str
                    if not pattern.startswith('^'):
                        pattern = '^' + pattern
                    if not pattern.endswith('$'):
                        pattern = pattern + '$'
                    
                    regex = re.compile(pattern)
                    matched = []
                    for roi in available_rois:
                        if regex.match(roi) and roi not in seen:
                            expanded.append(roi)
                            seen.add(roi)
                            matched.append(roi)
                    
                    if matched:
                        self._vprint(f"   🔍 '{item_str}': Matched {len(matched)} ROIs", level='simple')
                    else:
                        self._vprint(f"   ⚠️ '{item_str}': No matching ROIs found", level='simple')
                        
                except re.error as e:
                    self._vprint(f"   ⚠️ Invalid regex '{item_str}': {e}, treating as literal", level='simple')
                    if item_str not in seen:
                        expanded.append(item_str)
                        seen.add(item_str)
            else:
                # Literal ROI name
                if item_str not in seen:
                    expanded.append(item_str)
                    seen.add(item_str)
        
        return expanded

    def _get_available_rois(self, use_cache=True, fetch_online=True):
        """Query NeuPrint database for available ROIs in the current dataset.
        
        Caches results locally to avoid repeated API calls. Returns a list of ROI names
        that are available in the NeuPrint database for the current dataset.
        
        Parameters
        ----------
        use_cache : bool
            If True, use cached ROI list if available. If False, force refresh from API.
        fetch_online : bool
            If True, attempt to fetch from NeuPrint online. If False, only use local cache/meshes.
        
        Returns
        -------
        list
            List of available ROI names for the current dataset.
        
        References:
        - NeuPrint ROI documentation: https://neuprint.janelia.org/
        - navis neuprint interface: https://navis-org.github.io/navis/reference/navis/interfaces/neuprint/
        - neuprint-python API: https://github.com/connectome-neuprint/neuprint-python
        """
        # Cache file path in organized cache/ structure
        dataset_normalized = self.dataset.replace(':', '_').replace('.', '_')
        cache_dir = os.path.join(self.script_path, 'cache', dataset_normalized)
        cache_file = os.path.join(cache_dir, 'available_rois.json')
        
        # Try to load from cache first
        if use_cache and os.path.exists(cache_file):
            try:
                import json
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self._vprint(f'✓ Loaded {len(cached_data)} available ROIs from cache', level='full')
                    return cached_data
            except Exception as e:
                self._vprint(f'⚠️ Failed to load ROI cache: {e}, fetching from API...', level='full')
        
        # Fetch from NeuPrint API
        if fetch_online:
            # Special handling for FlyWire/FAFB: Do not use API, use local primary_rois or hemibrain cache
            if 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower():
                self._vprint('ℹ️  FlyWire/FAFB dataset detected: Skipping online API fetch for ROIs.', level='full')
                self._vprint('   Scanning local ROI meshes...', level='full')
                
                found_rois = set()
                
                # Scan primary_rois
                primary_dir = os.path.join(self.script_path, 'navis_roi_meshes_json', 'primary_rois')
                if os.path.exists(primary_dir):
                    for f in os.listdir(primary_dir):
                        if f.endswith('.json'):
                            found_rois.add(f[:-5])
                            
                # Scan hemibrain cache
                hb_cache = os.path.join(self.script_path, 'cache', 'hemibrain_v1_2_1', 'meshes')
                if os.path.exists(hb_cache):
                    for f in os.listdir(hb_cache):
                        if f.endswith('.json'):
                            found_rois.add(f[:-5])
                            
                roi_list = sorted(list(found_rois))
                self._vprint(f'✓ Found {len(roi_list)} available ROIs from local storage', level='full')
                
                # Cache the results
                if roi_list:
                    try:
                        import json
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, 'w') as f:
                            json.dump(roi_list, f, indent=2)
                    except Exception as e:
                        self._vprint(f'⚠️ Failed to cache ROI list: {e}', level='full')
                        
                return roi_list

            try:
                self._vprint('📥 Fetching available ROIs from NeuPrint online database...', level='full')
                
                # Initialize neuprint client using environment variable or global client
                from neuprint import Client, fetch_meta
                
                client = self.client
                
                if client is None:
                    # Try to get token from env/config files first
                    from utils.token_manager import token_manager
                    token = token_manager.get_neuprint_token()
                    
                    if token:
                        # Determine server URL based on dataset
                        if 'optic' in self.dataset.lower():
                            server = 'https://neuprint-optic-lobe.janelia.org'
                            dataset_name = self.dataset.split(':')[0]  # 'optic-lobe'
                        else:
                            server = 'https://neuprint.janelia.org'
                            dataset_name = 'hemibrain:v1.2.1'  # default
                        
                        try:
                            client = Client(server, dataset=dataset_name, token=token)
                        except Exception as e:
                            self._vprint(f'   Warning: Failed to create client with token: {e}', level='full')
                            self._vprint(f'   Attempting to use default/global client...', level='full')
                            client = None
                
                # Fetch metadata (will use client if provided, otherwise global)
                meta = fetch_meta(client=client)
                
                roi_list = []
                # Extract ROI list from meta info
                if 'roiInfo' in meta:
                    roi_list = list(meta['roiInfo'].keys())
                    self._vprint(f'   Found {len(roi_list)} ROIs from roiInfo', level='full')
                elif 'primaryRois' in meta:
                    roi_list = list(meta['primaryRois'])
                    self._vprint(f'   Found {len(roi_list)} primary ROIs', level='full')
                else:
                    self._vprint(f'   Warning: No roiInfo/primaryRois in metadata, falling back to local cache', level='full')
                
                roi_list = sorted(roi_list)
                
                # Cache the results (create directory only when needed)
                if roi_list:
                    try:
                        import json
                        os.makedirs(cache_dir, exist_ok=True)
                        with open(cache_file, 'w') as f:
                            json.dump(roi_list, f, indent=2)
                        self._vprint(f'✓ Cached {len(roi_list)} available ROIs to {cache_file}', level='full')
                    except Exception as e:
                        self._vprint(f'⚠️ Failed to cache ROI list: {e}', level='full')
                
                return roi_list
                
            except Exception as e:
                self._vprint(f'⚠️ Failed to fetch available ROIs from NeuPrint: {e}', level='full')
                self._vprint(f'   Tip: Set NEUPRINT_APPLICATION_CREDENTIALS environment variable', level='full')
                self._vprint(f'   Using ROIs from local mesh directory instead.', level='full')
        
        # Fallback: list available meshes from local directory
        mesh_dir = self._get_dataset_mesh_dir()
        if os.path.exists(mesh_dir):
            roi_list = [f.replace('.json', '') for f in os.listdir(mesh_dir) if f.endswith('.json')]
            roi_list = sorted(roi_list)
            self._vprint(f'✓ Found {len(roi_list)} ROIs in local cache: {mesh_dir}', level='full')
            
            # Cache the results from local scan
            if roi_list:
                try:
                    import json
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(roi_list, f, indent=2)
                    self._vprint(f'✓ Cached {len(roi_list)} available ROIs to {cache_file}', level='full')
                except Exception as e:
                    self._vprint(f'⚠️ Failed to cache ROI list: {e}', level='full')
            
            return roi_list
        else:
            self._vprint(f'⚠️ No ROI data available (online fetch failed and no local cache)', level='full')
            return []
    
    def _check_transform_requirements_early(self):
        """Check transform requirements at startup and advise user on options.
        
        This method checks if the current dataset and brain_mesh settings require
        coordinate transforms, and if so, verifies transform availability and
        advises the user on transformation-free alternatives.
        
        For hemibrain or FAFB with brain_mesh='whole', H5 transforms are required. 
        The user will be prompted to either:
        1. Download transforms (~13GB, enables JRC2018F whole brain view)
        2. Use brain_mesh='template' for transformation-free native view
        3. Use brain_mesh='none' for no brain mesh
        
        For all other datasets, native templates are used without transforms.
        """
        dataset_lower = self.dataset.lower()
        
        # Check if transforms are needed
        needs_transform = self._dataset_needs_transform()
        
        if not needs_transform:
            # Dataset uses native template - no transforms needed
            if self.brain_mesh in ['template', 'whole']:
                template_info = self._get_template_info()
                if template_info.get('skip_transform', False):
                    self._vprint(f'✓ Using native {template_info["mesh_name"]} - no coordinate transforms needed', level='full')
            return
        
        # Only hemibrain and FAFB with brain_mesh='whole' require H5 transforms
        is_hemibrain = 'hemibrain' in dataset_lower
        is_fafb = 'flywire' in dataset_lower or 'fafb' in dataset_lower
        
        if not (is_hemibrain or is_fafb):
            return
            
        # brain_mesh='whole' needs transform check
        if self.brain_mesh == 'none':
            return
            
        # Check if transforms are available
        import flybrains
        import navis
        
        template_info = self._get_template_info()
        source = template_info['source']
        target = template_info['target']
        
        # Try to find the transform path
        transforms_available = False
        try:
            path = navis.transforms.registry.find_bridging_path(source, target)
            transforms_available = True
        except (ValueError, KeyError):
            pass
        
        if transforms_available:
            if self.brain_mesh == 'whole':
                self._vprint(f'✓ Transforms available for {source} → {target}', level='full')
            return
            
        # Transforms not available - prompt user
        YELLOW = '\\033[93m'
        CYAN = '\\033[96m'
        GREEN = '\\033[92m'
        RED = '\\033[91m'
        RESET = '\\033[0m'
        
        dataset_name = 'FlyWire/FAFB' if is_fafb else 'Hemibrain'
        native_template = 'FLYWIRE' if is_fafb else 'JRCFIB2018F'
        
        print(f'\\n{YELLOW}{"="*70}')
        print(f'⚠️  Coordinate Transform Required for {dataset_name}')
        print(f'{"="*70}{RESET}')
        print()
        print(f'Your settings: dataset={self.dataset}, brain_mesh={self.brain_mesh}')
        print()
        print(f'The {dataset_name} dataset requires coordinate transforms for brain_mesh="whole":')
        print(f'  • Transform path: {source} → {target}')
        print(f'  • Requires downloading ~{CYAN}13 GB{RESET} of transform files')
        print(f'  • Transformation adds processing time to visualization')
        print()
        print(f'{GREEN}💡 Transformation-Free Alternatives:{RESET}')
        print()
        print(f'  Option 1: Use brain_mesh="template" (recommended)')
        print(f'            → Uses native {native_template} template mesh')
        print(f'            → No transforms needed, fast visualization')
        print()
        print(f'  Option 2: Use brain_mesh="none"')
        print(f'            → No brain mesh, only neurons and synapses')
        print()
        print(f'  Option 3: Use a different dataset')
        if is_fafb:
            print(f'            → male-cns: Native JRCFIB2022M template (male CNS)')
            print(f'            → hemibrain: Native JRCFIB2018F template')
        else:
            print(f'            → FlyWire/FAFB: Native FLYWIRE template (female brain)')
            print(f'            → male-cns: Native JRCFIB2022M template (male CNS)')
        print(f'            → These datasets have no transform requirements with brain_mesh="template"')
        print()
        print(f'{"="*70}')
        
        # Prompt user for choice
        print(f'\\nHow would you like to proceed?')
        print(f'  [1] Download transforms (~13GB) and continue with brain_mesh="whole"')
        print(f'  [2] Use brain_mesh="template" instead (no download, fast)')
        print(f'  [3] Use brain_mesh="none" (no brain mesh)')
        print(f'  [q] Quit')
        print()
        
        try:
            choice = input('Enter choice [1/2/3/q] (default: 2): ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = '2'
        
        if choice == '' or choice == '2':
            print(f'\\n{GREEN}✓ Using brain_mesh="template" (transformation-free){RESET}')
            self.brain_mesh = 'template'
        elif choice == '3':
            print(f'\\n{GREEN}✓ Using brain_mesh="none"{RESET}')
            self.brain_mesh = 'none'
        elif choice == '1':
            # Try to download transforms
            if self._check_and_download_transforms():
                print(f'\\n{GREEN}✓ Transforms downloaded successfully{RESET}')
            else:
                print(f'\\n{YELLOW}⚠️ Transform download failed or cancelled, using brain_mesh="template"{RESET}')
                self.brain_mesh = 'template'
        elif choice == 'q':
            print(f'\\n{RED}Exiting...{RESET}')
            sys.exit(0)
        else:
            print(f'\\n{YELLOW}⚠️ Invalid choice, using brain_mesh="template"{RESET}')
            self.brain_mesh = 'template'
    
    def _dataset_needs_transform(self):
        """Check if current dataset needs H5 transforms that require file downloads.
        
        Returns
        -------
        bool
            True if H5 transforms requiring file downloads are needed,
            False if only built-in affine transforms are needed
            
        Notes
        -----
        H5 transforms required (need ~13GB download):
        - hemibrain with brain_mesh='whole': JRCFIB2018Fraw → JRC2018F path includes H5transform
        - FlyWire/FAFB with brain_mesh='whole': FAFB → JRC2018F path includes H5transform
        
        Only affine transforms (built-in, no download):
        - hemibrain with brain_mesh='template': JRCFIB2018Fraw → JRCFIB2018F
        - male-cns: JRCFIB2022Mraw → JRCFIB2022M
        - manc: MANCraw → MANC
        - optic-lobe: JRCFIB2022Mraw → JRCFIB2022M
        
        No transforms at all:
        - FlyWire/FAFB with brain_mesh='template': Native FLYWIRE template (identity transform)
        """
        dataset_lower = self.dataset.lower()
        
        # Hemibrain with brain_mesh='whole' requires H5 transforms
        # because it needs to go from JRCFIB2018Fraw to JRC2018F (involves H5transform)
        if 'hemibrain' in dataset_lower and self.brain_mesh == 'whole':
            return True
        
        # FlyWire/FAFB with brain_mesh='whole' requires H5 transforms
        # because it needs to go from FAFB to JRC2018F (involves H5transform)
        if ('flywire' in dataset_lower or 'fafb' in dataset_lower) and self.brain_mesh == 'whole':
            return True
        
        # All other cases use only affine transforms or no transforms:
        # - hemibrain with template: Affine only
        # - FlyWire/FAFB with template: No transform (native FLYWIRE)
        # - male-cns, manc, optic-lobe: Affine only
        return False
    
    def _get_fafb_tilt_correction_matrix(self):
        """Get the affine transformation matrix to correct FAFB/FLYWIRE left-right tilt.
        
        The FLYWIRE template mesh has an intrinsic left-right tilt (~8-10 degrees rotation 
        around the Z-axis when viewed from the front). This method returns a rotation matrix
        that corrects this tilt.
        
        Returns
        -------
        numpy.ndarray
            4x4 affine transformation matrix for tilt correction.
            Returns identity matrix if not FAFB dataset or not using template mode.
        
        Notes
        -----
        The rotation is applied around the center of the FLYWIRE brain mesh to avoid
        shifting objects. The rotation combines:
        - Z-axis rotation (-4 degrees) to correct left-right tilt in front view
        - Y-axis rotation (-3 degrees) to correct tilt in top view
        """
        import numpy as np
        
        # Check if correction is needed
        is_fafb = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()
        if not is_fafb or self.brain_mesh != 'template':
            return np.eye(4)  # Return identity - no correction needed
        
        import math
        
        # Rotation angles
        angle_z = math.radians(-3)  # Front view tilt correction (Z-axis)
        angle_y = math.radians(-3)  # Top view tilt correction (Y-axis)
        
        # FLYWIRE brain mesh approximate center (computed from flybrains.FLYWIRE.mesh)
        # This ensures rotation happens around the brain center, not the origin
        center_x = 527652.0
        center_y = 240039.0
        center_z = 148110.0
        
        # Build transformation: translate to origin -> rotate -> translate back
        # T_back @ R @ T_to_origin
        
        # Translation to origin
        T_to_origin = np.array([
            [1, 0, 0, -center_x],
            [0, 1, 0, -center_y],
            [0, 0, 1, -center_z],
            [0, 0, 0, 1]
        ])
        
        # Translation back
        T_back = np.array([
            [1, 0, 0, center_x],
            [0, 1, 0, center_y],
            [0, 0, 1, center_z],
            [0, 0, 0, 1]
        ])
        
        # Rotation matrix around Z-axis (front view correction)
        cos_z = math.cos(angle_z)
        sin_z = math.sin(angle_z)
        Rz = np.array([
            [cos_z, -sin_z, 0, 0],
            [sin_z,  cos_z, 0, 0],
            [0,      0,     1, 0],
            [0,      0,     0, 1]
        ])
        
        # Rotation matrix around Y-axis (top view correction)
        cos_y = math.cos(angle_y)
        sin_y = math.sin(angle_y)
        Ry = np.array([
            [cos_y,  0, sin_y, 0],
            [0,      1, 0,     0],
            [-sin_y, 0, cos_y, 0],
            [0,      0, 0,     1]
        ])
        
        # Combined rotation: first Z, then Y
        R = Ry @ Rz
        
        # Full transformation: translate to origin, rotate, translate back
        rotation_matrix = T_back @ R @ T_to_origin
        
        return rotation_matrix
    
    def _apply_fafb_tilt_correction(self, obj, progress_bar=None, layer_label='layer'):
        """Apply FAFB tilt correction to navis objects or DataFrames with xyz coordinates.
        
        Parameters
        ----------
        obj : TreeNeuron, MeshNeuron, Volume, NeuronList, or pd.DataFrame
            Object to transform. DataFrames must have 'x', 'y', 'z' columns.
        progress_bar : tqdm, optional
            Shared layer progress bar; a NeuronList is transformed per neuron
            with progress reported through its postfix instead of navis
            spawning its own "Xforming" bar on a separate row.
        layer_label : str
            Label used in the shared bar's postfix while transforming.
            
        Returns
        -------
        same type as input
            Transformed object (in-place for most objects)
        """
        import numpy as np
        
        # Check if correction is needed
        is_fafb = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()
        if not is_fafb or self.brain_mesh != 'template' or not self.FAFB_template_correction:
            return obj  # No correction needed
        
        rotation_matrix = self._get_fafb_tilt_correction_matrix()
        
        # Check if it's identity (no transform needed)
        if np.allclose(rotation_matrix, np.eye(4)):
            return obj
        
        # Handle DataFrame separately (for synapse coordinates)
        if isinstance(obj, pd.DataFrame):
            try:
                # Apply full 4x4 affine transform to x, y, z columns
                if 'x' in obj.columns and 'y' in obj.columns and 'z' in obj.columns:
                    # Convert to homogeneous coordinates and apply full transform
                    coords = obj[['x', 'y', 'z']].values
                    n_points = coords.shape[0]
                    
                    # Add homogeneous coordinate (1s column)
                    homogeneous = np.hstack([coords, np.ones((n_points, 1))])
                    
                    # Apply transform: result = coords @ transform.T
                    transformed = homogeneous @ rotation_matrix.T
                    
                    obj = obj.copy()
                    obj['x'] = transformed[:, 0]
                    obj['y'] = transformed[:, 1]
                    obj['z'] = transformed[:, 2]
                return obj
            except Exception as e:
                self._vprint(f'⚠️  Failed to apply FAFB tilt correction to DataFrame: {e}', level='full')
                return obj
        
        # Apply transform using navis.xform for neurons/volumes
        try:
            # navis.xform requires an AffineTransform object, not a raw numpy array
            from navis.transforms import AffineTransform
            affine_transform = AffineTransform(rotation_matrix)

            # Xforming a whole NeuronList in one call makes navis spawn its
            # own "Xforming" progress bar on a separate row. Loop per neuron
            # instead and report through the shared layer bar so the layer's
            # transform + plot progress refreshes on a single row.
            if isinstance(obj, navis.NeuronList) and len(obj) > 1:
                def set_status(status):
                    if progress_bar is not None and hasattr(progress_bar, 'set_postfix_str'):
                        progress_bar.set_postfix_str(status)

                xf = []
                total = len(obj)
                for idx, n in enumerate(obj):
                    label = getattr(n, 'name', None) or str(getattr(n, 'id', idx))
                    set_status(f'{layer_label}: tilt correction {idx}/{total} | current={label}')
                    try:
                        xf.append(navis.xform(n, affine_transform))
                    except Exception:
                        xf.append(n)  # Keep original on per-neuron failure
                set_status(f'{layer_label}: tilt correction {total}/{total}')
                return obj.__class__(xf)

            transformed = navis.xform(obj, affine_transform)
            return transformed
        except Exception as e:
            self._vprint(f'⚠️  Failed to apply FAFB tilt correction: {e}', level='full')
            return obj
    
    def _get_template_info(self):
        """Get template brain/VNC information for current dataset.
        
        Handles transform paths for all NeuPrint datasets:
        - Brain datasets: hemibrain, optic-lobe
        - VNC datasets: manc (various versions)
        - Brain+VNC datasets: male-cns
        
        Returns
        -------
        dict
            Dictionary with 'source', 'target', 'template_obj', and 'mesh_name' keys
            
        Notes
        -----
        Transform paths by dataset:
        - hemibrain: JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F
        - optic-lobe: JRCFIB2018Fraw → JRCFIB2018F → JRCFIB2018Fum → JRC2018F (same as hemibrain)
        - manc: MANCraw → MANC (VNC only, no brain transform)
        - male-cns: JRCFIB2022Mraw → JRCFIB2022M (brain + VNC)
        
        Note: optic-lobe uses the same coordinate system as hemibrain because it's
        a focused reconstruction of the optic lobe region within the hemibrain volume.
        """
        dataset_lower = self.dataset.lower()
        import flybrains
        
        # Brain datasets
        if 'hemibrain' in dataset_lower:
            return {
                'source': 'JRCFIB2018Fraw',
                'target': 'JRC2018F' if self.brain_mesh == 'whole' else 'JRCFIB2018F',
                'template_obj': flybrains.JRC2018F if self.brain_mesh == 'whole' else flybrains.JRCFIB2018F,
                'mesh_name': 'JRC2018F (whole brain)' if self.brain_mesh == 'whole' else 'JRCFIB2018F (hemibrain)'
            }
        elif 'optic' in dataset_lower:
            # Optic-lobe dataset is part of the Male CNS (JRCFIB2022M) volume
            # It is NOT part of the hemibrain (JRCFIB2018F) volume
            # Stored in JRCFIB2022Mraw coordinates
            return {
                'source': 'JRCFIB2022Mraw',
                'target': 'JRCFIB2022M',  # Male CNS template
                'template_obj': flybrains.JRCFIB2022M,
                'mesh_name': 'JRCFIB2022M (Male CNS)'
            }
        
        # VNC datasets
        elif 'manc' in dataset_lower:
            # MANC (Male Adult Nerve Cord)
            if self.brain_mesh == 'whole':
                # Transform to Male CNS (brain + VNC) space
                return {
                    'source': 'MANC',
                    'target': 'JRCFIB2022M',
                    'template_obj': flybrains.JRCFIB2022M,
                    'mesh_name': 'JRCFIB2022M (male CNS: brain + VNC)'
                }
            else:
                # Use native MANC template (VNC only)
                return {
                    'source': 'MANC',
                    'target': 'MANC',
                    'template_obj': flybrains.MANC,
                    'mesh_name': 'MANC (VNC envelope)'
                }
        
        # Brain + VNC datasets
        elif 'male-cns' in dataset_lower or 'malecns' in dataset_lower:
            # Male CNS (JRCFIB2022M) - Brain + VNC
            # 'whole' shows full CNS envelope (brain + VNC)
            return {
                'source': 'JRCFIB2022Mraw',
                'target': 'JRCFIB2022M',
                'template_obj': flybrains.JRCFIB2022M,
                'mesh_name': 'JRCFIB2022M (male CNS: brain + VNC)'
            }
        
        # FlyWire / FAFB datasets
        # For 'template': No transform needed - use native FLYWIRE coordinates
        # For 'whole': Transform to JRC2018F (standard female brain template)
        elif 'flywire' in dataset_lower or 'fafb' in dataset_lower:
            if self.brain_mesh == 'whole':
                # Transform to JRC2018F standard whole-brain template
                # This requires H5 transforms (~580MB download)
                return {
                    'source': 'FAFB',  # FAFB native coordinates
                    'target': 'JRC2018F',  # Standard female brain template
                    'template_obj': flybrains.JRC2018F,
                    'mesh_name': 'JRC2018F (standard whole brain)',
                    'skip_transform': False  # Need to transform skeletons/synapses
                }
            else:
                # 'template' mode: Use native FLYWIRE coordinates (no transform)
                # FLYWIRE and FAFB14 share the same bounding box and are effectively the same space
                # for visualization purposes. Using native coordinates avoids:
                # 1. Downloading ~580MB transform file (JRC2018F_FAFB.h5)
                # 2. Slow transformation of all skeleton vertices and synapse coordinates
                # 3. Potential coordinate precision loss from warping
                return {
                    'source': 'FLYWIRE',  # Native space - no transform
                    'target': 'FLYWIRE',  # Same as source - identity transform
                    'template_obj': flybrains.FLYWIRE,
                    'mesh_name': 'FLYWIRE (native FAFB coordinates)',
                    'skip_transform': True  # Flag to skip skeleton/synapse transforms
                }
        
        # Fallback to hemibrain for unknown datasets
        else:
            self._vprint(f'⚠️  Unknown dataset "{self.dataset}", defaulting to hemibrain template')
            return {
                'source': 'JRCFIB2018Fraw',
                'target': 'JRCFIB2018F',
                'template_obj': flybrains.JRCFIB2018F,
                'mesh_name': 'JRCFIB2018F (hemibrain)'
            }
    
    def _get_vnc_template_info(self):
        """Get VNC template information for current dataset.
        
        Available for datasets with VNC data:
        - male-cns: VNC portion of JRCFIB2022M (extracted geometrically, Y >= 210000)
        - manc: MANC template (native VNC mesh)
        
        Returns
        -------
        dict or None
            Dictionary with 'mesh' (trimesh object) and 'mesh_name' keys,
            or None if VNC mesh is not available for the current dataset.
        """
        dataset_lower = self.dataset.lower()
        import flybrains
        import trimesh
        
        # Male CNS and MANC datasets - extract VNC portion from JRCFIB2022M mesh
        if 'male-cns' in dataset_lower or 'malecns' in dataset_lower or 'manc' in dataset_lower:
            # Check if flybrains provides mesh_vnc (flybrains >= 0.6.3)
            if hasattr(flybrains.JRCFIB2022M, 'mesh_vnc'):
                return {
                    'mesh': flybrains.JRCFIB2022M.mesh_vnc,
                    'mesh_name': 'JRCFIB2022M (VNC)'
                }
            else:
                # Extract VNC portion geometrically
                # VNC is in the posterior portion (higher Z values in JRCFIB2022M coordinates)
                # Z cutoff ~340000 separates brain from VNC (Neck region)
                try:
                    full_mesh = flybrains.JRCFIB2022M.mesh
                    vnc_z_cutoff = 340000  # Z coordinate separating brain from VNC
                    
                    # Get vertices in the VNC region (Z >= cutoff)
                    vnc_mask = full_mesh.vertices[:, 2] >= vnc_z_cutoff
                    
                    # Extract submesh by filtering faces that have all vertices in VNC region
                    vnc_faces = []
                    for face in full_mesh.faces:
                        if all(vnc_mask[v] for v in face):
                            vnc_faces.append(face)
                    
                    if vnc_faces:
                        # Create new mesh from VNC vertices only
                        vnc_mesh_trimesh = trimesh.Trimesh(
                            vertices=full_mesh.vertices,
                            faces=vnc_faces
                        )
                        # Clean up unused vertices
                        vnc_mesh_trimesh.remove_unreferenced_vertices()
                        vnc_mesh = navis.Volume(vnc_mesh_trimesh, name='JRCFIB2022M_vnc')
                        return {
                            'mesh': vnc_mesh,
                            'mesh_name': 'JRCFIB2022M (VNC)'
                        }
                except Exception as e:
                    self._vprint(f'⚠️  VNC mesh extraction failed: {e}', level='simple')
                    return None
        
        # MANC dataset - VNC only (has proper VNC mesh)
        elif 'manc' in dataset_lower:
            return {
                'mesh': flybrains.MANC.mesh,
                'mesh_name': 'MANC (VNC)'
            }
        
        # VNC mesh not available for other datasets
        return None
    
    def _needs_skeleton_transform(self):
        """Check if skeleton/synapse coordinate transforms are needed.
        
        This method determines whether coordinate transforms should be applied
        to skeleton and synapse data during visualization.
        
        Returns
        -------
        bool
            True if transforms should be applied, False if data is already in target space
            
        Notes
        -----
        Returns False (skip transform) for:
        - FlyWire/FAFB: Data and template mesh are both in FLYWIRE space (identity transform)
        - brain_mesh='none': No template mesh, no transform needed
        
        Returns True (apply transform) for:
        - hemibrain, male-cns, manc, optic-lobe: Data needs affine transform to template space
          (These are fast, built-in affine transforms, no file download needed)
        
        Note: This is different from _dataset_needs_transform() which checks if heavy
        H5 transforms requiring file downloads are needed.
        """
        if self.brain_mesh == 'none':
            return False
            
        template_info = self._get_template_info()
        
        # Check for skip_transform flag (set for FAFB/FlyWire - identity transform)
        if template_info.get('skip_transform', False):
            return False
        
        # No transform needed if source == target (identity transform)
        if template_info['source'] == template_info['target']:
            return False
            
        return True
    
    def _check_and_download_transforms(self):
        """Check if flybrains transforms exist locally, prompt user before downloading.
        
        Brain transforms are large files (multiple files, ~10GB total uncompressed). 
        This method checks if the required transforms exist locally before attempting 
        to download them, and prompts the user for confirmation.
        
        Transforms are stored in the default flybrains data directory:
        ~/flybrain-data/
        
        Returns
        -------
        bool
            True if transforms are available (already exist or successfully downloaded),
            False otherwise.
        
        References:
        - flybrains package: https://github.com/navis-org/navis-flybrains
        - JRC2018F brain template: https://www.janelia.org/open-science/jrc-2018-brain-templates
        """
        if not self.verbose:
            return False

        try:
            import flybrains
            
            # Get the transform directory from attribute or use default
            transforms_dir = os.path.expanduser(self.transforms_dir)
            
            # Set environment variable if custom path is specified
            if self.transforms_dir != '~/flybrain-data':
                os.environ['FLYBRAINS_DATA'] = transforms_dir
                self._vprint(f'Using custom transform directory: {transforms_dir}', level='full')
            
            # Get dataset-specific template info
            template_info = self._get_template_info()
            source = template_info['source']
            target = template_info['target']
            
            # ANSI color codes
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            
            # Check if the transformation path exists by attempting to find bridging path
            try:
                path = navis.transforms.registry.find_bridging_path(source, target)
                self._vprint(f'✓ Brain transforms already available', level='full')
                self._vprint(f'  Location: {YELLOW}{transforms_dir}{RESET}', level='full')
                self._vprint(f'  Transform path: {" -> ".join([str(p) for p in path])}', level='full')
                return True
            except (ValueError, KeyError):
                # Transform path not found, need to download
                pass
            
            # ANSI color codes
            YELLOW = '\033[93m'
            RESET = '\033[0m'
            
            # Prompt user for download confirmation
            self._vprint('\\n' + '='*70)
            self._vprint('⚠️  Brain Transformation Required')
            self._vprint('='*70)
            self._vprint(f'To use brain_mesh="whole" for {self.dataset}, you need brain transforms.')
            self._vprint(f'Transform path needed: {source} → JRCFIB2018F → JRCFIB2018Fum → {target}')
            self._vprint('')
            self._vprint('⚠️  IMPORTANT: flybrains downloads ALL JRC transforms as a bundle:')
            self._vprint('   • JRC2018F_JRCFIB2018F.h5   (~1.29 GB)  ← YOU NEED THIS for hemibrain/optic-lobe')
            self._vprint('   • JRC2018F_FAFB.h5          (~580 MB)   (enables FAFB dataset support)')
            self._vprint('   • JRC2018F_JFRC2013.h5      (~1.39 GB)  (enables JFRC2013 template)')
            self._vprint('   • JRC2018F_FCWB.h5          (~1.29 GB)  (enables FCWB template)')
            self._vprint('   • JRC2018U_JRC2018F.h5      (~717 MB)   (enables unisex template)')
            self._vprint('   • JRC2018U_JRC2018M.h5      (~1.10 GB)  (enables male template)')
            self._vprint('   • JRC2018F_JFRC2010.h5      (~1.65 GB)  (enables legacy template)')
            self._vprint('   • JRCFIB2022M_JRC2018M.h5   (~2.12 GB)  (enables male CNS registration)')
            self._vprint('')
            self._vprint('   Total download: ~10 GB (but only ~1.3 GB used for your dataset)')
            self._vprint('   Download time: ~1-2 hours (cannot download individual files)')
            self._vprint('   Why all files? The flybrains package bundles all transforms together.')
            self._vprint('')
            self._vprint('The transforms will be cached in:')
            self._vprint(f'  {YELLOW}{transforms_dir}/{RESET}')
            
            # Save transform path info to file
            info_file = os.path.join(self.output_dir, 'brain_transforms_info.txt')
            os.makedirs(self.output_dir, exist_ok=True)
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write('Brain Transforms Information\\n')
                f.write('='*70 + '\\n\\n')
                f.write(f'Dataset: {self.dataset}\\n')
                f.write(f'Transform path: {source} → JRCFIB2018F → JRCFIB2018Fum → {target}\\n\\n')
                f.write('Storage Location:\\n')
                f.write(f'  {transforms_dir}/\\n\\n')
                f.write('Transform Files (8 files, ~10 GB total):\\n')
                f.write('  • JRC2018F_JRCFIB2018F.h5   (~1.29 GB)\\n')
                f.write('  • JRC2018F_FAFB.h5          (~580 MB)\\n')
                f.write('  • JRC2018F_JFRC2013.h5      (~1.39 GB)\\n')
                f.write('  • JRC2018F_FCWB.h5          (~1.29 GB)\\n')
                f.write('  • JRC2018U_JRC2018F.h5      (~717 MB)\\n')
                f.write('  • JRC2018U_JRC2018M.h5      (~1.10 GB)\\n')
                f.write('  • JRC2018F_JFRC2010.h5      (~1.65 GB)\\n')
                f.write('  • JRCFIB2022M_JRC2018M.h5   (~2.12 GB)\\n\\n')
                f.write('To change the storage location:\\n')
                f.write('  1. Set transforms_dir attribute when creating VisualizeSkeleton\\n')
                f.write('  2. Set FLYBRAINS_DATA environment variable before importing flybrains\\n')
                f.write('  3. Or manually move files to the new location\\n\\n')
                f.write('More information:\\n')
                f.write('  https://github.com/navis-org/navis-flybrains\\n')
            self._vprint(f'\\n📄 Transform info saved to: {info_file}')
            self._vprint('')
            self._vprint('💡 Note: The flybrains.download_jrc_transforms() function downloads')
            self._vprint('   ALL 8 files as a bundle with no selective download option.')
            self._vprint('   This is by design in the flybrains library to provide complete')
            self._vprint('   cross-dataset registration capabilities.')
            self._vprint('')
            self._vprint('For more information, see:')
            self._vprint('  https://github.com/navis-org/navis-flybrains')
            self._vprint('='*70)
            
            response = input('Download all transforms now? [y/N]: ').strip().lower()
            
            if response in ['y', 'yes']:
                self._vprint('\\n📥 Downloading brain transforms...')
                self._vprint('This may take several minutes depending on your connection.')
                flybrains.download_jrc_transforms()
                
                # Re-register transforms after download
                self._vprint('📝 Registering downloaded transforms...')
                flybrains.register_transforms()
                
                # Verify the transform path is now available
                try:
                    path = navis.transforms.registry.find_bridging_path(source, target)
                    self._vprint(f'✓ Transforms downloaded and registered successfully!')
                    self._vprint(f'  Location: {YELLOW}{transforms_dir}{RESET}')
                    self._vprint(f'  Transform path: {" -> ".join([str(p) for p in path])}')
                    
                    # Update the saved info file with success status
                    info_file = os.path.join(self.output_dir, 'brain_transforms_info.txt')
                    with open(info_file, 'a', encoding='utf-8') as f:
                        f.write(f'\\nDownload Status: SUCCESS\\n')
                        f.write(f'Downloaded at: {pd.Timestamp.now()}\\n')
                    return True
                except (ValueError, KeyError) as e:
                    self._vprint(f'⚠️  Transforms downloaded but bridging path not found: {e}')
                    self._vprint(f'   This may indicate the transforms do not include {source} → {target}')
                    return False
            else:
                self._vprint('\\n⚠️  Download cancelled. Setting brain_mesh to "none".')
                return False
                
        except ImportError:
            self._vprint('\\n⚠️  flybrains package not installed.')
            self._vprint('   Install it with: pip install navis[flybrains]')
            self._vprint('   Setting brain_mesh to "none".')
            return False
        except Exception as e:
            self._vprint(f'\\n⚠️  Error checking brain transforms: {e}')
            self._vprint('   Setting brain_mesh to "none".')
            return False
    
    def plot_mesh(self):
        """Plot ROI meshes and brain meshes.
        
        Loads ROI meshes from dataset-specific cache directories, with fallback to
        primary_rois/ for backward compatibility. Supports brain mesh visualization
        with automatic transform handling.
        
        Dataset-specific mesh caching:
        - hemibrain:v1.2.1 -> navis_roi_meshes_json/hemibrain_v1_2_1/
        - optic-lobe:v1.1 -> navis_roi_meshes_json/optic-lobe_v1_1/
        - Fallback: navis_roi_meshes_json/primary_rois/
        
        Brain mesh options (dataset-aware):
        - 'none': Only plot ROI meshes specified in mesh_roi parameter
        - 'template': Plot native EM template mesh (JRCFIB2018F, MANC, or JRCFIB2022M)
        - 'whole': Plot standard template mesh (may require transforms for some datasets)
        
        Behavior with mesh_roi=[]:
        - When mesh_roi is an empty list [], no ROI meshes are plotted
        - But brain_mesh='whole' or 'template' will still plot the brain mesh
        - This allows showing neurons with only the whole brain outline
        
        References:
        - navis Volume API: https://navis.readthedocs.io/en/latest/source/api.html#navis.Volume
        - flybrains templates: https://github.com/navis-org/navis-flybrains
        - mesh optimization: use Volume.simplify() to reduce mesh complexity for faster rendering
        """
        # Skip if mesh_roi is None (explicitly disabled)
        # Note: Empty list [] means "no ROI meshes but maybe brain mesh"
        if self.mesh_roi is None:
            return
        
        # Check if we have any work to do (ROI meshes, brain mesh, or VNC mesh)
        has_roi_meshes = len(self.mesh_roi) > 0
        has_brain_mesh = self.brain_mesh in ['template', 'whole']
        has_vnc_mesh = self.vnc_mesh
        
        if not has_roi_meshes and not has_brain_mesh and not has_vnc_mesh:
            return
        
        # For FAFB with brain_mesh='whole' and mesh_roi specified
        # ROI transforms from male-cns (JRCFIB2022Mraw) to JRC2018F require elastix which is problematic
        is_flywire = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()
        if is_flywire and self.brain_mesh == 'whole' and has_roi_meshes:
            # Always skip ROI meshes in whole mode for FAFB/FlyWire to avoid elastix dependency
            self._vprint('')
            self._vprint('⚠️  WARNING: ROI mesh transformation is not supported in "whole" mode for FAFB.', level='simple')
            self._vprint('   ROI meshes (mesh_roi) will be skipped.', level='simple')
            self._vprint('', level='simple')
            self._vprint('   📌 Recommendation:', level='simple')
            self._vprint('   Use brain_mesh="template" instead. This mode supports ROI meshes natively.', level='simple')
            self._vprint('')
            # Clear mesh_roi to skip ROI plotting but continue with brain mesh
            self.mesh_roi = []
            has_roi_meshes = False
        
        # Ensure available_rois.json exists (generate if missing)
        # This checks cache first, and if missing, fetches from API or scans local meshes
        self._get_available_rois(use_cache=True, fetch_online=True)
        
        # Get dataset-specific mesh directory
        mesh_dir = self._get_dataset_mesh_dir()
        self._vprint(f'Using mesh directory: {mesh_dir}', level='full')
        
        roiunits = []
        roi_names = []
        roi_colors = []
        
        # Use mesh_roi list directly (no auto-mirroring suffix expansion)
        final_mesh_roi = self.mesh_roi
        
        # Handle colors - mesh_color is already standardized to rgba strings
        final_mesh_colors = []
        for i, roi in enumerate(final_mesh_roi):
            if isinstance(self.mesh_color, list):
                if i < len(self.mesh_color):
                    color = self.mesh_color[i]
                else:
                    # Fallback to default gray with current mesh_alpha
                    color = f'rgba(100, 100, 100, {self.mesh_alpha})'
            else:
                color = self.mesh_color
            
            # Ensure color is a string (standardize if needed)
            if not isinstance(color, str):
                color = standardize_color(color, default_alpha=self.mesh_alpha)
            
            final_mesh_colors.append(color)
        
        for i, roi in enumerate(final_mesh_roi):
            color = final_mesh_colors[i]
            source_info = "Dataset Cache"
            roi_source_space = None # Track the coordinate space of the ROI
            roi_needs_transform = False  # Track if ROI needs transform after loading
            
            # Determine if this is FlyWire/FAFB
            is_flywire = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()

            # Try dataset-specific directory first (with case-safe filename)
            mesh_file = self._get_mesh_file_path(mesh_dir, roi)
            
            # For FAFB: Check for pre-transformed ROI mesh cache first
            # Transformed meshes are stored in cache/{dataset}/meshes_transformed/{TARGET}/
            # where TARGET is FLYWIRE (for template mode) or JRC2018F (for whole mode)
            transformed_mesh_file = None
            if is_flywire:
                # Determine target space based on brain_mesh mode
                target_space = 'JRC2018F' if self.brain_mesh == 'whole' else 'FLYWIRE'
                transformed_cache_dir = os.path.join(self._get_cache_path('meshes_transformed'), target_space)
                transformed_mesh_file = self._get_mesh_file_path(transformed_cache_dir, roi)
                
                if os.path.exists(transformed_mesh_file):
                    # Load pre-transformed mesh - no further transform needed
                    mesh_file = transformed_mesh_file
                    source_info = f"FAFB Transformed Cache ({target_space})"
                    roi_needs_transform = False
                    self._vprint(f'  ✓ Loading "{roi}" from transformed cache ({target_space})', level='full')
            
            # Special handling for FlyWire/FAFB - fetch from male-cns if not found
            if is_flywire and not os.path.exists(mesh_file):
                    self._vprint(f'📥 ROI mesh "{roi}" not found locally, attempting to download...', level='full')
                    mesh_found = False
                    
                    # 1. Try male-cns:v0.9 (NeuPrint)
                    try:
                        import navis.interfaces.neuprint as neu
                        from neuprint import Client
                        from utils.token_manager import token_manager
                        
                        token = token_manager.get_neuprint_token() or self.token
                        if token:
                            try:
                                self._vprint(f'   Checking male-cns:v0.9...', level='full')
                                mc_client = Client('https://neuprint.janelia.org', dataset='male-cns:v0.9', token=token)
                                mesh = neu.fetch_roi(roi, client=mc_client)
                                if mesh:
                                    os.makedirs(mesh_dir, exist_ok=True)
                                    mesh.to_json(mesh_file)
                                    self._vprint(f'   ✓ Found in male-cns:v0.9', level='full')
                                    source_info = "male-cns:v0.9 (Downloaded)"
                                    roi_source_space = 'JRCFIB2022Mraw' # Use raw coordinates for male-cns ROIs
                                    mesh_found = True
                            except Exception as e:
                                # print(f'   (male-cns check failed: {e})')
                                pass
                    except ImportError:
                        pass
                    
                    # 2. Try generic navis fetch (fallback) - REMOVED as it causes errors if navis doesn't have fetch_roi
                    # if not mesh_found:
                    #     try:
                    #         print(f'   Attempting generic navis fetch...')
                    #         # This tries to use whatever client is default or configured in navis
                    #         # Usually fetches from Hemibrain if no dataset specified, or checks available clients
                    #         mesh = navis.fetch_roi(roi)
                    #         if mesh:
                    #             os.makedirs(mesh_dir, exist_ok=True)
                    #             mesh.to_json(mesh_file)
                    #             print(f'   ✓ Found via navis.fetch_roi')
                    #             source_info = "navis.fetch_roi"
                    #             roi_source_space = 'JRCFIB2018F' # Default for Hemibrain ROIs
                    #             mesh_found = True
                    #     except Exception as e:
                    #         print(f'   Warning: Failed to fetch "{roi}" via navis: {e}')

            # Standard logic for non-FlyWire or if file exists
            # Fallback to primary_rois if not found (only for non-FlyWire or if we want to support it)
            if not os.path.exists(mesh_file) and not is_flywire:
                # Try legacy fallback path (doesn't use case-safe encoding)
                mesh_file_fallback = os.path.join(self.script_path, 'navis_roi_meshes_json', 'primary_rois', roi + '.json')
                if os.path.exists(mesh_file_fallback):
                    mesh_file = mesh_file_fallback
                    source_info = "Primary ROIs (Local)"
                    roi_source_space = 'JRCFIB2018F'
                else:
                    # Try to download from NeuPrint (Hemibrain/Optic Lobe/Male CNS)
                    self._vprint(f'📥 ROI mesh "{roi}" not found locally, attempting to download from NeuPrint...', level='full')
                    source_info = "NeuPrint (Downloaded)"
                    try:
                        import navis.interfaces.neuprint as neu
                        from neuprint import Client
                        from utils.token_manager import token_manager
                        
                        client = self.client
                        
                        if client is None:
                            token = token_manager.get_neuprint_token() or self.token
                            
                            if token:
                                if 'optic' in self.dataset.lower():
                                    server = 'https://neuprint-optic-lobe.janelia.org'
                                    dataset_name = self.dataset.split(':')[0]
                                    roi_source_space = 'JRCFIB2022Mraw' # Optic lobe
                                elif 'male-cns' in self.dataset.lower() or 'malecns' in self.dataset.lower():
                                    server = 'https://neuprint.janelia.org'
                                    dataset_name = 'male-cns:v0.9' # Default for male-cns
                                    roi_source_space = 'JRCFIB2022Mraw' # Male CNS raw
                                else:
                                    server = 'https://neuprint.janelia.org'
                                    dataset_name = 'hemibrain:v1.2.1'
                                    roi_source_space = 'JRCFIB2018F'
                                
                                try:
                                    client = Client(server, dataset=dataset_name, token=token)
                                except Exception as e:
                                    self._vprint(f'   Warning: Failed to create client: {e}', level='full')
                        
                        mesh = neu.fetch_roi(roi, client=client)
                        os.makedirs(mesh_dir, exist_ok=True)
                        mesh.to_json(mesh_file)
                        self._vprint(f'✓ Downloaded and cached "{roi}" mesh to {mesh_file}', level='full')
                        
                        # Transform if needed (Hemibrain specific)
                        if self.brain_mesh in ['whole', 'template']:
                            template_info = self._get_template_info()
                            self._vprint(f'Transforming brain region {roi}...', end='', level='full')
                            with self._suppress_output():
                                mesh = navis.xform_brain(mesh, source=template_info['source'], target=template_info['target'])
                            # Note: We don't save the transformed mesh back to cache here to keep cache pure?
                            # Actually previous code didn't save transformed.
                    except Exception as e:
                        self._vprint(f'⚠️  Failed to download "{roi}" mesh: {e}', level='full')
            
            # Load and plot
            if os.path.exists(mesh_file):
                try:
                    mesh = navis.Volume.from_json(mesh_file)
                    self._vprint(f'✓ Loaded "{roi}" from {source_info}', level='full')
                    
                    # Transform if needed (skip if loaded from transformed cache)
                    if self.brain_mesh in ['whole', 'template'] and "Transformed Cache" not in source_info:
                        template_info = self._get_template_info()
                        target = template_info['target']
                        
                        # For FAFB with ROIs from male-cns:
                        # - 'template' mode: transform to FLYWIRE (native FAFB space)
                        # - 'whole' mode: transform to JRC2018F (standard whole brain template)
                        if is_flywire:
                            # ROIs from male-cns need to be transformed
                            if roi_source_space:
                                source = roi_source_space
                            else:
                                # If loading from raw cache (roi_source_space is None), assume it's from male-cns
                                source = 'JRCFIB2022Mraw'
                            
                            # Target depends on brain_mesh mode
                            if self.brain_mesh == 'whole':
                                target = 'JRC2018F'  # Match skeleton/template space
                            else:
                                target = 'FLYWIRE'  # Native FAFB space for template mode
                            
                            self._vprint(f'Transforming ROI {roi} ({source} -> {target})...', end='', level='full')
                            try:
                                with self._suppress_output():
                                    mesh = navis.xform_brain(mesh, source=source, target=target)
                                self._vprint(' Done', level='full')
                                
                                # Cache the transformed mesh for future use
                                # Use target space name for cache directory (FLYWIRE or JRC2018F)
                                transformed_cache_dir = os.path.join(self._get_cache_path('meshes_transformed'), target)
                                os.makedirs(transformed_cache_dir, exist_ok=True)
                                # Use case-safe filename for transformed mesh
                                transformed_mesh_file = os.path.join(transformed_cache_dir, self._roi_to_filename(roi))
                                try:
                                    mesh.to_json(transformed_mesh_file)
                                    self._vprint(f'  💾 Cached transformed ROI to {transformed_mesh_file}', level='full')
                                except Exception as cache_e:
                                    self._vprint(f'  ⚠️ Failed to cache transformed ROI: {cache_e}', level='full')
                            except Exception as e:
                                self._vprint(f' Failed: {e}', level='full')
                        else:
                            # Non-FAFB datasets: use standard transform
                            source = template_info['source']
                            
                            # Skip transform if source == target (identity)
                            if source != target:
                                self._vprint(f'Transforming brain region {roi} ({source} -> {target})...', end='', level='full')
                                try:
                                    with self._suppress_output():
                                        mesh = navis.xform_brain(mesh, source=source, target=target)
                                    self._vprint(' Done', level='full')
                                except Exception as e:
                                    self._vprint(f' Failed: {e}', level='full')
                    
                    # Simplify mesh if requested
                    if self.roi_mesh_simplification > 0:
                        try:
                            import trimesh
                            # Access underlying trimesh object
                            tm = None
                            if hasattr(mesh, 'trimesh'):
                                tm = mesh.trimesh
                            elif hasattr(mesh, 'mesh'):
                                tm = mesh.mesh
                            elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                                # Fallback: create trimesh from vertices/faces
                                # Note: navis.Volume properties might be numpy arrays
                                tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                            
                            if tm:
                                n_faces = len(tm.faces)
                                target_faces = int(n_faces * (1 - self.roi_mesh_simplification))
                                if target_faces < n_faces:
                                    # Use open3d for accurate simplification (trimesh 4.x fast_simplification only achieves ~60%)
                                    new_tm = self._simplify_mesh_open3d(tm, target_faces)
                                    
                                    # Re-instantiate Volume to ensure it's clean and updated
                                    # Preserving attributes
                                    old_name = getattr(mesh, 'name', roi)
                                    old_id = getattr(mesh, 'id', None)
                                    
                                    mesh = navis.Volume(new_tm, name=old_name, id=old_id)
                                    
                                    self._vprint(f' (simplified {self.roi_mesh_simplification*100:.0f}%: {n_faces}->{len(new_tm.faces)} faces)', end='', level='full')
                                else:
                                    self._vprint(f' (simplification skipped: target {target_faces} >= {n_faces} faces)', end='', level='full')
                            else:
                                # Debug: print available attributes to help diagnose
                                attrs = [a for a in dir(mesh) if not a.startswith('_')]
                                self._vprint(f' (simplification skipped: could not extract mesh from {type(mesh)}. Available attrs: {attrs[:10]}...)', end='', level='full')
                        except Exception as e:
                            self._vprint(f' (simplification failed: {e})', end='', level='full')

                    # Collect for export
                    try:
                        self._append_exportable_mesh(mesh, color=color, name=getattr(mesh, 'name', roi), role='roi')
                    except Exception as e:
                        self._vprint(f' (export collection failed: {e})', end='', level='full')

                    # Apply FAFB tilt correction if using template mode
                    # This corrects the left-right tilt in the FLYWIRE template mesh
                    if is_flywire and self.brain_mesh == 'template':
                        mesh = self._apply_fafb_tilt_correction(mesh)

                    roiunits.append(mesh)
                    roi_names.append(roi)
                    roi_colors.append(color)

                    # Mirror logic: SKIP if FlyWire
                    if not is_flywire:
                        contralateral_roi = roi.replace('(R)', '(L)')
                        should_mirror = (
                            self.mirror_on_contralateral and 
                            roi.endswith('(R)') and 
                            contralateral_roi not in final_mesh_roi
                        )
                        
                        if should_mirror:
                            try:
                                template = None
                                if self.brain_mesh == 'whole':
                                    template_info = self._get_template_info()
                                    template = template_info['target']
                                elif self.brain_mesh == 'template':
                                    if 'hemibrain' in self.dataset or 'optic-lobe' in self.dataset:
                                        template = 'JRCFIB2018F'
                                    elif 'male-cns' in self.dataset:
                                        template = 'JRCFIB2022M'
                                
                                if template:
                                    mirrored_mesh = navis.mirror_brain(mesh, template, mirror_axis='x')
                                    roiunits.append(mirrored_mesh)
                                    roi_names.append(contralateral_roi)
                                    roi_colors.append(color)
                            except Exception as e:
                                self._vprint(f' (mirror failed: {e})', end='', level='full')

                except Exception as e:
                    self._vprint(f'⚠️  Failed to load mesh {roi}: {e}', level='full')
            else:
                if not is_flywire: # Only warn if we expected to find it (FlyWire might just fail silently if not found)
                     self._vprint(f'⚠️  ROI mesh "{roi}" not found.', level='full')
        
        # Plot ROI meshes if any were loaded
        if roiunits:
            self._vprint('plotting mesh of brain regions...', level='full')
            for roi_i in range(len(roiunits)):
                # Colors are now standardized to rgba strings, convert to hex + alpha for navis
                color_str = roi_colors[roi_i]
                
                # Safeguard: If color is not a string (could be tuple/list from unexpected path),
                # standardize it now
                if not isinstance(color_str, str):
                    color_str = standardize_color(color_str, default_alpha=self.mesh_alpha)
                
                # Extract hex color and alpha from standardized rgba string
                color_hex = self._rgba_to_hex(color_str)
                alpha = self._extract_alpha_from_color(color_str)
                
                if self.backend == 'plotly':
                    with self._suppress_output():
                        fig_mesh = navis.plot3d(roiunits[roi_i], backend='plotly', color=color_hex, alpha=alpha)
                    mesh_traces = fig_mesh.data
                    
                    roi_name = roi_names[roi_i]
                    _configure_roi_mesh_traces(
                        mesh_traces, roi_name,
                        legend_rank=ROI_MESH_LEGEND_RANK_BASE + roi_i * 100,
                    )
                    self.fig_3d.add_traces(mesh_traces)
                elif self.backend == 'k3d':
                    try:
                        with self._suppress_output():
                            temp_plot = navis.plot3d(roiunits[roi_i], backend='k3d', inline=False, color=color_hex, alpha=alpha)
                        for obj in temp_plot.objects:
                            obj.name = f'brain region [{roi_names[roi_i]}]'
                            self.fig_3d += obj
                    except Exception as e:
                        self._vprint(f'⚠️  k3d mesh plotting failed: {e}', level='full')
        elif has_roi_meshes:
            # Only warn if user specified ROI meshes but none loaded
            self._vprint('⚠️  No valid ROI meshes loaded', level='full')

        # Plot brain mesh (whole brain or template) regardless of ROI mesh status
        if self.brain_mesh in ['template', 'whole']:
            template_info = self._get_template_info()
            mesh_display_name = template_info['mesh_name']
            
            # For male-cns with brain_mesh='template', use brain-only mesh if vnc_mesh is also True
            # This allows independent show/hide of brain and VNC in the interactive HTML
            # If vnc_mesh=False, show the full CNS mesh
            dataset_lower = self.dataset.lower()
            is_male_cns = 'male-cns' in dataset_lower or 'malecns' in dataset_lower
            # Note: 'manc' is NOT treated as male-cns here because its native template is VNC-only,
            # so we don't need to extract a "brain" portion from it.
            
            use_brain_only = is_male_cns and self.brain_mesh == 'template' and self.vnc_mesh
            
            if use_brain_only:
                mesh_display_name = 'JRCFIB2022M (brain)'
            
            self._vprint(f'Plotting {mesh_display_name} mesh...', level='full')
            try:
                import flybrains
                import trimesh
                
                # Select appropriate mesh
                if use_brain_only:
                    # For male-cns with vnc_mesh=True, extract brain-only portion
                    # flybrains doesn't provide mesh_brain, so we clip the mesh geometrically
                    # Brain is in the anterior portion (lower Z values in JRCFIB2022M coordinates)
                    # Z cutoff ~340000 separates brain from VNC based on geometry
                    full_mesh = flybrains.JRCFIB2022M.mesh
                    brain_z_cutoff = 340000  # Z coordinate separating brain from VNC
                    
                    # Get vertices in the brain region (Z < cutoff)
                    brain_mask = full_mesh.vertices[:, 2] < brain_z_cutoff
                    
                    # Extract submesh by filtering faces that have all vertices in brain region
                    brain_faces = []
                    for face in full_mesh.faces:
                        if all(brain_mask[v] for v in face):
                            brain_faces.append(face)
                    
                    if brain_faces:
                        # Create new mesh from brain vertices only
                        brain_mesh_trimesh = trimesh.Trimesh(
                            vertices=full_mesh.vertices,
                            faces=brain_faces
                        )
                        # Clean up unused vertices
                        brain_mesh_trimesh.remove_unreferenced_vertices()
                        brain_mesh = navis.Volume(brain_mesh_trimesh, name='JRCFIB2022M_brain')
                        self._vprint(f'   Extracted brain mesh: {len(brain_mesh_trimesh.vertices)} vertices', level='full')
                    else:
                        # Fallback to full mesh if extraction fails
                        brain_mesh = full_mesh
                        self._vprint('   ⚠️  Brain mesh extraction failed, using full CNS mesh', level='full')
                elif is_male_cns and hasattr(flybrains.JRCFIB2022M, 'mesh_brain'):
                    brain_mesh = flybrains.JRCFIB2022M.mesh_brain
                else:
                    brain_mesh = template_info['template_obj'].mesh if hasattr(template_info['template_obj'], 'mesh') else template_info['template_obj']
                
                # Apply FAFB tilt correction if using template mode
                # This corrects the left-right tilt in the FLYWIRE template mesh
                is_fafb = 'flywire' in self.dataset.lower() or 'fafb' in self.dataset.lower()
                if is_fafb and self.brain_mesh == 'template':
                    brain_mesh = self._apply_fafb_tilt_correction(brain_mesh)
                
                effective_brain_color = self._get_effective_mesh_color('brain')
                if self.backend == 'plotly':
                    with self._suppress_output():
                        fig_brain = navis.plot3d(brain_mesh, backend='plotly')
                    brain_traces = fig_brain.data
                    for trace in brain_traces:
                        trace.showlegend = True
                        trace.name = mesh_display_name
                        trace.hoverinfo = 'none'
                        trace.legendrank = BRAIN_MESH_LEGEND_RANK
                        self._apply_plotly_trace_color(trace, effective_brain_color)
                    self.fig_3d.add_traces(brain_traces)
                elif self.backend == 'k3d':
                    with self._suppress_output():
                        temp_plot = navis.plot3d(brain_mesh, backend='k3d', inline=False)
                    for obj in temp_plot.objects:
                        obj.name = mesh_display_name
                        self._apply_k3d_object_color(obj, effective_brain_color)
                        self.fig_3d += obj

                self._append_exportable_mesh(
                    brain_mesh,
                    color=effective_brain_color,
                    name=mesh_display_name,
                    role='brain',
                )
                        
                self._vprint(f'✓ {mesh_display_name} mesh loaded successfully', level='full')
            except Exception as e:
                self._vprint(f'⚠️  Failed to load {mesh_display_name} mesh: {e}', level='full')
                if self._dataset_needs_transform() and not self._check_and_download_transforms():
                    self._vprint('   Skipping brain/VNC mesh visualization', level='full')
                else:
                    # Retry after download - use template object mesh
                    try:
                        retry_mesh = template_info['template_obj'].mesh if hasattr(template_info['template_obj'], 'mesh') else template_info['template_obj']
                        effective_brain_color = self._get_effective_mesh_color('brain')
                        if self.backend == 'plotly':
                            with self._suppress_output():
                                fig_brain = navis.plot3d(retry_mesh, backend='plotly')
                            brain_traces = fig_brain.data
                            for trace in brain_traces:
                                trace.showlegend = True
                                trace.name = mesh_display_name
                                trace.hoverinfo = 'none'
                                trace.legendrank = BRAIN_MESH_LEGEND_RANK
                                self._apply_plotly_trace_color(trace, effective_brain_color)
                            self.fig_3d.add_traces(brain_traces)
                        elif self.backend == 'k3d':
                            with self._suppress_output():
                                temp_plot = navis.plot3d(retry_mesh, backend='k3d', inline=False)
                            for obj in temp_plot.objects:
                                obj.name = mesh_display_name
                                self._apply_k3d_object_color(obj, effective_brain_color)
                                self.fig_3d += obj
                        self._append_exportable_mesh(
                            retry_mesh,
                            color=effective_brain_color,
                            name=mesh_display_name,
                            role='brain',
                        )
                        self._vprint(f'✓ {mesh_display_name} mesh loaded successfully after download', level='full')
                    except Exception as retry_e:
                        self._vprint(f'⚠️  Still failed to load {mesh_display_name} mesh: {retry_e}', level='full')
                        self._vprint('   Skipping brain/VNC mesh visualization', level='full')
        
        # Plot VNC mesh if requested (only for manc and male-cns datasets)
        if self.vnc_mesh:
            dataset_lower = self.dataset.lower()
            
            # Check if VNC is already shown by brain_mesh (for male-cns/manc datasets using JRCFIB2022M)
            is_male_cns = 'male-cns' in dataset_lower or 'malecns' in dataset_lower
            is_manc = 'manc' in dataset_lower
            
            # use_brain_only logic from above (replicated here for clarity)
            use_brain_only = is_male_cns and self.brain_mesh == 'template'
            
            # MANC with template mode: The "brain mesh" (template) IS the VNC mesh.
            if is_manc and self.brain_mesh == 'template':
                self._vprint('ℹ️  VNC mesh already shown as template', level='full')
            
            # If using JRCFIB2022M and NOT splitting brain (e.g. brain_mesh='whole'), VNC is already included
            elif (is_male_cns or is_manc) and self.brain_mesh in ['template', 'whole'] and not use_brain_only:
                self._vprint('ℹ️  VNC mesh already shown via brain_mesh', level='full')
            else:
                vnc_info = self._get_vnc_template_info()
                if vnc_info:
                    vnc_display_name = vnc_info['mesh_name']
                    self._vprint(f'Plotting {vnc_display_name} mesh...', level='full')
                    try:
                        vnc_mesh = vnc_info['mesh']
                        
                        effective_vnc_color = self._get_effective_mesh_color('vnc')
                        if self.backend == 'plotly':
                            with self._suppress_output():
                                fig_vnc = navis.plot3d(vnc_mesh, backend='plotly')
                            vnc_traces = fig_vnc.data
                            for trace in vnc_traces:
                                trace.showlegend = True
                                trace.name = vnc_display_name
                                trace.hoverinfo = 'none'
                                trace.legendrank = VNC_MESH_LEGEND_RANK
                                self._apply_plotly_trace_color(trace, effective_vnc_color)
                            self.fig_3d.add_traces(vnc_traces)
                        elif self.backend == 'k3d':
                            with self._suppress_output():
                                temp_plot = navis.plot3d(vnc_mesh, backend='k3d', inline=False)
                            for obj in temp_plot.objects:
                                obj.name = vnc_display_name
                                self._apply_k3d_object_color(obj, effective_vnc_color)
                                self.fig_3d += obj

                        self._append_exportable_mesh(
                            vnc_mesh,
                            color=effective_vnc_color,
                            name=vnc_display_name,
                            role='vnc',
                        )
                        
                        self._vprint(f'✓ {vnc_display_name} mesh loaded successfully', level='full')
                    except Exception as e:
                        self._vprint(f'⚠️  Failed to load VNC mesh: {e}', level='full')
                else:
                    self._vprint('⚠️  VNC mesh is only available for manc and male-cns datasets', level='full')
        
        self._vprint('Done', level='full')
        return 0
    
    def save_figure(self):
        if self.backend == 'plotly':
            # Scatter synapse/pre-post-site modes expose a size slider in the
            # interactive HTML; solid marker modes leave this empty.
            sliders = getattr(self, '_plotly_sliders', [])
            
            # set layout
            # Always use frontal view camera regardless of brain_mesh setting
            # This ensures consistent viewing angle for all visualizations
            # Standard fly brain orientation: X: Left-Right, Y: Dorsal-Ventral, Z: Anterior-Posterior
            # Frontal view: Look from Anterior (negative Z direction)
            
            scene_camera_parameters = dict(
                up=dict(x=0, y=-1, z=0),  # Y is up (inverted in some templates)
                eye=dict(x=0, y=0, z=-2.5),  # Look from front
                # center=dict(x=0, y=0, z=0), # Let Plotly auto-center
            )
            
            # Fix for MANC dataset (VNC)
            # Default Front (-Z) shows Tail. We want to look from Neck (+Z).
            if 'manc' in self.dataset.lower():
                 scene_camera_parameters = dict(
                    up=dict(x=0, y=-1, z=0),  # Dorsal (-Y) is up
                    eye=dict(x=0, y=0, z=2.5),  # Look from Anterior (+Z)
                )

            # Fix for hemibrain template mode (JRCFIB2018F)
            # JRCFIB2018F has Y-axis pointing posterior→anterior, Z is Dorsal-Ventral.
            # To show Frontal view, we need to look from Anterior (positive Y axis).
            if 'hemibrain' in self.dataset.lower() and self.brain_mesh == 'template':
                 scene_camera_parameters = dict(
                    up=dict(x=0, y=0, z=-1),  # Z is up (Dorsal)
                    eye=dict(x=0, y=2.0, z=0),  # Look from Front (Anterior is +Y)
                )
            
            self.fig_3d.update_layout(
                colorway = self.synapse_colors,
                sliders=sliders,
                paper_bgcolor=self.background_color,
                plot_bgcolor=self.background_color,
                scene=dict(
                    dragmode='orbit',
                    xaxis={'visible':False}, 
                    yaxis={'visible':False},
                    zaxis={'visible':False},
                    # Use 'data' aspectmode to ensure equal axis scaling
                    # This prevents distortion when no meshes are plotted
                    aspectmode='data',
                    bgcolor=self.background_color,
                ),
                scene_camera=scene_camera_parameters,
                # Legend settings: use constant sizing so alpha doesn't affect legend swatches
                legend=dict(
                    itemsizing='constant',  # Fixed legend swatch size regardless of trace properties
                    traceorder='normal',  # Respect the explicit legendrank sequence
                    font=dict(color='white' if self._is_dark_background() else 'black'),
                    bgcolor='rgba(0,0,0,0)',  # Transparent legend background
                ),
            )

            # save figure
            self.fig_path = os.path.join(self.save_folder,self.saveas)
            
            # Ensure save folder exists
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder, exist_ok=True)
            
            self._vprint(f'saving figure to \033[34m{self.fig_path}.html\033[0m...', end='')
            
            # Configure interactive controls based on interactive_html setting
            if self.interactive_html:
                # Add view selection dropdown menu to the figure
                self._add_view_selection_menu()
                
                # Interactive config: show toolbar, enable scroll zoom, customize buttons
                html_config = {
                    'displayModeBar': True,
                    'displaylogo': False,  # Hide Plotly logo
                    'scrollZoom': True,
                    'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'select2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': self.saveas,
                        'height': 1800,
                        'width': 2400,
                        'scale': 2
                    }
                }
            else:
                # Minimal config: hide toolbar for cleaner appearance
                html_config = {'displayModeBar': False}
            
            # Keep each HTML self-contained so it remains portable when copied
            # without the surrounding output directory.
            self._write_plotly_html(
                self.fig_3d,
                self.fig_path + '.html',
                theme_toggle=self.html_theme_toggle,
                legend_tree=(self.legend_mode == 'tree'),
                auto_open=False,
                include_plotlyjs=True,
                config=html_config,
            )
            
            if self.show_fig:
                try:
                    import webbrowser
                    webbrowser.open('file://' + os.path.abspath(self.fig_path+'.html'))
                except Exception as e:
                    self._vprint(f'\\n⚠️  Failed to open browser: {e}')
            
            self._vprint('Done (HTML saved)')
            
            # Export multiple view angles as PNG
            # Skip if export_views is False
            if self.export_views is not False:
                try:
                    self._vprint('   Exporting static PNGs (multiple views)...')
                    
                    # Create exported_views subfolder
                    views_folder = os.path.join(self.save_folder, 'exported_views')
                    os.makedirs(views_folder, exist_ok=True)
                    
                    # Define camera angles for different views
                    # Based on the default front view: eye=(0, 0, -2), up=(0, -1, 0)
                    # X: Left-Right, Y: Dorsal-Ventral (up), Z: Anterior-Posterior (front-back)
                    view_cameras = {
                        'front': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                        'back': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                        'top': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                        'bottom': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                        'left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                        'right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                    }

                    # Adjust for MANC (Male Adult Nerve Cord)
                    # User reports default Front view (-Z) shows Tail (Posterior).
                    # This implies Tail is at -Z (closest to camera), so Anterior is at +Z.
                    # Fix: Reverse Z axis for Front/Back views.
                    if 'manc' in self.dataset.lower():
                         view_cameras = {
                            'front': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                            'back': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                            'top': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                            'bottom': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                            'left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                            'right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                        }

                    # Adjust for hemibrain template (JRCFIB2018F)
                    # JRCFIB2018F coordinate system:
                    #   X-axis: Left-Right (as normal fly brain)
                    #   Y-axis: Posterior→Anterior (front is +Y direction)
                    #   Z-axis: Ventral→Dorsal (up is +Z direction, but we use -Z as "up" for standard brain viewing)
                    #
                    # For front view: eye at +Y, looking at origin, with -Z as up (dorsal up)
                    # Key insight: When looking from +Y, the up vector should be -Z to match standard viewing
                    if 'hemibrain' in self.dataset.lower() and self.brain_mesh == 'template':
                         view_cameras = {
                            'front': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                            'back': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                            'top': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
                            'bottom': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
                            'left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                            'right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                        }
                    
                    # Update layout for static export to remove UI elements
                    self.fig_3d.update_layout(
                        margin=dict(l=0, r=0, b=0, t=0),
                        sliders=[],      # Remove sliders
                        updatemenus=[],  # Remove view selection dropdown
                        annotations=[],  # Remove controls hint and view label
                        paper_bgcolor=self.background_color,
                        plot_bgcolor=self.background_color,
                        scene=dict(bgcolor=self.background_color),
                        legend=dict(
                            bgcolor=self.background_color,
                            font=dict(color='white' if self._is_dark_background() else 'black'),
                        ),
                    )
                    
                    import shutil
                    front_view_path = None
                    
                    # Determine which views to export
                    views_to_export = []
                    if self.export_views is True:
                        views_to_export = list(view_cameras.keys())
                    elif isinstance(self.export_views, list):
                        views_to_export = self.export_views
                    elif isinstance(self.export_views, str):
                        views_to_export = [self.export_views]
                    
                    if not views_to_export:
                        self._vprint('   Skipping PNG export (export_views=False)')
                    else:
                        # Check HTML size
                        html_path = self.fig_path + '.html'
                        html_size_mb = 0
                        if os.path.exists(html_path):
                            html_size_mb = os.path.getsize(html_path) / 1024 / 1024
                        
                        # Determine if we should pre-simplify based on export_simplified_png setting
                        # Default (False): Only simplify on timeout
                        # True: Simplify if > 50MB
                        # int: Simplify if > that many MB
                        simplify_threshold_mb = None
                        if self.export_simplified_png is True:
                            simplify_threshold_mb = 50
                        elif isinstance(self.export_simplified_png, (int, float)) and self.export_simplified_png > 0:
                            simplify_threshold_mb = self.export_simplified_png
                        
                        # Create export figure (may be simplified)
                        export_fig = self.fig_3d
                        using_simplified = False
                        html_size_cap = self._get_html_size_cap()
                        
                        # Auto-simplify for kaleido if HTML > size cap (prevents timeout)
                        if self.export_method not in ('webdriver', 'webdriver-fast') and html_size_mb > html_size_cap:
                            # Target 50% file size reduction
                            # HTML overhead (Plotly.js, layout, etc.) is ~20-30MB
                            # To get 50% total reduction, we need more aggressive mesh simplification
                            # Factor = (target_size - overhead) / (original_size - overhead)
                            html_overhead_mb = 25  # Estimated fixed overhead
                            target_size_mb = html_size_mb * 0.5  # 50% target
                            data_size = html_size_mb - html_overhead_mb
                            target_data_size = target_size_mb - html_overhead_mb
                            simplification_factor = max(0.1, target_data_size / data_size) if data_size > 0 else 0.5
                            
                            self._vprint(f'      ⚠️  HTML size ({html_size_mb:.0f}MB) exceeds {html_size_cap}MB - auto-simplifying for kaleido')
                            self._vprint(f'         Target: {target_size_mb:.0f}MB (50% reduction), simplification factor: {simplification_factor:.2f}')
                            export_fig = self._simplify_figure_for_kaleido(self.fig_3d, simplification_factor)
                            using_simplified = True
                            
                            # Save simplified HTML for reuse by export_video, plot_individuals, etc.
                            simplified_html_path = os.path.join(self.save_folder, f"{self.saveas}_simplified.html")
                            self._write_plotly_html(
                                export_fig,
                                simplified_html_path,
                                theme_toggle=self.html_theme_toggle,
                                legend_tree=(self.legend_mode == 'tree'),
                                auto_open=False,
                                include_plotlyjs=True,
                                config={'displayModeBar': False},
                            )
                            simplified_size_mb = os.path.getsize(simplified_html_path) / 1024 / 1024
                            self._vprint(f'      ✓ Saved simplified HTML: {os.path.basename(simplified_html_path)} ({simplified_size_mb:.0f}MB)')
                            
                            # Store for reuse
                            self._simplified_export_fig = export_fig
                            self._simplified_html_path = simplified_html_path
                            
                            self._vprint(f'      💡 For better quality, use export_method="webdriver"')
                        elif simplify_threshold_mb and html_size_mb > simplify_threshold_mb:
                            self._vprint(f'      ⚠️  HTML size: {html_size_mb:.1f}MB > {simplify_threshold_mb}MB - creating simplified copy for export')
                            export_fig = self._create_simplified_export_figure()
                            using_simplified = True
                        elif html_size_mb > 50:
                            self._vprint(f'      ℹ️  HTML size: {html_size_mb:.1f}MB - export may take 1-2 minutes')
                        
                        # Track if any export fails to skip remaining views
                        export_failed = False
                        exported_views = []
                        use_kaleido_fallback = False
                        
                        # Choose export method
                        if self.export_method in ('webdriver', 'webdriver-fast'):
                            # WebDriver method - OPTIMIZED: open browser once for all views
                            exported_views = self._export_views_with_webdriver_session(
                                export_fig, views_to_export, view_cameras, views_folder
                            )
                            export_failed = len(exported_views) == 0
                            if exported_views and 'front' in exported_views:
                                front_view_path = os.path.join(views_folder, f"{self.saveas}_front.png")
                            
                            # Fallback to kaleido if webdriver failed
                            if export_failed:
                                self._vprint(f'      ⚠️  WebDriver export failed. Falling back to kaleido...')
                                self._vprint(f'      💡 Tip: Set export_method="kaleido" for future use if WebDriver continues to fail.')
                                use_kaleido_fallback = True
                                export_failed = False  # Reset to try kaleido
                        
                        if self.export_method == 'kaleido' or use_kaleido_fallback:
                            # Default: kaleido export with timeout (per-view)
                            for view_name in views_to_export:
                                if export_failed:
                                    self._vprint(f'      ⚠️  Skipping {view_name} (previous export failed)')
                                    continue
                                    
                                if view_name not in view_cameras:
                                    self._vprint(f'      ⚠️  Skipping invalid view: {view_name}')
                                    continue
                                    
                                camera = view_cameras[view_name]
                                view_path = os.path.join(views_folder, f"{self.saveas}_{view_name}.png")
                                export_fig.update_layout(scene_camera=camera)
                                
                                success, msg, final_scale = self._export_png_with_timeout(
                                    export_fig, view_path, 
                                    width=1200, height=900, 
                                    scale=self.export_scale, 
                                    timeout=300
                                )
                                
                                if success:
                                    self._vprint(f'      {view_name}: {msg}', level='full')
                                    exported_views.append(view_name)
                                    
                                    # Save front view path for copying to root
                                    if view_name == 'front':
                                        front_view_path = view_path
                                else:
                                    self._vprint(f'      ⚠️  {view_name} export failed: {msg}')
                                    export_failed = True
                                    self._vprint(f'      💡 Skipping remaining views. Try reducing export_scale or increasing skeleton_mesh_simplification.')
                        
                        # Copy front view to root folder without '_front' suffix
                        if front_view_path and os.path.exists(front_view_path):
                            root_png_path = os.path.join(self.save_folder, f"{self.saveas}.png")
                            shutil.copy2(front_view_path, root_png_path)
                            self._vprint(f'   ✓ Copied front view to root: {self.saveas}.png')
                        
                        if exported_views:
                            self._vprint(f'   ✓ Exported {len(exported_views)} view PNGs to exported_views/ ({", ".join(exported_views)})')
                        
                except Exception as e:
                    self._vprint(f'\\n   ⚠️  PNG/SVG export failed: {e}. Continuing without static images...')
            
        elif self.backend == 'k3d':
            self.fig_path = os.path.join(self.save_folder,self.saveas)
            self._vprint(f'saving figure to \033[34m{self.fig_path}.html\033[0m...', end='')
            
            try:
                from ipywidgets.embed import embed_minimal_html
                embed_minimal_html(
                    self.fig_path+'.html', 
                    views=[self.fig_3d], 
                    title=self.saveas
                )
                self._record_large_html_warning(self.fig_path + '.html')
                self._vprint('Done')
                
                if self.show_fig:
                    self._vprint('Note: k3d plots cannot be automatically opened from script. Please open the HTML file manually.')
                    
            except ImportError:
                self._vprint('\\n⚠️  ipywidgets not installed. Cannot save k3d plot to HTML.')
                self._vprint('   Please install it with `pip install ipywidgets`')
            except Exception as e:
                self._vprint(f'\\n⚠️  Failed to save k3d plot: {e}')
    
    @staticmethod
    def _resolve_plotly_trace_identities(neuron_vols, fig_traces):
        """Map each navis-generated trace to its owning neuron.

        navis names skeleton and connector traces after their neuron but
        leaves companion meshes (soma spheres) unnamed. A positional
        fallback counts traces rather than neurons, so once a companion
        mesh appears every later identity slips onto a neighboring bodyId
        and the soma ends up with a different legend label than its own
        skeleton. Match names against the plotted neuron identities first;
        otherwise give named traces their legacy positional entry and let
        unnamed companions inherit whatever named trace immediately
        precedes them. Leading orphan meshes fall back to pure position.

        Returns a list of ``(neuron_id, source_index)`` aligned with
        ``fig_traces``.
        """
        identity_keys = {}
        for pos, vol in enumerate(neuron_vols or []):
            for key in (getattr(vol, 'id', None), getattr(vol, 'name', None)):
                if key is not None:
                    identity_keys.setdefault(str(key), pos)

        fig_traces = list(fig_traces)

        resolved = []
        # Identity of the most recent named trace. navis emits each neuron's
        # skeleton first, then its unnamed companions (soma meshes), so an
        # unnamed trace belongs to whatever named trace precedes it -- even
        # when that name did not map back to a plotted neuron identity.
        last_named_identity = None
        for j, trace in enumerate(fig_traces):
            existing_name = getattr(trace, 'name', None)
            pos = None
            if existing_name:
                pos = identity_keys.get(str(existing_name))
                identity = (
                    str(existing_name), pos if pos is not None else j,
                )
                resolved.append(identity)
                last_named_identity = identity
            elif last_named_identity is not None:
                resolved.append(last_named_identity)
            elif j < len(neuron_vols or []):
                resolved.append((
                    str(getattr(neuron_vols[j], 'id', f'neuron_{j}')), j,
                ))
            else:
                resolved.append((f'neuron_{j}', j))

        return resolved

    @staticmethod
    def _dedupe_profile_name(safe_name, taken):
        """Suffix a sanitized profile stem until it does not collide.

        Distinct legend keys can sanitize to the same filename; without
        this the later export silently overwrites the earlier one.
        """
        base = safe_name or 'individual'
        candidate = base
        suffix = 2
        while candidate in taken:
            candidate = f'{base}_{suffix}'
            suffix += 1
        taken.add(candidate)
        return candidate

    def plot_neurons(self):
        import time
        start_time = time.time()
        
        # Reset simplified figure and HTML path from previous export
        self._simplified_export_fig = None
        self._simplified_html_path = None
        
        self._vprint('\n' + '='*60)
        self._vprint(f'🧠 VisualizeSkeleton: Plotting {self.dataset}')
        self._vprint('='*60)

        # Structured step events (web UI only): the total covers this method's
        # three phases plus the optional export phases that follow it in the
        # generated script.  progress_total is None for direct callers, which
        # keeps the previous quiet behavior.
        total = self.progress_total
        if total:
            self._progress(1, total, 'Loading skeletons')
        self.plot_skeleton()
        if total:
            self._progress(
                2, total,
                'Loading meshes' if self.skip_synapse else 'Loading synapses and meshes',
            )
        self.plot_synapses()
        self.plot_mesh()
        if total:
            self._progress(3, total, 'Rendering and saving the 3D scene')
        self.save_figure()
        
        elapsed = time.time() - start_time
        self._vprint(f'\n✅ Complete! Total time: {elapsed:.1f}s')
        self._vprint(f'📁 Output: {self.save_folder}')
        self._vprint('='*60 + '\n')

    def plot_individuals(
        self,
        output_format: str | list = 'png',
        views: str | list = 'front',
        scale: int = None,
        pdf_images_per_page: tuple = (3, 2),
        pdf_title: str = None,
        neuron_alpha: float = None,
        summary_format: str | list = 'pdf',
        auto_crop: bool = True,
        crop_margin: int = 30,
        export_method: str = None,
    ):
        """
        Plot individual neurons/types independently based on the main figure's legend entries.
        
        This method should be called AFTER plot_neurons() to ensure all necessary data is available.
        It iterates through the legend entries in the main figure and generates separate plots
        for each individual legend item by hiding other neuron traces (efficient, no duplication).
        
        Behavior varies by legend_mode:
        - 'single': plots individual neurons (each neuron separate)
        - 'type': plots by neuron type (grouped by type within layers)
        - 'layer': plots by layer (all neurons in a layer grouped)
        
        Parameters
        ----------
        output_format : str or list, default 'png'
            Output format(s) for individual plots.
            Options: 'png', 'html', or list like ['png', 'html']
        views : str or list, default 'front'
            View angle(s) for PNG exports.
            Options: 'front', 'back', 'top', 'bottom', 'left', 'right'
            Can be a single string or list like ['front', 'top']
        scale : int, default None (uses self.export_scale)
            Scale factor for PNG export resolution. Overrides self.export_scale if provided.
            Higher values produce larger, higher-quality images.
        pdf_images_per_page : tuple, default (3, 2)
            (columns, rows) - number of images per page when generating PDF/PPTX.
        pdf_title : str, optional
            Custom title for PDF/PPTX pages. If None, uses the layer/neuron name.
        neuron_alpha : float, optional
            Opacity for neuron traces in individual plots (0.0-1.0).
            If None, defaults to 0.8 for better visibility in individual views.
        summary_format : str or list, default 'pdf'
            Format(s) for summary file generation.
            Options: 'pdf', 'pptx', or list like ['pdf', 'pptx']
        auto_crop : bool, default True
            If True, automatically crop whitespace/background from PNG exports
            to minimize empty margins around content.
        crop_margin : int, default 30
            Margin (in pixels) to preserve around content when auto_crop=True.
        export_method : str, optional
            Export method for PNG generation. Options: 'webdriver', 'kaleido'.
            If None, uses self.export_method (default from class initialization).
            - 'webdriver': Uses Selenium Chrome for rendering (better for large figures)
            - 'kaleido': Uses kaleido for rendering (faster for small figures)
            
        Returns
        -------
        str or None
            Path to the output folder containing individual plots,
            or None if no plots were generated.
            
        Example
        -------
        >>> vs = VisualizeSkeleton(...)
        >>> vs.plot_neurons()
        >>> vs.plot_individuals(output_format=['png', 'html'], views=['front', 'top'])
        >>> vs.plot_individuals(summary_format=['pdf', 'pptx'])  # Generate both PDF and PPTX
        >>> vs.plot_individuals(export_method='webdriver')  # Force webdriver for large figures
        """
        import copy
        
        if not hasattr(self, 'fig_3d') or self.fig_3d is None:
            self._vprint('⚠️  No figure found. Please run plot_neurons() first.')
            return None
            
        if self.backend != 'plotly':
            self._vprint('⚠️  plot_individuals() only supports plotly backend.')
            return None
        
        # Use specified export_method or fall back to self.export_method
        actual_export_method = export_method if export_method is not None else self.export_method
        
        # Use self.export_scale if scale not specified
        # For webdriver modes, cap inherited scale at 5 for stability (user can override explicitly)
        if scale is None:
            scale = self.export_scale
            if actual_export_method in ('webdriver', 'webdriver-fast') and scale > 5:
                scale = 5
                self._vprint(f'   ℹ️  Capped inherited scale to 5 for webdriver mode (explicit scale= overrides)')
        
        # Normalize inputs
        if isinstance(output_format, str):
            output_format = [output_format]
        if isinstance(views, str):
            views = [views]
            
        # Validate inputs
        valid_formats = {'png', 'html'}
        output_format = [f.lower() for f in output_format]
        for fmt in output_format:
            if fmt not in valid_formats:
                self._vprint(f'⚠️  Invalid output format: {fmt}. Use "png" or "html".')
                return None
                
        valid_views = {'front', 'back', 'top', 'bottom', 'left', 'right'}
        views = [v.lower() for v in views]
        for view in views:
            if view not in valid_views:
                self._vprint(f'⚠️  Invalid view: {view}. Use one of {valid_views}.')
                return None
        
        # Create output directory
        output_dir = os.path.join(self.save_folder, 'individual_profiles')
        os.makedirs(output_dir, exist_ok=True)
        
        self._vprint(f'\n📊 Generating individual plots...')
        self._vprint(f'   Output: {output_dir}')
        
        # View cameras for PNG export
        view_cameras = {
            'front': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            'back': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            'top': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
            'bottom': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
            'left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            'right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
        }
        
        # Adjust for MANC (Male Adult Nerve Cord)
        # Fix Front view (-Z shows Tail) -> Should look from +Z (Neck)
        if 'manc' in self.dataset.lower():
             view_cameras = {
                'front': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                'back': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                'top': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                'bottom': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
                'left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
                'right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            }

        # Adjust for hemibrain template (JRCFIB2018F) - front/back swapped due to Y-axis orientation
        # JRCFIB2018F coordinate system:
        #   X-axis: Left-Right (as normal fly brain)
        #   Y-axis: Posterior→Anterior (front is +Y direction)
        #   Z-axis: Ventral→Dorsal (up is +Z direction, but we use -Z as "up" for standard brain viewing)
        if 'hemibrain' in self.dataset.lower() and self.brain_mesh == 'template':
            view_cameras = {
                'front': dict(eye=dict(x=0, y=2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                'back': dict(eye=dict(x=0, y=-2.5, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                'top': dict(eye=dict(x=0, y=0, z=-2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
                'bottom': dict(eye=dict(x=0, y=0, z=2.5), center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
                'left': dict(eye=dict(x=-2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
                'right': dict(eye=dict(x=2.5, y=0, z=0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=-1)),
            }
        
        # Get all traces from the main figure
        all_traces = list(self.fig_3d.data)
        n_traces = len(all_traces)
        
        # Individual exports inherit the application-wide neuron opacity unless
        # the caller explicitly requests another value.
        individual_alpha = neuron_alpha if neuron_alpha is not None else 0.2
        
        # Helper function to modify alpha after decoding any supported color
        # format. Lists/arrays of per-point colors are handled element-wise;
        # a 3/4-channel numeric sequence remains one color.
        def _modify_color_alpha(color_str, new_alpha):
            """Apply a new alpha after decoding any supported color format."""
            if color_str is None:
                return None
            if hasattr(color_str, 'tolist') and not isinstance(color_str, (str, tuple, list)):
                color_str = color_str.tolist()
            if isinstance(color_str, (list, tuple)):
                try:
                    return color_to_rgba_string(color_str, alpha=new_alpha)
                except (TypeError, ValueError):
                    return [
                        _modify_color_alpha(item, new_alpha)
                        for item in color_str
                    ]
            try:
                return color_to_rgba_string(color_str, alpha=new_alpha)
            except (TypeError, ValueError):
                return str(color_str)

        def _set_trace_color_alpha(trace, new_alpha):
            """Apply an export alpha to every Plotly color-bearing slot."""
            for target in (
                trace,
                getattr(trace, 'marker', None),
                getattr(trace, 'line', None),
            ):
                if target is None or not hasattr(target, 'color'):
                    continue
                current = getattr(target, 'color', None)
                if current is not None:
                    target.color = _modify_color_alpha(current, new_alpha)
        
        # Store original visibility, opacity, and color states to restore later
        original_visibility = []
        original_opacity = []
        original_colors = []
        original_nested_colors = []
        for trace in all_traces:
            original_visibility.append(getattr(trace, 'visible', True))
            original_opacity.append(getattr(trace, 'opacity', None))
            original_colors.append(getattr(trace, 'color', None))
            original_nested_colors.append({
                'marker': getattr(getattr(trace, 'marker', None), 'color', None),
                'line': getattr(getattr(trace, 'line', None), 'color', None),
            })
        
        # Store original camera and layout settings
        original_layout = copy.deepcopy(self.fig_3d.layout)
        
        # Identify unique legend entries (excluding hidden legends and mesh/synapse traces)
        legend_entries = {}  # {legend_name: [trace_indices]}
        background_indices = []  # mesh/synapse traces to always show
        pre_post_site_entries = []  # [(trace_index, owning_skeleton_legend)]
        
        # Get mesh_roi names for matching
        mesh_roi_names = [r.lower() for r in self.mesh_roi] if self.mesh_roi else []
        
        # Template/mesh names to always include as background
        mesh_keywords = ['mesh', 'brain region', 'template', 'vnc']
        template_names = ['JRCFIB', 'MANC', 'JRC2018', 'FLYWIRE', 'FAFB', 'jrcfib', 'flywire', 'fafb']
        
        for idx, trace in enumerate(all_traces):
            trace_name = getattr(trace, 'name', '')
            show_legend = getattr(trace, 'showlegend', True)
            legend_group = getattr(trace, 'legendgroup', None)
            trace_name_lower = trace_name.lower() if trace_name else ''
            
            # Identify mesh/roi traces (keep visible as background)
            # Include brain-region traces, standard templates, and user-specified mesh_roi
            is_background = False
            
            if trace_name:
                # Check for mesh-related keywords
                if any(kw in trace_name_lower for kw in mesh_keywords):
                    is_background = True
                # Check for template names (case-insensitive for most, case-sensitive for acronyms)
                elif any(tn in trace_name for tn in template_names):
                    is_background = True
                # Check for user-specified ROI meshes
                elif any(roi_name in trace_name_lower for roi_name in mesh_roi_names):
                    is_background = True
            
            if is_background:
                background_indices.append(idx)
                continue

            # Pre/post site traces intentionally use a distinct legend group
            # (``pre_post:{role}:{owner}``) so their pre/post entries remain
            # separate in the main figure.  For individual profiles they
            # belong with the owning morphology trace, rather than being
            # treated as global synapse background or dropped as hidden
            # legend entries.
            site_group = str(legend_group or '')
            if site_group.startswith('pre_post:'):
                parts = site_group.split(':', 2)
                owner = parts[2] if len(parts) == 3 else ''
                pre_post_site_entries.append((idx, owner))
                continue
                
            # Identify synapse traces (keep visible as background)
            if trace_name and ('synapse' in trace_name_lower or 'pre-syn' in trace_name_lower or 'post-syn' in trace_name_lower):
                background_indices.append(idx)
                continue
            
            # Use legendgroup as key if available (for merged traces), else use name
            key = legend_group if legend_group else trace_name
            if key and show_legend:
                if key not in legend_entries:
                    legend_entries[key] = []
                legend_entries[key].append(idx)
            elif key and not show_legend and legend_group:
                # Traces with same legendgroup but showlegend=False
                if key not in legend_entries:
                    legend_entries[key] = []
                legend_entries[key].append(idx)

        # Attach each pre/post site trace to the same profile as its owner.
        # Exact matching is expected because the site renderer resolves its
        # owner identity from ``_neuron_legend_labels_by_layer``.  Keep an
        # unmatched site visible as background as a defensive fallback for a
        # partially rendered skeleton, rather than silently omitting it from
        # every exported profile.
        for trace_index, owner in pre_post_site_entries:
            if owner in legend_entries:
                legend_entries[owner].append(trace_index)
            else:
                background_indices.append(trace_index)
        
        if not legend_entries:
            self._vprint('⚠️  No legend entries found to plot individually.')
            return None
        
        self._vprint(f'   Found {len(legend_entries)} individual legend entries')
        
        # Generate individual plots by hiding/showing traces
        generated_files = {'png': {}, 'html': []}
        
        # Track if PNG export fails to skip remaining individuals
        png_export_failed = False
        
        # Helper to sanitize filenames
        def _sanitize_filename(name):
            safe_name = "".join(c if c.isalnum() or c in '.+_- ' else '_' for c in str(name))
            safe_name = safe_name.strip().replace(' ', '_')
            while '__' in safe_name:
                safe_name = safe_name.replace('__', '_')
            return safe_name.rstrip('_')

        # Distinct legend keys can sanitize to the same filename; keep every
        # exported profile instead of letting the later one overwrite the
        # earlier one.
        taken_profile_names = set()
        
        from tqdm import tqdm
        legend_names = list(legend_entries.keys())
        
        # ==============================================================================
        # OPTIMIZED WEBDRIVER EXPORT: Open browser once, toggle visibility via JavaScript
        # ==============================================================================
        if actual_export_method in ('webdriver', 'webdriver-fast') and 'png' in output_format:
            self._vprint(f'   Using optimized WebDriver export (single browser session)')
            
            # Use simplified HTML if available from previous export (e.g., export_views)
            simplified_html = getattr(self, '_simplified_html_path', None)
            if simplified_html and os.path.exists(simplified_html):
                self._vprint(f'   ✓ Reusing simplified HTML from previous export')
                temp_html = simplified_html
            else:
                # Use simplified figure if available, otherwise use original
                export_fig = getattr(self, '_simplified_export_fig', None) or self.fig_3d
                if export_fig is not self.fig_3d:
                    self._vprint(f'   ✓ Reusing simplified figure from previous export')
                
                # Save main HTML with all traces visible for WebDriver
                temp_html = os.path.join(output_dir, '_temp_main_figure.html')
                self._write_plotly_html(
                    export_fig,
                    temp_html,
                    auto_open=False,
                    include_plotlyjs=True,
                    config={'displayModeBar': False},
                )
            
            # Use the optimized helper function
            result = export_individuals_webdriver(
                html_path=temp_html,
                output_dir=output_dir,
                legend_entries=legend_entries,
                background_indices=background_indices,
                total_traces=n_traces,
                views=views,
                view_cameras=view_cameras,
                scale=scale,
                width=900,
                height=900,
                timeout=300,
                verbose=True,
                auto_crop=auto_crop,
                crop_margin=crop_margin,
                background_color=self.background_color
            )
            
            # Clean up temp HTML (but not the permanent simplified HTML)
            if temp_html and not temp_html.endswith('_simplified.html'):
                try:
                    os.remove(temp_html)
                except:
                    pass
            
            if result['success']:
                generated_files['png'] = result['files']
            else:
                png_export_failed = True
                self._vprint(f'   ⚠️  WebDriver export failed: {result["error"]}')
            
            # Export HTML files separately if requested (in traditional loop)
            if 'html' in output_format:
                for legend_name in tqdm(legend_names, desc='Exporting HTML files'):
                    trace_indices = legend_entries[legend_name]
                    safe_name = self._dedupe_profile_name(
                        _sanitize_filename(legend_name), taken_profile_names,
                    )
                    
                    # Set visibility for this individual
                    for idx in range(n_traces):
                        if idx in trace_indices or idx in background_indices:
                            self.fig_3d.data[idx].visible = True
                            if idx in trace_indices:
                                trace = self.fig_3d.data[idx]
                                _set_trace_color_alpha(trace, individual_alpha)
                                if hasattr(trace, 'opacity'):
                                    trace.opacity = individual_alpha
                        else:
                            self.fig_3d.data[idx].visible = False
                    
                    self.fig_3d.update_layout(
                        title=dict(text='', x=0.5),
                        margin=dict(l=0, r=0, b=0, t=0),
                        showlegend=False,
                        updatemenus=[],  # Remove view selection dropdown
                        annotations=[],  # Remove controls hint
                        paper_bgcolor=self.background_color,
                        plot_bgcolor=self.background_color,
                        scene=dict(bgcolor=self.background_color),
                    )
                    
                    html_filename = f'{safe_name}.html'
                    html_path = os.path.join(output_dir, html_filename)
                    self._write_plotly_html(
                        self.fig_3d,
                        html_path,
                        include_plotlyjs=True,
                        full_html=True,
                    )
                    generated_files['html'].append(html_path)
        
        # ==============================================================================
        # FALLBACK: Traditional loop for kaleido or HTML-only export
        # ==============================================================================
        else:
            for legend_name in tqdm(legend_names, desc='Plotting individuals'):
                # Skip if PNG export already failed
                if png_export_failed and 'png' in output_format and 'html' not in output_format:
                    continue
                
                trace_indices = legend_entries[legend_name]
                safe_name = self._dedupe_profile_name(
                    _sanitize_filename(legend_name), taken_profile_names,
                )
                
                # Hide all neuron traces, show only this legend's traces + background
                for idx in range(n_traces):
                    if idx in trace_indices or idx in background_indices:
                        self.fig_3d.data[idx].visible = True
                        if idx in trace_indices:
                            trace = self.fig_3d.data[idx]
                            _set_trace_color_alpha(trace, individual_alpha)
                            if hasattr(trace, 'opacity'):
                                trace.opacity = individual_alpha
                    else:
                        self.fig_3d.data[idx].visible = False
                
                self.fig_3d.update_layout(
                    title=dict(text='', x=0.5),
                    margin=dict(l=0, r=0, b=0, t=0),
                    sliders=[],
                    updatemenus=[],
                    annotations=[],  # Remove controls hint
                    showlegend=False,
                    paper_bgcolor=self.background_color,
                    plot_bgcolor=self.background_color,
                    scene=dict(
                        domain=dict(x=[0, 1], y=[0, 1]),  # Full viewport, no margin
                        bgcolor=self.background_color,
                    ),
                )
                
                # Export HTML if requested
                if 'html' in output_format:
                    html_filename = f'{safe_name}.html'
                    html_path = os.path.join(output_dir, html_filename)
                    self._write_plotly_html(
                        self.fig_3d,
                        html_path,
                        include_plotlyjs=True,
                        full_html=True,
                    )
                    generated_files['html'].append(html_path)
                
                # Export PNG(s) if requested (kaleido path)
                if 'png' in output_format and not png_export_failed:
                    if safe_name not in generated_files['png']:
                        generated_files['png'][safe_name] = []
                    
                    for view_name in views:
                        if png_export_failed:
                            break
                            
                        camera = view_cameras[view_name]
                        self.fig_3d.update_layout(scene_camera=camera)
                        
                        png_filename = f'{view_name}_{safe_name}.png'
                        png_path = os.path.join(output_dir, png_filename)
                        
                        success, msg, final_scale = self._export_png_with_timeout(
                            self.fig_3d, png_path, 
                            width=900, height=900, 
                            scale=scale, 
                            timeout=300,
                            auto_crop=auto_crop,
                            crop_margin=crop_margin
                        )
                        
                        if success:
                            generated_files['png'][safe_name].append((png_path, view_name))
                        else:
                            self._vprint(f'   ⚠️  PNG export failed for {legend_name} ({view_name}): {msg}')
                            png_export_failed = True
                            self._vprint(f'   💡 Skipping remaining exports. Try reducing scale or skeleton_mesh_simplification.')
                            break
        
        # Restore original figure state (visibility, opacity, and colors)
        for idx in range(n_traces):
            self.fig_3d.data[idx].visible = original_visibility[idx]
            # Restore opacity (even if None, to reset any changes)
            if hasattr(self.fig_3d.data[idx], 'opacity'):
                self.fig_3d.data[idx].opacity = original_opacity[idx]
            # Restore color (which contains alpha in RGBA format for navis traces)
            if hasattr(self.fig_3d.data[idx], 'color') and original_colors[idx] is not None:
                self.fig_3d.data[idx].color = original_colors[idx]
            for slot_name, original_color in original_nested_colors[idx].items():
                container = getattr(self.fig_3d.data[idx], slot_name, None)
                if container is not None and hasattr(container, 'color') and original_color is not None:
                    container.color = original_color
        
        # Restore original layout (includes resetting scene domain)
        self.fig_3d.update_layout(original_layout)
        
        # Normalize summary_format
        if summary_format is True:
            # Default to PDF when True
            summary_format = ['pdf']
        elif isinstance(summary_format, str):
            summary_format = [summary_format.lower()]
        elif summary_format:
            summary_format = [f.lower() for f in summary_format]
        else:
            # False or None - no summary
            summary_format = []
        
        # Generate PDF/PPTX summaries if PNG images were created
        if 'png' in output_format and generated_files['png']:
            # Save summaries in parent folder (parallel to individual_profiles/)
            parent_dir = os.path.dirname(output_dir)
            base_title = pdf_title or self.saveas
            
            # Generate PDF if requested
            if 'pdf' in summary_format:
                # For single view, generate one PDF without suffix (organized by view)
                # For multiple views, generate both _by_view and _by_name PDFs
                if len(views) == 1:
                    self._vprint(f'\n📄 Generating PDF summary...')
                    pdf_path = self._create_individual_pdf(
                        output_dir=parent_dir,
                        images_dict=generated_files['png'],
                        images_per_page=pdf_images_per_page,
                        title=base_title,
                        organize_by='view',  # Organize by view for single-view PDF
                        views=views,
                        pdf_suffix='',
                        background_color=self.background_color,
                    )
                    if pdf_path:
                        self._vprint(f'   ✅ PDF saved: {pdf_path}')
                else:
                    self._vprint(f'\n📄 Generating PDF summaries...')
                    # Generate PDF organized by view
                    pdf_path_view = self._create_individual_pdf(
                        output_dir=parent_dir,
                        images_dict=generated_files['png'],
                        images_per_page=pdf_images_per_page,
                        title=base_title,
                        organize_by='view',
                        views=views,
                        pdf_suffix='_by_view',
                        background_color=self.background_color,
                    )
                    if pdf_path_view:
                        self._vprint(f'   ✅ PDF saved: {pdf_path_view}')
                    
                    # Generate PDF organized by name
                    pdf_path_name = self._create_individual_pdf(
                        output_dir=parent_dir,
                        images_dict=generated_files['png'],
                        images_per_page=pdf_images_per_page,
                        title=base_title,
                        organize_by='name',
                        views=views,
                        pdf_suffix='_by_name',
                        background_color=self.background_color,
                    )
                    if pdf_path_name:
                        self._vprint(f'   ✅ PDF saved: {pdf_path_name}')
            
            # Generate PPTX if requested
            if 'pptx' in summary_format:
                if len(views) == 1:
                    self._vprint(f'\n📊 Generating PPTX summary...')
                    pptx_path = self._create_individual_pptx(
                        output_dir=parent_dir,
                        images_dict=generated_files['png'],
                        images_per_page=pdf_images_per_page,
                        title=base_title,
                        organize_by='view',
                        views=views,
                        pptx_suffix='',
                        background_color=self.background_color,
                    )
                    if pptx_path:
                        self._vprint(f'   ✅ PPTX saved: {pptx_path}')
                else:
                    self._vprint(f'\n📊 Generating PPTX summaries...')
                    # Generate PPTX organized by view
                    pptx_path_view = self._create_individual_pptx(
                        output_dir=parent_dir,
                        images_dict=generated_files['png'],
                        images_per_page=pdf_images_per_page,
                        title=base_title,
                        organize_by='view',
                        views=views,
                        pptx_suffix='_by_view',
                        background_color=self.background_color,
                    )
                    if pptx_path_view:
                        self._vprint(f'   ✅ PPTX saved: {pptx_path_view}')
                    
                    # Generate PPTX organized by name
                    pptx_path_name = self._create_individual_pptx(
                        output_dir=parent_dir,
                        images_dict=generated_files['png'],
                        images_per_page=pdf_images_per_page,
                        title=base_title,
                        organize_by='name',
                        views=views,
                        pptx_suffix='_by_name',
                        background_color=self.background_color,
                    )
                    if pptx_path_name:
                        self._vprint(f'   ✅ PPTX saved: {pptx_path_name}')
        
        # Summary
        n_png = sum(len(v) for v in generated_files['png'].values())
        n_html = len(generated_files['html'])
        self._vprint(f'\n✅ Individual plots complete!')
        self._vprint(f'   PNG files: {n_png}')
        self._vprint(f'   HTML files: {n_html}')
        self._vprint(f'   Output folder: {output_dir}')
        
        return output_dir

    def _create_individual_pdf(
        self,
        output_dir: str,
        images_dict: dict,
        images_per_page: tuple = (4, 3),
        title: str = None,
        organize_by: str = 'name',
        views: list = None,
        pdf_suffix: str = '',
        background_color: str = 'white',
    ) -> str | None:
        """
        Create a PDF summary from individual profile PNG images.
        
        Parameters
        ----------
        output_dir : str
            Directory where PDF will be saved.
        images_dict : dict
            Dictionary mapping legend names to list of (image_path, view_name) tuples.
            e.g., {'neuron1': [('/path/front.png', 'front'), ('/path/top.png', 'top')], ...}
        images_per_page : tuple
            (columns, rows) - number of images per page.
        title : str, optional
            Title for the PDF document.
        organize_by : str
            How images are organized: 'name' or 'view'
        views : list, optional
            List of view names for organizing by view
        pdf_suffix : str, optional
            Suffix to add to PDF filename (e.g., '_by_view', '_by_name')
            
        Returns
        -------
        str or None
            Path to created PDF, or None if creation failed.
        """
        try:
            from reportlab.lib.pagesizes import A4, landscape as rl_landscape
            from reportlab.lib.units import inch
            from reportlab.pdfgen import canvas
            from PIL import Image
        except ImportError:
            self._vprint('⚠️  PDF generation requires reportlab and Pillow.')
            self._vprint('   Install with: pip install reportlab Pillow')
            return None
        
        from pathlib import Path
        
        # Natural sort function for rank-based names like r1, r2, ..., r10, r11
        def natural_sort_key(s):
            """Sort strings containing numbers in natural order (r1, r2, ..., r10, r11)."""
            import re
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
        
        # Organize images based on organize_by option
        # Group by view when organize_by='view', otherwise by name
        if organize_by == 'view' and views:
            # Group images by view, each view gets its own section
            images_by_category = {}
            for view_name in views:
                images_by_category[view_name] = []
            
            for legend_name, img_info_list in sorted(images_dict.items(), key=lambda x: natural_sort_key(x[0])):
                for img_info in img_info_list:
                    if isinstance(img_info, tuple):
                        img_path, view_name = img_info
                    else:
                        img_path = img_info
                        view_name = views[0] if views else 'front'
                    if os.path.exists(img_path) and view_name in images_by_category:
                        images_by_category[view_name].append((legend_name, img_path, view_name))
        else:
            # Group images by name (default) with natural sorting
            images_by_category = {}
            for legend_name, img_info_list in sorted(images_dict.items(), key=lambda x: natural_sort_key(x[0])):
                if legend_name not in images_by_category:
                    images_by_category[legend_name] = []
                for img_info in img_info_list:
                    if isinstance(img_info, tuple):
                        img_path, view_name = img_info
                    else:
                        img_path = img_info
                        view_name = ''
                    if os.path.exists(img_path):
                        images_by_category[legend_name].append((legend_name, img_path, view_name))
        
        # Flatten but keep category boundaries for page breaks
        all_categories = list(images_by_category.keys())
        
        if not any(images_by_category.values()):
            self._vprint('⚠️  No images found for PDF generation.')
            return None
        
        # Output path
        pdf_path = os.path.join(output_dir, f'individual_profiles_summary{pdf_suffix}.pdf')
        
        # Page setup (landscape A4)
        page_width, page_height = rl_landscape(A4)
        cols, rows = images_per_page
        margin = 0.3 * inch  # Reduced from 0.5 to minimize blank space
        title_height = 20  # Reduced from 25 to minimize blank space
        
        # Determine if dark background for text color
        is_dark_bg = self._is_dark_background(background_color)
        text_color = (1, 1, 1) if is_dark_bg else (0, 0, 0)  # RGB 0-1 for reportlab
        
        # Parse background color for reportlab
        def parse_color_for_reportlab(color_str):
            """Convert any supported color to reportlab's RGB tuple."""
            try:
                r, g, b = extract_rgb_tuple(color_str)
                return (r / 255.0, g / 255.0, b / 255.0)
            except (TypeError, ValueError):
                return (1, 1, 1)
        
        bg_color = parse_color_for_reportlab(background_color)
        
        # Calculate cell dimensions
        usable_width = page_width - 2 * margin
        usable_height = page_height - 2 * margin - title_height
        cell_width = usable_width / cols
        cell_height = usable_height / rows
        
        # Create PDF
        c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
        
        images_per_full_page = cols * rows
        
        # Process each category separately (don't mix categories on same page)
        for category_name in all_categories:
            category_images = images_by_category[category_name]
            if not category_images:
                continue
            
            # Calculate pages needed for this category
            total_pages_for_category = (len(category_images) + images_per_full_page - 1) // images_per_full_page
            
            for page_idx in range(total_pages_for_category):
                # Draw background rectangle
                c.setFillColorRGB(*bg_color)
                c.rect(0, 0, page_width, page_height, fill=True, stroke=False)
                
                # Page title with category info
                c.setFillColorRGB(*text_color)
                c.setFont("Helvetica-Bold", 14)
                if organize_by == 'view':
                    # Use '{view} view' as title when organized by view
                    page_title = f"{category_name} view"
                else:
                    # Use layer_name as title when organized by name
                    page_title = str(category_name)
                if total_pages_for_category > 1:
                    page_title += f" ({page_idx + 1}/{total_pages_for_category})"
                c.drawCentredString(page_width / 2, page_height - margin - 5, page_title)
                
                # Get images for this page from the category
                start_idx = page_idx * images_per_full_page
                end_idx = min(start_idx + images_per_full_page, len(category_images))
                page_images = category_images[start_idx:end_idx]
                
                # Draw images
                for i, (legend_name, img_path, view_name) in enumerate(page_images):
                    row = i // cols
                    col = i % cols
                    
                    # Calculate position
                    x = margin + col * cell_width
                    y = page_height - margin - title_height - (row + 1) * cell_height
                    
                    try:
                        with Image.open(img_path) as img:
                            img_width, img_height = img.size
                            
                            # Calculate scaling - minimize padding between images
                            padding = 0  # Reduced from 5 to minimize blank space
                            label_height = 12  # Reduced from 15 to minimize blank space
                            max_width = cell_width - 2 * padding
                            max_height = cell_height - 2 * padding - label_height
                            
                            scale_w = max_width / img_width
                            scale_h = max_height / img_height
                            scale_factor = min(scale_w, scale_h)
                            
                            draw_width = img_width * scale_factor
                            draw_height = img_height * scale_factor
                            
                            # Center horizontally in cell, leave space at top for label
                            draw_x = x + (cell_width - draw_width) / 2
                            draw_y = y + (cell_height - label_height - draw_height) / 2
                            
                            # Draw image
                            c.drawImage(
                                img_path,
                                draw_x, draw_y,
                                width=draw_width,
                                height=draw_height,
                                preserveAspectRatio=True
                            )
                            
                            # Draw label on TOP of image
                            c.setFillColorRGB(*text_color)
                            c.setFont("Helvetica", 12)
                            label = str(legend_name)
                            if view_name and organize_by != 'view':
                                # Only add view suffix if not organizing by view
                                label += f" ({view_name})"
                            if len(label) > 30:
                                label = label[:27] + '...'
                            label_y = y + cell_height - label_height + 2  # Reduced from 3
                            c.drawCentredString(x + cell_width / 2, label_y, label)
                            
                    except Exception as e:
                        self._vprint(f'   ⚠️  Could not process: {img_path} - {e}', level='full')
                
                c.showPage()
        
        c.save()
        return pdf_path

    def _create_individual_pptx(
        self,
        output_dir: str,
        images_dict: dict,
        images_per_page: tuple = (4, 3),
        title: str = None,
        organize_by: str = 'name',
        views: list = None,
        pptx_suffix: str = '',
        label_fontsize: int = 20,
        title_fontsize: int = 24,
        background_color: str = 'white',
    ) -> str | None:
        """
        Create a PPTX summary from individual profile PNG images.
        
        Parameters
        ----------
        output_dir : str
            Directory where PPTX will be saved.
        images_dict : dict
            Dictionary mapping legend names to list of (image_path, view_name) tuples.
            e.g., {'neuron1': [('/path/front.png', 'front'), ('/path/top.png', 'top')], ...}
        images_per_page : tuple
            (columns, rows) - number of images per slide.
        title : str, optional
            Title for the PPTX document.
        organize_by : str
            How images are organized: 'name' or 'view'
        views : list, optional
            List of view names for organizing by view
        pptx_suffix : str, optional
            Suffix to add to PPTX filename (e.g., '_by_view', '_by_name')
        label_fontsize : int, default 20
            Font size for image labels in points.
        title_fontsize : int, default 24
            Font size for slide titles in points.
        background_color : str, default 'white'
            Background color for slides. Supports 'black', 'white', hex codes, etc.
            
        Returns
        -------
        str or None
            Path to created PPTX, or None if creation failed.
        """
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
            from pptx.enum.text import PP_ALIGN
            from pptx.dml.color import RGBColor
        except ImportError:
            self._vprint('⚠️  PPTX generation requires python-pptx.')
            self._vprint('   Install with: pip install python-pptx')
            return None
        
        from PIL import Image
        
        # Natural sort function for rank-based names like r1, r2, ..., r10, r11
        def natural_sort_key(s):
            """Sort strings containing numbers in natural order (r1, r2, ..., r10, r11)."""
            import re
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]
        
        # Organize images based on organize_by option
        if organize_by == 'view' and views:
            images_by_category = {}
            for view_name in views:
                images_by_category[view_name] = []
            
            for legend_name, img_info_list in sorted(images_dict.items(), key=lambda x: natural_sort_key(x[0])):
                for img_info in img_info_list:
                    if isinstance(img_info, tuple):
                        img_path, view_name = img_info
                    else:
                        img_path = img_info
                        view_name = views[0] if views else 'front'
                    if os.path.exists(img_path) and view_name in images_by_category:
                        images_by_category[view_name].append((legend_name, img_path, view_name))
        else:
            images_by_category = {}
            for legend_name, img_info_list in sorted(images_dict.items(), key=lambda x: natural_sort_key(x[0])):
                if legend_name not in images_by_category:
                    images_by_category[legend_name] = []
                for img_info in img_info_list:
                    if isinstance(img_info, tuple):
                        img_path, view_name = img_info
                    else:
                        img_path = img_info
                        view_name = ''
                    if os.path.exists(img_path):
                        images_by_category[legend_name].append((legend_name, img_path, view_name))
        
        all_categories = list(images_by_category.keys())
        
        if not any(images_by_category.values()):
            self._vprint('⚠️  No images found for PPTX generation.')
            return None
        
        # Output path
        pptx_path = os.path.join(output_dir, f'individual_profiles_summary{pptx_suffix}.pptx')
        
        # Determine if dark background for text color
        is_dark_bg = self._is_dark_background(background_color)
        font_color = RGBColor(255, 255, 255) if is_dark_bg else RGBColor(0, 0, 0)
        
        # Parse background color for pptx
        def parse_color_for_pptx(color_str):
            """Convert any supported color to a PowerPoint RGBColor."""
            try:
                r, g, b = extract_rgb_tuple(color_str)
                return RGBColor(r, g, b)
            except (TypeError, ValueError):
                return RGBColor(255, 255, 255)
        
        bg_color = parse_color_for_pptx(background_color)
        
        # Slide setup (widescreen 16:9)
        slide_width, slide_height = 13.333, 7.5
        cols, rows = images_per_page
        margin = 0.3  # inches
        title_height_inches = 0.5
        label_height_inches = (label_fontsize / 72) * 1.5
        
        # Create presentation
        prs = Presentation()
        prs.slide_width = Inches(slide_width)
        prs.slide_height = Inches(slide_height)
        blank_layout = prs.slide_layouts[6]  # Blank slide
        
        # Calculate cell dimensions
        usable_width = slide_width - 2 * margin
        usable_height = slide_height - margin - title_height_inches - margin
        cell_width = usable_width / cols
        cell_height = usable_height / rows
        
        images_per_full_page = cols * rows
        
        # Process each category separately
        for category_name in all_categories:
            category_images = images_by_category[category_name]
            if not category_images:
                continue
            
            total_pages_for_category = (len(category_images) + images_per_full_page - 1) // images_per_full_page
            
            for page_idx in range(total_pages_for_category):
                slide = prs.slides.add_slide(blank_layout)
                
                # Set slide background color
                background = slide.background
                fill = background.fill
                fill.solid()
                fill.fore_color.rgb = bg_color
                
                # Build title
                if organize_by == 'view':
                    slide_title = f"{category_name} view"
                else:
                    slide_title = str(category_name)
                if total_pages_for_category > 1:
                    slide_title += f" ({page_idx + 1}/{total_pages_for_category})"
                
                # Add title
                txBox = slide.shapes.add_textbox(
                    Inches(margin),
                    Inches(margin / 2),
                    Inches(slide_width - 2 * margin),
                    Inches(title_height_inches)
                )
                tf = txBox.text_frame
                p = tf.paragraphs[0]
                p.text = slide_title
                p.font.size = Pt(title_fontsize)
                p.font.bold = True
                p.font.color.rgb = font_color
                p.alignment = PP_ALIGN.CENTER
                
                # Get images for this page
                start_idx = page_idx * images_per_full_page
                end_idx = min(start_idx + images_per_full_page, len(category_images))
                page_images = category_images[start_idx:end_idx]
                
                content_top = margin + title_height_inches
                
                for i, (legend_name, img_path, view_name) in enumerate(page_images):
                    row = i // cols
                    col = i % cols
                    
                    cell_left = margin + col * cell_width
                    cell_top = content_top + row * cell_height
                    
                    try:
                        with Image.open(img_path) as img:
                            img_width, img_height = img.size
                            
                            # Calculate scaling
                            max_width = cell_width - 0.1
                            max_height = cell_height - label_height_inches - 0.1
                            
                            scale_w = max_width / (img_width / 96)
                            scale_h = max_height / (img_height / 96)
                            scale_factor = min(scale_w, scale_h)
                            
                            final_width = (img_width / 96) * scale_factor
                            final_height = (img_height / 96) * scale_factor
                            
                            # Center image in cell
                            img_left = cell_left + (cell_width - final_width) / 2
                            img_top = cell_top + (cell_height - label_height_inches - final_height) / 2
                            
                            # Add image
                            slide.shapes.add_picture(
                                img_path,
                                Inches(img_left),
                                Inches(img_top),
                                Inches(final_width),
                                Inches(final_height)
                            )
                            
                            # Add label
                            label = str(legend_name)
                            if view_name and organize_by != 'view':
                                label += f" ({view_name})"
                            max_chars = int(cell_width * 8)
                            if len(label) > max_chars:
                                label = label[:max_chars-3] + '...'
                            
                            txBox = slide.shapes.add_textbox(
                                Inches(cell_left),
                                Inches(cell_top + cell_height - label_height_inches),
                                Inches(cell_width),
                                Inches(label_height_inches)
                            )
                            tf = txBox.text_frame
                            p = tf.paragraphs[0]
                            p.text = label
                            p.font.size = Pt(label_fontsize)
                            p.font.color.rgb = font_color
                            p.alignment = PP_ALIGN.CENTER
                            
                    except Exception as e:
                        self._vprint(f'   ⚠️  Could not process: {img_path} - {e}', level='full')
        
        prs.save(pptx_path)
        return pptx_path

    def _to_rgba(self, color, alpha=None):
        # Convert color to uint8 RGBA for trimesh.
        import numpy as np

        try:
            r, g, b, a = extract_rgba_tuple(color, default_alpha=1.0)
            if alpha is not None:
                # Reuse the shared alpha parser so export overrides accept
                # normalized, byte, and percentage notation consistently.
                a = extract_rgba_tuple(set_alpha(color, alpha))[3]
            c = np.array([r, g, b, max(0.0, min(1.0, a))], dtype=float)
        except:
            c = np.array([128.0, 128.0, 128.0, 255.0], dtype=float)
            return c.astype(np.uint8)

        return np.array([
            int(np.clip(round(c[0]), 0, 255)),
            int(np.clip(round(c[1]), 0, 255)),
            int(np.clip(round(c[2]), 0, 255)),
            int(np.clip(round(c[3] * 255), 0, 255)),
        ], dtype=np.uint8)

    def _extract_trimesh(self, mesh):
        """Extract a trimesh.Trimesh from navis/trimesh-like objects."""
        import trimesh

        if mesh is None:
            return None
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        if isinstance(mesh, go.Mesh3d):
            vertices = np.column_stack([np.asarray(mesh.x), np.asarray(mesh.y), np.asarray(mesh.z)])
            faces = np.column_stack([np.asarray(mesh.i), np.asarray(mesh.j), np.asarray(mesh.k)]).astype(np.int64)
            if len(vertices) == 0 or len(faces) == 0:
                return None
            return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if hasattr(mesh, 'trimesh'):
            return mesh.trimesh
        if hasattr(mesh, 'mesh') and isinstance(mesh.mesh, trimesh.Trimesh):
            return mesh.mesh
        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
        return None

    def _append_exportable_mesh(self, mesh, color=None, alpha=None, name=None, role='mesh'):
        """Append a colored mesh copy to the cached export scene."""
        try:
            tm = self._extract_trimesh(mesh)
            if tm is None:
                return False

            tm = tm.copy()
            rgba = self._to_rgba(color if color is not None else 'gray', alpha=alpha)
            tm.visual.face_colors = rgba
            tm.metadata['export_rgba'] = rgba.tolist()
            if name:
                tm.metadata['name'] = name
            tm.metadata['export_role'] = role
            self.exportable_meshes.append(tm)
            return True
        except Exception:
            return False

    def _get_glb_export_rgba(self, mesh):
        """Return the exact cached RGBA color for GLB export."""
        metadata_rgba = getattr(mesh, 'metadata', {}).get('export_rgba')
        if metadata_rgba is not None:
            return np.asarray(metadata_rgba, dtype=np.uint8)

        face_colors = np.asarray(mesh.visual.face_colors)
        if face_colors.ndim == 2 and len(face_colors) > 0:
            return face_colors[0].astype(np.uint8).copy()

        return np.array([128, 128, 128, 255], dtype=np.uint8)

    def _prepare_glb_mesh(self, mesh, geometry_name):
        """Convert a cached mesh into a GLB-friendly mesh with explicit PBR material."""
        import trimesh

        tm = mesh.copy()
        rgba = self._get_glb_export_rgba(tm)
        material = trimesh.visual.material.PBRMaterial(
            name=geometry_name,
            baseColorFactor=rgba,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            doubleSided=True,
            alphaMode='BLEND' if rgba[3] < 255 else 'OPAQUE',
        )
        tm.visual = trimesh.visual.texture.TextureVisuals(material=material)
        tm.metadata.update(getattr(mesh, 'metadata', {}))
        return tm

    def export_3d_model(self, filename=None, format='glb'):
        """Export the 3D scene to a model file (GLB, OBJ, STL).
        
        Exports the current scene (neurons, synapses, ROIs) to a 3D model file.
        Useful for importing into Blender, Unity, or other 3D software.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, uses self.saveas.
        format : str, default 'glb'
            Output format: 'glb' (binary glTF, recommended), 'obj', 'stl', 'ply'.
        
        Returns
        -------
        str
            Path to the saved file.
        """
        if not self.exportable_meshes and not hasattr(self, 'fig_3d'):
            self._vprint('⚠️  No meshes to export. Run plot_neurons() first.')
            return None
            
        if filename is None:
            filename = self.saveas
            
        # Ensure extension
        if not filename.lower().endswith(f'.{format}'):
            filename += f'.{format}'
            
        filepath = os.path.join(self.save_folder, filename)
        
        self._vprint(f'Exporting 3D model to {filepath}...', level='full')
        
        try:
            import trimesh
            
            # Collect all meshes
            meshes = []
            if hasattr(self, 'exportable_meshes'):
                for index, mesh in enumerate(self.exportable_meshes):
                    geometry_name = getattr(mesh, 'metadata', {}).get('name', f'mesh_{index}')
                    meshes.append(self._prepare_glb_mesh(mesh, geometry_name))
                
            if not meshes:
                self._vprint('⚠️  No meshes found to export.')
                return None
                
            # Combine scene
            scene = trimesh.Scene()
            for index, mesh in enumerate(meshes):
                geometry_name = getattr(mesh, 'metadata', {}).get('name', f'mesh_{index}')
                scene.add_geometry(mesh, geom_name=geometry_name, node_name=geometry_name)
            
            # Export
            scene.export(filepath)
            self._vprint(f'✓ Saved {filepath}')
            return filepath
            
        except ImportError:
            self._vprint('⚠️  trimesh not installed. Cannot export 3D model.')
            self._vprint('   pip install trimesh')
            return None
        except Exception as e:
            self._vprint(f'⚠️  Export failed: {e}')
            return None

    def export_video(self, fps=30, degree_per_frame=1.0, rotate='horizontal', rotate_plane=None, 
                    view_direction=None, view_distance=None, synapse_size=1, 
                    html_file=None, output_dir=None, use_existing_images=True, 
                    export_gif=True, gif_scale=0.2, gif_optimize=True,
                    auto_crop=True, crop_margin=30, export_method: str = None, **kwargs):
        '''
        Export a rotating 3D visualization to MP4 video.
        
        Can be used in two modes:
        1. After plot_neurons(): Uses the current figure in memory
        2. Standalone with html_file: Loads figure from existing HTML file
        
        For standalone usage without VisualizeSkeleton initialization, use the
        module-level function `export_video_from_html()` instead.
        
        Parameters
        ----------
        fps : int, default 30
            Frames per second for the output video.
        degree_per_frame : float, default 1.0
            Rotation angle in degrees per frame. Controls rotation speed.
            - 1.0 → 360 frames for full rotation (12 sec video at 30 fps)
            - 2.0 → 180 frames for full rotation (6 sec video at 30 fps)
            - 0.5 → 720 frames for full rotation (24 sec video at 30 fps)
        rotate : str, default 'horizontal'
            Rotation direction:
            - 'horizontal': Rotate around Y-axis (turntable motion)
            - 'vertical': Rotate around X-axis (tumbling motion)
        rotate_plane : str, optional (deprecated)
            Legacy parameter. Use 'rotate' instead.
            Plane to rotate: 'xy', 'xz', or 'yz'.
        view_direction : tuple, optional, default (1, -1)
            Camera direction multipliers for sin/cos components.
            Options: (1, 1), (1, -1), (-1, 1), or (-1, -1).
        view_distance : float, optional, default 2.2
            Relative camera distance from center (1.0 = close, 3.0 = far).
        synapse_size : int, default 1
            Size of synapse markers in the video (1-10 recommended).
        html_file : str, optional
            Path to existing HTML file to load figure data from.
            Enables standalone usage without calling plot_neurons() first.
            Example: '/path/to/my_neurons.html'
        output_dir : str, optional
            Directory to save video output. If None and html_file is provided,
            uses the directory containing the html_file.
            If None and using plot_neurons(), uses self.save_folder.
        use_existing_images : bool, default True
            If True, skip rendering and reuse cached images from previous export.
            Useful for regenerating video with different fps without re-rendering.
        export_gif : bool, default True
            If True, automatically convert videos to GIF format after export.
        gif_scale : float, default 0.2
            Scale factor for GIF resolution (0.1-1.0). Lower values = smaller file size.
            Example: 0.2 = 20% of original video resolution.
        gif_optimize : bool, default True
            Enable GIF compression optimization for smaller file sizes.
        auto_crop : bool, default True
            If True, automatically crop whitespace/background from each frame
            with consistent sizing across all frames (computes max projection bounds).
            This ensures uniform frame dimensions for smooth video playback.
        crop_margin : int, default 30
            Margin (in pixels) to preserve around content when auto_crop=True.
            Has no effect unless auto_crop=True.
        export_method : str, optional
            Export method to use for frame rendering: 'webdriver' or 'kaleido'.
            If None, uses the class-level export_method attribute.
            - 'webdriver': Uses Chrome WebDriver (recommended for large figures)
            - 'kaleido': Uses Kaleido static image export (default)
        **kwargs : dict
            Additional arguments for plotly write_image():
            - scale : int - Resolution multiplier. Defaults to min(self.export_scale, 3).
              For video, scale is capped at 3 for reasonable performance.
              To use scale=4 (very slow), pass it explicitly: scale=4
            - width : int - Video width in pixels (default 1200)
            - height : int - Video height in pixels (default 900)
        
        Returns
        -------
        int
            0 on success, 1 on failure
        
        Output Files
        ------------
        - {output_dir}/pics_{fps}fps_{rotate_plane}/ : Cached frame images
        - {output_dir}/{name}_video_h_forward.mp4 : Forward rotation video (horizontal)
        - {output_dir}/{name}_video_h_backward.mp4 : Reverse rotation video (horizontal)
        - {output_dir}/{name}_video_v_forward.mp4 : Forward rotation video (vertical)
        - {output_dir}/{name}_video_v_backward.mp4 : Reverse rotation video (vertical)
        - {output_dir}/{name}_video_h_forward.gif : Forward rotation GIF (horizontal, if export_gif=True)
        - {output_dir}/{name}_video_h_backward.gif : Reverse rotation GIF (horizontal, if export_gif=True)
        - {output_dir}/{name}_video_v_forward.gif : Forward rotation GIF (vertical, if export_gif=True)
        - {output_dir}/{name}_video_v_backward.gif : Reverse rotation GIF (vertical, if export_gif=True)
        
        Examples
        --------
        # Mode 1: After plot_neurons()
        vs = VisualizeSkeleton(dataset='hemibrain:v1.2.1', neuron_layers=['EB'])
        vs.plot_neurons()
        vs.export_video(fps=30, degree_per_frame=1.0)
        
        # Faster rotation (shorter video)
        vs.export_video(fps=30, degree_per_frame=2.0)
        
        # Vertical rotation
        vs.export_video(fps=30, rotate='vertical')
        
        # High quality export
        vs.export_video(fps=30, scale=4, width=1920, height=1080)
        
        # Mode 2: From existing HTML file (output to same directory)
        vs.export_video(html_file='/path/to/existing_plot.html')
        
        # Mode 3: Standalone function (no VisualizeSkeleton needed)
        from visualize_skeleton import export_video_from_html
        export_video_from_html('/path/to/plot.html', fps=30, degree_per_frame=1.0)
        
        # Reuse cached images (fast video regeneration)
        vs.export_video(fps=60, use_existing_images=True)
        '''
        # Handle rotate parameter - overrides rotate_plane
        if rotate == 'horizontal':
            rotate_plane = 'xz'  # Rotate around vertical (Y) axis
        elif rotate == 'vertical':
            rotate_plane = 'yz'  # Rotate around horizontal (X) axis
        elif rotate_plane is None:
            # Default to horizontal rotation if neither specified
            rotate_plane = 'xz'
        # else: use the explicitly provided rotate_plane
        
        # Determine which export method to use
        actual_export_method = export_method if export_method else self.export_method
        
        if view_direction is None:
            if 'manc' in self.dataset.lower():
                view_direction = (1, 1)  # MANC: +Z is front
            else:
                view_direction = (1, -1)
        if view_distance is None:
            view_distance = 2.2
        
        # Warn if crop_margin is set but auto_crop is disabled
        if crop_margin != 30 and not auto_crop:
            self._vprint(f'⚠️  crop_margin={crop_margin} has no effect without auto_crop=True')
            self._vprint(f'   Add auto_crop=True to enable automatic frame cropping')
        
        # Warn about kaleido limitations for video export
        if actual_export_method == 'kaleido':
            self._vprint('💡 Tip: For video export, consider using export_method="webdriver" for better stability.')
            self._vprint('   Kaleido can timeout on complex figures. WebDriver is more reliable for animations.')
        
        # Set default scale from self.export_scale if not specified in kwargs
        # For kaleido: cap at scale=3 for reasonable performance (scale=4 is very slow)
        # For webdriver/webdriver-fast: cap at scale=5 for stability
        # Users can explicitly set higher scale to override the cap
        if kwargs.get('scale') is None and kwargs.get('width') is None and kwargs.get('height') is None:
            if actual_export_method in ('webdriver', 'webdriver-fast'):
                # WebDriver can handle higher scales - cap at 5
                default_scale = min(getattr(self, 'export_scale', 5), 5)
            else:
                # Kaleido - cap at 3
                default_scale = min(getattr(self, 'export_scale', 3), 3)
            kwargs['scale'] = default_scale
        elif kwargs.get('scale') is not None:
            # User explicitly provided scale - check if > cap and warn
            user_scale = kwargs['scale']
            cap = 5 if actual_export_method in ('webdriver', 'webdriver-fast') else 3
            if user_scale > cap:
                self._vprint(f'⚠️  Using scale={user_scale} for video export (uncapped). This may be slow.')
            elif user_scale > getattr(self, 'export_scale', 3):
                # User requested higher than default, allow it
                pass
        
        # Use explicit degree_per_frame instead of calculating from fps
        step = degree_per_frame
        
        # Determine output directory and filename
        if output_dir is not None:
            save_folder = output_dir
            # Extract filename from html_file or use default
            if html_file is not None:
                saveas = os.path.splitext(os.path.basename(html_file))[0]
            else:
                saveas = 'video_export'
            os.makedirs(save_folder, exist_ok=True)
        elif html_file is not None:
            # Use the directory containing the html_file
            save_folder = os.path.dirname(os.path.abspath(html_file))
            saveas = os.path.splitext(os.path.basename(html_file))[0]
        elif hasattr(self, 'save_folder') and self.save_folder:
            save_folder = self.save_folder
            saveas = self.saveas if hasattr(self, 'saveas') and self.saveas else 'video_export'
        else:
            raise ValueError(
                'No output directory specified. Either:\n'
                '  1. Run plot_neurons() first, or\n'
                '  2. Provide html_file parameter (output goes to same directory), or\n'
                '  3. Provide output_dir parameter explicitly'
            )
        
        # Load figure from existing HTML file if provided (OPTIMIZATION)
        if html_file is not None:
            self._vprint(f'📂 Loading figure from existing HTML: {html_file}')
            if not os.path.exists(html_file):
                raise FileNotFoundError(f'HTML file not found: {html_file}')
            
            # Read and parse the HTML file to extract figure data
            import plotly.io as pio
            try:
                fig_loaded = pio.read_html(html_file)
                fig_traces = fig_loaded.data
                self._vprint(f'✓ Loaded {len(fig_traces)} traces from HTML file')
            except Exception as e:
                raise RuntimeError(f'Failed to load figure from HTML: {e}')
        else:
            # Use current figure
            if not hasattr(self, 'fig_path') or not os.path.exists(self.fig_path+'.html'):
                raise RuntimeError(
                    'No figure found. Either run plot_neurons() first or provide html_file parameter.'
                )
            html_size = os.path.getsize(self.fig_path+'.html') / 1024 / 1024 # in MB
            html_size_cap = self._get_html_size_cap()
            if html_size > html_size_cap:
                self._vprint(f'⚠️  Figure is large ({html_size:.1f} MB). Rendering may be slow.')
                if actual_export_method == 'kaleido':
                    self._vprint(f'   ⚠️  Kaleido may timeout on figures > {html_size_cap}MB.')
                    self._vprint(f'   💡 Recommended: Use export_method="webdriver" for large figures.')
                    self._vprint(f'   💡 Or increase skeleton_mesh_simplification (e.g., 0.97-0.99).')
                else:
                    self._vprint(f'   Consider using higher skeleton_mesh_simplification if export is slow.')
            
            # Use simplified HTML file if available from previous export (e.g., export_views)
            simplified_fig = getattr(self, '_simplified_export_fig', None)
            if simplified_fig is not None:
                self._vprint(f'   ✓ Using simplified figure from previous export')
                fig_traces = simplified_fig.data
            else:
                # Use original figure
                fig_traces = self.fig_3d.data
        # Configure figure for video export
        for trace in fig_traces:
            trace.showlegend = False
            if hasattr(trace,'marker'):
                trace.marker.size = synapse_size
        
        fig_layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
        )
        fig_new = go.Figure(data=fig_traces, layout=fig_layout)
        
        # Set camera parameters - always use frontal view for consistency
        scene_camera_parameters = dict(
            up=dict(x=0, y=-1, z=0),
            eye=dict(x=0, y=0, z=-view_distance),
        )
        
        if 'hemibrain' in self.dataset.lower() and self.brain_mesh == 'template':
             scene_camera_parameters = dict(
                up=dict(x=0, y=0, z=-1),
                eye=dict(x=0, y=view_distance, z=0),  # +Y is front (anterior)
            )
        
        # Adjust for MANC (Male Adult Nerve Cord)
        if 'manc' in self.dataset.lower():
             # MANC: Anterior is +Z, Posterior is -Z. Dorsal is -Y.
             # "Front" view should be looking from Anterior -> Posterior, so eye at +Z.
             scene_camera_parameters = dict(
                up=dict(x=0, y=-1, z=0),
                eye=dict(x=0, y=0, z=view_distance),
            )

        
        fig_new.update_layout(
            sliders=[],  # Remove sliders for cleaner video
            updatemenus=[],  # Remove view selection dropdown
            annotations=[],  # Remove controls hint
            paper_bgcolor=self.background_color,
            plot_bgcolor=self.background_color,
            scene=dict(
                dragmode='orbit',
                xaxis={'visible':False}, 
                yaxis={'visible':False},
                zaxis={'visible':False},
                bgcolor=self.background_color,
            ),
            scene_camera=scene_camera_parameters,
            legend=dict(
                bgcolor=self.background_color,
                font=dict(color='white' if self._is_dark_background() else 'black'),
            ),
        )
        
        # Set up image folder
        pic_folder = os.path.join(save_folder, f'pics_{fps}fps_{rotate_plane}')
        
        # Calculate rotation steps
        if step > 0:
            steps_to_write = np.linspace(0, 360, int(360/step), endpoint=False)
        elif step < 0:
            steps_to_write = np.linspace(360, 0, int(360/step), endpoint=False)
        
        # OPTIMIZATION: Skip image rendering if use_existing_images=True
        # For WebDriver modes, also check for partial completion (resume support)
        resume_mode = False
        if use_existing_images and os.path.exists(pic_folder):
            existing_images = [f for f in os.listdir(pic_folder) 
                             if f.startswith('deg_') and f.endswith('.jpeg') 
                             and os.path.getsize(os.path.join(pic_folder, f)) >= 1024]
            if len(existing_images) == len(steps_to_write):
                self._vprint(f'✓ Using {len(existing_images)} existing images from {pic_folder}')
                self._vprint(f'  Skipping image rendering (use_existing_images=True)')
            elif len(existing_images) > 0 and actual_export_method in ('webdriver', 'webdriver-fast'):
                # Partial completion - enable resume mode for WebDriver
                self._vprint(f'🔄 Found {len(existing_images)}/{len(steps_to_write)} completed frames')
                self._vprint(f'   Resuming from frame {len(existing_images) + 1}...')
                resume_mode = True
                use_existing_images = False  # Continue rendering
            else:
                self._vprint(f'⚠️  Found {len(existing_images)} images but need {len(steps_to_write)}')
                self._vprint(f'  Re-rendering images...')
                use_existing_images = False
        else:
            use_existing_images = False
        
        # Render images if needed
        if not use_existing_images:
            if os.path.exists(pic_folder) and not resume_mode:
                # Only delete folder if NOT in resume mode
                shutil.rmtree(pic_folder)
            if not os.path.exists(pic_folder):
                os.makedirs(pic_folder)
            
            self._vprint(f'🎬 Rendering {len(steps_to_write)} frames at {fps} fps...')
            self._vprint(f'   Resolution: scale={kwargs.get("scale", "auto")}', end='')
            if 'width' in kwargs and 'height' in kwargs:
                self._vprint(f', size={kwargs["width"]}x{kwargs["height"]}')
            else:
                self._vprint('')
            
            # Ensure dimensions are set to avoid blank images if not provided
            if 'width' not in kwargs: kwargs['width'] = 1200
            if 'height' not in kwargs: kwargs['height'] = 900
            
            t0 = time.time()
            current_scale = kwargs.get('scale', 2)
            frame_export_failed = False
            use_kaleido_fallback = False
            
            # Choose export method
            if actual_export_method in ('webdriver', 'webdriver-fast'):
                # WebDriver method - efficient: open browser once for all frames
                # Uses canvas.toDataURL() for fast, high-quality capture
                self._vprint(f'   Using WebDriver for frame export (canvas.toDataURL)...')
                
                # Get render_wait from attribute (None = auto-calibrate, 0 = fastest)
                render_wait = getattr(self, 'webdriver_render_wait', None)
                if render_wait is not None:
                    self._vprint(f'   Render wait: {render_wait}s')
                
                # Save figure to temp HTML
                temp_html = os.path.join(pic_folder, '_temp_video.html')
                self._write_plotly_html(
                    fig_new,
                    temp_html,
                    auto_open=False,
                    include_plotlyjs=True,
                    config={'displayModeBar': False},
                )
                
                # Only save simplified HTML copy if we actually used a simplified figure
                # Check if _simplified_export_fig exists AND was used for this export
                simplified_fig = getattr(self, '_simplified_export_fig', None)
                if simplified_fig is not None:
                    html_size_mb = os.path.getsize(temp_html) / 1024 / 1024
                    simplified_html_path = os.path.join(save_folder, f"{saveas}_simplified.html")
                    if not os.path.exists(simplified_html_path):
                        import shutil
                        shutil.copy(temp_html, simplified_html_path)
                        self._vprint(f'   ✓ Saved simplified HTML: {os.path.basename(simplified_html_path)} ({html_size_mb:.1f}MB)')
                
                # Retry logic for Chrome crashes - with RESUME capability
                # Retries reset when progress is made past previous crash point
                max_retries = 3
                webdriver_success = False
                last_error = None
                resume_from_frame = 0  # Track where to resume from
                last_crash_frame = -1  # Track where last crash occurred
                consecutive_crashes_at_same_point = 0  # Only count crashes at same position
                
                while consecutive_crashes_at_same_point < max_retries:
                    # Check how many frames were completed before this attempt
                    existing_frames = [f for f in os.listdir(pic_folder) 
                                      if f.startswith('deg_') and f.endswith('.jpeg')
                                      and os.path.getsize(os.path.join(pic_folder, f)) >= 1024]
                    frames_completed = len(existing_frames)
                    
                    if frames_completed > resume_from_frame:
                        # Progress was made - this is a resume, not a retry at same point
                        resume_from_frame = frames_completed
                        if consecutive_crashes_at_same_point > 0:
                            self._vprint(f'\n   🔄 RESUMING from frame {frames_completed + 1} ({frames_completed} frames saved)...')
                        
                        # Brief pause before resume to let system resources settle
                        time.sleep(2)
                        time.sleep(2)
                    
                    try:
                        with WebDriverExportSession(
                            width=kwargs['width'], height=kwargs['height'],
                            scale=current_scale,
                            timeout=300,
                            render_wait=render_wait
                        ) as session:
                            session.load_html(temp_html, wait_for_render=True, render_wait=3, background_color=self.background_color)
                            self._vprint(f'   ✓ HTML loaded in browser (render_wait={session._render_wait:.2f}s)')
                            
                            # Get initial camera from HTML - this defines "front view"
                            initial_camera = session.get_current_camera()
                            if initial_camera:
                                initial_eye = initial_camera.get('eye', {'x': 0, 'y': 0, 'z': -view_distance})
                                initial_up = initial_camera.get('up', {'x': 0, 'y': -1, 'z': 0})
                                self._vprint(f'   Initial camera: eye=({initial_eye.get("x", 0):.2f}, {initial_eye.get("y", 0):.2f}, {initial_eye.get("z", 0):.2f})')
                            else:
                                initial_eye = {'x': 0, 'y': 0, 'z': -view_distance}
                                initial_up = {'x': 0, 'y': -1, 'z': 0}
                            
                            # Compute view distance from initial camera
                            cam_distance = np.sqrt(
                                initial_eye.get('x', 0)**2 + 
                                initial_eye.get('y', 0)**2 + 
                                initial_eye.get('z', 0)**2
                            )
                            if cam_distance < 0.1:
                                cam_distance = view_distance
                            
                            for i, deg in enumerate(steps_to_write):
                                if frame_export_failed:
                                    break
                                
                                # Skip frames that are already completed (resume support)
                                fig_path = os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg')
                                if os.path.exists(fig_path) and os.path.getsize(fig_path) >= 1024:
                                    # Frame already exists and is valid, skip
                                    continue
                                
                                rad_i = np.deg2rad(deg)
                                sin_val = np.sin(rad_i)
                                cos_val = np.cos(rad_i)
                                
                                # Small offset to avoid gimbal lock
                                offset = cam_distance * 0.01
                                
                                # Rotation logic - independent from kaleido
                                # Rotate camera around the object based on rotation plane
                                # Note: Initial camera is eye={0,0,-z}, up={0,-1,0}
                                if rotate_plane == 'xy':
                                    # Horizontal rotation around Z axis
                                    eye = {
                                        'x': cam_distance * sin_val,
                                        'y': cam_distance * cos_val,
                                        'z': offset
                                    }
                                    up = {'x': 0, 'y': 0, 'z': 1}
                                    
                                elif rotate_plane == 'yz':
                                    # Vertical rotation around X axis (front→bottom→back→top)
                                    eye = {
                                        'x': offset,
                                        'y': cam_distance * sin_val,
                                        'z': -cam_distance * cos_val
                                    }
                                    up = {
                                        'x': 0,
                                        'y': -cos_val,
                                        'z': -sin_val
                                    }
                                    
                                elif rotate_plane == 'xz':
                                    # Horizontal rotation around Y axis
                                    z_sign = 1 if 'manc' in self.dataset.lower() else -1
                                    eye = {
                                        'x': cam_distance * sin_val,
                                        'y': offset,
                                        'z': z_sign * cam_distance * cos_val
                                    }
                                    up = {'x': 0, 'y': -1, 'z': 0}
                                
                                # Rotate camera via JavaScript
                                session.set_camera(eye=eye, up=up)
                                
                                # Decode the configured background through the
                                # shared parser for screenshot compositing.
                                try:
                                    bg_rgb = extract_rgb_tuple(self.background_color)
                                except (TypeError, ValueError):
                                    bg_rgb = (255, 255, 255)
                                
                                # Take screenshot
                                session.screenshot(fig_path, convert_to_jpeg=True, jpeg_quality=95,
                                                 auto_crop=False, background_color=bg_rgb)
                                
                                # Verify
                                if not os.path.exists(fig_path) or os.path.getsize(fig_path) < 1024:
                                    self._vprint(f'\n⚠️  Frame {i+1} export failed')
                                    frame_export_failed = True
                                    break
                                
                                # Count completed frames for progress display
                                completed_count = len([f for f in os.listdir(pic_folder) 
                                                      if f.startswith('deg_') and f.endswith('.jpeg') 
                                                      and os.path.getsize(os.path.join(pic_folder, f)) >= 1024])
                                
                                elapsed = time.time() - t0
                                avg_time = elapsed / max(1, completed_count - resume_from_frame)
                                remaining = avg_time * (len(steps_to_write) - completed_count)
                                print(f'\r  Frame {completed_count}/{len(steps_to_write)} | '
                                      f'Elapsed: {elapsed:.1f}s | '
                                      f'ETA: {remaining:.1f}s | '
                                      f'{avg_time:.2f}s/frame', end='    ', flush=True)
                        
                        # Check if all frames completed
                        final_frame_count = len([f for f in os.listdir(pic_folder) 
                                                if f.startswith('deg_') and f.endswith('.jpeg')
                                                and os.path.getsize(os.path.join(pic_folder, f)) >= 1024])
                        
                        if final_frame_count >= len(steps_to_write) and not frame_export_failed:
                            webdriver_success = True
                            break  # Exit retry loop on success
                        elif frame_export_failed:
                            # Frame export failed but not due to crash - don't retry
                            break
                            
                    except Exception as e:
                        last_error = e
                        error_msg = str(e).lower()
                        # Check if this is a Chrome crash (empty message or specific patterns)
                        is_chrome_crash = (
                            'Message: \n' in str(e) or 
                            str(e) == '' or
                            'chrome not reachable' in error_msg or
                            'session deleted' in error_msg or
                            'no such window' in error_msg or
                            'tab crashed' in error_msg or
                            'target window already closed' in error_msg or
                            'disconnected' in error_msg
                        )
                        
                        if is_chrome_crash:
                            # Count how many frames were saved before crash
                            frames_saved = len([f for f in os.listdir(pic_folder) 
                                              if f.startswith('deg_') and f.endswith('.jpeg')
                                              and os.path.getsize(os.path.join(pic_folder, f)) >= 1024])
                            
                            # Check if this crash is at a new position (progress was made)
                            if frames_saved > last_crash_frame:
                                # Progress was made! Reset retry counter
                                consecutive_crashes_at_same_point = 1
                                last_crash_frame = frames_saved
                                self._vprint(f'\n   ⚠️  Chrome crashed unexpectedly at frame {frames_saved}. Will resume from there...')
                            else:
                                # Crashed at same position again
                                consecutive_crashes_at_same_point += 1
                                self._vprint(f'\n   ⚠️  Chrome crashed again at frame {frames_saved} (attempt {consecutive_crashes_at_same_point}/{max_retries})')
                            
                            # Do NOT delete frames - keep them for resume
                            # Reset timer for next attempt's ETA calculation
                            t0 = time.time()
                            continue
                        else:
                            # Not a crash - some other error, don't retry
                            break
                
                # Handle final result
                if not webdriver_success:
                    if last_error:
                        self._vprint(f'\n⚠️  WebDriver export failed after {consecutive_crashes_at_same_point} consecutive crashes at same point: {last_error}')
                    self._vprint(f'      Falling back to kaleido...')
                    self._vprint(f'      💡 Tip: Set export_method="kaleido" for future use if WebDriver continues to fail.')
                    use_kaleido_fallback = True
                    frame_export_failed = False  # Reset to try kaleido
                    # Reset timer for kaleido attempt
                    t0 = time.time()
                
                # Clean up temp HTML
                try:
                    os.remove(temp_html)
                except:
                    pass
                
            if actual_export_method == 'kaleido' or use_kaleido_fallback:
                # Default: kaleido export with timeout
                # Check HTML size and auto-simplify if > size cap
                html_size_mb = 0
                if hasattr(self, 'fig_path') and os.path.exists(self.fig_path + '.html'):
                    html_size_mb = os.path.getsize(self.fig_path + '.html') / 1024 / 1024
                
                html_size_cap = self._get_html_size_cap()
                
                # Auto-simplify for kaleido if HTML > size cap
                # But first check if we already have a simplified figure from previous export
                if html_size_mb > html_size_cap:
                    simplified_fig = getattr(self, '_simplified_export_fig', None)
                    simplified_html = getattr(self, '_simplified_html_path', None)
                    
                    if simplified_fig is not None:
                        self._vprint(f'⚠️  HTML size ({html_size_mb:.0f}MB) exceeds {html_size_cap}MB - using previously simplified figure')
                        # Rebuild fig_new from simplified figure traces
                        fig_new = go.Figure(data=simplified_fig.data, layout=fig_layout)
                        fig_new.update_layout(
                            sliders=[],
                            scene=dict(
                                dragmode='orbit',
                                xaxis={'visible':False}, 
                                yaxis={'visible':False},
                                zaxis={'visible':False},
                            ),
                            scene_camera=scene_camera_parameters,
                        )
                    else:
                        # Target 50% file size reduction with overhead estimation
                        html_overhead_mb = 25  # Estimated fixed overhead
                        target_size_mb = html_size_mb * 0.5  # 50% target
                        data_size = html_size_mb - html_overhead_mb
                        target_data_size = target_size_mb - html_overhead_mb
                        simplification_factor = max(0.1, target_data_size / data_size) if data_size > 0 else 0.5
                        
                        self._vprint(f'⚠️  HTML size ({html_size_mb:.0f}MB) exceeds {html_size_cap}MB - kaleido may timeout')
                        self._vprint(f'   Target: {target_size_mb:.0f}MB (50% reduction), simplification factor: {simplification_factor:.2f}...')
                        
                        # Create simplified copy with dynamic reduction
                        fig_new = self._simplify_figure_for_kaleido(fig_new, simplification_factor)
                        
                        # Save simplified HTML for future reuse
                        simplified_html_path = os.path.join(save_folder, f"{saveas}_simplified.html")
                        self._write_plotly_html(
                            fig_new,
                            simplified_html_path,
                            theme_toggle=self.html_theme_toggle,
                            legend_tree=(self.legend_mode == 'tree'),
                            auto_open=False,
                            include_plotlyjs=True,
                            config={'displayModeBar': False},
                        )
                        simplified_size_mb = os.path.getsize(simplified_html_path) / 1024 / 1024
                        self._vprint(f'   ✓ Saved simplified HTML: {os.path.basename(simplified_html_path)} ({simplified_size_mb:.0f}MB)')
                        
                        # Store for reuse
                        self._simplified_export_fig = fig_new
                        self._simplified_html_path = simplified_html_path
                        
                        self._vprint(f'   💡 For better quality, consider using export_method="webdriver"')
                
                # Set up timeout handler (Unix only)
                old_handler = None
                has_alarm_support = False
                try:
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    has_alarm_support = True
                except (AttributeError, ValueError):
                    has_alarm_support = False
                
                # Get base timeout from attribute
                base_timeout = getattr(self, 'export_timeout', 60)
                frame1_success = False  # Track if frame 1 exported successfully
                
                # Sequential rendering with timeout control
                self._vprint(f'   Rendering frames... (First frame may take longer to initialize Kaleido)')
                
                try:
                    for i, deg in enumerate(steps_to_write):
                        if frame_export_failed:
                            break
                            
                        rad_i = np.deg2rad(deg)
                        x = view_distance * np.sin(rad_i) * view_direction[0]
                        z = view_distance * np.cos(rad_i) * view_direction[1]
                        
                        # Add small offset to avoid gimbal lock at axis-aligned positions
                        # This prevents camera flipping at 90, 180, 270 degrees
                        y_offset = view_distance * 0.01  # 1% offset
                        
                        if rotate_plane == 'xy':
                            fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=z, z=y_offset)))
                        elif rotate_plane == 'yz':
                            fig_new.update_layout(scene_camera=dict(eye=dict(x=y_offset, y=x, z=z)))
                        elif rotate_plane == 'xz':
                            fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=y_offset, z=z)))
                        
                        fig_path = os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg')
                        
                        try:
                            # Timeout: first frame gets base_timeout, subsequent frames get 2x if frame1 succeeded
                            # This is because frame 1 success verifies export is working (just slower)
                            if i == 0:
                                timeout_sec = base_timeout
                            else:
                                timeout_sec = base_timeout * 2 if frame1_success else base_timeout
                            
                            if has_alarm_support:
                                signal.alarm(timeout_sec)
                            
                            kwargs_copy = kwargs.copy()
                            kwargs_copy['scale'] = current_scale
                            fig_new.write_image(fig_path, **kwargs_copy)
                            
                            if has_alarm_support:
                                signal.alarm(0)  # Cancel timeout
                            
                            # Mark frame 1 as successful - allows 2x timeout tolerance for subsequent frames
                            if i == 0:
                                frame1_success = True
                                
                        except PNGExportTimeout:
                            if has_alarm_support:
                                signal.alarm(0)
                            
                            if current_scale > 1:
                                self._vprint(f'\n⚠️  Frame {i+1} timed out at scale={current_scale}, retrying with scale=1...')
                                current_scale = 1
                                try:
                                    if has_alarm_support:
                                        signal.alarm(base_timeout * 2)
                                    kwargs_copy = kwargs.copy()
                                    kwargs_copy['scale'] = 1
                                    fig_new.write_image(fig_path, **kwargs_copy)
                                    if has_alarm_support:
                                        signal.alarm(0)
                                    self._vprint(f'   ✓ Frame {i+1} exported with scale=1')
                                    if i == 0:
                                        frame1_success = True
                                except (PNGExportTimeout, Exception) as retry_e:
                                    if has_alarm_support:
                                        signal.alarm(0)
                                    self._vprint(f'\n⚠️  Frame {i+1} failed even at scale=1: {retry_e}')
                                    self._vprint('   💡 Figure is too complex. Recommendations:')
                                    self._vprint('      1. Use export_method=\"webdriver\" (more reliable for complex figures)')
                                    self._vprint('      2. Increase skeleton_mesh_simplification (e.g., 0.97-0.99)')
                                    self._vprint('      3. Increase export_timeout (current: {base_timeout}s)')
                                    frame_export_failed = True
                                    break
                            else:
                                self._vprint(f'\n⚠️  Frame {i+1} timed out at scale=1. Figure is too complex.')
                                self._vprint('   💡 Recommendations:')
                                self._vprint('      1. Use export_method=\"webdriver\" (more reliable for complex figures)')
                                self._vprint('      2. Increase skeleton_mesh_simplification (e.g., 0.97-0.99)')
                                self._vprint(f'      3. Increase export_timeout (current: {base_timeout}s)')
                                frame_export_failed = True
                                break
                                
                        except Exception as e:
                            if has_alarm_support:
                                signal.alarm(0)
                            self._vprint(f'\n⚠️  Frame {i+1} failed: {e}')
                            if i == 0 and current_scale > 1:
                                self._vprint(f'   Retrying with scale=1...')
                                current_scale = 1
                                try:
                                    kwargs_copy = kwargs.copy()
                                    kwargs_copy['scale'] = 1
                                    fig_new.write_image(fig_path, **kwargs_copy)
                                    self._vprint(f'   ✓ Frame {i+1} exported with scale=1')
                                    frame1_success = True
                                except Exception as retry_e:
                                    self._vprint(f'   ⚠️  Retry failed: {retry_e}')
                                    self._vprint('   💡 Use export_method=\"webdriver\" or increase skeleton_mesh_simplification.')
                                    frame_export_failed = True
                                    break
                            else:
                                frame_export_failed = True
                                break
                        
                        elapsed = time.time() - t0
                        avg_time = elapsed / (i + 1)
                        remaining = avg_time * (len(steps_to_write) - i - 1)
                        print(f'\r  Frame {i+1}/{len(steps_to_write)} | '
                              f'Elapsed: {elapsed:.1f}s | '
                              f'ETA: {remaining:.1f}s | '
                              f'{avg_time:.2f}s/frame', end='    ', flush=True)
                              
                finally:
                    # Always cancel any pending alarm and restore original handler
                    if has_alarm_support:
                        try:
                            signal.alarm(0)  # Cancel any pending alarm
                            if old_handler is not None:
                                signal.signal(signal.SIGALRM, old_handler)
                        except:
                            pass
            
            if frame_export_failed:
                self._vprint(f'\n⚠️  Frame rendering aborted. Skipping video generation.')
                return 1
            
            print('\n✓ Image rendering complete')
            
            # Apply consistent cropping if auto_crop is enabled
            # This ensures all frames have the same dimensions during rotation
            if auto_crop:
                self._vprint(f'   Applying consistent auto-crop across all frames...')
                crop_result = self._apply_consistent_crop(pic_folder, margin=crop_margin)
                if crop_result:
                    crop_w, crop_h = crop_result
                    self._vprint(f'   ✓ All frames cropped consistently to {crop_w}x{crop_h}')
                else:
                    self._vprint(f'   ⚠️  Auto-crop failed, using original frame sizes')
        
        # Generate videos from images
        self._vprint(f'\nGenerating videos...')
        imglist = os.listdir(pic_folder)
        img_eg = cv2.imread(os.path.join(pic_folder, imglist[0]))
        height, width, layers = img_eg.shape
        
        self._vprint(f'   Video resolution: {width}x{height}')

        # Add rotation direction indicator to filename
        rotation_suffix = 'h' if rotate_plane == 'xz' else 'v' if rotate_plane == 'yz' else rotate_plane
        
        # Forward video - OPTIMIZED with faster codec
        video_path_forward = os.path.join(save_folder, f'{saveas}_video_{rotation_suffix}_forward.mp4')
        # Use H.264 codec for better compression and compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec (faster than mp4v)
        out = cv2.VideoWriter(video_path_forward, fourcc, fps, frameSize=(width, height))
        
        t0 = time.time()
        for i, deg in enumerate(steps_to_write):
            img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            if (i + 1) % 10 == 0 or i == len(steps_to_write) - 1:
                print(f'\r  Forward video: {i+1}/{len(steps_to_write)} frames', end='  ')
        out.release()
        t1 = time.time()
        print(f'\n✓ Forward video: {video_path_forward} ({t1-t0:.1f}s)')
        
        # Backward video
        video_path_backward = os.path.join(save_folder, f'{saveas}_video_{rotation_suffix}_backward.mp4')
        out = cv2.VideoWriter(video_path_backward, fourcc, fps, frameSize=(width, height))
        
        t0 = time.time()
        for i, deg in enumerate(steps_to_write[::-1]):
            img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
            out.write(img)
            if (i + 1) % 10 == 0 or i == len(steps_to_write) - 1:
                print(f'\r  Backward video: {i+1}/{len(steps_to_write)} frames', end='  ')
        out.release()
        t1 = time.time()
        print(f'\n✓ Backward video: {video_path_backward} ({t1-t0:.1f}s)')
        
        print(f'\n✅ Video export complete!')
        self._vprint(f'   Image cache: {pic_folder}')
        self._vprint(f'   Tip: Use use_existing_images=True to skip re-rendering next time')
        
        # Convert to GIF if requested
        if export_gif:
            self._vprint(f'\n🎞️  Converting videos to GIF format...')
            self._vprint(f'   Scale: {gif_scale} | Optimize: {gif_optimize}')
            
            # Convert forward video to GIF
            gif_path_forward = video_path_forward.replace('.mp4', '.gif')
            try:
                video2gif(
                    video_path_forward,
                    gif_path_forward,
                    fps=fps,
                    scale=gif_scale,
                    optimize=gif_optimize
                )
                self._vprint(f'   ✓ Forward GIF: {gif_path_forward}')
            except Exception as e:
                self._vprint(f'   ⚠️  Forward GIF conversion failed: {e}')
            
            # Convert backward video to GIF
            gif_path_backward = video_path_backward.replace('.mp4', '.gif')
            try:
                video2gif(
                    video_path_backward,
                    gif_path_backward,
                    fps=fps,
                    scale=gif_scale,
                    optimize=gif_optimize
                )
                self._vprint(f'   ✓ Backward GIF: {gif_path_backward}')
            except Exception as e:
                self._vprint(f'   ⚠️  Backward GIF conversion failed: {e}')
        
        return 0


def _detect_content_bounds_standalone(img, background_color=(255, 255, 255)):
    """Detect the bounding box of content in an image (standalone version)."""
    from PIL import Image
    import numpy as np
    
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, background_color)
        bg.paste(img, mask=img.split()[3])
        img_rgb = bg
    else:
        img_rgb = img.convert('RGB')
    
    arr = np.array(img_rgb)
    tolerance = 10
    bg_r, bg_g, bg_b = background_color
    
    non_bg_mask = (
        (np.abs(arr[:, :, 0].astype(int) - bg_r) > tolerance) |
        (np.abs(arr[:, :, 1].astype(int) - bg_g) > tolerance) |
        (np.abs(arr[:, :, 2].astype(int) - bg_b) > tolerance)
    )
    
    rows = np.any(non_bg_mask, axis=1)
    cols = np.any(non_bg_mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    
    return (row_min, row_max, col_min, col_max)


def _apply_consistent_crop_standalone(pic_folder, margin=20, background_color=(255, 255, 255)):
    """
    Apply consistent cropping to all images in a folder (standalone version).
    
    This finds the maximum extent of content across all frames and crops
    all frames to the same unified bounds + margin.
    
    Parameters
    ----------
    pic_folder : str
        Folder containing the frame images (deg_*.jpeg)
    margin : int
        Margin to preserve around content
    background_color : tuple
        RGB tuple for background color
        
    Returns
    -------
    tuple or None
        (width, height) of the cropped images, or None if failed
    """
    from PIL import Image
    import glob
    
    # Find all JPEG images in folder
    image_paths = sorted(glob.glob(os.path.join(pic_folder, 'deg_*.jpeg')))
    
    if not image_paths:
        return None
    
    # Sample frames for efficiency (every 10 degrees for 360 rotation)
    sample_count = min(36, len(image_paths))
    if sample_count < len(image_paths):
        indices = [int(i * len(image_paths) / sample_count) for i in range(sample_count)]
        sampled_paths = [image_paths[i] for i in indices]
    else:
        sampled_paths = image_paths
    
    # Compute unified bounds across sampled frames
    unified_row_min = None
    unified_row_max = None
    unified_col_min = None
    unified_col_max = None
    
    for path in sampled_paths:
        try:
            with Image.open(path) as img:
                bounds = _detect_content_bounds_standalone(img, background_color)
            
            if bounds is None:
                continue
                
            row_min, row_max, col_min, col_max = bounds
            
            if unified_row_min is None:
                unified_row_min = row_min
                unified_row_max = row_max
                unified_col_min = col_min
                unified_col_max = col_max
            else:
                unified_row_min = min(unified_row_min, row_min)
                unified_row_max = max(unified_row_max, row_max)
                unified_col_min = min(unified_col_min, col_min)
                unified_col_max = max(unified_col_max, col_max)
        except Exception:
            continue
    
    if unified_row_min is None:
        return None
    
    # Get image dimensions from first image
    with Image.open(image_paths[0]) as img:
        img_height = img.height
        img_width = img.width
    
    # Add margin to unified bounds
    row_min = max(0, unified_row_min - margin)
    row_max = min(img_height - 1, unified_row_max + margin)
    col_min = max(0, unified_col_min - margin)
    col_max = min(img_width - 1, unified_col_max + margin)
    
    final_width = col_max - col_min + 1
    final_height = row_max - row_min + 1
    
    # Apply consistent crop to all images
    for path in image_paths:
        try:
            with Image.open(path) as img:
                cropped = img.crop((col_min, row_min, col_max + 1, row_max + 1))
                if cropped.mode == 'RGBA':
                    rgb_img = Image.new('RGB', cropped.size, background_color)
                    rgb_img.paste(cropped, mask=cropped.split()[3])
                    cropped = rgb_img
                elif cropped.mode != 'RGB':
                    cropped = cropped.convert('RGB')
                cropped.save(path, 'JPEG', quality=95)
        except Exception:
            pass
    
    return (final_width, final_height)


def export_video_from_html(html_file, fps=30, degree_per_frame=1.0, rotate='horizontal',
                           output_dir=None, use_existing_images=True, 
                           export_gif=True, gif_scale=0.2, gif_optimize=True,
                           auto_crop=False, crop_margin=30, **kwargs):
    '''
    Standalone function to export a rotating video from an existing Plotly HTML file.
    
    This function does NOT require VisualizeSkeleton initialization or NeuPrint client.
    It directly loads the HTML figure and renders the video.
    
    Parameters
    ----------
    html_file : str
        Path to existing Plotly HTML file to load figure data from.
    fps : int, default 30
        Frames per second for the output video.
    degree_per_frame : float, default 1.0
        Rotation angle in degrees per frame.
        - 1.0 → 360 frames for full rotation (12 sec video at 30 fps)
        - 2.0 → 180 frames for full rotation (6 sec video at 30 fps)
    rotate : str, default 'horizontal'
        Rotation direction: 'horizontal' or 'vertical'.
    output_dir : str, optional
        Directory to save video output. If None, uses the directory containing html_file.
    use_existing_images : bool, default True
        If True, reuse cached images from previous export if available.
    export_gif : bool, default True
        If True, automatically convert videos to GIF format after export.
    gif_scale : float, default 0.2
        Scale factor for GIF resolution (0.1-1.0). Lower values = smaller file size.
    gif_optimize : bool, default True
        Enable GIF compression optimization for smaller file sizes.
    auto_crop : bool, default False
        If True, auto-crop frames to content bounds with consistent sizing across all frames.
        This ensures uniform frame dimensions during rotation for smooth video playback.
    crop_margin : int, default 30
        Margin in pixels around content when auto_crop is enabled.
    **kwargs : dict
        Additional arguments for plotly write_image():
        - scale : int, default 2
        - width : int, default 1200
        - height : int, default 900
    
    Returns
    -------
    int
        0 on success, 1 on failure
    
    Examples
    --------
    # Basic usage - output to same directory as HTML file
    from visualize_skeleton import export_video_from_html
    export_video_from_html('/path/to/my_neurons.html')
    
    # Custom settings
    export_video_from_html(
        '/path/to/my_neurons.html',
        fps=60,
        degree_per_frame=0.5,  # Slower rotation
        rotate='vertical',
        scale=4  # Higher quality
    )
    
    # Specify output directory
    export_video_from_html(
        '/path/to/my_neurons.html',
        output_dir='/path/to/output/'
    )
    '''
    import plotly.io as pio
    import plotly.graph_objects as go
    import cv2
    import shutil
    import time
    
    # Validate input
    if not os.path.exists(html_file):
        raise FileNotFoundError(f'HTML file not found: {html_file}')
    
    # Determine output directory
    if output_dir is None:
        save_folder = os.path.dirname(os.path.abspath(html_file))
    else:
        save_folder = output_dir
        os.makedirs(save_folder, exist_ok=True)
    
    saveas = os.path.splitext(os.path.basename(html_file))[0]
    
    # Handle rotate parameter
    if rotate == 'horizontal':
        rotate_plane = 'xz'
    elif rotate == 'vertical':
        rotate_plane = 'yz'
    else:
        rotate_plane = 'xz'
    
    # Set defaults
    view_direction = kwargs.pop('view_direction', (1, -1))
    view_distance = kwargs.pop('view_distance', 2.2)
    synapse_size = kwargs.pop('synapse_size', 1)
    
    if kwargs.get('scale') is None and kwargs.get('width') is None and kwargs.get('height') is None:
        kwargs['scale'] = 2
    
    # Load figure from HTML
    print(f'📂 Loading figure from: {html_file}')
    try:
        fig_loaded = pio.read_html(html_file)
        fig_traces = fig_loaded.data
        print(f'✓ Loaded {len(fig_traces)} traces from HTML file')
    except Exception as e:
        raise RuntimeError(f'Failed to load figure from HTML: {e}')
    
    # Configure figure for video
    for trace in fig_traces:
        trace.showlegend = False
        if hasattr(trace, 'marker'):
            trace.marker.size = synapse_size
    
    fig_layout = go.Layout(margin=dict(l=1, r=1, b=1, t=1, pad=0))
    fig_new = go.Figure(data=fig_traces, layout=fig_layout)
    
    fig_new.update_layout(
        sliders=[],
        scene=dict(
            dragmode='orbit',
            xaxis={'visible': False},
            yaxis={'visible': False},
            zaxis={'visible': False},
        ),
        scene_camera=dict(
            up=dict(x=0, y=-1, z=0),
            eye=dict(x=0, y=0, z=-view_distance),
        ),
    )
    
    # Set up image folder
    pic_folder = os.path.join(save_folder, f'pics_{fps}fps_{rotate_plane}')
    
    # Calculate rotation steps
    step = degree_per_frame
    steps_to_write = np.linspace(0, 360, int(360/step), endpoint=False)
    
    # Check for existing images
    if use_existing_images and os.path.exists(pic_folder):
        existing_images = [f for f in os.listdir(pic_folder) if f.endswith('.jpeg')]
        if len(existing_images) == len(steps_to_write):
            print(f'✓ Using {len(existing_images)} existing images from {pic_folder}')
        else:
            print(f'⚠️  Found {len(existing_images)} images but need {len(steps_to_write)}, re-rendering...')
            use_existing_images = False
    else:
        use_existing_images = False
    
    # Render images if needed
    if not use_existing_images:
        if os.path.exists(pic_folder):
            shutil.rmtree(pic_folder)
        os.makedirs(pic_folder)
        
        if 'width' not in kwargs:
            kwargs['width'] = 1200
        if 'height' not in kwargs:
            kwargs['height'] = 900
        
        print(f'🎬 Rendering {len(steps_to_write)} frames at {fps} fps...')
        t0 = time.time()
        
        for i, deg in enumerate(steps_to_write):
            rad_i = np.deg2rad(deg)
            x = view_distance * np.sin(rad_i) * view_direction[0]
            y = view_distance * np.cos(rad_i) * view_direction[1]
            
            if rotate_plane == 'xy':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=y, z=0)))
            elif rotate_plane == 'yz':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=0, y=x, z=y)))
            elif rotate_plane == 'xz':
                fig_new.update_layout(scene_camera=dict(eye=dict(x=x, y=0, z=y)))
            
            fig_path = os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg')
            
            try:
                fig_new.write_image(fig_path, **kwargs)
            except Exception as e:
                print(f'\n⚠️  Frame {i+1} failed: {e}')
                if i == 0:
                    print('   Try reducing "scale" (e.g. scale=1)')
                    return 1
            
            elapsed = time.time() - t0
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(steps_to_write) - i - 1)
            print(f'\r  Frame {i+1}/{len(steps_to_write)} | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s', end='  ', flush=True)
        
        print('\n✓ Image rendering complete')
        
        # Apply consistent cropping if auto_crop is enabled
        if auto_crop:
            print(f'   Applying consistent auto-crop across all frames...')
            crop_result = _apply_consistent_crop_standalone(pic_folder, margin=crop_margin)
            if crop_result:
                crop_w, crop_h = crop_result
                print(f'   ✓ All frames cropped consistently to {crop_w}x{crop_h}')
            else:
                print(f'   ⚠️  Auto-crop failed, using original frame sizes')
    
    # Generate videos
    print(f'\nGenerating videos...')
    imglist = os.listdir(pic_folder)
    img_eg = cv2.imread(os.path.join(pic_folder, imglist[0]))
    height, width, layers = img_eg.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    # Add rotation direction indicator to filename
    rotation_suffix = 'h' if rotate_plane == 'xz' else 'v' if rotate_plane == 'yz' else rotate_plane
    
    # Forward video
    video_path_forward = os.path.join(save_folder, f'{saveas}_video_{rotation_suffix}_forward.mp4')
    out = cv2.VideoWriter(video_path_forward, fourcc, fps, frameSize=(width, height))
    for deg in steps_to_write:
        img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
        out.write(img)
    out.release()
    print(f'✓ Forward video: {video_path_forward}')
    
    # Backward video
    video_path_backward = os.path.join(save_folder, f'{saveas}_video_{rotation_suffix}_backward.mp4')
    out = cv2.VideoWriter(video_path_backward, fourcc, fps, frameSize=(width, height))
    for deg in steps_to_write[::-1]:
        img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
        out.write(img)
    out.release()
    print(f'✓ Backward video: {video_path_backward}')
    
    print(f'\n✅ Video export complete!')
    
    # Convert to GIF if requested
    if export_gif:
        print(f'\n🎞️  Converting videos to GIF format...')
        print(f'   Scale: {gif_scale} | Optimize: {gif_optimize}')
        
        # Convert forward video to GIF
        gif_path_forward = video_path_forward.replace('.mp4', '.gif')
        try:
            video2gif(
                video_path_forward,
                gif_path_forward,
                fps=fps,
                scale=gif_scale,
                optimize=gif_optimize
            )
            print(f'   ✓ Forward GIF: {gif_path_forward}')
        except Exception as e:
            print(f'   ⚠️  Forward GIF conversion failed: {e}')
        
        # Convert backward video to GIF
        gif_path_backward = video_path_backward.replace('.mp4', '.gif')
        try:
            video2gif(
                video_path_backward,
                gif_path_backward,
                fps=fps,
                scale=gif_scale,
                optimize=gif_optimize
            )
            print(f'   ✓ Backward GIF: {gif_path_backward}')
        except Exception as e:
            print(f'   ⚠️  Backward GIF conversion failed: {e}')
    
    return 0


def export_video_webdriver(
    html_file: str,
    fps: int = 30,
    degree_per_frame: float = 1.0,
    rotate: str = 'horizontal',
    output_dir: str = None,
    width: int = 1200,
    height: int = 900,
    scale: int = 2,
    view_distance: float = 2.2,
    export_gif: bool = True,
    gif_scale: float = 0.2,
    gif_optimize: bool = True,
    timeout: int = 120,
    auto_crop: bool = False,
    crop_margin: int = 30,
) -> int:
    """
    Export a rotating video from an existing Plotly HTML file using WebDriver.
    
    This is an EFFICIENT alternative to export_video_from_html() that:
    - Opens the browser ONCE and keeps it open
    - Rotates the camera using JavaScript (no figure regeneration)
    - Takes screenshots in series without reopening the browser
    
    This method is significantly faster for large/complex 3D figures because:
    1. The HTML only needs to load once (not per-frame)
    2. Camera rotation uses JavaScript instead of Python figure update + kaleido export
    3. Works with WebGL for smooth rendering
    
    Parameters
    ----------
    html_file : str
        Path to existing Plotly HTML file to load.
    fps : int, default 30
        Frames per second for the output video.
    degree_per_frame : float, default 1.0
        Rotation angle in degrees per frame.
        - 1.0 → 360 frames for full rotation (12 sec video at 30 fps)
        - 2.0 → 180 frames for full rotation (6 sec video at 30 fps)
    rotate : str, default 'horizontal'
        Rotation direction: 'horizontal' or 'vertical'.
    output_dir : str, optional
        Directory to save video output. If None, uses the directory containing html_file.
    width : int, default 1200
        Browser viewport width.
    height : int, default 900
        Browser viewport height.
    scale : int, default 2
        Scale factor for screenshot resolution (actual size = width*scale x height*scale).
    view_distance : float, default 2.2
        Camera distance from the center (affects zoom level).
    export_gif : bool, default True
        If True, automatically convert videos to GIF format after export.
    gif_scale : float, default 0.2
        Scale factor for GIF resolution (0.1-1.0).
    gif_optimize : bool, default True
        Enable GIF compression optimization.
    timeout : int, default 120
        Maximum time in seconds to wait for page load.
    auto_crop : bool, default False
        If True, auto-crop frames to content bounds with consistent sizing across all frames.
        This ensures uniform frame dimensions during rotation for smooth video playback.
    crop_margin : int, default 30
        Margin in pixels around content when auto_crop is enabled.
    
    Returns
    -------
    int
        0 on success, 1 on failure
    
    Notes
    -----
    Requires: selenium, webdriver-manager
    
    On macOS, headless Chrome doesn't support WebGL, so we use an offscreen
    window (positioned at -10000,-10000) instead of true headless mode.
    
    Examples
    --------
    # Basic usage
    from visualize_skeleton import export_video_webdriver
    export_video_webdriver('/path/to/my_neurons.html')
    
    # Faster rotation, higher quality
    export_video_webdriver(
        '/path/to/my_neurons.html',
        fps=60,
        degree_per_frame=2.0,  # Faster rotation
        scale=3  # Higher quality
    )
    
    # Vertical rotation
    export_video_webdriver(
        '/path/to/my_neurons.html',
        rotate='vertical',
        view_distance=2.5
    )
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
    except ImportError:
        print("❌ Error: selenium is required for export_video_webdriver()")
        print("   Install with: pip install selenium webdriver-manager")
        return 1
    
    import shutil
    import time
    
    # Validate input
    if not os.path.exists(html_file):
        raise FileNotFoundError(f'HTML file not found: {html_file}')
    
    # Determine output directory
    if output_dir is None:
        save_folder = os.path.dirname(os.path.abspath(html_file))
    else:
        save_folder = output_dir
        os.makedirs(save_folder, exist_ok=True)
    
    saveas = os.path.splitext(os.path.basename(html_file))[0]
    
    # Handle rotate parameter
    if rotate == 'horizontal':
        rotate_plane = 'xz'
    elif rotate == 'vertical':
        rotate_plane = 'yz'
    else:
        rotate_plane = 'xz'
    
    # Calculate rotation steps
    step = degree_per_frame
    steps_to_write = np.linspace(0, 360, int(360/step), endpoint=False)
    
    # Set up image folder
    pic_folder = os.path.join(save_folder, f'pics_{fps}fps_{rotate_plane}_webdriver')
    if os.path.exists(pic_folder):
        shutil.rmtree(pic_folder)
    os.makedirs(pic_folder)
    
    # Calculate actual browser dimensions
    actual_width = width * scale
    actual_height = height * scale
    
    # Set up Chrome options
    # Use --headless=new (Chrome 109+) for WebGL support in headless mode
    chrome_options = ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument(f'--window-size={actual_width},{actual_height}')
    chrome_options.add_argument('--headless=new')  # Modern headless with WebGL support
    
    # Initialize ChromeDriver using webdriver-manager (cross-platform)
    driver = None
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except ImportError:
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"❌ Error: Could not initialize Chrome WebDriver: {e}")
            print(f"   Install dependencies: pip install selenium webdriver-manager")
            return 1
    except Exception as e:
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e2:
            print(f"❌ Error: Could not initialize Chrome WebDriver: {e2}")
            print(f"   Ensure Chrome 109+ is installed and webdriver-manager is up to date")
            return 1
    
    try:
        print(f'📂 Loading HTML file: {html_file}')
        file_url = f'file://{os.path.abspath(html_file)}'
        driver.get(file_url)
        
        # Wait for Plotly to render
        print(f'   Waiting for Plotly to render...')
        wait = WebDriverWait(driver, timeout)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "plotly")))
        
        # Additional wait for WebGL rendering
        time.sleep(3)
        print(f'✓ Page loaded and rendered')
        
        # JavaScript to update camera position
        # Plotly stores camera in layout.scene.camera.eye
        js_set_camera = """
        var gd = document.querySelector('.js-plotly-plot');
        if (gd && gd.layout && gd.layout.scene) {
            Plotly.relayout(gd, {
                'scene.camera.eye': {x: %f, y: %f, z: %f},
                'scene.camera.up': {x: 0, y: -1, z: 0}
            });
        }
        """
        
        print(f'🎬 Rendering {len(steps_to_write)} frames at {fps} fps...')
        t0 = time.time()
        
        for i, deg in enumerate(steps_to_write):
            rad_i = np.deg2rad(deg)
            
            # Add small offset to avoid gimbal lock at axis-aligned positions
            # This prevents camera flipping at 90, 180, 270 degrees
            offset = view_distance * 0.01  # 1% offset
            
            # Use view_direction (1, -1) to match original export_video_from_html behavior
            # The -1 for cos component ensures consistent rotation direction
            sin_component = view_distance * np.sin(rad_i)  # * 1
            cos_component = view_distance * np.cos(rad_i) * (-1)  # * -1
            
            if rotate_plane == 'xz':  # Horizontal rotation
                eye_x = sin_component
                eye_y = offset  # Small offset instead of 0
                eye_z = cos_component
            elif rotate_plane == 'yz':  # Vertical rotation
                eye_x = offset  # Small offset instead of 0
                eye_y = sin_component
                eye_z = cos_component
            else:  # xy plane
                eye_x = sin_component
                eye_y = cos_component
                eye_z = offset  # Small offset instead of 0
            
            # Update camera via JavaScript
            driver.execute_script(js_set_camera % (eye_x, eye_y, eye_z))
            
            # Brief wait for rendering
            time.sleep(0.1)
            
            # Take screenshot
            frame_path = os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg')
            
            # Get screenshot as PNG, then convert to JPEG
            screenshot_png = os.path.join(pic_folder, f'temp_{deg:.1f}.png')
            driver.save_screenshot(screenshot_png)
            
            # Convert PNG to JPEG for consistency with existing code
            from PIL import Image
            img = Image.open(screenshot_png)
            img = img.convert('RGB')
            img.save(frame_path, 'JPEG', quality=95)
            os.remove(screenshot_png)
            
            # Progress update
            elapsed = time.time() - t0
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(steps_to_write) - i - 1)
            print(f'\r   Frame {i+1}/{len(steps_to_write)} | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s', end='  ', flush=True)
        
        print('\n✓ Image rendering complete')
        
        # Apply consistent cropping if auto_crop is enabled
        if auto_crop:
            print(f'   Applying consistent auto-crop across all frames...')
            crop_result = _apply_consistent_crop_standalone(pic_folder, margin=crop_margin)
            if crop_result:
                crop_w, crop_h = crop_result
                print(f'   ✓ All frames cropped consistently to {crop_w}x{crop_h}')
            else:
                print(f'   ⚠️  Auto-crop failed, using original frame sizes')
        
    finally:
        driver.quit()
    
    # Generate videos
    print(f'\nGenerating videos...')
    imglist = os.listdir(pic_folder)
    imglist = [f for f in imglist if f.endswith('.jpeg')]
    
    img_eg = cv2.imread(os.path.join(pic_folder, imglist[0]))
    frame_height, frame_width, layers = img_eg.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    # Forward video
    video_path_forward = os.path.join(save_folder, f'{saveas}_video_forward_webdriver.mp4')
    out = cv2.VideoWriter(video_path_forward, fourcc, fps, frameSize=(frame_width, frame_height))
    for deg in steps_to_write:
        img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
        out.write(img)
    out.release()
    print(f'✓ Forward video: {video_path_forward}')
    
    # Backward video
    video_path_backward = os.path.join(save_folder, f'{saveas}_video_backward_webdriver.mp4')
    out = cv2.VideoWriter(video_path_backward, fourcc, fps, frameSize=(frame_width, frame_height))
    for deg in steps_to_write[::-1]:
        img = cv2.imread(os.path.join(pic_folder, f'deg_{deg:.1f}.jpeg'))
        out.write(img)
    out.release()
    print(f'✓ Backward video: {video_path_backward}')
    
    print(f'\n✅ Video export complete!')
    
    # Convert to GIF if requested
    if export_gif:
        print(f'\n🎞️  Converting videos to GIF format...')
        print(f'   Scale: {gif_scale} | Optimize: {gif_optimize}')
        
        # Convert forward video to GIF
        gif_path_forward = video_path_forward.replace('.mp4', '.gif')
        try:
            video2gif(
                video_path_forward,
                gif_path_forward,
                fps=fps,
                scale=gif_scale,
                optimize=gif_optimize
            )
            print(f'   ✓ Forward GIF: {gif_path_forward}')
        except Exception as e:
            print(f'   ⚠️  Forward GIF conversion failed: {e}')
        
        # Convert backward video to GIF
        gif_path_backward = video_path_backward.replace('.mp4', '.gif')
        try:
            video2gif(
                video_path_backward,
                gif_path_backward,
                fps=fps,
                scale=gif_scale,
                optimize=gif_optimize
            )
            print(f'   ✓ Backward GIF: {gif_path_backward}')
        except Exception as e:
            print(f'   ⚠️  Backward GIF conversion failed: {e}')
    
    return 0


def export_png_webdriver(
    html_file: str,
    output_path: str = None,
    width: int = 1200,
    height: int = 900,
    scale: int = 3,
    timeout: int = 60,
) -> str:
    """
    Export a single PNG from an existing Plotly HTML file using WebDriver.
    
    This is an alternative to kaleido-based PNG export that works reliably
    with WebGL for complex 3D scenes.
    
    Parameters
    ----------
    html_file : str
        Path to existing Plotly HTML file to load.
    output_path : str, optional
        Path for the output PNG file. If None, uses the same path as html_file
        with .png extension.
    width : int, default 1200
        Browser viewport width.
    height : int, default 900
        Browser viewport height.
    scale : int, default 3
        Scale factor for screenshot resolution (actual size = width*scale x height*scale).
    timeout : int, default 60
        Maximum time in seconds to wait for page load.
    
    Returns
    -------
    str
        Path to the created PNG file on success, None on failure.
    
    Notes
    -----
    Requires: selenium, webdriver-manager
    
    On macOS, headless Chrome doesn't support WebGL, so we use an offscreen
    window (positioned at -10000,-10000) instead of true headless mode.
    
    Examples
    --------
    # Basic usage
    from visualize_skeleton import export_png_webdriver
    export_png_webdriver('/path/to/my_neurons.html')
    
    # Higher resolution
    export_png_webdriver('/path/to/my_neurons.html', scale=4)
    
    # Custom output path
    export_png_webdriver('/path/to/my_neurons.html', output_path='/path/to/output.png')
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
    except ImportError:
        print("❌ Error: selenium is required for export_png_webdriver()")
        print("   Install with: pip install selenium webdriver-manager")
        return None
    
    import time
    
    # Validate input
    if not os.path.exists(html_file):
        raise FileNotFoundError(f'HTML file not found: {html_file}')
    
    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(html_file)[0] + '_webdriver.png'
    
    # Calculate actual browser dimensions
    actual_width = width * scale
    actual_height = height * scale
    
    # Set up Chrome options
    # Use --headless=new (Chrome 109+) for WebGL support in headless mode
    chrome_options = ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument(f'--window-size={actual_width},{actual_height}')
    chrome_options.add_argument('--headless=new')  # Modern headless with WebGL support
    
    # Initialize ChromeDriver using webdriver-manager (cross-platform)
    driver = None
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except ImportError:
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"❌ Error: Could not initialize Chrome WebDriver: {e}")
            print(f"   Install dependencies: pip install selenium webdriver-manager")
            return None
    except Exception as e:
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e2:
            print(f"❌ Error: Could not initialize Chrome WebDriver: {e2}")
            print(f"   Ensure Chrome 109+ is installed and webdriver-manager is up to date")
            return None
    
    try:
        print(f'📂 Loading HTML file: {html_file}')
        file_url = f'file://{os.path.abspath(html_file)}'
        driver.get(file_url)
        
        # Wait for Plotly to render
        print(f'   Waiting for Plotly to render...')
        wait = WebDriverWait(driver, timeout)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "plotly")))
        
        # Additional wait for WebGL rendering
        time.sleep(3)
        print(f'✓ Page loaded and rendered')
        
        # Take screenshot
        driver.save_screenshot(output_path)
        
        print(f'✅ PNG exported: {output_path}')
        
        # Report file size
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f'   Resolution: {actual_width}x{actual_height}')
        print(f'   File size: {output_size:.2f} MB')
        
        return output_path
        
    except Exception as e:
        print(f'❌ Error during export: {e}')
        return None
        
    finally:
        driver.quit()


def video2gif(
    input_video: str,
    output_gif: str = None,
    fps: int = None,
    scale: float = 1.0,
    optimize: bool = True,
    loop: int = 0,
) -> str:
    """
    Convert a video file (MP4) to an animated GIF with adjustable compression and fps.
    
    This is a static helper function that can be called independently.
    
    Parameters
    ----------
    input_video : str
        Path to the input video file (MP4 or other formats supported by cv2).
    output_gif : str, optional
        Path for the output GIF file. If None, uses the same path as input with .gif extension.
    fps : int, optional
        Target frames per second for the GIF. If None, uses the original video fps.
        Lower fps = smaller file size, choppier animation.
    scale : float, default 1.0
        Scale factor for the output dimensions (0.0-1.0 for compression).
        - 1.0: Original resolution
        - 0.5: Half resolution (75% file size reduction)
        - 0.25: Quarter resolution
    optimize : bool, default True
        Whether to optimize the GIF palette for smaller file size.
        When True and ffmpeg is on PATH, uses ffmpeg palettegen/paletteuse
        (scene-wide palette, ~10-15% smaller than PIL on rendered scenes).
        Otherwise falls back to PIL's optimize and disposal settings.
    loop : int, default 0
        Number of times the GIF should loop.
        - 0: Loop forever
        - 1: Play once
        - n: Loop n times
    
    Returns
    -------
    str
        Path to the created GIF file.
    
    Examples
    --------
    # Basic conversion
    from visualize_skeleton import video2gif
    video2gif('/path/to/video.mp4')
    
    # With compression (half size, 15 fps)
    video2gif('/path/to/video.mp4', fps=15, scale=0.5)
    
    # Custom output path
    video2gif('/path/to/video.mp4', output_gif='/path/to/output.gif', scale=0.75)
    """
    from PIL import Image
    import shutil
    
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    # Set output path
    if output_gif is None:
        output_gif = os.path.splitext(input_video)[0] + '.gif'
    
    # Open video with cv2
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_video}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use original fps if not specified
    target_fps = fps if fps is not None else int(original_fps)
    
    # Calculate frame skip for target fps
    if target_fps >= original_fps:
        frame_skip = 1
    else:
        frame_skip = int(original_fps / target_fps)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    print(f'🎬 Converting video to GIF...')
    print(f'   Input: {input_video}')
    print(f'   Original: {width}x{height} @ {original_fps:.1f} fps, {frame_count} frames')
    print(f'   Output: {new_width}x{new_height} @ {target_fps} fps')
    
    # Preferred path: ffmpeg palettegen/paletteuse builds one scene-wide
    # optimized palette, producing smaller GIFs than PIL's per-frame
    # adaptive palettes. Fall back to PIL if ffmpeg is missing or fails.
    if optimize and shutil.which('ffmpeg') is not None:
        try:
            print(f'   Using ffmpeg palettegen/paletteuse...')
            _video2gif_ffmpeg(
                input_video,
                output_gif,
                fps=target_fps if target_fps < original_fps else None,
                scale=scale,
                loop=loop,
            )
            input_size = os.path.getsize(input_video) / (1024 * 1024)
            output_size = os.path.getsize(output_gif) / (1024 * 1024)
            print(f'✅ GIF created: {output_gif}')
            print(f'   Input size: {input_size:.2f} MB')
            print(f'   Output size: {output_size:.2f} MB')
            print(f'   Compression ratio: {output_size/input_size:.2%}')
            return output_gif
        except Exception as e:
            print(f'   ⚠️  ffmpeg conversion failed ({e}); falling back to PIL')
    
    # Read frames
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if scale != 1.0:
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), 
                                       interpolation=cv2.INTER_AREA)
            
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
        
        frame_idx += 1
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames extracted from video")
    
    print(f'   Extracted {len(frames)} frames')
    
    # Calculate frame duration in milliseconds
    duration = int(1000 / target_fps)
    
    # Save as GIF with progress bar
    # PIL's save doesn't have a progress callback, so we save frame by frame
    print(f'   Saving GIF ({len(frames)} frames)...')
    
    import io
    
    # For large GIFs, save incrementally to show progress
    total_frames = len(frames)
    
    # Use a temporary buffer approach with progress reporting
    print(f'   [', end='', flush=True)
    bar_width = 40
    
    # We'll save all at once but show a simple progress indicator during optimization
    # Since PIL doesn't support progress callbacks, we simulate with frame processing info
    for i, frame in enumerate(frames):
        # Show progress bar
        progress = (i + 1) / total_frames
        filled = int(bar_width * progress)
        print(f'\r   [{"="*filled}{">" if filled < bar_width else ""}{" "*(bar_width-filled-1 if filled < bar_width else 0)}] {i+1}/{total_frames}', end='', flush=True)
    
    print(f'\r   [{"="*bar_width}] Optimizing...', end='', flush=True)
    
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=optimize,
        disposal=2,  # Clear frame before drawing next (better for animations)
    )
    print(f'\r   [{"="*bar_width}] Done!          ')
    
    # Report file sizes
    input_size = os.path.getsize(input_video) / (1024 * 1024)
    output_size = os.path.getsize(output_gif) / (1024 * 1024)
    
    print(f'✅ GIF created: {output_gif}')
    print(f'   Input size: {input_size:.2f} MB')
    print(f'   Output size: {output_size:.2f} MB')
    print(f'   Compression ratio: {output_size/input_size:.2%}')
    
    return output_gif


def _video2gif_ffmpeg(input_video, output_gif, fps=None, scale=1.0, loop=0):
    """
    Convert a video to GIF using ffmpeg palettegen/paletteuse.
    
    Two-pass conversion: pass 1 builds a scene-wide 256-color palette
    weighted toward pixels that change between frames (stats_mode=diff),
    pass 2 renders all frames against that palette. This yields smaller
    files than PIL's per-frame adaptive palettes.
    
    Raises on any ffmpeg failure so callers can fall back to PIL.
    """
    import subprocess
    import tempfile
    
    filters = []
    if scale != 1.0:
        # Keep dimensions even for encoder safety
        filters.append(
            f'scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=lanczos'
        )
    if fps is not None:
        filters.append(f'fps={fps}')
    
    palette_path = os.path.join(
        tempfile.gettempdir(), f'drocat_gif_palette_{os.getpid()}.png'
    )
    try:
        # Pass 1: build the palette
        r = subprocess.run(
            ['ffmpeg', '-y', '-v', 'error', '-i', input_video,
             '-vf', ','.join(filters + ['palettegen=stats_mode=diff']),
             palette_path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f'palettegen failed: {r.stderr.strip().splitlines()[-1] if r.stderr else r.returncode}')
        
        # Pass 2: render frames against the palette
        if filters:
            lavfi = f"{','.join(filters)}[x];[x][1:v]paletteuse=dither=sierra2_4a"
        else:
            lavfi = 'paletteuse=dither=sierra2_4a'
        r = subprocess.run(
            ['ffmpeg', '-y', '-v', 'error', '-i', input_video, '-i', palette_path,
             '-lavfi', lavfi, '-loop', str(loop), output_gif],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f'paletteuse failed: {r.stderr.strip().splitlines()[-1] if r.stderr else r.returncode}')
    finally:
        if os.path.exists(palette_path):
            os.remove(palette_path)


def img2pptx(
    input_path: str | list,
    output_pptx: str = None,
    images_per_slide: tuple = (4, 2),
    slide_title: str = None,
    slide_size: str = 'widescreen',
    margin: float = 0.3,
    title_height: int = 60,
    label_fontsize: int = 20,
    title_fontsize: int = 24,
    label_position: str = 'below',
    label_overlay_alpha: float = 0.7,
    cell_padding: float = 0.05,
    include_subfolders: bool = False,
    group_by_subfolder: bool = True,
    font_color: tuple = (0, 0, 0),
    font: str = 'Arial',
    background_color: tuple | str = None,
) -> str:
    """
    Aggregate images to PowerPoint (PPTX) with proper layout, or convert PDF pages to PPTX.
    
    This function has been moved to src/utils/report_utils.py.
    This wrapper is provided for backward compatibility.
    """
    try:
        from utils.report_utils import img2pptx as _img2pptx
    except ImportError:
        try:
            from src.utils.report_utils import img2pptx as _img2pptx
        except ImportError:
             raise ImportError("Could not import img2pptx from utils.report_utils. Please ensure the file exists.")
    
    return _img2pptx(
        input_path=input_path,
        output_pptx=output_pptx,
        images_per_slide=images_per_slide,
        slide_title=slide_title,
        slide_size=slide_size,
        margin=margin,
        title_height=title_height,
        label_fontsize=label_fontsize,
        title_fontsize=title_fontsize,
        label_position=label_position,
        label_overlay_alpha=label_overlay_alpha,
        cell_padding=cell_padding,
        include_subfolders=include_subfolders,
        group_by_subfolder=group_by_subfolder,
        font_color=font_color,
        font=font,
        background_color=background_color
    )
