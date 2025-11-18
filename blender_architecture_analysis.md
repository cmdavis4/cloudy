# Blender Module Architecture Analysis

## Overview
The blender module is a comprehensive system for creating atmospheric data visualizations in Blender. It orchestrates a complete pipeline: importing time-varying mesh data, managing animations via frame change callbacks, configuring rendering, and managing scene hierarchy.

**Key Entry Point:** `run_atmospheric_animation()` in `blender_run.py`

---

## 1. OVERALL ARCHITECTURE

### Module Structure
```
blender/
├── __init__.py                 # Public API exports
├── blender_run.py             # Main entry point: run_atmospheric_animation()
├── blender_animation.py       # Frame change callbacks & animation logic
├── blender_import.py          # Mesh data loading & importing
├── blender_core.py            # Scene manipulation & utilities
├── blender_camera.py          # Camera setup
├── blender_render.py          # Render settings configuration
└── blender_config.py          # Default configuration constants
```

### Key Components
1. **Entry Point** - `run_atmospheric_animation()`
2. **Callback System** - Frame change handlers via `bpy.app.handlers`
3. **Data Import** - Loading timestamped mesh files (PLY, VDB, OBJ, VTP)
4. **Scene Management** - Collections, objects, materials, and geometry
5. **Animation** - Dynamic mesh updates via callback on each frame

---

## 2. HOW ANIMATION CURRENTLY WORKS WITH CALLBACKS

### Callback Registration Flow

**Location:** `blender_animation.py:20-38`

```python
def register_frame_change_handlers(frame_change_callback: Any) -> None:
    """Register frame change handlers for animation."""
    # Clear existing handlers
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.render_pre.clear()
    bpy.app.handlers.render_post.clear()
    bpy.app.handlers.render_init.clear()

    # Add handlers for both post, in both viewport and render
    bpy.app.handlers.frame_change_post.append(frame_change_callback)
    bpy.app.handlers.render_pre.append(frame_change_callback)
```

### Callback Architecture

**Where Defined:** `blender_run.py:251-260`

```python
@bpy.app.handlers.persistent
def callback_function(scene, depsgraph=None):
    return update_for_new_frame(
        data_directory=data_dir,
        frame_to_time_mapping=frame_to_time_mapping,
        kwargs_data=kwargs_data,
        global_scale=global_scale,
        scene=scene,
        depsgraph=depsgraph,
    )

register_frame_change_handlers(frame_change_callback=callback_function)
```

### What Triggers the Callback
- **Event**: Every frame change during playback or rendering
- **Handlers Triggered**: 
  - `bpy.app.handlers.frame_change_post` (after frame set in viewport)
  - `bpy.app.handlers.render_pre` (before each frame renders)
- **Frequency**: Once per frame (minimum)

### Callback Processing Flow
1. User plays animation or starts render
2. Blender engine changes frame
3. Handler triggers `callback_function(scene, depsgraph)`
4. Calls `update_for_new_frame()` with current context
5. Function updates mesh geometry for the new frame

---

## 3. WHERE `run_atmospheric_animation` IS DEFINED

**File:** `/home/cmdavis4/projects/common/blender/blender_run.py:33-292`

### Function Signature
```python
def run_atmospheric_animation(
    data_dir: PathLike,
    output_dir: PathLike,
    kwargs_data: Dict[str, Dict[str, Any]],
    config: ConfigDict = {},
    use_grass=True,
    simulation_minutes_per_second: float = 5,
    fps: int = 24,
    render: bool = False,
    assets_libraries: list[str] = [],
    global_scale: float | Tuple[float, float, float] = 0.001,
    limit: int | None = None,
    use_time_stretching: bool = False,
    output_format="PNG",
) -> None:
```

### What It Does (Main Responsibilities)

1. **Scene Setup** (line 97)
   - Resets scene to clean state
   - Clears existing handlers and objects

2. **Configuration Management** (lines 99-111)
   - Merges user config with defaults (DEFAULT_BLENDER_CONFIG)
   - Sets render output format (PNG/FFMPEG)

3. **Camera & Render Setup** (lines 135-144)
   - Calls `setup_camera()`
   - Calls `setup_render_settings()` with quality/resolution
   - Sets render engine

4. **Animation Time Analysis** (lines 146-177)
   - Scans data directory for timestamped files
   - Extracts unique simulation times (dt values)
   - Validates even spacing of timesteps
   - Calculates FPS and frames_per_timestep

5. **Frame Mapping Creation** (lines 220-230)
   - Creates `frame_to_time_mapping` list
   - Maps Blender frame numbers (1-based) to simulation times
   - Each simulation timestep gets `frames_per_timestep_int` frames

6. **Scene Collection Structure** (lines 246-249)
   - Creates "data" collection
   - Creates sub-collections for file types: "plys", "vdbs", "objs", "vtps"

7. **Callback Registration** (lines 251-262)
   - Defines `callback_function` with captured variables
   - Registers callback via `register_frame_change_handlers()`

8. **Scene Initialization** (lines 264-271)
   - Sets frame to 1 (triggers initial callback)
   - Creates grass field if `use_grass=True`
   - Sets sky background

9. **Render Execution** (lines 291-292)
   - Optionally starts rendering with `bpy.ops.render.render(animation=True)`

---

## 4. DATA IMPORT AND ORGANIZATION

### How Data is Organized (Time-Varying Objects)

**File:** `/home/cmdavis4/projects/common/blender/blender_import.py`

#### File Naming Convention
Files follow pattern: `dt-YYYYMMDDHHMMSS_category-varname.suffix`

Example: `dt-20230101000000_Rcondensate-var1.ply`

#### Data Import Pipeline
1. **Discovery** (`get_data_filepaths()` line 23-27)
   - Scans directory for files with suffixes: `.ply`, `.vdb`, `.vtk`
   - Returns list of Path objects

2. **Time Extraction** (`run_atmospheric_animation()` line 147-154)
   - Parses datetime from filename using `to_kv_pairs()`
   - Extracts unique times: `animation_times = sorted(list(set([...])))` 
   - Must be evenly spaced (raises error if not)

3. **Single File Import** (`import_single_data_file()` line 93-150)
   - Calls appropriate import function based on suffix
   - Detects newly added objects
   - Extracts metadata:
     - `data_file_suffix`: File type (.ply, .vdb, etc)
     - `category`: From filename (e.g., "Rcondensate")
     - `varname`: Variable name from filename
   - Returns single BlenderObject

4. **Time-Specific Import** (`import_data_for_time()` line 152-229)
   - Scans for all files matching current timestep pattern
   - Filters by category (only imports categories in `kwargs_data`)
   - Filters by variable name if specified
   - Applies global scaling to non-VDB objects
   - Returns dict: `{object_name: BlenderObject}`

#### Data Structure in Scene

**Hierarchy:**
```
Scene Collection
├── Data Collection
│   ├── plys/
│   │   ├── category-varname (permanent mesh updated per frame)
│   │   └── ...
│   ├── vdbs/
│   │   ├── category-varname (permanent mesh updated per frame)
│   │   └── ...
│   ├── objs/
│   └── vtps/
├── date_text (datetime display)
├── grass_plane (if use_grass=True)
└── Camera
```

#### Permanent vs Temporary Objects

**Permanent Objects:**
- Created once on first frame they appear
- Name format: `category-varname` (without timestamp)
- Stored in appropriate collection (plys, vdbs, etc)
- Geometry updated via `update_object_geometry()` each frame

**Temporary Objects:**
- Created during import_single_data_file()
- Have full timestamped name
- Used to extract geometry
- Deleted after updating permanent object

---

## 5. FRAME CHANGE CALLBACKS: REGISTRATION AND USAGE

### Handler Registration Points

**Primary Registration** (`blender_animation.py:20-38`)
```python
def register_frame_change_handlers(frame_change_callback: Any) -> None:
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_post.clear()
    bpy.app.handlers.render_pre.clear()
    bpy.app.handlers.render_post.clear()
    bpy.app.handlers.render_init.clear()

    bpy.app.handlers.frame_change_post.append(frame_change_callback)
    bpy.app.handlers.render_pre.append(frame_change_callback)
```

**Two handlers are used:**
1. `frame_change_post`: Triggers after frame changes in viewport
2. `render_pre`: Triggers before each frame is rendered

**Why both?**
- Viewport playback needs to update visible geometry
- Rendering needs to ensure correct geometry before rendering frame

### Handler Lifecycle

1. **Registration** (blender_run.py:262)
   - Called once during setup
   - Clears all existing handlers
   - Appends new callback

2. **Execution** (Every frame)
   - Frame changes to N
   - Handlers fire: `callback_function(scene, depsgraph)`
   - `update_for_new_frame()` executes
   - Meshes updated for frame N

3. **Cleanup** (blender_core.py:63-89)
   - `reset_scene()` clears all handlers when resetting

### Handler Behavior Notes

- **Persistent Decorator**: `@bpy.app.handlers.persistent` keeps callback alive
- **Global Capture**: Callback captures `data_dir`, `frame_to_time_mapping`, etc from enclosing scope
- **Depsgraph**: Optional parameter for dependency graph updates
- **Scene Parameter**: Always receives current scene object

---

## 6. COMPLETE WORKFLOW FROM SETUP TO EXECUTION

### Initialization Phase

```
run_atmospheric_animation() called
    ↓
1. Reset Scene (blender_core.reset_scene)
   - Clear handlers
   - Delete non-camera objects
   - Clear collections
   
2. Parse Configuration
   - Merge user config with defaults
   - Extract resolution, engine, quality
   
3. Analyze Time Data
   - get_data_filepaths(data_dir) → list of files
   - Extract unique times from filenames
   - Validate even spacing
   - Calculate FPS and frames per timestep
   
4. Create Frame Mapping
   - frame_to_time_mapping[frame_index] = simulation_time
   - Blender frame 1 → simulation_times[0]
   - Blender frame 2 → simulation_times[0 or 1, depending on frames_per_timestep]
   
5. Setup Scene Structure
   - setup_camera() → Camera object
   - setup_render_settings(resolution, engine, quality)
   - create_collection("data") → data collection
   - create_collection("plys" | "vdbs" | "objs" | "vtps")
   - Add date_text object and position it
   
6. Register Animation Callback
   - Define callback_function with @bpy.app.handlers.persistent
   - Callback captures: data_dir, frame_to_time_mapping, kwargs_data, global_scale
   - register_frame_change_handlers(callback_function)
   
7. Initialize Scene
   - bpy.context.scene.frame_set(1) → triggers callback for frame 1
   - Callback imports data for frame 1
   
8. Setup Grass (optional)
   - create_grass_field() if use_grass=True
   
9. Set Background
   - set_sky_background("sun")
   
10. Deselect All Objects
```

### Playback/Render Phase

```
User plays animation or calls: bpy.ops.render.render(animation=True)
    ↓
For each frame in range(frame_start, frame_end + 1):
    ↓
    A. Frame Change Event
       Blender sets current_frame to N
       ↓
    B. Handler Triggers (frame_change_post and/or render_pre)
       callback_function(scene, depsgraph)
       ↓
    C. update_for_new_frame() executes:
       - Get frame_index = current_frame - 1
       - Get simulation_time = frame_to_time_mapping[frame_index]
       - Convert time to string for file lookup
       - Update date_text display
       ↓
    D. Import Data for Time
       import_data_for_time(data_dir, simulation_time, kwargs_data, global_scale)
       ↓
    E. For each file matching this timestep:
       - Import mesh (creates temporary object)
       - Get or create permanent object
       - Update permanent object geometry from temporary
       - Delete temporary object
       ↓
    F. Geometry Updated
       Viewport or renderer sees updated geometry
       ↓
    G. Repeat for next frame
```

### Data Flow on Frame Change

```
Frame change triggered
    ↓
callback_function receives:
    - scene (current Blender scene)
    - depsgraph (dependency graph for updates)
    
update_for_new_frame receives:
    - data_directory: /path/to/data
    - frame_to_time_mapping: [dt1, dt1, dt2, dt2, ...]
    - kwargs_data: {"category": {properties}}
    - global_scale: (0.001, 0.001, 0.001)
    - scene: scene object
    - depsgraph: for updates
    
Inside update_for_new_frame:
    - Get current_frame from scene.frame_current
    - Convert to 0-based index: frame_index = current_frame - 1
    - Look up simulation_time: current_time = frame_to_time_mapping[frame_index]
    - Format filename pattern: f"dt-{dt_to_str(current_time)}*"
    - Find matching data files
    
    For each file:
        - Import it → temporary object
        - Extract category and varname
        - Check if in kwargs_data
        - Get permanent object name (category-varname, without timestamp)
        - Try to find existing permanent object
        - If not found:
            - Use temporary as permanent
            - Setup with kwargs
        - If found:
            - Update geometry from temporary
            - Delete temporary
            - Ensure all properties preserved
```

---

## 7. CONFIGURATION FLAGS HANDLING

### Default Configuration

**File:** `blender_config.py:3-14`

```python
DEFAULT_BLENDER_CONFIG: ConfigDict = {
    "frames_per_timestep": 12,
    "grass_field_size": 150,
    "grass_density": 1500,
    "resolution_x": 1920,
    "resolution_y": 1080,
    "render_engine": "BLENDER_EEVEE_NEXT",
    "render_samples": 128,
    "sun_energy": 4,
    "background_color": (0.02, 0.05, 0.2, 1.0),
    "background_strength": 0.4,
}
```

### Configuration Merging

**Location:** `blender_run.py:105-111`

```python
if config:
    # Use our default settings except for what was passed
    passed_config = copy(config)
    config = copy(DEFAULT_BLENDER_CONFIG)
    config.update(passed_config)  # Override defaults with user values
else:
    config = copy(DEFAULT_BLENDER_CONFIG)
```

### Function Parameters (Configuration Points)

**run_atmospheric_animation() parameters:**
- `config`: Dict merged with defaults
- `use_grass`: Boolean flag to create grass field
- `simulation_minutes_per_second`: Timing parameter
- `fps`: Frames per second
- `render`: Boolean to auto-start rendering
- `assets_libraries`: List of library names to load
- `global_scale`: Scaling factor (float or tuple)
- `limit`: Max timesteps to include
- `use_time_stretching`: Boolean for time stretching mode
- `output_format`: "PNG" or "FFMPEG"

### Key Configuration Usage Points

1. **Render Engine** (line 137)
   - `bpy.context.scene.render.engine = config["render_engine"]`

2. **Resolution** (line 142)
   - `resolution=(config.get("resolution_x"), config.get("resolution_y"))`

3. **Quality Settings** (line 143)
   - `quality=config.get("quality", "MEDIUM")`

4. **Grass Field** (line 267)
   - `if use_grass: create_grass_field()`

5. **Time Stretching** (lines 194-218)
   - `if animation and use_time_stretching:`
   - Adjusts frame_map_old and frame_map_new

6. **Output Format** (lines 100-103)
   - Sets `bpy.context.scene.render.image_settings.file_format`
   - Sets FFMPEG format if applicable

---

## 8. KEY DATA STRUCTURES

### frame_to_time_mapping

**Type:** `List[datetime.datetime]`

**Purpose:** Maps Blender frame numbers to simulation times

**Structure:**
```python
frame_to_time_mapping = [
    datetime(2023, 1, 1, 0, 0, 0),   # Frame 1
    datetime(2023, 1, 1, 0, 0, 0),   # Frame 2 (same time, multiple frames per timestep)
    datetime(2023, 1, 1, 0, 5, 0),   # Frame 3 (next timestep)
    datetime(2023, 1, 1, 0, 5, 0),   # Frame 4 (same time)
    ...
]
```

**Length:** `len(animation_times) * frames_per_timestep_int`

### kwargs_data

**Type:** `Dict[str, Dict[str, Any]]`

**Purpose:** Per-category object configuration

**Structure:**
```python
kwargs_data = {
    "Rcondensate": {
        "scale": (0.0002, 0.0002, 0.0002),
        "location": (0.0, 0.0, 5.0),
        "material": "cloud_material",
        "collections": ["data", "plys"],
        "varnames": ["var1", "var2"],  # Optional filter
    },
    "thetadeficit": {
        "scale": (0.0002, 0.0002, 0.001),
        "location": (0.0, 0.0, 0.1),
        ...
    },
}
```

### Object Metadata

**Attached to each BlenderObject as custom properties:**
```python
obj["data_file_suffix"] = ".ply"  # or .vdb, .obj, .vtp
obj["category"] = "Rcondensate"   # From filename
obj["varname"] = "var1"           # From filename
obj.name = "Rcondensate-var1"     # Permanent name (without timestamp)
```

---

## 9. ANIMATION METHODS ALREADY IN PLACE

### Callback-Based Animation (`update_for_new_frame`)

- **Pros:**
  - Dynamic: Load different meshes each frame
  - Memory efficient: Only one copy of each mesh in memory
  - Flexible: Can respond to user timeline changes

- **Cons:**
  - Slower: Every frame triggers file I/O and import
  - Requires callback registration
  - Difficult to preview without rendering

### Visibility Keyframe Functions (Already Exist!)

**Location:** `blender_animation.py:67-259`

These functions ALREADY support keyframe-based visibility animation:

```python
def _animate_object_visibility_helper(objects, frame, visible):
    """Set visibility keyframes at specific frame"""
    bpy.context.scene.frame_set(frame)
    for object in objects:
        object.hide_viewport = not visible
        object.hide_render = not visible
        object.keyframe_insert(data_path="hide_viewport", frame=frame)
        object.keyframe_insert(data_path="hide_render", frame=frame)

def animate_objects_visibility(objects, start_frame, end_frame):
    """Set visibility keyframes for a frame range"""
    if start_frame > 1:
        _animate_hide_objects(start_frame - 1)
    _animate_show_objects(start_frame)
    _animate_show_objects(end_frame)
    if end_frame < frame_end:
        _animate_hide_objects(end_frame + 1)
```

---

## 10. ARCHITECTURE RECOMMENDATIONS FOR KEYFRAME MODE

### Current Bottleneck
- **Callback approach**: File I/O happens at render time for EVERY frame
- Frame 1: Load mesh 1
- Frame 2: Load mesh 1 (again, wasted I/O)
- Frame 3: Load mesh 2
- Etc.

### Proposed Keyframe Mode Benefits
- Pre-load all meshes into scene at setup time
- Create visibility keyframes for each object
- Render without callbacks (faster, more predictable)
- Use built-in Blender keyframe/NLA system
- Better for preview and debugging

### Implementation Pattern

```python
def run_atmospheric_animation_keyframe_mode(
    data_dir: PathLike,
    output_dir: PathLike,
    kwargs_data: Dict[str, Dict[str, Any]],
    config: ConfigDict = {},
    animation_mode: str = "callback",  # NEW: "callback" or "keyframe"
    ...
) -> None:
    # ... existing setup code ...
    
    if animation_mode == "callback":
        # Existing code
        register_frame_change_handlers(callback_function)
    elif animation_mode == "keyframe":
        # New code path
        for simulation_time in animation_times:
            # Import all data for this time
            objects = import_data_for_time(...)
            # Get frames where this data is visible
            start_frame = calculate_frame_for_time(simulation_time)
            end_frame = calculate_frame_for_time(next_time or end)
            # Set visibility keyframes
            animate_objects_visibility(objects, start_frame, end_frame)
        # No callback registration
```

---

## Summary

The Blender module uses a **callback-based animation system** where:
1. Scene is analyzed at startup to determine animation times
2. A persistent callback is registered on frame change events
3. Each frame change triggers mesh loading and geometry updates
4. Data organization uses timestamped files matched to frame numbers
5. Configuration is managed through defaults + user overrides
6. Collections and hierarchy organize time-varying objects by type

The architecture is flexible for adding an alternative **keyframe-based mode** that would pre-load all meshes and use Blender's native animation system.
