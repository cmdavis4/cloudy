# Blender Module Architecture Exploration - Complete Analysis

## Exploration Overview

This is a very thorough exploration of the Blender atmospheric visualization module architecture. The exploration covers the complete workflow from initialization through animation execution, with detailed analysis of the callback-based animation system and data management.

**Location:** `/home/cmdavis4/projects/common/blender/`

---

## Key Documents Generated

### 1. **blender_architecture_analysis.md** - MAIN COMPREHENSIVE GUIDE
   - Complete overview of all architecture components
   - Detailed explanation of each system
   - Full workflow descriptions with code references
   - Data structures and configuration handling
   - Recommendation for keyframe mode implementation

### 2. **blender_workflow_diagrams.txt** - VISUAL ARCHITECTURE MAPS
   - Diagram 1: High-level initialization flow
   - Diagram 2: Per-frame callback execution flow
   - Diagram 3: Scene hierarchy organization
   - Diagram 4: Frame-to-time mapping structure
   - Diagram 5: Configuration flow and merging
   - Diagram 6: Comparison of callback vs keyframe modes

### 3. **blender_architecture_summary.txt** - QUICK REFERENCE
   - File locations and key functions
   - Core concepts summary
   - Execution flow summary
   - Configuration flags reference
   - Handler registration details
   - Code patterns and debugging tips

---

## Quick Navigation

### Understanding the Architecture

1. **Start Here:** Read **blender_architecture_analysis.md** Sections 1-3
   - Overview of module structure
   - How callbacks work
   - Main entry point function

2. **Understand the Data:** Read **blender_architecture_analysis.md** Sections 4-5
   - How time-varying data is organized
   - How frames map to simulation times
   - How callbacks are registered

3. **Complete Flow:** Read **blender_architecture_analysis.md** Section 6
   - Complete initialization workflow
   - Playback/render phase
   - Data flow on frame changes

4. **Configuration:** Read **blender_architecture_analysis.md** Section 7
   - Configuration handling
   - Configuration merging
   - Key parameters

5. **Visual Understanding:** Review **blender_workflow_diagrams.txt**
   - See Diagram 1 for initialization
   - See Diagram 2 for callback execution
   - See Diagram 3 for scene structure

### Key Functions Quick Lookup

Use **blender_architecture_summary.txt** "FILE LOCATIONS & KEY FUNCTIONS" section:
- Line numbers for every important function
- Cross-reference with actual source files

### Configuration Reference

Use **blender_architecture_summary.txt** "CONFIGURATION FLAGS" section:
- All available configuration options
- Default values
- Where they are used

---

## Core Architecture Summary

### Entry Point
**File:** `/home/cmdavis4/projects/common/blender/blender_run.py`  
**Function:** `run_atmospheric_animation()` (lines 33-292)

### Animation System
**Callback-Based Approach:**
- Persistent callback registered on frame changes
- Executes on every frame during playback/render
- Loads mesh data on-demand per frame
- Two handlers: `frame_change_post` and `render_pre`

### Data Organization
**File Structure:**
- Timestamped files: `dt-YYYYMMDDHHMMSS_category-varname.suffix`
- Supported formats: `.ply`, `.vdb`, `.obj`, `.vtk`, `.vtp`
- Multiple files per timestep (different variables)

**Scene Hierarchy:**
```
Scene Collection
├── data/
│   ├── plys/       (Permanent PLY meshes, updated per frame)
│   ├── vdbs/       (Permanent VDB volumes, updated per frame)
│   ├── objs/
│   └── vtps/
├── date_text       (Current simulation time display)
├── grass_plane     (Optional grass field)
└── Camera
```

### Frame Mapping
**Concept:** Blender frames (1-based) mapped to simulation times via list

**Example:**
```
animation_times = [T0, T1, T2]
frames_per_timestep = 2

frame_to_time_mapping = [T0, T0, T1, T1, T2, T2]

Frame 1 → Index 0 → T0
Frame 2 → Index 1 → T0
Frame 3 → Index 2 → T1
Frame 4 → Index 3 → T1
...
```

### Configuration System
**Default Configuration:** `/home/cmdavis4/projects/common/blender/blender_config.py`

**User Configuration:** Merged with defaults via dict.update()

**Parameters:**
- Scene settings (resolution, engine, quality)
- Animation settings (use_grass, fps, output_format)
- Timing settings (simulation_minutes_per_second, limit)

---

## Critical Code Paths

### Initialization Path
```
run_atmospheric_animation()
  ├── reset_scene()                     (blender_core.py)
  ├── Parse & merge configuration       (blender_config.py)
  ├── get_data_filepaths()              (blender_import.py)
  ├── Extract animation_times
  ├── Create frame_to_time_mapping
  ├── setup_camera()                    (blender_camera.py)
  ├── setup_render_settings()           (blender_render.py)
  ├── create_collection()               (blender_core.py)
  ├── Define callback_function with captured vars
  ├── register_frame_change_handlers()  (blender_animation.py)
  ├── frame_set(1)                      [Triggers first callback!]
  ├── create_grass_field()              (blender_core.py)
  ├── set_sky_background()              (blender_core.py)
  └── Optionally render                 (bpy.ops.render.render)
```

### Per-Frame Callback Path
```
callback_function() [registered handler]
  ├── update_for_new_frame()            (blender_animation.py)
  │   ├── Get current_frame from scene
  │   ├── Convert to 0-based index
  │   ├── Look up simulation_time from frame_to_time_mapping
  │   ├── Update date_text display
  │   └── import_data_for_time()        (blender_import.py)
  │       ├── Format glob pattern: f"dt-{dt_string}*"
  │       ├── Find matching files
  │       └── For each file:
  │           ├── import_single_data_file()
  │           ├── Get or create permanent object
  │           ├── Update geometry OR setup object
  │           ├── Delete temporary import
  │           └── Update date_text
  └── Return to Blender engine
```

### Data Import Path
```
import_data_for_time() [Called per-frame]
  ├── Scan for files matching timestamp pattern
  ├── Sort by file type (VTK first)
  └── For each file:
      ├── import_single_data_file()
      │   ├── Call appropriate importer by suffix
      │   ├── Detect newly added object
      │   ├── Extract metadata (category, varname)
      │   └── Return BlenderObject
      ├── Try to find existing permanent object
      ├── If found:
      │   ├── update_object_geometry()
      │   └── Delete temporary object
      ├── If not found:
      │   ├── setup_object()
      │   ├── Rename to permanent name
      │   └── (temporary becomes permanent)
      └── Store in return dict
```

---

## Key Variables & Data Structures

### In Closure (Captured by Callback)
- **data_dir**: Root directory with timestamped files
- **frame_to_time_mapping**: List mapping frame index → simulation time
- **kwargs_data**: Dict of category configs (scale, location, etc.)
- **global_scale**: Tuple of (x, y, z) scaling factors

### Scene Objects
```python
# Permanent objects have:
obj["data_file_suffix"] = ".ply"  (or .vdb, .obj, .vtp)
obj["category"] = "Rcondensate"
obj["varname"] = "var1"
obj.name = "Rcondensate-var1"
```

### Configuration Dict
```python
config = {
    "frames_per_timestep": 12,
    "resolution_x": 1920,
    "resolution_y": 1080,
    "render_engine": "BLENDER_EEVEE_NEXT",
    "render_samples": 128,
    "grass_field_size": 150,
    "grass_density": 1500,
    "sun_energy": 4,
    "background_color": (0.02, 0.05, 0.2, 1.0),
    "background_strength": 0.4,
}
```

---

## Handler Registration Details

### Two Handlers Are Registered

1. **frame_change_post** - Fires after frame changes in viewport
2. **render_pre** - Fires before each frame is rendered

### Handler Callback Requirements
- Must be decorated with `@bpy.app.handlers.persistent`
- Must accept: `(scene, depsgraph=None)` parameters
- Gets cleared and re-registered on each run

### Handler Lifecycle
1. All existing handlers cleared
2. New callback appended to both handler lists
3. Scene set to frame 1 (triggers callback immediately)
4. Callback executes on every subsequent frame change

---

## Understanding the Time System

### File Naming Convention
```
dt-YYYYMMDDHHMMSS_category-varname.suffix

Example: dt-20230101123000_Rcondensate-var1.ply
         dt-20230101123000_wind-data.vdb
         dt-20230101123005_Rcondensate-var1.ply
```

### Time Extraction Process
1. `get_data_filepaths()` finds all files with supported suffixes
2. `to_kv_pairs(filepath, parse_datetimes=True)` extracts datetime and metadata
3. Unique datetimes extracted: `set(all_datetimes)`
4. Sorted and validated for even spacing

### Frame Calculation
```python
animation_times_timestep = animation_times[1] - animation_times[0]
fps = max(1, int((simulation_minutes_per_second * 60) / timestep_seconds))

# Creates frame_to_time_mapping where each time appears
# frames_per_timestep_int times consecutively
```

---

## Alternative Animation Mode: Keyframe-Based (Proposed)

### Current Limitations (Callback Mode)
- File I/O happens on every frame
- Same meshes loaded repeatedly (frames 1-2 both load T0)
- Callback overhead
- Difficult to preview without rendering

### Proposed Keyframe Mode Benefits
- Pre-load all meshes at startup
- Use Blender's native keyframe system for visibility
- Faster rendering (no per-frame I/O)
- Better preview capability
- Simpler workflow

### Implementation Strategy
See **blender_architecture_analysis.md** Section 10 for detailed recommendations

Key points:
1. Add `animation_mode` parameter to `run_atmospheric_animation()`
2. In keyframe mode: Import all data upfront
3. Create visibility keyframes using existing `animate_objects_visibility()`
4. Skip callback registration entirely

The infrastructure is already in place:
- `animate_objects_visibility()` exists in `blender_animation.py`
- `_animate_object_visibility_helper()` sets keyframes
- Just needs orchestration to tie it all together

---

## File Reference

### Source Files
- `/home/cmdavis4/projects/common/blender/blender_run.py` - Main entry point
- `/home/cmdavis4/projects/common/blender/blender_animation.py` - Callback system
- `/home/cmdavis4/projects/common/blender/blender_import.py` - Data loading
- `/home/cmdavis4/projects/common/blender/blender_core.py` - Scene manipulation
- `/home/cmdavis4/projects/common/blender/blender_camera.py` - Camera setup
- `/home/cmdavis4/projects/common/blender/blender_render.py` - Render settings
- `/home/cmdavis4/projects/common/blender/blender_config.py` - Configuration
- `/home/cmdavis4/projects/common/blender/__init__.py` - Public API

### Supporting Files
- `/home/cmdavis4/projects/common/types_core.py` - Type definitions
- `/home/cmdavis4/projects/common/utils.py` - Utility functions

---

## Key Insights for Keyframe Mode Planning

### Existing Infrastructure Already Supports It
1. **Visibility Keyframe Functions** exist: `animate_objects_visibility()`, `_animate_object_visibility_helper()`
2. **Frame Mapping** already calculated: `frame_to_time_mapping` tells us which frames show which times
3. **Data Import** already works: `import_data_for_time()` can be called multiple times at startup
4. **Configuration** already flexible: Easy to add new animation_mode parameter

### Implementation Approach
1. **Setup Phase Changes:**
   - Instead of registering callback, run import loop
   - Import data for all simulation times
   - Create visibility keyframes for each time range

2. **Render Phase Changes:**
   - No callback execution
   - Blender's native animation system handles visibility
   - Simpler dependency graph

3. **Main Loop Structure:**
   ```python
   for i, simulation_time in enumerate(animation_times):
       objects = import_data_for_time(data_dir, time, kwargs_data, scale)
       start_frame = i * frames_per_timestep_int + 1
       end_frame = (i + 1) * frames_per_timestep_int
       animate_objects_visibility(objects, start_frame, end_frame)
   ```

---

## Debugging & Testing Strategy

### Check Callback Execution
```python
# Monitor callback triggers
print(len(bpy.app.handlers.frame_change_post))  # Should be 1
print(len(bpy.app.handlers.render_pre))  # Should be 1
```

### Check Data Discovery
```python
from blender.blender_import import get_data_filepaths
files = get_data_filepaths(data_dir)
print(f"Found {len(files)} files")
for f in files:
    print(f.name)
```

### Check Object State
```python
for obj in bpy.context.scene.objects:
    print(f"{obj.name}: category={obj.get('category')}, "
          f"varname={obj.get('varname')}, "
          f"suffix={obj.get('data_file_suffix')}")
```

### Add Logging to Callback
Edit `update_for_new_frame()` to add print statements:
```python
print(f"Frame {current_frame} -> Time {current_time}")
print(f"Looking for: {glob_pattern}")
print(f"Found {len(this_time_data_filepaths)} files")
```

---

## Next Steps for Development

### Short Term
1. Review this analysis to understand architecture
2. Study actual source code with diagrams as reference
3. Run existing tests to verify callback mode

### Medium Term (For Keyframe Mode)
1. Create new function: `run_atmospheric_animation_keyframe_mode()`
2. Refactor common initialization code into helper functions
3. Implement pre-load loop and keyframe creation
4. Add tests for both animation modes

### Long Term
1. Add animation_mode parameter to main function
2. Route to appropriate implementation based on mode
3. Profile both modes to compare performance
4. Document best practices for each mode

---

## Summary

This exploration provides a complete understanding of the Blender module architecture:

1. **Architecture**: Callback-based animation with on-demand mesh loading
2. **Data Flow**: Timestamped files → frame mapping → per-frame callbacks
3. **Configuration**: Flexible merging of defaults and user config
4. **Organization**: Scene hierarchy with collections for different mesh types
5. **Extensibility**: Clear path to add keyframe-based animation mode

The codebase is well-structured and ready for the proposed keyframe mode enhancement, with existing infrastructure (visibility keyframe functions, frame mapping, configuration system) providing a solid foundation.

All code is malware-safe and well-documented. Ready for implementation planning.
