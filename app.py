import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import pyvista as pv
from stpyvista import stpyvista
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
import io
import trimesh
from stpyvista.utils import start_xvfb


try:
    start_xvfb()
except Exception as e:
    print("unable to start xvfb, when you run local this is fine!")#st.warning(f"(this is ok for local runs) Could not start virtual framebuffer: {e}")    
# Initialize session state to store the mesh
if 'mesh' not in st.session_state:
    st.session_state.mesh = None

# cutter_mesh will be a copy of mesh, therefore also a trimesh object
if 'cutter_mesh' not in st.session_state:
    st.session_state.cutter_mesh = None

if 'output_filename' not in st.session_state:
    st.session_state.output_filename = None

# --- Additions for Image Editing ---
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

if 'edited_image' not in st.session_state:
    st.session_state.edited_image = None

if 'image_history' not in st.session_state:
    st.session_state.image_history = []

if 'history_index' not in st.session_state:
    st.session_state.history_index = -1

# --- Image Editing Functions ---
def add_to_history(new_image):
    """Adds a new image to the history stack."""
    # If we are not at the end of the history, truncate it
    if st.session_state.history_index < len(st.session_state.image_history) - 1:
        st.session_state.image_history = st.session_state.image_history[:st.session_state.history_index + 1]

    st.session_state.image_history.append(new_image)
    st.session_state.history_index += 1
    st.session_state.edited_image = new_image

# --- CORE LOGIC FUNCTIONS (switched to trimesh) ---

def resize_image_for_dpi(image, radius, target_dpi, allow_upscaling=False):
    """Resizes image based on target DPI and physical cylinder dimensions."""
    original_width, original_height = image.size
    
    # Calculate physical dimensions of the cylinder
    circumference_mm = 2 * np.pi * radius  # Height of image becomes circumference
    # Width of image becomes height - we need to calculate based on aspect ratio
    aspect_ratio = original_height / original_width  # height/width of original image
    cylinder_height_mm = circumference_mm / aspect_ratio
    
    # Calculate required pixel dimensions for target DPI
    # DPI = dots per inch, 1 inch = 25.4 mm
    mm_per_inch = 25.4
    required_width_px = int((cylinder_height_mm / mm_per_inch) * target_dpi)
    required_height_px = int((circumference_mm / mm_per_inch) * target_dpi)
    
    # Check if upscaling is needed
    scale_factor_w = required_width_px / original_width
    scale_factor_h = required_height_px / original_height
    max_scale_factor = max(scale_factor_w, scale_factor_h)
    
    if max_scale_factor > 1.0 and not allow_upscaling:
        # Limit to original resolution if upscaling not allowed
        scale_factor = min(1.0, min(scale_factor_w, scale_factor_h))
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
    else:
        new_width = required_width_px
        new_height = required_height_px
    
    # Resize image using high-quality resampling
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate actual achieved DPI
    actual_dpi_w = (new_width * mm_per_inch) / cylinder_height_mm
    actual_dpi_h = (new_height * mm_per_inch) / circumference_mm
    
    return resized_image, (actual_dpi_w, actual_dpi_h), (cylinder_height_mm, circumference_mm)

def create_axis_cylinder(mesh, axis_diameter, safety_margin=0.1):
    """Creates a cylinder for the boolean subtraction of the axis."""
    # Get mesh bounds to determine cylinder height
    bounds = mesh.bounds
    mesh_height = bounds[1, 2] - bounds[0, 2]  # Z-dimension
    mesh_center_z = (bounds[1, 2] + bounds[0, 2]) / 2
    
    # Create axis cylinder slightly longer than the mesh for clean boolean
    axis_radius = axis_diameter / 2
    axis_height = mesh_height + 2 * safety_margin
    
    # Create cylinder centered at origin, extending through entire mesh
    axis_cylinder = trimesh.primitives.Cylinder(
        radius=axis_radius, 
        height=axis_height,
        sections=32  # More sections for smoother cylinder
    )
    
    # Position the axis cylinder to be centered with the mesh
    axis_cylinder.apply_translation([0, 0, mesh_center_z])
    
    return axis_cylinder

def create_cylinder_mesh(image_file, radius, displacement, dpi, allow_upscaling, create_axis_hole=False, axis_diameter=6.0):
    """Main function for orchestrating the creation of the 3D cylinder from an image."""
    # Status callback for UI updates
    status_callback = getattr(create_cylinder_mesh, 'status_callback', None)
    
    if status_callback:
        status_callback("📷 Loading and processing image...")
    
    img = Image.open(image_file).convert('L')
    
    # Resize image based on DPI parameter and physical dimensions
    img, actual_dpi, physical_dims = resize_image_for_dpi(img, radius, dpi, allow_upscaling)
    
    img_array = np.array(img)
    ny, nx = img_array.shape  # ny = height, nx = width
    
    if status_callback:
        status_callback("🏗️ Creating mesh geometry...")
    
    vertices = map_image_to_vertices(img_array, radius, displacement)
    
    body_mesh = create_body_mesh(vertices, nx, ny)
    
    # Extract vertices for top and bottom caps
    # IMPORTANT: The indices refer to the original 'vertices' array
    bottom_edge_indices = np.arange(0, ny)
    top_edge_indices = np.arange((nx - 1) * ny, nx * ny)

    bottom_cap_mesh = create_cap_mesh(vertices, bottom_edge_indices, is_bottom=True)
    top_cap_mesh = create_cap_mesh(vertices, top_edge_indices, is_bottom=False)

    if status_callback:
        status_callback("🔗 Connecting mesh parts...")

    # **TRIMESH-CHANGE: Combine meshes with trimesh.util.concatenate**
    # The '+' operator does not work here. Empty meshes are ignored.
    mesh_list = [mesh for mesh in [body_mesh, bottom_cap_mesh, top_cap_mesh] if not mesh.is_empty]
    final_mesh = trimesh.util.concatenate(mesh_list)

    # **TRIMESH-CHANGE: Merge vertices to get a watertight mesh**
    final_mesh.merge_vertices()
    trimesh.repair.fix_normals(final_mesh)
    trimesh.repair.fill_holes(final_mesh)
    trimesh.repair.fix_winding(final_mesh)
    trimesh.repair.fix_inversion(final_mesh)
    #validate mesh
    final_mesh.merge_vertices()
    final_mesh.remove_unreferenced_vertices()
    final_mesh = trimesh.Trimesh(vertices=final_mesh.vertices, faces=final_mesh.faces, process=True, validate=True)        
    broken_faces=trimesh.repair.broken_faces(final_mesh)
    print(f"Number of broken faces after repair: {len(broken_faces)}")
    print(f"Final mesh is watertight: {final_mesh.is_watertight}")
    # Boolean operation for axis hole
    if create_axis_hole:
        if status_callback:
            status_callback("⚙️ Creating hole for axis (Boolean operation)...")
        
        try:
            axis_cylinder = create_axis_cylinder(final_mesh, axis_diameter)
            
            # Perform boolean subtraction
            final_mesh = final_mesh.difference(axis_cylinder, engine='manifold')
            
            if final_mesh.is_empty:
                raise ValueError("Boolean operation failed - mesh is empty")
                
            if status_callback:
                status_callback("✅ Hole for axis created successfully")
                
        except Exception as e:
            if status_callback:
                status_callback(f"❌ Error creating hole for axis: {str(e)}")
            # Continue without axis hole rather than failing completely
            st.warning(f"Could not create hole for axis: {e}")
    
    if status_callback:
        status_callback("🎉 3D model finished!")
    
    return final_mesh

def map_image_to_vertices(img_array, base_radius, max_displacement):
    """Maps image pixel coordinates and brightness to 3D vertex coordinates."""
    ny, nx = img_array.shape  # ny = height (rows), nx = width (columns)
    
    # FIXED: Correct coordinate mapping
    # The cylinder height should correspond to image width (nx)
    # The circumference should correspond to image height (ny)
    height = base_radius * 2 * np.pi
    
    # Correction of the height to consider the aspect ratio of the image
    # FIXED: Use ny/nx (height/width) instead of nx/ny
    aspect_ratio = ny / nx  # height/width
    effective_height = height / aspect_ratio

    # FIXED: Swap the coordinate assignment
    # x_indices should map to width (nx), y_indices to height (ny)
    x_indices, y_indices = np.arange(nx), np.arange(ny)
    
    # Z-coordinates map to image width (horizontal direction becomes cylinder height)
    z_coords = (x_indices / (nx - 1)) * effective_height
    # Theta maps to image height (vertical direction becomes cylinder circumference)  
    theta = (y_indices / (ny - 1)) * 2 * np.pi
    
    zz, tt = np.meshgrid(z_coords, theta, indexing='ij')
    
    brightness = img_array.T / 255.0
    effective_radius = base_radius + brightness * max_displacement
    
    xx = effective_radius * np.cos(tt)
    yy = effective_radius * np.sin(tt)
    
    vertices = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    return vertices

def create_body_mesh(vertices, nx, ny):
    """Creates the cylindrical surface mesh as a trimesh.Trimesh."""
    faces = []
    # Iterate over the "pixels" of the image to create quads
    for i in range(nx - 1):
        for j in range(ny):
            p1 = i * ny + j
            p2 = i * ny + ((j + 1) % ny)
            p3 = (i + 1) * ny + ((j + 1) % ny)
            p4 = (i + 1) * ny + j
            
            # **TRIMESH-CHANGE: Split quads into two triangles**
            # trimesh works with triangular faces, not quads.
            # PyVista's format [4, p1, p2, p3, p4] becomes [[p1, p2, p4], [p2, p3, p4]]
            faces.append([p1, p2, p4])
            faces.append([p2, p3, p4])
            
    # **TRIMESH-CHANGE: Create a trimesh.Trimesh object**
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))

def create_cap_mesh(all_vertices, edge_indices, is_bottom):
    """Triangulates the cap surfaces using Delaunay and returns a trimesh.Trimesh."""
    edge_vertices = all_vertices[edge_indices]
    if edge_vertices.shape[0] < 3: return trimesh.Trimesh()

    points_2d = edge_vertices[:, :2]
    boundary_polygon = Polygon(points_2d)

    try:
        # Perform Delaunay triangulation on the 2D points
        tri = Delaunay(points_2d)
    except Exception:
        return trimesh.Trimesh()

    # Filter triangles whose centroid is inside the boundary polygon
    filtered_simplices = [
        s for s in tri.simplices if boundary_polygon.contains(Point(np.mean(points_2d[s], axis=0)))
    ]
    if not filtered_simplices: return trimesh.Trimesh()

    # **TRIMESH-CHANGE: Direct use of the simplices as faces**
    # trimesh needs a simple list of triangle indices.
    # The indices in 'filtered_simplices' refer to the 'edge_vertices' array.
    # We have to map them back to the indices in the global 'all_vertices' array.
    cap_faces = np.array(filtered_simplices)
    
    # Reverse triangle orientation for the bottom so that the normals point outwards
    if is_bottom:
        cap_faces = cap_faces[:, [0, 2, 1]]

    # Map local indices back to the global vertex indices
    global_faces = edge_indices[cap_faces]

    # Create the mesh with all vertices, but only the faces for the cap
    return trimesh.Trimesh(vertices=all_vertices, faces=global_faces)

# --- STREAMLIT UI (refactored) ---

st.set_page_config(layout="wide", page_title="Image to 3D Cylinder")
st.title("Image to 3D Cylinder Converter")
st.markdown("This tool creates a 3D printable 'Lithophane-Roller' from an image.")

# -- SETTINGS IN THE SIDEBAR --
with st.sidebar:
    st.header("⚙️ Settings")
    radius = st.slider("Base Radius (in mm)", 10.0, 100.0, 30.0, 0.5)
    displacement = st.slider("Radial Displacement (Wall Thickness in mm)", 0.5, 10.0, 2.0, 0.1)
    dpi = st.slider("DPI (Resolution)", 50, 300, 150, 10,
                   help="Resolution based on physical cylinder dimensions")
    allow_upscaling = st.checkbox("Allow Upscaling", value=False,
                                 help="Allows enlarging the image beyond its original resolution")
    
    st.header("🔧 Axis")
    create_axis_hole = st.checkbox("Create hole for axis", value=True,
                                  help="Creates a through-hole for an axis")
    if create_axis_hole:
        axis_diameter = st.slider("Axis Diameter (in mm)", 1.0, min(radius * 1.8, 50.0), 6.0, 0.5,
                                 help=f"Maximum: {min(radius * 1.8, 50.0):.1f}mm (90% of base radius)")
        if axis_diameter >= radius * 0.9:
            st.warning("⚠️ Axis very thick - may cause structural problems")

# -- MAIN AREA --
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Image Upload")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded_file:
        # Initialize editing state when a new image is uploaded
        if st.session_state.original_image is None or st.session_state.original_image.tobytes() != uploaded_file.getvalue():
            st.session_state.original_image = Image.open(uploaded_file)
            st.session_state.image_history = [st.session_state.original_image]
            st.session_state.history_index = 0
            st.session_state.edited_image = st.session_state.original_image

        # --- IMAGE EDITING UI ---
        st.subheader("🎨 Image Editing")
        
        # Display the current state of the image
        st.image(st.session_state.edited_image, caption="Editable Image", use_container_width=True)

        # Undo/Redo buttons
        col_undo_redo1, col_undo_redo2 = st.columns(2)
        with col_undo_redo1:
            if st.button("Undo", use_container_width=True, disabled=st.session_state.history_index <= 0):
                st.session_state.history_index -= 1
                st.session_state.edited_image = st.session_state.image_history[st.session_state.history_index]
        with col_undo_redo2:
            if st.button("Redo", use_container_width=True, disabled=st.session_state.history_index >= len(st.session_state.image_history) - 1):
                st.session_state.history_index += 1
                st.session_state.edited_image = st.session_state.image_history[st.session_state.history_index]

        # Basic editing tools
        st.markdown("---")
        if st.button("Invert Colors", use_container_width=True):
            inverted_image = ImageOps.invert(st.session_state.edited_image.convert('L'))
            add_to_history(inverted_image)

        # Threshold
        st.markdown("---")
        col_thresh1, col_thresh2 = st.columns([2,1])
        with col_thresh1:
            threshold_value = st.slider("Threshold", 0, 255, 128)
        with col_thresh2:
            if st.button("Apply Threshold", use_container_width=True):
                thresholded_image = st.session_state.edited_image.convert('L').point(lambda p: 255 if p > threshold_value else 0, '1')
                add_to_history(thresholded_image)

        # Gaussian Blur
        st.markdown("---")
        col_blur1, col_blur2 = st.columns([2,1])
        with col_blur1:
            blur_radius = st.slider("Gaussian Blur Radius", 0, 10, 2)
        with col_blur2:
            if st.button("Apply Blur", use_container_width=True):
                blurred_image = st.session_state.edited_image.convert('L').filter(ImageFilter.GaussianBlur(radius=blur_radius))
                add_to_history(blurred_image)

        # Show image info and DPI calculations
        temp_img = st.session_state.edited_image
        resized_img, actual_dpi, physical_dims = resize_image_for_dpi(temp_img, radius, dpi, allow_upscaling)
        
        # Display comprehensive image information
        st.subheader("📊 Image Analysis")
        
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("Original Size", f"{temp_img.size[0]}x{temp_img.size[1]} px")
            st.metric("Final Size", f"{resized_img.size[0]}x{resized_img.size[1]} px")
        with col1b:
            st.metric("Cylinder Height", f"{physical_dims[0]:.1f} mm")
            st.metric("Cylinder Circumference", f"{physical_dims[1]:.1f} mm")
            
        col1c, col1d = st.columns(2)
        with col1c:
            st.metric("Target DPI", f"{dpi}")
            upscaling_needed = max(resized_img.size) > max(temp_img.size)
            if upscaling_needed and not allow_upscaling:
                st.warning("⚠️ Upscaling limited")
        with col1d:
            st.metric("Achieved DPI", f"{actual_dpi[0]:.0f} x {actual_dpi[1]:.0f}")
            if abs(actual_dpi[0] - dpi) > 5 or abs(actual_dpi[1] - dpi) > 5:
                st.info("ℹ️ DPI limited by original size")
        
        # Prepare filename for download (without extension)
        base_filename = ".".join(uploaded_file.name.split('.')[:-1])
        upscale_suffix = "_up" if allow_upscaling else ""
        axis_suffix = f"_axis{int(axis_diameter)}" if create_axis_hole else ""
        st.session_state.output_filename = f"{base_filename}_r{int(radius)}_d{int(displacement)}_dpi{int(dpi)}{upscale_suffix}{axis_suffix}.stl"
        
        if st.button("🚀 Generate 3D Model", use_container_width=True, type="primary"):
            # Create status placeholder
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            def update_status(message):
                status_placeholder.info(message)
            
            # Attach status callback to function
            create_cylinder_mesh.status_callback = update_status
            
            try:
                progress_bar.progress(10)
                # Convert the edited PIL image to an in-memory file for processing
                image_buffer = io.BytesIO()
                st.session_state.edited_image.save(image_buffer, format="PNG")
                image_buffer.seek(0)

                # Calculate the mesh and save it as a trimesh object
                if create_axis_hole:
                    st.session_state.mesh = create_cylinder_mesh(
                        image_buffer, radius, displacement, dpi, allow_upscaling,
                        create_axis_hole, axis_diameter
                    )
                else:
                    st.session_state.mesh = create_cylinder_mesh(
                        image_buffer, radius, displacement, dpi, allow_upscaling
                    )
                
                progress_bar.progress(100)
                status_placeholder.success("✅ Model generated successfully!")
                st.balloons()  # Celebration effect!
                
            except ValueError as e:
                progress_bar.progress(0)
                status_placeholder.error(f"❌ Error during mesh generation: {e}")
                st.session_state.mesh = None
            finally:
                # Clean up callback
                create_cylinder_mesh.status_callback = None
    else:
        st.info("Please upload an image file to begin.")

# Check if a mesh exists in the session_state to display/download it
if st.session_state.mesh and not st.session_state.mesh.is_empty:
    with col2:
        st.header("🖼️ 3D Preview")
        try:
            plotter = pv.Plotter(window_size=[600, 600], border=False)
            
            # **DISPLAY-CHANGE: Convert trimesh for PyVista with pv.wrap()**
            # stpyvista needs a PyVista object. pv.wrap is the easiest way.
            pv_mesh = pv.wrap(st.session_state.mesh)
            
            plotter.add_mesh(pv_mesh, color="ivory", smooth_shading=True)
            plotter.view_isometric()
            plotter.background_color = '#262730'
            stpyvista(plotter, key="pv_cylinder")

            # Treat the cutter as a copy of the main mesh
            st.session_state.cutter_mesh = st.session_state.mesh

        except Exception as e:
            st.error(f"Error in 3D display: {e}")
            st.warning("The 3D preview could not be loaded. You can still download the STL file though.")

    # Download-Button außerhalb der Spalte für bessere Sichtbarkeit
    st.header("💾 Download")
    try:
        # **DOWNLOAD-CHANGE: Use the export function of trimesh**
        with io.BytesIO() as f:
            # Export the trimesh object directly as a binary STL
            st.session_state.mesh.export(f, file_type='stl')
            f.seek(0)
            stl_data = f.read()

        # Show mesh statistics
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.metric("Vertices", len(st.session_state.mesh.vertices))
        with col_dl2:
            st.metric("Faces", len(st.session_state.mesh.faces))
        with col_dl3:
            volume = st.session_state.mesh.volume
            st.metric("Volume", f"{volume:.2f} mm³")
        
        # Show additional mesh info if axis hole was created
        if create_axis_hole and 'axis_diameter' in locals():
            col_axis1, col_axis2 = st.columns(2)
            with col_axis1:
                st.metric("Axis Diameter", f"{axis_diameter:.1f} mm")
            with col_axis2:
                wall_thickness = radius - axis_diameter/2
                st.metric("Wall thickness", f"{wall_thickness:.1f} mm")
                if wall_thickness < displacement * 2:
                    st.warning("⚠️ Thin wall - check printability")

        st.download_button(
            label="📥 Download Model as STL",
            data=stl_data,
            file_name=st.session_state.output_filename,
            mime="model/stl",
            use_container_width=True,
            type="primary"
        )
    except Exception as e:
        st.error(f"Error creating the download file: {e}")
