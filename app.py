import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import pyvista as pv
from stpyvista import stpyvista
from shapely.geometry import Polygon
import io
import trimesh
from stpyvista.utils import start_xvfb

from gyroid_coexistence import (CoexistenceConfig, image_to_blade_mask,
                                build_gyroid_dual_cylinder)


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

# --- Additions for the two-part rotational ejector system ---
if 'shell_mesh' not in st.session_state:
    st.session_state.shell_mesh = None
if 'core_mesh' not in st.session_state:
    st.session_state.core_mesh = None
if 'ejector_report' not in st.session_state:
    st.session_state.ejector_report = None
if 'shell_filename' not in st.session_state:
    st.session_state.shell_filename = None
if 'core_filename' not in st.session_state:
    st.session_state.core_filename = None

# --- Additions for Image Editing ---
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

if 'edited_image' not in st.session_state:
    st.session_state.edited_image = None

if 'image_history' not in st.session_state:
    st.session_state.image_history = []

if 'history_index' not in st.session_state:
    st.session_state.history_index = -1

# --- Additions for Radius/Width Sync ---
if 'radius' not in st.session_state:
    st.session_state.radius = 30.0
if 'width' not in st.session_state:
    st.session_state.width = 30.0 * 2 * np.pi

# --- Image Editing Functions ---
def add_to_history(new_image):
    """Adds a new image to the history stack."""
    # If we are not at the end of the history, truncate it
    if st.session_state.history_index < len(st.session_state.image_history) - 1:
        st.session_state.image_history = st.session_state.image_history[:st.session_state.history_index + 1]

    st.session_state.image_history.append(new_image)
    st.session_state.history_index += 1
    st.session_state.edited_image = new_image

# --- Callbacks for Radius/Width Sync ---
def ejector_min_radius():
    """Kleinster Radius, den der Zweiteiler mit den aktuellen Einstellungen
    noch hergibt.

    Die Rechnung steht in CoexistenceConfig.min_radius_mm(). Sie haengt an
    Hub, Schneidentiefe, Ueberblendung und Achsdurchmesser -- deshalb wird
    sie hier aus dem Session-State neu ausgewertet, sobald einer dieser
    Werte sich aendert. Unterhalb davon bleibt fuer die Gyroid-Zone weniger
    als eine Masche uebrig, und eingeschlossene Musterflaechen finden in der
    Tiefe keinen Weg mehr zueinander. Das ist eine Frage des Platzes und
    nicht der Aufloesung, laesst sich also auch mit feineren Voxeln nicht
    heilen -- deshalb wird es gesperrt statt hinterher gemeldet.
    """
    if not st.session_state.get('generate_ejector_system', True):
        return 10.0
    axis = (st.session_state.get('axis_diameter', 6.0)
            if st.session_state.get('create_axis_hole', True) else None)
    cfg = CoexistenceConfig(
        voxel_mm=st.session_state.get('ejector_voxel_mm', 0.9),
        travel_mm=st.session_state.get('ejector_travel', 3.0),
        cut_depth_mm=st.session_state.get('ejector_cut_depth', 4.0),
        min_wall_mm=st.session_state.get('ejector_min_wall', 1.2),
        print_clearance_mm=st.session_state.get('ejector_clearance', 0.4),
        blend_mm=st.session_state.get('ejector_blend', 6.0),
        axis_diameter_mm=axis,
    )
    # Auf halbe Millimeter aufrunden, damit der Wert zur Schrittweite des
    # Reglers passt.
    return float(np.ceil(cfg.min_radius_mm() * 2) / 2)


def sync_radius_from_width():
    """Setzt den Radius aus der eingegebenen Breite -- aber nie unter die
    Untergrenze, die der Zweiteiler braucht."""
    width = st.session_state.width
    radius = width / (2 * np.pi)
    st.session_state.radius = max(radius, ejector_min_radius())


def correct_overhangs(image, angle_deg, radius, displacement, dpi, allow_upscaling,
                      l_to_r=True, r_to_l=True, t_to_b=True, b_to_t=True):
    """
    Adjusts image brightness to prevent overhangs steeper than a given angle
    by raising the brightness of lower pixels.
    """
    # 1. Get Physical Dimensions and final resolution
    resized_image, _, physical_dims = resize_image_for_dpi(
        image, radius, dpi, allow_upscaling
    )
    cylinder_height_mm, circumference_mm = physical_dims
    final_width_px, final_height_px = resized_image.size

    # 2. Calculate Maximum Allowed Brightness Change for each direction
    max_slope = np.tan(np.deg2rad(angle_deg))

    # Horizontal (Left/Right) correction corresponds to the cylinder's main axis
    pixel_width_mm = cylinder_height_mm / final_width_px
    max_delta_brightness_lr = (max_slope * pixel_width_mm * 255) / displacement

    # Vertical (Top/Bottom) correction corresponds to the cylinder's circumference
    pixel_height_mm = circumference_mm / final_height_px
    max_delta_brightness_tb = (max_slope * pixel_height_mm * 255) / displacement

    # 3. Apply Correction Algorithm
    img_array = np.array(resized_image.convert("L")).astype(np.float32)

    # The core logic is to ensure that for any two adjacent pixels,
    # their brightness difference does not violate the maximum slope.
    # We raise the lower pixel's brightness to meet this requirement.
    # Formula: pixel_B_new = max(pixel_B_old, pixel_A - max_delta_brightness)

    if l_to_r:
        for i in range(1, final_width_px):
            prev_col = img_array[:, i - 1]
            img_array[:, i] = np.maximum(img_array[:, i], prev_col - max_delta_brightness_lr)

    if r_to_l:
        for i in range(final_width_px - 2, -1, -1):
            next_col = img_array[:, i + 1]
            img_array[:, i] = np.maximum(img_array[:, i], next_col - max_delta_brightness_lr)

    if t_to_b:
        for j in range(1, final_height_px):
            prev_row = img_array[j - 1, :]
            img_array[j, :] = np.maximum(img_array[j, :], prev_row - max_delta_brightness_tb)

    if b_to_t:
        for j in range(final_height_px - 2, -1, -1):
            next_row = img_array[j + 1, :]
            img_array[j, :] = np.maximum(img_array[j, :], next_row - max_delta_brightness_tb)

    # 4. Return Corrected Image
    corrected_image_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected_image_array)

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
    # Theta maps to image height (vertical direction becomes cylinder circumference).
    # BUGFIX: theta is PERIODIC (row 0 and row ny-1 are adjacent on the cylinder,
    # not the same angle) so it must use an endpoint-exclusive sampling (divide by
    # ny, not ny-1). The previous "/(ny-1)" made the last row hit theta=2*pi,
    # i.e. the exact same angle as row 0. For a seamlessly tileable image (where
    # row 0 and row ny-1 differ slightly, by design, to continue the pattern) this
    # created near-duplicate boundary vertices at different radii right at the
    # theta=0 seam -- a degenerate fold that breaks the mesh's watertightness.
    theta = (y_indices / ny) * 2 * np.pi
    
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
    """Triangulates the cap surfaces and returns a trimesh.Trimesh.

    BUGFIX: this used to run an unconstrained scipy Delaunay triangulation
    over the ring points and keep only the triangles whose centroid tested
    inside the boundary polygon. That is unreliable for jagged/star-shaped
    cap rings (busy image patterns produce exactly this kind of boundary):
    legitimate triangles near concave notches can have a centroid that
    falls just outside the polygon and get dropped, leaving the cap with
    holes -- i.e. a non-watertight mesh. Downstream this shows up as
    broken normals/geometry and, for the dual-cylinder ejector, boolean
    overlap checks against a non-watertight mesh producing NaN.

    Fix: triangulate the ring itself via ear clipping (mapbox_earcut,
    through trimesh.creation.triangulate_polygon), which always closes the
    cap for any simple polygon regardless of how concave it is.
    """
    edge_vertices = all_vertices[edge_indices]
    if edge_vertices.shape[0] < 3: return trimesh.Trimesh()

    points_2d = edge_vertices[:, :2]
    boundary_polygon = Polygon(points_2d)
    if boundary_polygon.area == 0:
        return trimesh.Trimesh()

    try:
        tri_vertices_2d, cap_faces = trimesh.creation.triangulate_polygon(
            boundary_polygon, engine="earcut"
        )
    except Exception:
        return trimesh.Trimesh()

    if cap_faces is None or len(cap_faces) == 0:
        return trimesh.Trimesh()

    # earcut triangulates using exactly the input ring vertices (in order,
    # with the closing/repeated first point appended) -- no Steiner points
    # are inserted -- so face indices map 1:1 back onto ``edge_indices``.
    n_ring = points_2d.shape[0]
    if tri_vertices_2d.shape[0] != n_ring + 1 or cap_faces.max() >= n_ring:
        return trimesh.Trimesh()

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
    st.number_input(
        "set width (in mm)",
        key='width',
        on_change=sync_radius_from_width,
        step=1.0,
        format="%.2f",
        help="Enter the width to set the radius accordingly, respecting image aspect ratio"
    )

    st.slider(
        "Base Radius (in mm)",
        10.0,
        100.0,
        key='radius',
        step=0.5,
        help="Grundradius des Zylinders. Der Zweiteiler braucht eine "
             "Untergrenze -- sie steht bei den Auswerfer-Einstellungen und "
             "haengt von Hub, Schneidentiefe und Achse ab."
    )
    radius = st.session_state.radius
    displacement = st.slider("Radial Displacement (Wall Thickness in mm)", 0.5, 10.0, 2.0, 0.1)
    dpi = st.slider("DPI (Resolution)", 50, 300, 150, 10,
                   help="Resolution based on physical cylinder dimensions")
    allow_upscaling = st.checkbox("Allow Upscaling", value=False,
                                 help="Allows enlarging the image beyond its original resolution")
    
    st.header("🔧 Axis")
    create_axis_hole = st.checkbox("Create hole for axis", value=True,
                                  help="Creates a through-hole for an axis",
                                  key="create_axis_hole")
    if create_axis_hole:
        axis_diameter = st.slider("Axis Diameter (in mm)", 1.0, min(radius * 1.8, 50.0), 6.0, 0.5,
                                 key="axis_diameter",
                                 help=f"Maximum: {min(radius * 1.8, 50.0):.1f}mm (90% of base radius)")
        if axis_diameter >= radius * 0.9:
            st.warning("⚠️ Axis very thick - may cause structural problems")

    st.header("🔄 Schneide + Ausstoesser")
    generate_ejector_system = st.checkbox(
        "Zweiteiler erzeugen (Schneide + Ausstoesser)", value=True,
        help="Die dunklen Linien des Bildes werden zur KLINGE, die hellen "
             "Flaechen dazwischen zum AUSSTOESSER, der das geschnittene Teil "
             "herausdrueckt. Beide Koerper teilen sich denselben Bauraum und "
             "werden ineinander gedruckt (gyroid_coexistence.py). Braucht "
             "eine Achsbohrung.",
        key="generate_ejector_system",
    )
    radius_too_small = False
    if generate_ejector_system:
        ejector_threshold = st.slider(
            "Klingen-Schwellwert (Helligkeit)", 0, 255, 128, 1,
            help="Pixel dunkler als dieser Wert werden zur Klinge.",
            key="ejector_threshold",
        )
        ejector_travel = st.slider(
            "Auswerferhub (mm)", 0.5, 6.0, 3.0, 0.1,
            help="Der volle Weg, den eine Ausstoesserplatte zuruecklegt. Der "
                 "Ausstoesser sitzt exzentrisch und wandert aus seiner "
                 "Mittellage um die HALBE Strecke nach jeder Seite -- so viel "
                 "Freiraum muss ueberall zwischen beiden Koerpern bleiben, "
                 "und so grob faellt die Gyroid-Struktur im Inneren aus.",
            key="ejector_travel",
        )
        ejector_cut_depth = st.slider(
            "Schneidentiefe (mm)", 1.0, 12.0, 4.0, 0.5,
            help="So tief steht die Klinge ueber dem Ausstoesser. Bis zu "
                 "dieser Tiefe bleibt die Zuordnung fest -- das ist das "
                 "Produkt und wird nicht wegoptimiert.",
            key="ejector_cut_depth",
        )
        ejector_voxel_mm = st.slider(
            "Voxelgroesse (mm)", 0.5, 2.0, 0.9, 0.1,
            help="Aufloesung des Voxelgitters. Feiner = genauere Muster und "
                 "duennere Waende moeglich, aber deutlich laengere "
                 "Rechenzeit (die Voxelzahl waechst mit der dritten Potenz).",
            key="ejector_voxel_mm",
        )
        ejector_min_wall = st.slider(
            "Mindestwandstaerke (mm)", 0.4, 3.0, 1.2, 0.1,
            help="Duennste Wand, die als druckbar gilt. Klingenlinien, die "
                 "im Bild duenner sind, werden darauf aufgedickt.",
            key="ejector_min_wall",
        )
        ejector_clearance = st.slider(
            "Druckspiel (mm)", 0.0, 1.0, 0.4, 0.05,
            help="Zusaetzlicher Spalt in ALLE Richtungen (auch in z), damit "
                 "die ineinander gedruckten Teile nicht verschmelzen.",
            key="ejector_clearance",
        )
        with st.expander("Feineinstellung"):
            ejector_blend = st.slider(
                "Ueberblendstrecke (mm)", 2.0, 20.0, 6.0, 0.5,
                help="Auf dieser Tiefe geht das Muster in die "
                     "Gyroid-Struktur ueber.",
                key="ejector_blend",
            )
            ejector_period = st.slider(
                "Gyroid-Periode (mm, 0 = automatisch)", 0.0, 60.0, 0.0, 1.0,
                help="Maschenweite der Gyroid-Struktur. 0 laesst sie suchen: "
                     "so fein wie moeglich, aber grob genug, dass beide "
                     "Haelften nach dem Freischneiden des Hubs noch "
                     "zusammenhaengen.",
                key="ejector_period",
            )
        # Untergrenze fuer den Radius. Bewusst als SPERRE und nicht als
        # Slider-Minimum: aendert man das Minimum eines Reglers mit Key,
        # setzt Streamlit seinen Wert auf eben dieses Minimum zurueck -- der
        # Radius waere dann bei jeder Aenderung an Hub oder Achse
        # unbemerkt auf die Untergrenze gesprungen.
        ejector_radius_min = ejector_min_radius()
        radius_too_small = radius < ejector_radius_min
        if radius_too_small:
            st.error(
                f"⛔ Radius {radius:.1f} mm ist zu klein fuer diese "
                f"Einstellungen -- mindestens {ejector_radius_min:.1f} mm. "
                f"Zwischen Nabe und Schneidentiefe bleibt sonst zu wenig "
                f"Platz fuer die Gyroid-Zone, in der sich die "
                f"eingeschlossenen Musterflaechen verbinden; der Ausstoesser "
                f"kaeme in mehreren Teilen heraus. Abhilfe: groesserer "
                f"Radius, kleinerer Hub, geringere Schneidentiefe, duennere "
                f"Achse oder feineres Voxelgitter."
            )
        if not create_axis_hole:
            st.warning("⚠️ Der Ausstoesser benoetigt eine Achsbohrung "
                       "('Create hole for axis').")

# -- MAIN AREA --
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Image Upload")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"],
                                      key="uploaded_file")
    
    if uploaded_file:
        # Initialize editing state when a new image is uploaded
        if st.session_state.original_image is None:
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
                st.rerun()
        with col_undo_redo2:
            if st.button("Redo", use_container_width=True, disabled=st.session_state.history_index >= len(st.session_state.image_history) - 1):
                st.session_state.history_index += 1
                st.session_state.edited_image = st.session_state.image_history[st.session_state.history_index]
                st.rerun()

        # Basic editing tools
        st.markdown("---")
        if st.button("Invert Colors", use_container_width=True):
            inverted_image = ImageOps.invert(st.session_state.edited_image.convert('L'))
            add_to_history(inverted_image)
            st.rerun()

        # Threshold
        st.markdown("---")
        col_thresh1, col_thresh2 = st.columns([2,1])
        with col_thresh1:
            threshold_value = st.slider("Threshold", 0, 255, 128)
        with col_thresh2:
            if st.button("Apply Threshold", use_container_width=True):
                thresholded_image = st.session_state.edited_image.convert('L').point(lambda p: 255 if p > threshold_value else 0, '1')
                add_to_history(thresholded_image)
                st.rerun()

        # Gaussian Blur
        st.markdown("---")
        col_blur1, col_blur2 = st.columns([2,1])
        with col_blur1:
            blur_radius = st.slider("Gaussian Blur Radius", 0, 10, 2)
        with col_blur2:
            if st.button("Apply Blur", use_container_width=True):
                blurred_image = st.session_state.edited_image.convert('L').filter(ImageFilter.GaussianBlur(radius=blur_radius))
                add_to_history(blurred_image)
                st.rerun()

        # Overhang Correction
        st.markdown("---")
        st.subheader("Overhang Correction")
        overhang_angle = st.slider(
            "Max Overhang Angle (°)", 1, 90, 45,
            help="Corrects steep slopes that would create unprintable overhangs. 90° is vertical (no correction), 45° is a standard recommendation."
        )

        st.write("Correction Directions:")
        col_dir1, col_dir2, col_dir3, col_dir4 = st.columns(4)
        with col_dir1:
            correct_l_to_r = st.checkbox("L→R", value=True, help="Left to Right")
        with col_dir2:
            correct_r_to_l = st.checkbox("R→L", value=True, help="Right to Left")
        with col_dir3:
            correct_t_to_b = st.checkbox("T→B", value=True, help="Top to Bottom")
        with col_dir4:
            correct_b_to_t = st.checkbox("B→T", value=True, help="Bottom to Top")

        if st.button("Apply Overhang Correction", use_container_width=True):
            with st.spinner("Applying overhang correction..."):
                corrected_image = correct_overhangs(
                    image=st.session_state.edited_image,
                    angle_deg=overhang_angle,
                    radius=radius,
                    displacement=displacement,
                    dpi=dpi,
                    allow_upscaling=allow_upscaling,
                    l_to_r=correct_l_to_r,
                    r_to_l=correct_r_to_l,
                    t_to_b=correct_t_to_b,
                    b_to_t=correct_b_to_t
                )
                add_to_history(corrected_image)
                st.rerun()

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
        
        if generate_ejector_system:
            st.session_state.shell_filename = f"{base_filename}_r{int(radius)}_schneide.stl"
            st.session_state.core_filename = f"{base_filename}_r{int(radius)}_ausstoesser.stl"

        generate_disabled = generate_ejector_system and (
            not create_axis_hole or radius_too_small)
        if st.button("🚀 Generate 3D Model", use_container_width=True, type="primary",
                     disabled=generate_disabled, key="generate_button"):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Reset both output modes so preview/download reflects only the
            # freshly generated result.
            st.session_state.mesh = None
            st.session_state.shell_mesh = None
            st.session_state.core_mesh = None
            st.session_state.ejector_report = None

            if generate_ejector_system:
                try:
                    progress_bar.progress(10)
                    status_placeholder.info("📷 Bild wird aufbereitet...")
                    resized_img, _, physical_dims = resize_image_for_dpi(
                        st.session_state.edited_image, radius, dpi, allow_upscaling
                    )
                    cylinder_height_mm, _ = physical_dims
                    blade_mask = image_to_blade_mask(
                        np.array(resized_img.convert('L')), ejector_threshold
                    )

                    progress_bar.progress(40)
                    status_placeholder.info(
                        "🏗️ Voxelgitter, Gyroid und Reparaturen..."
                    )
                    cfg = CoexistenceConfig(
                        voxel_mm=ejector_voxel_mm,
                        cut_depth_mm=ejector_cut_depth,
                        travel_mm=ejector_travel,
                        print_clearance_mm=ejector_clearance,
                        min_wall_mm=ejector_min_wall,
                        blend_mm=ejector_blend,
                        axis_diameter_mm=axis_diameter,
                        gyroid_period_mm=(ejector_period or None),
                    )
                    shell_mesh, core_mesh, ejector_report = build_gyroid_dual_cylinder(
                        blade_mask,
                        radius_mm=radius,
                        height_mm=cylinder_height_mm,
                        cfg=cfg,
                    )
                    st.session_state.shell_mesh = shell_mesh
                    st.session_state.core_mesh = core_mesh
                    st.session_state.ejector_report = ejector_report

                    progress_bar.progress(100)
                    if ejector_report.get("ok"):
                        status_placeholder.success(
                            "✅ Schneide und Ausstoesser erfolgreich erzeugt!"
                        )
                        st.balloons()
                    else:
                        status_placeholder.warning(
                            "⚠️ Erzeugt, aber nicht alle Bedingungen erfuellt "
                            "-- siehe Report."
                        )

                    with st.expander("🔍 Report", expanded=not ejector_report.get("ok")):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric(
                                "Bewegungsspalt eingehalten",
                                "ja" if ejector_report["xy_travel_ok"] else "nein",
                                help=f"Verschiebung um {ejector_travel} mm in "
                                     f"jeder XY-Richtung ohne Beruehrung.",
                            )
                        with col_b:
                            st.metric(
                                "Koerper (Soll: 1 / 1)",
                                f"{ejector_report['blade_bodies']} / "
                                f"{ejector_report['ejector_bodies']}",
                            )
                        with col_c:
                            st.metric(
                                "Muster vollstaendig",
                                f"{ejector_report['pattern_completeness'] * 100:.1f} %",
                                help="Anteil der Klingenpixel, die im "
                                     "fertigen Koerper wirklich auftauchen. "
                                     "Soll: 100 %.",
                            )
                        st.caption(
                            f"Schwebende Voxel (Soll je 0): "
                            f"{ejector_report['blade_floating_voxels']} / "
                            f"{ejector_report['ejector_floating_voxels']} · "
                            f"Auslenkung ±"
                            f"{ejector_travel / 2:.2f} mm bei {ejector_travel:.1f} mm Hub"
                        )
                        gyroid = ejector_report.get("gyroid", {})
                        st.caption(
                            f"Gyroid: Periode {gyroid.get('period_mm', '?')} mm, "
                            f"z-Streckung {gyroid.get('z_stretch', '?')}, "
                            f"Zone {ejector_report.get('gyroid_zone_mm', '?')} mm · "
                            f"Gitter {ejector_report['grid']['n_theta']}×"
                            f"{ejector_report['grid']['n_z']}×"
                            f"{ejector_report['grid']['n_r']} Voxel à "
                            f"{ejector_report['grid']['voxel_mm']} mm"
                        )
                        st.caption(
                            f"Volumen: Schneide "
                            f"{ejector_report['blade_volume_mm3']:.0f} mm³, "
                            f"Ausstoesser "
                            f"{ejector_report['ejector_volume_mm3']:.0f} mm³ · "
                            f"Material dicker als {ejector_min_wall} mm: "
                            f"{ejector_report['blade_wall_ratio'] * 100:.0f} % / "
                            f"{ejector_report['ejector_wall_ratio'] * 100:.0f} %"
                        )
                        dropped = ejector_report.get(
                            "ejector_volume_dropped_mm3", 0.0)
                        if dropped > 0:
                            st.caption(
                                f"Aufgegeben: "
                                f"{ejector_report.get('ejector_parts_dropped', 0)} "
                                f"Ausstoesserteile mit zusammen {dropped:.0f} mm³ "
                                f"waren weder anzubinden noch zu stuetzen und "
                                f"wurden entfernt -- der Ausstoesser drueckt dort "
                                f"schwaecher, bleibt dafuer ein Stueck."
                            )
                        reps = ejector_report.get("repairs", {})
                        for label, key in (("Schneide", "blade"),
                                           ("Ausstoesser", "ejector")):
                            info = reps.get(key, {})
                            st.caption(
                                f"{label}: {info.get('support_voxels_added', 0)} "
                                f"Stuetzvoxel, {info.get('links_added', 0)} "
                                f"Verbindungen, "
                                f"{info.get('diagonal_contacts_closed', 0)} "
                                f"Kantenkontakte geschlossen / "
                                f"{info.get('diagonal_contacts_separated', 0)} "
                                f"getrennt, {info.get('rounds', 0)} Reparaturrunden"
                            )
                        for warning in ejector_report.get("warnings", []):
                            st.warning(warning)
                except Exception as e:
                    progress_bar.progress(0)
                    status_placeholder.error(f"❌ Error during ejector generation: {e}")
                    st.session_state.shell_mesh = None
                    st.session_state.core_mesh = None
            else:
                # Create status placeholder
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

has_ejector_result = (
    st.session_state.shell_mesh is not None and not st.session_state.shell_mesh.is_empty
    and st.session_state.core_mesh is not None and not st.session_state.core_mesh.is_empty
)
has_single_result = st.session_state.mesh and not st.session_state.mesh.is_empty

if has_ejector_result:
    with col2:
        st.header("🖼️ 3D Preview")
        try:
            plotter = pv.Plotter(window_size=[600, 600], border=False)
            pv_shell = pv.wrap(st.session_state.shell_mesh)
            pv_core = pv.wrap(st.session_state.core_mesh)
            plotter.add_mesh(pv_shell, color="lightblue", opacity=0.55,
                              smooth_shading=True, name="shell")
            plotter.add_mesh(pv_core, color="ivory", smooth_shading=True, name="core")
            plotter.view_isometric()
            plotter.background_color = '#262730'
            stpyvista(plotter, key="pv_ejector")
        except Exception as e:
            st.error(f"Error in 3D display: {e}")
            st.warning("The 3D preview could not be loaded. You can still download the STL files though.")

    st.header("💾 Download")
    try:
        col_dl_1, col_dl_2 = st.columns(2)

        with col_dl_1:
            st.subheader("Schneide")
            with io.BytesIO() as f:
                st.session_state.shell_mesh.export(f, file_type='stl')
                f.seek(0)
                shell_data = f.read()
            st.metric("Faces", len(st.session_state.shell_mesh.faces))
            st.download_button("📥 Download Schneide", shell_data,
                                st.session_state.shell_filename, "model/stl",
                                use_container_width=True, type="primary")

        with col_dl_2:
            st.subheader("Ausstoesser")
            with io.BytesIO() as f:
                st.session_state.core_mesh.export(f, file_type='stl')
                f.seek(0)
                core_data = f.read()
            st.metric("Faces", len(st.session_state.core_mesh.faces))
            st.download_button("📥 Download Ausstoesser", core_data,
                                st.session_state.core_filename, "model/stl",
                                use_container_width=True, type="primary")
    except Exception as e:
        st.error(f"Error creating the download file: {e}")

elif has_single_result:
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
