import streamlit as st
import numpy as np
from PIL import Image
import pyvista as pv
from stpyvista import stpyvista
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
import io
import trimesh # Wichtig: trimesh ist jetzt die primäre Bibliothek

# Initialize session state to store the mesh
if 'mesh' not in st.session_state:
    st.session_state.mesh = None

# cutter_mesh wird eine Kopie von mesh sein, daher auch ein trimesh-Objekt
if 'cutter_mesh' not in st.session_state:
    st.session_state.cutter_mesh = None

if 'output_filename' not in st.session_state:
    st.session_state.output_filename = None

# --- CORE LOGIC FUNCTIONS (auf trimesh umgestellt) ---

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
    """Erstellt einen Zylinder für die Boolean-Subtraktion der Achse."""
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
    """Hauptfunktion zur Orchestrierung der Erstellung des 3D-Zylinders aus einem Bild."""
    # Status callback für UI Updates
    status_callback = getattr(create_cylinder_mesh, 'status_callback', None)
    
    if status_callback:
        status_callback("📷 Lade und verarbeite Bild...")
    
    img = Image.open(image_file).convert('L')
    
    # Resize image based on DPI parameter and physical dimensions
    img, actual_dpi, physical_dims = resize_image_for_dpi(img, radius, dpi, allow_upscaling)
    
    img_array = np.array(img)
    ny, nx = img_array.shape  # ny = height, nx = width
    
    if status_callback:
        status_callback("🏗️ Erstelle Netzgeometrie...")
    
    vertices = map_image_to_vertices(img_array, radius, displacement)
    
    body_mesh = create_body_mesh(vertices, nx, ny)
    
    # Vertices für Deckel und Boden extrahieren
    # WICHTIG: Die Indizes beziehen sich auf den ursprünglichen 'vertices'-Array
    bottom_edge_indices = np.arange(0, ny)
    top_edge_indices = np.arange((nx - 1) * ny, nx * ny)

    bottom_cap_mesh = create_cap_mesh(vertices, bottom_edge_indices, is_bottom=True)
    top_cap_mesh = create_cap_mesh(vertices, top_edge_indices, is_bottom=False)

    if status_callback:
        status_callback("🔗 Verbinde Meshteile...")

    # **TRIMESH-ÄNDERUNG: Meshes mit trimesh.util.concatenate kombinieren**
    # Der '+' Operator funktioniert hier nicht. Leere Meshes werden ignoriert.
    mesh_list = [mesh for mesh in [body_mesh, bottom_cap_mesh, top_cap_mesh] if not mesh.is_empty]
    final_mesh = trimesh.util.concatenate(mesh_list)

    # **TRIMESH-ÄNDERUNG: Vertices verschmelzen, um ein wasserdichtes Netz zu erhalten**
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
    print(f"Anzahl defekter Flächen nach Reparatur: {len(broken_faces)}")
    print(f"Final mesh is watertight: {final_mesh.is_watertight}")
    # Boolean operation for axis hole
    if create_axis_hole:
        if status_callback:
            status_callback("⚙️ Erstelle Loch für Achse (Boolean-Operation)...")
        
        try:
            axis_cylinder = create_axis_cylinder(final_mesh, axis_diameter)
            
            # Perform boolean subtraction
            final_mesh = final_mesh.difference(axis_cylinder, engine='manifold')
            
            if final_mesh.is_empty:
                raise ValueError("Boolean-Operation fehlgeschlagen - Mesh ist leer")
                
            if status_callback:
                status_callback("✅ Loch für Achse erfolgreich erstellt")
                
        except Exception as e:
            if status_callback:
                status_callback(f"❌ Fehler beim Loch für Achse: {str(e)}")
            # Continue without axis hole rather than failing completely
            st.warning(f"Loch für Achse konnte nicht erstellt werden: {e}")
    
    if status_callback:
        status_callback("🎉 3D-Modell fertiggestellt!")
    
    return final_mesh

def map_image_to_vertices(img_array, base_radius, max_displacement):
    """Mappt Bildpixelkoordinaten und Helligkeit auf 3D-Vertex-Koordinaten."""
    ny, nx = img_array.shape  # ny = height (rows), nx = width (columns)
    
    # FIXED: Correct coordinate mapping
    # The cylinder height should correspond to image width (nx)
    # The circumference should correspond to image height (ny)
    height = base_radius * 2 * np.pi
    
    # Korrektur der Höhe, um das Seitenverhältnis des Bildes zu berücksichtigen
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
    """Erstellt das zylindrische Oberflächennetz als trimesh.Trimesh."""
    faces = []
    # Iteriere über die "Pixel" des Bildes, um Quads zu erstellen
    for i in range(nx - 1):
        for j in range(ny):
            p1 = i * ny + j
            p2 = i * ny + ((j + 1) % ny)
            p3 = (i + 1) * ny + ((j + 1) % ny)
            p4 = (i + 1) * ny + j
            
            # **TRIMESH-ÄNDERUNG: Quads in zwei Dreiecke aufteilen**
            # trimesh arbeitet mit Dreiecksflächen, nicht mit Quads.
            # PyVista's Format [4, p1, p2, p3, p4] wird zu [[p1, p2, p4], [p2, p3, p4]]
            faces.append([p1, p2, p4])
            faces.append([p2, p3, p4])
            
    # **TRIMESH-ÄNDERUNG: Erstelle ein trimesh.Trimesh Objekt**
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))

def create_cap_mesh(all_vertices, edge_indices, is_bottom):
    """Trianguliert die Deckelflächen mittels Delaunay und gibt ein trimesh.Trimesh zurück."""
    edge_vertices = all_vertices[edge_indices]
    if edge_vertices.shape[0] < 3: return trimesh.Trimesh()

    points_2d = edge_vertices[:, :2]
    boundary_polygon = Polygon(points_2d)

    try:
        # Führe Delaunay-Triangulation auf den 2D-Punkten durch
        tri = Delaunay(points_2d)
    except Exception:
        return trimesh.Trimesh()

    # Filtere Dreiecke, deren Mittelpunkt innerhalb des Randpolygons liegt
    filtered_simplices = [
        s for s in tri.simplices if boundary_polygon.contains(Point(np.mean(points_2d[s], axis=0)))
    ]
    if not filtered_simplices: return trimesh.Trimesh()

    # **TRIMESH-ÄNDERUNG: Direkte Verwendung der Simplices als Faces**
    # trimesh benötigt eine einfache Liste von Dreiecks-Indizes.
    # Die Indizes in 'filtered_simplices' beziehen sich auf den 'edge_vertices'-Array.
    # Wir müssen sie auf die Indizes im globalen 'all_vertices'-Array zurückführen.
    cap_faces = np.array(filtered_simplices)
    
    # Dreiecks-Orientierung für den Boden umkehren, damit die Normalen nach außen zeigen
    if is_bottom:
        cap_faces = cap_faces[:, [0, 2, 1]]

    # Map local indices back to the global vertex indices
    global_faces = edge_indices[cap_faces]

    # Erstelle das Mesh mit allen Vertices, aber nur den Faces für den Deckel
    return trimesh.Trimesh(vertices=all_vertices, faces=global_faces)

# --- STREAMLIT UI (refactored) ---

st.set_page_config(layout="wide", page_title="Bild-zu-3D-Zylinder")
st.title("Bild zu 3D-Zylinder Konverter")
st.markdown("Dieses Tool erstellt eine 3D-druckbare 'Lithophane-Rolle' aus einem Bild.")

# -- EINSTELLUNGEN IN DER SIDEBAR --
with st.sidebar:
    st.header("⚙️ Einstellungen")
    radius = st.slider("Basisradius (in mm)", 10.0, 100.0, 30.0, 0.5)
    displacement = st.slider("Radiale Verschiebung (Wanddicke in mm)", 0.5, 10.0, 2.0, 0.1)
    dpi = st.slider("DPI (Auflösung)", 50, 300, 150, 10, 
                   help="Auflösung basierend auf physischen Zylinderabmessungen")
    allow_upscaling = st.checkbox("Erlaube Upscaling", value=False,
                                 help="Erlaubt Vergrößerung des Bildes über die Originalauflösung hinaus")
    
    st.header("🔧 Achse")
    create_axis_hole = st.checkbox("Loch für Achse erstellen", value=True,
                                  help="Erstellt ein durchgehendes Loch für eine Achse")
    if create_axis_hole:
        axis_diameter = st.slider("Achsendurchmesser (in mm)", 1.0, min(radius * 1.8, 50.0), 6.0, 0.5,
                                 help=f"Maximum: {min(radius * 1.8, 50.0):.1f}mm (90% des Basisradius)")
        if axis_diameter >= radius * 0.9:
            st.warning("⚠️ Achse sehr dick - kann Strukturprobleme verursachen")

# -- HAUPTBEREICH --
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Bild Upload")
    uploaded_file = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg", "bmp"])
    
    if uploaded_file:
        # Show original image
        st.image(uploaded_file, caption="Hochgeladenes Bild", use_container_width=True)
        
        # Show image info and DPI calculations
        temp_img = Image.open(uploaded_file)
        resized_img, actual_dpi, physical_dims = resize_image_for_dpi(temp_img, radius, dpi, allow_upscaling)
        
        # Display comprehensive image information
        st.subheader("📊 Bildanalyse")
        
        col1a, col1b = st.columns(2)
        with col1a:
            st.metric("Original Größe", f"{temp_img.size[0]}x{temp_img.size[1]} px")
            st.metric("Finale Größe", f"{resized_img.size[0]}x{resized_img.size[1]} px")
        with col1b:
            st.metric("Zylinder Höhe", f"{physical_dims[0]:.1f} mm")
            st.metric("Zylinder Umfang", f"{physical_dims[1]:.1f} mm")
            
        col1c, col1d = st.columns(2)
        with col1c:
            st.metric("Ziel DPI", f"{dpi}")
            upscaling_needed = max(resized_img.size) > max(temp_img.size)
            if upscaling_needed and not allow_upscaling:
                st.warning("⚠️ Upscaling begrenzt")
        with col1d:
            st.metric("Erreichte DPI", f"{actual_dpi[0]:.0f} x {actual_dpi[1]:.0f}")
            if abs(actual_dpi[0] - dpi) > 5 or abs(actual_dpi[1] - dpi) > 5:
                st.info("ℹ️ DPI durch Originalgröße begrenzt")
        
        # Dateinamen für den Download vorbereiten (ohne Erweiterung)
        base_filename = ".".join(uploaded_file.name.split('.')[:-1])
        upscale_suffix = "_up" if allow_upscaling else ""
        axis_suffix = f"_axis{int(axis_diameter)}" if create_axis_hole else ""
        st.session_state.output_filename = f"{base_filename}_r{int(radius)}_d{int(displacement)}_dpi{int(dpi)}{upscale_suffix}{axis_suffix}.stl"
        
        if st.button("🚀 3D-Modell generieren", use_container_width=True, type="primary"):
            # Create status placeholder
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            def update_status(message):
                status_placeholder.info(message)
            
            # Attach status callback to function
            create_cylinder_mesh.status_callback = update_status
            
            try:
                progress_bar.progress(10)
                # Berechne das Netz und speichere es als trimesh-Objekt
                if create_axis_hole:
                    st.session_state.mesh = create_cylinder_mesh(
                        uploaded_file, radius, displacement, dpi, allow_upscaling, 
                        create_axis_hole, axis_diameter
                    )
                else:
                    st.session_state.mesh = create_cylinder_mesh(
                        uploaded_file, radius, displacement, dpi, allow_upscaling
                    )
                
                progress_bar.progress(100)
                status_placeholder.success("✅ Modell erfolgreich generiert!")
                st.balloons()  # Celebration effect!
                
            except ValueError as e:
                progress_bar.progress(0)
                status_placeholder.error(f"❌ Fehler bei der Netz-Generierung: {e}")
                st.session_state.mesh = None
            finally:
                # Clean up callback
                create_cylinder_mesh.status_callback = None
    else:
        st.info("Bitte laden Sie eine Bilddatei hoch, um zu beginnen.")

# Prüfe, ob ein Netz im session_state vorhanden ist, um es anzuzeigen/herunterzuladen
if st.session_state.mesh and not st.session_state.mesh.is_empty:
    with col2:
        st.header("🖼️ 3D-Vorschau")
        try:
            plotter = pv.Plotter(window_size=[600, 600], border=False)
            
            # **ANZEIGE-ÄNDERUNG: Konvertiere trimesh für PyVista mit pv.wrap()**
            # stpyvista benötigt ein PyVista-Objekt. pv.wrap ist der einfachste Weg.
            pv_mesh = pv.wrap(st.session_state.mesh)
            
            plotter.add_mesh(pv_mesh, color="ivory", smooth_shading=True)
            plotter.view_isometric()
            plotter.background_color = '#262730'
            stpyvista(plotter, key="pv_cylinder")

            # Den Cutter als Kopie des Haupt-Meshes behandeln
            st.session_state.cutter_mesh = st.session_state.mesh

        except Exception as e:
            st.error(f"Fehler bei der 3D-Anzeige: {e}")
            st.warning("Die 3D-Vorschau konnte nicht geladen werden. Sie können die STL-Datei aber trotzdem herunterladen.")

    # Download-Button außerhalb der Spalte für bessere Sichtbarkeit
    st.header("💾 Download")
    try:
        # **DOWNLOAD-ÄNDERUNG: Nutze die export-Funktion von trimesh**
        with io.BytesIO() as f:
            # Exportiere das trimesh-Objekt direkt als binäre STL
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
            st.metric("Volumen", f"{volume:.2f} mm³")
        
        # Show additional mesh info if axis hole was created
        if create_axis_hole and 'axis_diameter' in locals():
            col_axis1, col_axis2 = st.columns(2)
            with col_axis1:
                st.metric("Achsendurchmesser", f"{axis_diameter:.1f} mm")
            with col_axis2:
                wall_thickness = radius - axis_diameter/2
                st.metric("Wandstärke", f"{wall_thickness:.1f} mm")
                if wall_thickness < displacement * 2:
                    st.warning("⚠️ Dünne Wand - prüfen Sie die Druckbarkeit")

        st.download_button(
            label="📥 Modell als STL herunterladen",
            data=stl_data,
            file_name=st.session_state.output_filename,
            mime="model/stl",
            use_container_width=True,
            type="primary"
        )
    except Exception as e:
        st.error(f"Fehler beim Erstellen der Download-Datei: {e}")