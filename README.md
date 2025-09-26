# Image to Patterned Roller Generator

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Trimesh](https://img.shields.io/badge/Trimesh-powerful_3D_mesh_library-f06529.svg?style=for-the-badge)

![Banner](https://via.placeholder.com/1200x300.png?text=Image+to+Patterned+Roller+Generator)

A Streamlit web application to convert any 2D image into a 3D-printable patterned roller. Use it to create custom textures for applying paint, stamping clay, or making impressions in other materials.

## ✨ Features

- **Image Upload:** Upload any pattern or texture image (`.png`, `.jpg`, `.bmp`).
- **Real-time 3D Preview:** See a live preview of your patterned roller as you adjust the settings.
- **Customizable Parameters:**
    - **Radius:** Control the overall size of the roller.
    - **Pattern Depth:** Adjust the depth of the pattern based on image brightness.
    - **Resolution (DPI):** Define the output resolution for the 3D model.
    - **Axis Hole:** Optionally create a central hole for an axle.
- **STL Export:** Download the final model as an `.stl` file, ready for any 3D slicer.
- **Watertight Mesh:** The generated mesh is processed to be watertight, ensuring high printability.

## 🚀 How to Use the Application

1.  **Upload an Image:** Use the file uploader in the sidebar to select an image.
2.  **Adjust Settings:** Use the sliders and checkboxes in the sidebar to configure the model to your liking.
3.  **Generate Model:** Click the "Generate 3D Model" button.
4.  **Preview:** Interact with the 3D model in the preview pane.
5.  **Download:** Once you are happy with the result, click the "Download Model as STL" button.

## 🛠️ How to Run Locally

To run this application on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    Make sure you have Python 3.8+ installed. Then, install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The application should now be open in your web browser.

## ⚙️ How it Works

This tool works by mapping the pixels of an image onto the surface of a cylinder to create a 3D pattern.

1.  **Image Processing:** The input image is converted to grayscale.
2.  **Vertex Mapping:** The application iterates through each pixel of the image. The pixel's position is mapped to a 3D coordinate on a cylinder's surface, and its brightness determines the radial displacement (pattern depth). Brighter pixels create a deeper pattern, and darker pixels create a shallower one.
3.  **Mesh Generation:** The `trimesh` library is used to create a 3D mesh from these vertices. It generates the main body of the roller and adds top and bottom caps to create a solid, watertight object.
4.  **3D Preview:** The `stpyvista` library renders the `trimesh` object in the Streamlit interface, providing an interactive 3D view.
5.  **STL Export:** The final `trimesh` object is exported to a binary STL format, which is a standard file type for 3D printing.

## 💻 Technologies Used

- **Python**
- **Streamlit:** For the web application interface.
- **Trimesh:** For robust 3D mesh creation and manipulation.
- **PyVista:** For 3D visualization.
- **Pillow:** For image processing.
- **NumPy:** For numerical operations.
- **Shapely:** For geometric operations on the cap surfaces.
- **Scipy:** For scientific computing, specifically for Delaunay triangulation.
- **manifold3d:** For robust boolean operations.
- **networkx:** For graph operations on the mesh.