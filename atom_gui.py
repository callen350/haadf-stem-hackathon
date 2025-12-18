import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
import hyperspy.api as hs
import atomai as aoi
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import umap
import os

# --- Configuration ---
MODEL_PATH = "haadf_model_01.tar" # AtomAI models are usually .pt
MIN_BLOB_SIZE = 30  # Filter out atoms smaller than this (pixels)
PATCH_SIZE = 28     # Size of atom patches for defect finding
CLASS_NAMES = ['Mo', 'W', 'S2', 'SSe', 'Se2'] # Classes 1-5

# --- Helper Functions ---

def load_dm3_file(filepath):
    """Loads a DM3 file using Hyperspy and returns the normalized numpy array."""
    s = hs.load(filepath)
    img = s.data
    # If 3D stack, take sum (integrate)
    if img.ndim == 3:
        img = np.sum(img, axis=0)
    
    # Normalize 0-1
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def filter_predictions(pred_mask, min_size=30):
    """
    Takes a raw prediction mask (H, W) with integers 0-5.
    Removes blobs smaller than min_size for each class.
    Returns: Filtered Mask, DataFrame of centroids
    """
    clean_mask = np.zeros_like(pred_mask)
    atoms_data = []

    # Iterate classes 1-5 (Skip 0: Background)
    for class_id in range(1, 6):
        # Create binary mask for this class
        binary = (pred_mask == class_id).astype(bool)
        
        # 1. Remove small objects (Noise filter)
        clean_binary = remove_small_objects(binary, min_size=min_size, connectivity=2)
        
        # 2. Add back to main mask
        clean_mask[clean_binary] = class_id
        
        # 3. CRITICAL FIX: Label the blobs
        # This assigns unique IDs (1, 2, 3...) to each distinct atom
        labeled_blobs = label(clean_binary, connectivity=2)
        
        # 4. Extract properties from the LABELED image
        props = regionprops(labeled_blobs)
        
        for p in props:
            atoms_data.append({
                'y': p.centroid[0],
                'x': p.centroid[1],
                'class_id': class_id,
                'class_name': CLASS_NAMES[class_id-1],
                'area': p.area
            })
            
    return clean_mask, pd.DataFrame(atoms_data)

def find_defects_unsupervised(image, atom_df, window_size=28):
    """
    Runs the PCA -> UMAP -> GMM pipeline to find defect outliers.
    Returns: outlier_mask (boolean array corresponding to atom_df rows)
    """
    patches = []
    valid_indices = [] # Indices in atom_df that were successfully extracted
    
    h, w = image.shape
    r = window_size // 2
    
    # Extract patches
    for idx, row in atom_df.iterrows():
        y, x = int(row['y']), int(row['x'])
        if y-r >= 0 and y+r < h and x-r >= 0 and x+r < w:
            patch = image[y-r:y+r, x-r:x+r]
            patches.append(patch.flatten())
            valid_indices.append(idx)
            
    if len(patches) < 10:
        return np.zeros(len(atom_df), dtype=bool) # Not enough data

    patches_norm = StandardScaler().fit_transform(patches)
    
    # PCA
    pca = PCA(n_components=min(32, len(patches)), random_state=42)
    pca_features = pca.fit_transform(patches_norm)
    
    # UMAP (Need enough neighbors)
    n_neighbors = min(15, len(patches)-1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    # We run UMAP just to ensure the GMM input space is manifold-aligned (optional, but helps)
    # Actually, running GMM on PCA features is more robust for outlier detection.
    # Let's run GMM on PCA features directly to save time and stability.
    
    # GMM
    n_classes = len(atom_df['class_id'].unique()) + 2 # Approx classes + defects
    gmm = GaussianMixture(n_components=min(n_classes, len(patches)), random_state=42)
    gmm.fit(pca_features)
    
    # Score samples (Log Probability)
    log_prob = gmm.score_samples(pca_features)
    
    # Bottom 5% are outliers
    threshold = np.percentile(log_prob, 5)
    is_outlier_subset = log_prob < threshold
    
    # Map back to original dataframe length
    full_outlier_mask = np.zeros(len(atom_df), dtype=bool)
    full_outlier_mask[valid_indices] = is_outlier_subset
    
    return full_outlier_mask

# --- Main GUI Class ---

class AtomAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AtomAI Defect Analyzer (HAADF-STEM)")
        self.root.geometry("1300x850")
        
        self.model = None
        self.current_image = None
        self.current_filename = None
        self.results_df = None
        self.clean_mask = None
        self.outlier_mask = None
        
        # --- Top Controls ---
        control_frame = tk.Frame(root, padx=10, pady=10, bg="#f0f0f0")
        control_frame.pack(fill=tk.X)
        
        btn_load = tk.Button(control_frame, text="Load .dm3 File", command=self.load_image, bg="white", height=2)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_run = tk.Button(control_frame, text="Run Segmentation & Defect Discovery", command=self.run_analysis, bg="#d1ffbd", height=2)
        btn_run.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(control_frame, text="Status: Waiting to load model...", bg="#f0f0f0", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # --- Main Content (Tabs) ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Segmentation
        self.tab_seg = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_seg, text="1. Segmentation Map")
        
        # Tab 2: Distribution
        self.tab_stats = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_stats, text="2. Class Distribution")
        
        # Tab 3: Defect Map
        self.tab_defects = tk.Frame(self.notebook, bg="white")
        self.notebook.add(self.tab_defects, text="3. Defect Map (Unsupervised)")
        
        # Initialize Canvas placeholders
        self.canvases = {} 

        # Load Model immediately
        self.root.after(100, self.load_model)

    def load_model(self):
        try:
            self.status_label.config(text="Loading AtomAI Model... (This may take a moment)")
            self.root.update()
            
            # Load AtomAI Segmentor
            # Note: Ensure you have the .pt or .h5 file. 
            if os.path.exists(MODEL_PATH):
                self.model = aoi.load_model(MODEL_PATH)
                self.status_label.config(text=f"Model Loaded: {MODEL_PATH}")
            else:
                messagebox.showwarning("Model Missing", f"Could not find model file:\n{MODEL_PATH}\nPlease place it in the folder.")
                self.status_label.config(text="Model Not Found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("DM3 Files", "*.dm3"), ("All Files", "*.*")])
        if file_path:
            try:
                self.current_filename = os.path.basename(file_path)
                self.current_image = load_dm3_file(file_path)
                
                self.status_label.config(text=f"Loaded: {self.current_filename}")
                self.plot_figure(self.current_image, "Input Image", self.tab_seg)
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Could not load DM3 file:\n{str(e)}")

    def run_analysis(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded.")
            return

        self.status_label.config(text="Running AtomAI Segmentation...")
        self.root.update()
        
        try:
            # 1. Prediction
            # AtomAI expects (N, H, W, 1) or (H, W) depending on version, usually robust to standard inputs
            # predict() returns (prob_masks, coordinates) or just prob_masks depending on settings.
            # We want the probability masks to perform manual filtering.
            output, _ = self.model.predict(self.current_image, verbose=0)
            
            # Convert Probability Map (N, H, W, C) -> Integer Mask (H, W)
            # Take index 0 (single image)
            pred_mask = np.argmax(output[0], axis=-1) 
            
            # 2. Filtering (< 30px)
            self.status_label.config(text="Filtering Small Blobs & Extracting Centroids...")
            self.root.update()
            
            self.clean_mask, self.results_df = filter_predictions(pred_mask, min_size=MIN_BLOB_SIZE)
            
            if len(self.results_df) == 0:
                messagebox.showinfo("Info", "No atoms found after filtering.")
                return

            # 3. Unsupervised Defect Discovery
            self.status_label.config(text="Running Unsupervised Defect Detection (PCA/GMM)...")
            self.root.update()
            
            self.outlier_mask = find_defects_unsupervised(self.current_image, self.results_df, window_size=PATCH_SIZE)
            
            # 4. Visualization
            self.plot_segmentation_tab()
            self.plot_distribution_tab()
            self.plot_defect_tab()
            
            self.status_label.config(text="Analysis Complete. Check Tabs.")
            
        except Exception as e:
            print(e)
            messagebox.showerror("Analysis Error", f"An error occurred:\n{str(e)}")

    # --- Plotting Functions ---

    def plot_figure(self, img, title, master_frame, cmap='gray'):
        """Generic plotter to clear a tab and plot a simple image."""
        for widget in master_frame.winfo_children(): widget.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, master=master_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_segmentation_tab(self):
        """Tab 1: Original Image with superimposed colored atoms."""
        for widget in self.tab_seg.winfo_children(): widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.current_image, cmap='gray')
        
        # Color map for classes 1-5
        colors = {1: 'red', 2: 'purple', 3: 'lime', 4: 'yellow', 5: 'cyan'}
        
        # Plot circles
        for _, row in self.results_df.iterrows():
            c_id = row['class_id']
            color = colors.get(c_id, 'white')
            circle = Circle((row['x'], row['y']), radius=4, color=color, fill=False, linewidth=1)
            ax.add_patch(circle)
            
        # Legend
        patches_list = [Circle((0,0), radius=1, color=c, label=n) for i, (n, c) in enumerate(zip(CLASS_NAMES, colors.values()))]
        ax.legend(handles=patches_list, loc='upper right', title="Species")
        
        ax.set_title(f"Segmentation (Filtered > {MIN_BLOB_SIZE}px)")
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab_seg)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_distribution_tab(self):
        """Tab 2: Bar Chart."""
        for widget in self.tab_stats.winfo_children(): widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        counts = self.results_df['class_name'].value_counts()
        # Reorder to match CLASS_NAMES if present
        counts = counts.reindex(CLASS_NAMES, fill_value=0)
        
        colors = ['red', 'purple', 'lime', 'yellow', 'cyan']
        counts.plot(kind='bar', color=colors, ax=ax, edgecolor='black')
        
        ax.set_title(f"Atom Distribution ({len(self.results_df)} total)")
        ax.set_ylabel("Count")
        ax.set_xticklabels(counts.index, rotation=45)
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab_stats)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_defect_tab(self):
        """Tab 3: Defect Map (GMM Outliers)."""
        for widget in self.tab_defects.winfo_children(): widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.current_image, cmap='gray')
        
        # Split data
        defects = self.results_df[self.outlier_mask]
        normal = self.results_df[~self.outlier_mask]
        
        # Plot Normal
        ax.scatter(normal['x'], normal['y'], c='cyan', s=10, alpha=0.3, label='Lattice Atoms')
        
        # Plot Defects
        if len(defects) > 0:
            ax.scatter(defects['x'], defects['y'], 
                       c='red', s=50, marker='o', edgecolors='white', linewidth=1.5, 
                       label=f'Potential Defects ({len(defects)})')
            
        ax.legend(loc='upper right')
        ax.set_title("Unsupervised Defect Discovery (GMM Outliers)")
        ax.axis('off')
        
        canvas = FigureCanvasTkAgg(fig, master=self.tab_defects)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = AtomAnalysisApp(root)
    root.mainloop()