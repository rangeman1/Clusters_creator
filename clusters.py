import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import deque
import tkinter as tk
from tkinter import filedialog

# ---------------------------------------------------------
# Connected Components
# ---------------------------------------------------------

def connected_components(mask):
    labels = -np.ones(mask.shape, dtype=int)
    current = 0
    h, w = mask.shape
    neighbors = [(1,0),(-1,0),(0,1),(0,-1)]

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 1 and labels[y, x] == -1:
                queue = deque([(y, x)])
                labels[y, x] = current

                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in neighbors:
                        yy, xx = cy + dy, cx + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            if mask[yy, xx] == 1 and labels[yy, xx] == -1:
                                labels[yy, xx] = current
                                queue.append((yy, xx))

                current += 1

    return labels, current

# ---------------------------------------------------------
# Główna funkcja klastrująca
# ---------------------------------------------------------

def run_clustering(image_path, eps_value):

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    h, w, _ = img_np.shape

    # Binaryzacja
    gray = img_np.mean(axis=2)
    points_mask = ((gray >= 80) & (gray < 180)).astype(np.uint8)

    # Connected Components
    point_labels, n_points = connected_components(points_mask)

    # Centroidy
    point_centroids = []
    for i in range(n_points):
        ys, xs = np.where(point_labels == i)
        if len(xs) == 0:
            continue
        point_centroids.append([xs.mean(), ys.mean()])

    point_centroids = np.array(point_centroids)

    # DBSCAN
    db = DBSCAN(eps=eps_value, min_samples=1).fit(point_centroids)
    labels = db.labels_
    n_clusters = labels.max() + 1

    # Okręgi
    outline_img = img_np.copy()

    for cluster in range(n_clusters):
        cluster_points = point_centroids[labels == cluster]

        cx = cluster_points[:,0].mean()
        cy = cluster_points[:,1].mean()

        r = np.sqrt((cluster_points[:,0] - cx)**2 + (cluster_points[:,1] - cy)**2).max()

        angles = np.linspace(0, 2*np.pi, 2000)
        for a in angles:
            x = int(cx + r * np.cos(a))
            y = int(cy + r * np.sin(a))
            if 0 <= x < w and 0 <= y < h:
                outline_img[y, x] = [255, 0, 0]

    # Wizualizacja
    plt.figure(figsize=(8, 6))
    plt.imshow(outline_img)
    plt.title(f"Klastry punktów (eps={eps_value})")
    plt.axis("off")
    plt.show()

    # Wypisanie współrzędnych
    clusters = {}
    for cid in range(n_clusters):
        clusters[cid] = point_centroids[labels == cid].tolist()

    print("\nKlastry (współrzędne punktów):")
    for cid, pts in clusters.items():
        print(f"Klaster {cid}: {pts}")

# ---------------------------------------------------------
# GUI (Tkinter)
# ---------------------------------------------------------

def select_file():
    path = filedialog.askopenfilename(
        title="Wybierz obraz punktów",
        filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*")]
    )
    if path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, path)

def start_clustering():
    path = file_entry.get()
    eps_value = eps_slider.get()
    run_clustering(path, eps_value)

root = tk.Tk()
root.title("Klastrowanie punktów")

# Pole wyboru pliku
tk.Label(root, text="Plik obrazu:").pack()
file_entry = tk.Entry(root, width=50)
file_entry.pack()

tk.Button(root, text="Wybierz plik", command=select_file).pack()

# Suwak eps
tk.Label(root, text="Czułość klastrów (eps):").pack()
eps_slider = tk.Scale(root, from_=10, to=150, orient=tk.HORIZONTAL)
eps_slider.set(55)
eps_slider.pack()

# Start
tk.Button(root, text="Uruchom klastrowanie", command=start_clustering).pack()

root.mainloop()
