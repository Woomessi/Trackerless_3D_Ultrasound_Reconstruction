# Functions used during plotting

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _series_to_corners(series):
    """Convert series (N, 3, 3) → corners (N, 4, 3).

    series[:, 0] = center, series[:, 1] = lower-left, series[:, 2] = lower-right.

    The four corners (in order: ll → lr → ur → ul) are derived as:
        upper-right = 2*center - lower-left
        upper-left  = 2*center - lower-right
    """
    if isinstance(series, torch.Tensor):
        series = series.detach().cpu().numpy()
    center = series[:, 0, :]   # (N, 3)
    ll     = series[:, 1, :]   # lower-left
    lr     = series[:, 2, :]   # lower-right
    ur     = 2 * center - ll   # upper-right
    ul     = 2 * center - lr   # upper-left
    return np.stack([ll, lr, ur, ul], axis=1)  # (N, 4, 3)


def add_series_rects(plotter, series, indices=None, colors=None,
                     opacity=0.5, edge_width=2, frames=None):
    """Add series frame rectangles to an existing PyVista Plotter.

    Parameters
    ----------
    plotter  : pv.Plotter
        Target plotter.
    series   : (N, 3, 3) tensor or ndarray
        [center, lower-left, lower-right] in world mm.
    indices  : list[int] or None
        Which frames to draw.  None = all frames.
    colors   : list[str/tuple] or str or None
        Per-frame colours, a single colour for all, or None (uses colormap).
    opacity  : float
        Face opacity.
    edge_width : int
        Outline width in pixels.
    frames   : (N, H, W) ndarray or None
        If given, texture-map the image onto each rectangle via pv.Texture.
    """
    import pyvista as pv

    corners = _series_to_corners(series)      # (N, 4, 3)
    if indices is None:
        indices = list(range(len(corners)))

    # Resolve colour list
    if colors is None:
        cmap_fn = plt.get_cmap('coolwarm')
        colors = [cmap_fn(i / max(len(indices) - 1, 1))[:3]
                  for i in range(len(indices))]
    elif isinstance(colors, str):
        colors = [colors] * len(indices)

    for rank, i in enumerate(indices):
        c = corners[i]          # (4, 3): ll, lr, ur, ul
        color = colors[rank]

        # ── Filled quad ──────────────────────────────────────────────────
        if frames is not None:
            img = frames[i]
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            # pyvista Texture requires RGB.
            # Flip vertically: VTK/OpenGL textures have t=0 at the bottom,
            # but image arrays have row-0 at the top, which causes an upside-down
            # display without this flip.
            rgb = np.stack([img, img, img], axis=-1)[::-1]      # (H, W, 3)
            tex = pv.Texture(np.ascontiguousarray(rgb))

            # Build a StructuredGrid (2×2) for texture mapping
            # corners order: ll(0,1) lr(1,1) ur(1,0) ul(0,0)
            #   → grid[row, col]: [0,0]=ul, [0,1]=ur, [1,0]=ll, [1,1]=lr
            pts = np.array([c[3], c[2], c[0], c[1]], dtype=np.float32
                           ).reshape(2, 2, 3)
            grid = pv.StructuredGrid(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2])
            grid.texture_map_to_plane(inplace=True)
            plotter.add_mesh(grid, texture=tex, opacity=opacity,
                             show_edges=False)
        else:
            points = c.astype(np.float32)
            faces  = np.array([4, 0, 1, 2, 3])
            quad   = pv.PolyData(points, faces)
            plotter.add_mesh(quad, color=color, opacity=opacity,
                             show_edges=False)

        # ── Outline ──────────────────────────────────────────────────────
        # add_lines requires pairs of points (each segment = 2 points)
        segments = np.array([
            c[0], c[1],   # ll → lr
            c[1], c[2],   # lr → ur
            c[2], c[3],   # ur → ul
            c[3], c[0],   # ul → ll
        ], dtype=np.float32)
        plotter.add_lines(segments, color=color, width=edge_width)

        # ── Frame index label at centre ───────────────────────────────────
        centre = corners[i].mean(axis=0)
        plotter.add_point_labels(
            [centre], [str(i)],
            font_size=10, text_color=color,
            fill_shape=False, margin=0, always_visible=True,
        )


def visualize_series(series, step=1, cmap='coolwarm', opacity=0.4,
                     frames=None, title='Series'):
    """Visualize US frame positions in 3D using PyVista.

    Parameters
    ----------
    series : (N, 3, 3) tensor or ndarray
        [center, lower-left, lower-right] in world mm.
    step   : int
        Render every `step`-th frame (default 1 = all).
    cmap   : str
        Colormap for frame-index coloring.
    opacity: float
        Quad face opacity.
    frames : (N, H, W) ndarray or None
        Grayscale images (uint8 or float [0,1]) to texture-map onto each quad.
        Requires pyvista >= 0.38.
    title  : str
        Window title.
    """
    import pyvista as pv

    idx = list(range(0, len(_series_to_corners(series)), step))
    cmap_fn = plt.get_cmap(cmap)
    colors  = [cmap_fn(i / max(len(idx) - 1, 1))[:3] for i in range(len(idx))]

    plotter = pv.Plotter(title=title)
    add_series_rects(plotter, series, indices=idx, colors=colors,
                     opacity=opacity, frames=frames)
    plotter.show_axes()
    plotter.set_background('black')
    plotter.show()


def visualize_series_mpl(series, step=1, cmap='coolwarm', alpha=0.3,
                         ax=None, label=None):
    """Visualize US frame positions in 3D using Matplotlib.

    Parameters
    ----------
    series : (N, 3, 3) tensor or ndarray
        [center, lower-left, lower-right] in world mm.
    step   : int
        Render every `step`-th frame (default 1 = all).
    cmap   : str
        Colormap for frame-index coloring.
    alpha  : float
        Face transparency.
    ax     : Axes3D or None
        Existing axes to draw into; creates a new figure if None.
    label  : str or None
        Legend label (applied to the first/last frame outline only).

    Returns
    -------
    ax : Axes3D
    """
    corners = _series_to_corners(series)  # (N, 4, 3)
    idx = np.arange(0, len(corners), step)
    corners = corners[idx]
    M = len(corners)

    colormap = plt.get_cmap(cmap)
    colors   = colormap(np.linspace(0, 1, M))

    show = ax is None
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')

    # Draw quads as filled polygons
    polys = [c for c in corners]           # list of (4, 3) arrays
    poly_colors = [(r, g, b, alpha) for r, g, b, _ in colors]
    collection = Poly3DCollection(polys, facecolors=poly_colors,
                                  edgecolors='none')
    ax.add_collection3d(collection)

    # Outline first (blue) and last (red) frame
    for frm_corners, color, lw in [(corners[0], 'blue', 2),
                                   (corners[-1], 'red',  2)]:
        loop = np.vstack([frm_corners, frm_corners[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                color=color, linewidth=lw,
                label=label if color == 'blue' else None)

    # Centre-point trajectory
    centres = corners.mean(axis=1)         # (M, 3)
    ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2],
               c=np.linspace(0, 1, M), cmap=cmap, s=4, alpha=0.6)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_box_aspect([1, 1, 1])
    if label:
        ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return ax

def reference_image_points(image_size, density=2):
    """
    :param image_size: (x, y), used for defining default grid image_points
    :param density: (x, y), point sample density in each of x and y, default n=2
    """
    if isinstance(density,int):
        density=(density,density)

    image_points = torch.flip(torch.cartesian_prod(
        torch.linspace(1, image_size[0], density[0]),
        torch.linspace(1, image_size[1] , density[1])
    ).t(),[0])
    
    image_points = torch.cat([
        image_points, 
        torch.zeros(1,image_points.shape[1])*image_size[0]/2,
        torch.ones(1,image_points.shape[1])
        ], axis=0)
    
    return image_points

def transform_t2t(tforms, tforms_inv,pairs):
    # get the transformation between two tools, calculated from NDI recorded transformation, which is the transformation between the tool and the world
    tforms_world_to_tool0 = tforms_inv[pairs[:,0],:,:]
    tforms_tool1_to_world = tforms[pairs[:,1],:,:]
    return torch.matmul(tforms_world_to_tool0, tforms_tool1_to_world)  # tform_tool1_to_tool0

def data_pairs_adjacent(num_frames):
    # obtain the data_pairs to compute the tarnsfomration between frames and the reference (first) frame
    
    return torch.tensor([[0,n0] for n0 in range(num_frames)])

def data_pairs_local(num_frames):
    # obtain the data_pairs to compute the tarnsfomration between frames and the reference (the immediate previous) frame
    
    return torch.tensor([[n0,n0+1] for n0 in range(num_frames)])

def read_calib_matrices(filename_calib):
    # read the calibration matrices from the csv file
    # T{image->tool} = T{image_mm -> tool} * T{image_pix -> image_mm}}
    tform_calib = np.empty((8,4), np.float32)
    with open(filename_calib,'r') as csv_file:
        txt = [i.strip('\n').split(',') for i in csv_file.readlines()]
        tform_calib[0:4,:]=np.array(txt[1:5]).astype(np.float32)
        tform_calib[4:8,:]=np.array(txt[6:10]).astype(np.float32)
    return torch.tensor(tform_calib[0:4,:]),torch.tensor(tform_calib[4:8,:]), torch.tensor(tform_calib[4:8,:] @ tform_calib[0:4,:])

def plot_scan(gt,frame,saved_name,step,color,width = 4, scatter = 8, legend_size=50,legend = None):
    # plot the scan in 3D

    fig = plt.figure(figsize=(35,15))
    axs=[]
    for i in range(2):
        axs.append(fig.add_subplot(1,2,i+1,projection='3d'))
    plt.tight_layout()

    plotting(gt,frame,axs,step,color,width, scatter, legend_size,legend = legend)
    plt.savefig(saved_name +'.png')
    plt.close()

def plotting(gt,frame,axs,step,color,width = 4, scatter = 8, legend_size=50,legend = None): 
    # plot surface
    ysize, xsize = frame.shape[-2:]
    grid=np.meshgrid(np.linspace(0,1,ysize),np.linspace(0,1,xsize),indexing='ij')
    coord = np.zeros((3,ysize,xsize))

    for i_frame in range(0,gt.shape[0],step): 
        gx, gy, gz = [gt[i_frame, ii, :] for ii in range(3)]
        gx, gy, gz = gx.reshape(2, 2), gy.reshape(2, 2), gz.reshape(2, 2)
        coord[0]=gx[0,0]+(gx[1,0]-gx[0,0])*(grid[0])+(gx[0,1]-gx[0,0])*(grid[1])
        coord[1]=gy[0,0]+(gy[1,0]-gy[0,0])*(grid[0])+(gy[0,1]-gy[0,0])*(grid[1])
        coord[2]=gz[0,0]+(gz[1,0]-gz[0,0])*(grid[0])+(gz[0,1]-gz[0,0])*(grid[1])
         
        pix_intensities = (frame[i_frame, ...]/frame[i_frame, ...].max())
        for i,ax in enumerate(axs):
            ax.plot_surface(coord[0], coord[1], coord[2], facecolors=plt.cm.gray(pix_intensities), shade=False,linewidth=0, antialiased=True, alpha=0.5)
    # plot gt
    gx_all, gy_all, gz_all = [gt[:, ii, :] for ii in range(3)]
    for i,ax in enumerate(axs):
        ax.scatter(gx_all[...,0], gy_all[...,0], gz_all[...,0],  alpha=0.5, c = color, s=scatter, label=legend)
        ax.scatter(gx_all[...,1], gy_all[...,1], gz_all[...,1],  alpha=0.5,c = color, s=scatter)
        ax.scatter(gx_all[...,2], gy_all[...,2], gz_all[...,2],  alpha=0.5, c = color,s=scatter)
        ax.scatter(gx_all[...,3], gy_all[...,3], gz_all[...,3],  alpha=0.5,c = color, s=scatter)
        # plot the first frame and the last frame
        ax.plot(gt[0,0,0:2], gt[0,1,0:2], gt[0,2,0:2], 'b', linewidth = width)
        ax.plot(gt[0,0,[1,3]], gt[0,1,[1,3]], gt[0,2,[1,3]], 'b', linewidth = width) 
        ax.plot(gt[0,0,[3,2]], gt[0,1,[3,2]], gt[0,2,[3,2]], 'b', linewidth = width) 
        ax.plot(gt[0,0,[2,0]], gt[0,1,[2,0]], gt[0,2,[2,0]], 'b', linewidth = width)
        ax.plot(gt[-1,0,0:2], gt[-1,1,0:2], gt[-1,2,0:2], 'r', linewidth = width)
        ax.plot(gt[-1,0,[1,3]], gt[-1,1,[1,3]], gt[-1,2,[1,3]], 'r', linewidth = width) 
        ax.plot(gt[-1,0,[3,2]], gt[-1,1,[3,2]], gt[-1,2,[3,2]], 'r', linewidth = width) 
        ax.plot(gt[-1,0,[2,0]], gt[-1,1,[2,0]], gt[-1,2,[2,0]], 'r', linewidth = width)


        ax.axis('equal')
        ax.grid(False)
        ax.legend(fontsize = legend_size,markerscale = 5,scatterpoints = 5)
        # ax.axis('off')
        ax.set_xlabel('x',fontsize=legend_size)
        ax.set_ylabel('y',fontsize=legend_size)
        ax.set_zlabel('z',fontsize=legend_size)
        plt.rc('xtick', labelsize=legend_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=legend_size)    # fontsize of the tick labels
        if i==0:
            ax.view_init(10,30,0)
        else:
            ax.view_init(30,30,0)


def plot_scan_label_pred(gt,pred,frame,color,saved_name,step,width = 4, scatter = 8, legend_size=50):
    # plot the scan in 3D

    fig = plt.figure(figsize=(35,15))
    axs=[]
    for i in range(2):
        axs.append(fig.add_subplot(1,2,i+1,projection='3d'))
    plt.tight_layout()

    plotting(gt,frame,axs,step,color[0],width = 4, scatter = 8, legend_size=50,legend = 'GT')
    plotting(pred,frame,axs,step,color[1],width = 4, scatter = 8, legend_size=50,legend = 'Pred')
    
    plt.savefig(saved_name +'.png')
    plt.close()
                