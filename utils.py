import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

IX, IY, IVX, IVY, IAX, IAY = 0, 1, 2, 3, 4, 5  # indexes in state CA
IPHI, IV, IW = 2, 3, 4                         # indexes in state CTRV
IMR, IMPHI, IMD, IMX, IMY = 0, 1, 2, 3, 4      # indexes in measuremnents

def wrap_angle2(angle_rad: float):
    return np.arctan2(np.sin(angle_rad), np.cos(angle_rad))


def circular_mean(angles: np.ndarray, weights: np.ndarray = None):
    if weights is None:
        weights = np.ones_like(angles)
    return np.arctan2(np.dot(np.sin(angles), weights), np.dot(np.cos(angles), weights))


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts




# To make filterpy UKF work
def fx_ctrv(x: np.ndarray, T: float):
    w = x[IW]
    if np.abs(w) > np.radians(0.5):
        phi_p = x[IPHI] + w*T
        rc = x[IV]/w
        x[IX] += rc * (np.sin(phi_p) - np.sin(x[IPHI])) 
        x[IY] += rc * (-np.cos(phi_p) + np.cos(x[IPHI]))
        x[IPHI] = phi_p 
    else:
        x[IX] += x[IV]*np.cos(x[IPHI]) * T
        x[IY] += x[IV]*np.sin(x[IPHI]) * T
    return x


def hx_pos_only(x):
    x, y = x[IX], x[IY]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([r, phi])


def get_process_noise_matrix(x:np.ndarray, var_v, var_w, T: float):
    q = np.diag([var_v, var_w])
    phi = x[IPHI]
    gamma = np.array([
        [0.5*np.cos(phi)*T**2, 0],
        [0.5*np.sin(phi)*T**2, 0],
        [0, 0.5*T**2],
        [T, 0],
        [0, T],
    ])
    Q = gamma @ q @ gamma.T
    return Q