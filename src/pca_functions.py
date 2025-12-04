import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from adjustText import adjust_text
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Arc, PathPatch
from matplotlib.path import Path
import os

def load_Carmona_et_al_data():
    root_traits0 = pd.read_csv("data/Root_traits.txt", sep=' ')
    root_traits = root_traits0[['SRL', 'D', 'RTD', 'N']].dropna()
    ############################
    above_traits0 = pd.read_csv('data/Above_traits.txt', sep=' ')
    above_traits = above_traits0[['la', 'ln', 'ph', 'sla', 'ssd', 'sm']].dropna()
    above_traits['lma'] = 1/above_traits['sla']
    above_traits = above_traits.drop(['sla'],axis=1)
    above_traits = np.log10(above_traits)
    ############################
    common_species = root_traits.index.intersection(above_traits.index)
    root_traits_common    = root_traits.loc[common_species]
    above_traits_common   = above_traits.loc[common_species]
    data_df = pd.concat([above_traits_common, root_traits_common], axis=1)
    ############################
    return data_df 
###############################################################################
###############################################################################
def compute_residuals(df, target_cols, predictor_cols):
    """
    Regress each target column on predictor columns and return residuals.
    """
    X = df[predictor_cols]
    residuals = pd.DataFrame(index=df.index)
    for col in target_cols:
        y = df[col]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals[col] = y - y_pred
    return residuals
#####################################
def conditional_pca(data, group_control=['D','ssd','la'], n_components=4):
    """
    Perform PCA on:
      1. Raw main variables
      2. Residuals of main variables after regressing out control variables
    """
    group_main = [c for c in data.columns if c not in group_control]


    resid_df = compute_residuals(data,target_cols=group_main,predictor_cols=group_control)
    raw_df = data[group_main].copy(deep=True)
    
    comp_names = [f"PC{i+1}" for i in range(n_components)]
    
    ################
    scaler_raw = StandardScaler()
    raw_scaled = scaler_raw.fit_transform(raw_df)
    raw_scaled_df = pd.DataFrame(raw_scaled,index=raw_df.index,columns=raw_df.columns)
    ################
    scaler_res = StandardScaler()
    resid_scaled = scaler_res.fit_transform(resid_df)
    resid_scaled_df = pd.DataFrame(resid_scaled,index=resid_df.index,columns=resid_df.columns)
    ################
    # PCA on raw data of the main group columns
    pca_raw = PCA(n_components=n_components)
    scores_raw = pca_raw.fit_transform(raw_scaled)
    scores_raw_df = pd.DataFrame(scores_raw, index=data.index, columns=comp_names)
    loadings_raw_df = pd.DataFrame(pca_raw.components_,index=comp_names,columns=group_main)

    # PCA on residuals
    pca_res = PCA(n_components=n_components)
    scores_res = pca_res.fit_transform(resid_scaled)
    scores_res_df = pd.DataFrame(scores_res, index=data.index, columns=comp_names)
    loadings_res_df = pd.DataFrame(pca_res.components_,index=comp_names,columns=resid_df.columns)

    results = {
        "control_variables": group_control,
        "main_variables": group_main,
        "n_components": n_components,

        "residual_scaled_df": resid_scaled_df,
        "residual_res": {
            "scores": scores_res_df,
            "loadings": loadings_res_df,
            "variance_explained": pca_res.explained_variance_ratio_},

        "raw_scaled_df": raw_scaled_df,
        "raw_res": {
            "scores": scores_raw_df,
            "loadings": loadings_raw_df,
            "variance_explained": pca_raw.explained_variance_ratio_},
    }

    return results

###############################################################################
def pca_biplot(results, pairs = [(0, 1), (2, 3)], arrow_scale = 4, 
                       c1='gray', c2='red', is_residual = False):
    """
    Plots PCA scatter plots with biplot arrows, KDE density heatmap, and quantile contours.
    """
    
    # If is_residual=True, the function plots residual PCA; otherwise, it plots raw PCA.
    key = "residual_res" if is_residual else "raw_res"
    data_dict = results.get(key, {})
    
    scores_main = data_dict['scores'].values
    components_main = data_dict['loadings'].values
    explained_main = data_dict['variance_explained']
    features_main = data_dict['loadings'].columns.tolist()
  
    
    fig_c, axes_c = plt.subplots(1, 2, figsize=(7, 3), dpi=500)
    axes_c = axes_c.flatten()

    quantiles = [0.5, 0.95, 0.97]
    lw_map = {0.5: 0.75, 0.95: 0.5, 0.97: 0.75} 
    
    white_yellow_orange = LinearSegmentedColormap.from_list('white_yellow_orange',['white', 'gold', 'orange'])
    
    colors = {'lma':'r', 'ln':'r', 'N':'r', 'ph':'darkgreen', 'ssd':'darkgreen',
              'sm':'darkgreen', 'SRL':'b', 'D':'b', 'RTD':'b'}
      
    for idx, (i, j) in enumerate(pairs):
        ax = axes_c[idx]
        
        x = scores_main[:, i]
        y = scores_main[:, j]
        ax.scatter(x, y, alpha=0.6, color=c1, s=2, edgecolor=c1)
        
        ######################
        # Density 
        kde = gaussian_kde(np.vstack([x, y]))

        pad_x = 0.05 * (x.max() - x.min())
        pad_y = 0.05 * (y.max() - y.min())
        x_min, x_max = x.min() - pad_x, x.max() + pad_x
        y_min, y_max = y.min() - pad_y, y.max() + pad_y

        x_grid, y_grid = np.mgrid[x_min:x_max:250j, y_min:y_max:250j]
        grid = np.vstack([x_grid.ravel(), y_grid.ravel()])
        z_grid = kde(grid).reshape(x_grid.shape)  
        z_grid /= z_grid.sum()
        ax.imshow(np.rot90(z_grid),cmap=white_yellow_orange, extent=[x_min, x_max, y_min, y_max], alpha=0.6, aspect='auto')
        ######################
        ######################
        # Contours
        sorted_densities = np.sort(z_grid.ravel())[::-1]
        cumsum = np.cumsum(sorted_densities)
        cumsum /= cumsum[-1] # Cumulative sum of densities (normalized to 1)

        # Density values of the probability quantiles
        value_at_quantile = {} 
        for q in quantiles:
            q_index = np.searchsorted(cumsum, q)
            value_at_quantile[q] = sorted_densities[min(q_index, len(sorted_densities)-1)]

        levels = np.unique(np.sort(list(value_at_quantile.values())))
        ax.contour(x_grid, y_grid, z_grid, levels=levels, colors='black', linewidths=[lw_map[q] for q in quantiles], alpha=0.9)
        ######################
        
        texts = []
        # Biplot
        for k in range(components_main.shape[1]):
            loading_x = components_main[i, k]
            loading_y = components_main[j, k]
            
            x_arrow, y_arrow = loading_x * arrow_scale, loading_y * arrow_scale
            
            ax.arrow(0, 0, x_arrow, y_arrow, color=colors.get(features_main[k], 'black'), alpha=0.9,
                      head_width=0.3, head_length=0.4,
                      length_includes_head=True, zorder=10)
            
            offset = 0.5
            vec = np.array([loading_x, loading_y])
            norm_vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else np.array([1, 0])
            
            xt = x_arrow + norm_vec[0] * offset
            yt = y_arrow + norm_vec[1] * offset
            
            texts.append(ax.text(xt, yt, features_main[k], 
                    color='k', 
                    fontsize=12, zorder=15, ha='center', va='center'))
        adjust_text(texts, 
                    ax=ax, 
                    arrowprops=dict(arrowstyle="-", color='lightgray', lw=0.5, alpha=0.5))

        if idx==0: 
            ax.set_title('a.',fontsize=14,x=-0.05,y=1.01,fontweight='bold') 
        else: 
            ax.set_title('b.',fontsize=14,x=-0.05,y=1.01,fontweight='bold') 

            
        ax.set_xlabel(f'PC{i+1} ({explained_main[i]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC{j+1} ({explained_main[j]*100:.1f}%)', fontsize=12)
        ax.axhline(0, color='grey', lw=1, zorder=0)
        ax.axvline(0, color='grey', lw=1, zorder=0)
        ax.grid(False)
        ax.set_ylim(-5, 5)
        ax.set_xlim(-5, 5)

    plt.tight_layout()
    plt.savefig(f'figs/{key}_pca_biplot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
###############################################################################
###############################################################################
def angle_between(v1, v2):
    """Angle in degrees between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosang = np.clip(cosang, -1.0, 1.0) 
    return np.degrees(np.arccos(cosang))
#######################################################
class HandlerVshape(HandlerPatch):
    """Legend handler drawing 2 diverging lines + small semi-circle."""
    def __init__(self, color1, color2, lw=3,
                 width_frac=0.8, height_frac=0.8,
                 arc_lw=1, arc_offset=0.15, **kwargs):
        super().__init__(**kwargs)
        self.color1, self.color2 = color1, color2
        self.lw, self.arc_lw = lw, arc_lw
        self.width_frac, self.height_frac = width_frac, height_frac
        self.arc_offset = arc_offset

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):
        
        cx = width * 0.5
        y0 = 0.2 * height
        y1 = y0 + self.height_frac * height
        hw = 0.5 * self.width_frac * width

        # Two branches
        pts = [
            PathPatch(Path([(cx, y0), (cx - hw, y1)], [Path.MOVETO, Path.LINETO]),
                      facecolor='none', lw=self.lw, edgecolor=self.color1, transform=trans),
            PathPatch(Path([(cx, y0), (cx + hw, y1)], [Path.MOVETO, Path.LINETO]),
                      facecolor='none', lw=self.lw, edgecolor=self.color2, transform=trans)
        ]

        # Arc (angle indicator)
        arc = Arc((cx, y0 + self.arc_offset * height),
                  0.25 * width, 0.25 * width,
                  theta1=0, theta2=180,
                  edgecolor='k', lw=self.arc_lw,
                  transform=trans)

        return pts + [arc]
####################################################
def plot_3d_double_arrow(ax, start, end, color='k', length_ratio=0.08, linewidth=12,
                         alpha=0.2):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)


    vec = end - start
    length = np.linalg.norm(vec)
    direction = vec / length

    # Plot arrow from start to end
    ax.quiver(*start, *direction,
              length=length, arrow_length_ratio=length_ratio,
              color=color, linewidth=linewidth, alpha=alpha)
    # Plot arrow from end to start (opposite direction)
    ax.quiver(*end, *(-direction),
              length=length, arrow_length_ratio=length_ratio,
              color=color, linewidth=linewidth, alpha=alpha)

#############################################################
def compute_group_vector(components, features, group):
    idx = {f: i for i, f in enumerate(features)}
    unit_vecs = []

    for name in group:
        flip = name.startswith('-')
        feat = name[1:] if flip else name
        
        if feat not in idx:
            continue

        v = components[:3, idx[feat]]
        v = v / np.linalg.norm(v)
        
        if flip: v = -v
        unit_vecs.append(v)

    if not unit_vecs:
        return None

    # Mean composite direction
    mean_vec = np.mean(unit_vecs, axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec) 

    return mean_vec
###############################################################
def plot_pca_3d(results, scale=5, pad=0.2, is_residual=True,
                feature_color='k', dpi=150, ax=None, title=None, 
                azim0=10, elev0=35, roll0=0, c1='royalblue', c2='k'):

    key = "residual_res" if is_residual else "raw_res"
    d = results[key]

    scores     = d['scores'].values
    components = d['loadings'].values
    explained  = d['variance_explained']
    features   = d['loadings'].columns.tolist()

    if ax is None:
        fig = plt.figure(figsize=(10, 8), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], alpha=0.0, color=c1)

    groups = {
        "red":   ["-lma", "ln", "N"],
        "green": ["ssd", "ph", "sm"],
        "blue":  ["RTD", "-D", "-SRL"],
    }

    load_vecs = components[:3, :] * scale
    arrow_scale = np.max(np.linalg.norm(load_vecs.T, axis=1)) + pad

    # Composite vectors
    arrows = {}
    for color, group in groups.items():
        g = compute_group_vector(components, features, group)
        if g is None:
            continue
        arrows[color] = arrow_scale * g
        plot_3d_double_arrow(ax, -arrows[color], arrows[color], color=color)

    # Loading vectors
    for k, feat in enumerate(features):
        vec = load_vecs[:, k]
        ax.quiver(0, 0, 0, *vec, color=c2,
                  arrow_length_ratio=0.15, linewidth=2)

        offset = 0.25 if feat in ['ln', 'N'] else 0.05
        label_pos = vec + offset * np.linalg.norm(vec)
        ax.text(*label_pos, feat, color='k', fontsize=12, zorder=100)

    pairs = [("green", "red"), ("green", "blue"), ("red", "blue")]
    handles, labels, handler_map = [], [], {}

    for c1_, c2_ in pairs:
        if c1_ not in arrows or c2_ not in arrows:
            continue
        v1, v2 = arrows[c1_], arrows[c2_]
        ang = angle_between(v1, v2)
        dummy = object()
        handles.append(dummy)
        labels.append(f"{ang:.1f}°")
        handler_map[dummy] = HandlerVshape(c1_, c2_)

    ax.legend(handles, labels,
              loc="upper left", bbox_to_anchor=(0.7, 0.9),
              handler_map=handler_map, frameon=False,
              handlelength=4, handleheight=2, fontsize=14)

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=14, labelpad=-4)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=14, labelpad=-4)
    ax.set_zlabel(f"PC3 ({explained[2]*100:.1f}%)", fontsize=14, labelpad=-4)
    ax.set_title(title, fontsize=18, x=0.05, y=0.9, fontweight='bold')

    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_zlim(-4, 4)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([]) 
    ax.view_init(elev=elev0, azim=azim0, roll=roll0)
    plt.tight_layout()
    plt.savefig(f'figs/{key}_pca_3d.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
###############################################################################
###############################################################################
fig_dir ='figs' 
os.makedirs(fig_dir, exist_ok=True)

data_complete = load_Carmona_et_al_data()
conditional_res = conditional_pca(data_complete,group_control=['SRL','ssd','la'])
pca_biplot(conditional_res,is_residual=True)
plot_pca_3d(conditional_res,is_residual=True)


