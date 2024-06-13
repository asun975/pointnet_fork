#%%
import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from source import utils as su
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from path import Path
import plotly.graph_objects as go
import plotly.express as px

"""
https://colab.research.google.com/github/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb#scrollTo=y9hL_IOoMVzP
"""

random.seed = 42
path = Path(r"../ModelNet10")
#%%
folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};
classes

#%%
with open(path/"bed/train/bed_0001.off", 'r') as f:
    verts, faces = su.read_off(f)
i,j,k = np.array(faces).T
x,y,z = np.array(verts).T
len(x)

#%%
def visualize_rotate(data):
    """A function to display animated rotation of meshes and point clouds."""
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                    showactive=False,
                                    y=1,
                                    x=0.8,
                                    xanchor='left',
                                    yanchor='bottom',
                                    pad=dict(t=45, r=10),
                                    buttons=[dict(label='Play',
                                                    method='animate',
                                                    args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                    transition=dict(duration=0),
                                                                    fromcurrent=True,
                                                                    mode='immediate'
                                                                    )]
                                                    )
                                            ]
                                    )
                                ]
                    ),
                    frames=frames
            )

    return fig

def pcshow(xs,ys,zs):
    """A function to accurately visualize point clouds so we could see vertices better."""
    data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                   mode='markers')]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                      line=dict(width=2,
                      color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()
#%%
visualize_rotate([go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50, i=i,j=j,k=k)]).show()
#%%
visualize_rotate([go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers')]).show()
pcshow(x,y,z)
#%% Transforms with point sampling
pointcloud = su.PointSampler(3000)((verts, faces))
pcshow(*pointcloud.T)
#%% Normalize
norm_pointcloud = su.Normalize()(pointcloud)
pcshow(*norm_pointcloud.T)
#%% Augmentations
rot_pointcloud = su.RandRotation_z()(norm_pointcloud)
noisy_rot_pointcloud = su.RandomNoise()(rot_pointcloud)
pcshow(*noisy_rot_pointcloud.T)
