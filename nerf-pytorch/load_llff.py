import numpy as np
import os, imageio


def _load_data(basedir, factor, load_imgs=True):
  poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
  #poses, bds都先沒做transpose

  poses = poses_arr[:, :-2].reshape([-1, 3, 5])
  bds = poses_arr[:, -2:]

  sfx = f'_{factor}'

  imgdir = os.path.join(basedir, 'images' + sfx)
  if not os.path.exists(imgdir):
    print(imgdir, '不存在，return')
    return

  imgfiles = []
  for f in sorted(os.listdir(imgdir)):
    if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'):
      imgfiles.append(os.path.join(imgdir, f))

  if poses.shape[0] != len(imgfiles):
    print(f'Mismatch between imgs {len(imgfiles)} and poses {poses.shape[-1]}')
    return

  sh = imageio.imread(imgfiles[0]).shape
  poses[:, 0, 4] = sh[0]
  poses[:, 1, 4] = sh[1]
  poses[:, 2, 4] = poses[:, 2, 4] / factor

  if not load_imgs:
    return poses, bds

  imgs = []
  for f in imgfiles:
    imgs.append(imageio.imread(f)[:, :, :3]/255)

  imgs = np.array(imgs)

  print('Loaded image data', imgs.shape, poses[0, :, -1])
  return poses, bds, imgs

def normalize(x):
  return x / np.linalg.norm(x)

def viewmatrix(z, y, camera_center):
  z = normalize(z)
  y_avg = y
  x = normalize(np.cross(y_avg, z))
  y = normalize(np.cross(z, x))
  matrix = np.stack([x, y, z, camera_center], 1)
  return matrix

def poses_avg(poses):
  hwf = poses[0, :3, -1:]
  camera_center_mean = poses[:, :3, 3].mean(0)
  z_sum = poses[:, :3, 2].sum(0)
  y_sum = poses[:, :3, 1].sum(0)
  c2w = np.concatenate([viewmatrix(z_sum, y_sum, camera_center_mean), hwf], 1)
  return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
  render_poses = []
  rads = np.array(list(rads)+[1])
  hwf = c2w[:, 4:5]

  for theta in np.linspace(0, 2 * np.pi * rots, N+1)[:-1]:
    c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1]) *rads)
    z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1])))
    render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
  return render_poses

def recenter_poses(poses):
  poses_ = poses
  bottom = np.reshape([0, 0, 0, 1], [1, 4])
  c2w = poses_avg(poses)
  c2w = np.concatenate([c2w[:3, :4], bottom], 0)
  bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])

  poses = np.concatenate([poses[:, :3, :4], bottom], 1)
  poses = np.linalg.inv(c2w) @ poses
  poses_[:, :3, :4] = poses[:, :3, :4]
  poses = poses_
  return poses

def spherify_poses(poses, bds):
  def p3x4_to_4x4(p):
    p_0001 = [[[0,0,0,1]]]
    N_p0001 = np.tile(p_0001, [p.shape[0], 1, 1])
    p4x4 = np.concatenate([p, N_p0001], 1)
    return p4x4

  rays_z_camera_direction = poses[:, :3, 2:3]
  rays_o_camera_center = poses[:, :3, 3:4]

  #求場景中心
  def target_center_position(rays_o_camera_center, rays_z_camera_direction):
    A_i = np.eye(3) - rays_z_camera_direction * np.transpose(rays_z_camera_direction, [0, 2, 1])
    b_i = -A_i @ rays_o_camera_center
    pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist
  target_center = target_center_position(rays_o_camera_center, rays_z_camera_direction)

  center = target_center
  up = (poses[:, :3, 3] - center).mean(0)

  #camera方向變得與世界坐標系一樣，target_center移動到世界坐標系原點
  new_camera_z = normalize(up)
  new_camera_x = normalize(np.cross([0.1, 0.2, 0.3], new_camera_z))
  new_camera_y = normalize(np.cross(new_camera_z, new_camera_x))
  pos = center
  c2w = np.stack([new_camera_x, new_camera_y, new_camera_z, pos], 1)

  poses_reset = np.linalg.inv(p3x4_to_4x4(c2w.reshape(1, 3, 4))) @ p3x4_to_4x4(poses[:, :3, :4])

  rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

  sc = 1/rad
  poses_reset[:, :3, 3] *= sc
  bds *= sc
  rad *= sc

  #質心
  centroid = np.mean(poses_reset[:, :3, 3], 0)
  z_height = centroid[2]
  radcircle = np.sqrt(rad**2 - z_height**2)

  new_poses = []

  for theta in np.linspace(0, 2*np.pi, 120): #在0~2pi中取120個點
    camera_position = np.array([radcircle * np.cos(theta), radcircle * np.sin(theta), z_height])

    up = np.array([0, 0, -1])

    z = normalize(camera_position) #此時z軸是相機朝向
    x = normalize(np.cross(z, up))
    y = normalize(np.cross(z, x))
    pos = camera_position
    p = np.stack([x, y, z, pos], 1)

    new_poses.append(p)

  new_poses = np.array(new_poses)

  new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
  poses_reset = np.concatenate([poses_reset[:, :3, :], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)
  return poses_reset, new_poses, bds

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=0.75, spherify=False, path_zflat=False):

  poses, bds, imgs = _load_data(basedir, factor)
  print('Loaded', basedir, bds.min(), bds.max())

  poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], 2)

  poses = poses.astype(np.float32)
  images = imgs.astype(np.float32)
  bds = bds.astype(np.float32)

  if bd_factor is None:
    sc = 1
  else:
    sc = 1 / (bds.min() * bd_factor)

  poses[:, :3, 3] *= sc
  bds *= sc

  if recenter:
    poses = recenter_poses(poses)
  if spherify:
    poses, render_poses, bds = spherify_poses(poses, bds)
  else:
    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3, :4])

    up = normalize(poses[:, :3, 1].sum(0))

    close_depth = bds.min() * 0.9
    inf_depth = bds.max() * 5
    dt = 0.75
    focal = 1 / (((1-dt) / close_depth + dt / inf_depth))

    shrink_factor = 0.8
    zdelta = close_depth * 0.2
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    if path_zflat:
      zloc = -close_depth * 0.1
      c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
      rads[2] = 0
      N_rots = 1
      N_views /= 2
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=0.5, rots = N_rots, N = N_views)

  render_poses = np.array(render_poses).astype(np.float32)
  c2w = poses_avg(poses)
  print('Data:')
  print(poses.shape, images.shape, bds.shape)

  dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
  i_test = np.argmin(dists)
  print('HOLDOUT view is', i_test)

  images = images.astype(np.float32)
  poses = poses.astype(np.float32)
  return images, poses, bds, render_poses, i_test
