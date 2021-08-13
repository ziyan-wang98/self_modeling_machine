import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

def _get_coord(features, axis):
  """Returns the keypoint coordinate encoding for the given axis.
  Args:
    features: A tensor of shape [B, F_h, F_w, K] where K is the number of
      keypoints to extract.
    axis: `int` which axis to extract the coordinate for. Has to be axis 1 or 2.
  Returns:
    A tensor of shape [B, K] containing the keypoint centers along the given
      axis. The location is given in the range [-1, 1].
  """

  if axis != 3 and axis != 2:
    raise ValueError("Axis needs to be 3 or 2.")

  other_axis = 3 if axis == 2 else 3
  axis_size = features.shape[axis]

  # Compute the normalized weight for each row/column along the axis
  g_c_prob = features.mean(dim=other_axis, keepdim=False)
  g_c_prob = F.softmax(g_c_prob, dim=-1)

  # Linear combination of the interval [-1, 1] using the normalized weights to
  # give a single coordinate in the same interval [-1, 1]
  scale = th.linspace(-1.0, 1.0, axis_size).reshape(1, 1, axis_size).cuda()

  # print(g_c_prob.shape, scale.shape)
  # coordinate = (g_c_prob * scale).sum(dim=1, keepdim=False)

  coordinate = (g_c_prob * scale)

  return coordinate

def _get_keypoint_mus(keypoint_features):
  """Returns the keypoint center points.
  Args:
    keypoint_features: A tensor of shape [B, F_h, F_w, K] where K is the number
      of keypoints to extract.
  Returns:
    A tensor of shape [B, K, 2] of the y, x center points of each keypoint. Each
      center point are in the range [-1, 1]^2. Note: the first element is the y
      coordinate, the second is the x coordinate.
  """
  gauss_y = _get_coord(keypoint_features, 3)
  gauss_x = _get_coord(keypoint_features, 2)
  # print('gx', gauss_x.shape, 'gy', gauss_y.shape)
  gauss_mu = th.stack([gauss_y, gauss_x], dim=-1)
  return gauss_mu

def _get_gaussian_maps(mu, map_size, inv_std, power=2):
  """Transforms the keypoint center points to a gaussian masks."""
  # mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
  bs, nk, _, _ = mu.shape
  device = "cuda" if th.cuda.is_available() else "cpu"

  y = th.linspace(-1.0, 1.0, map_size[0]).resize(1, 1, map_size[0], 1).repeat(bs, nk, 1, 1).to(device)
  x = th.linspace(-1.0, 1.0, map_size[1]).resize(1, 1, 1, map_size[1]).repeat(bs, nk, 1, 1).to(device)

  mu_y = mu[:, :, :, 0].reshape(bs, nk, map_size[0], 1)
  mu_x = mu[:, :, :, 1].reshape(bs, nk, 1, map_size[1])
  # print('mx', mu_x.shape, 'my', mu_y.shape)


  g_y = th.pow(y - mu_y, power)
  g_x = th.pow(x - mu_x, power)
  # print('gx', g_x.shape, 'gy', g_y.shape)
  dist = (g_y + g_x) * math.pow(inv_std, power)
  g_yx = th.exp(-dist)
  # print('gyx', g_yx.shape)

  # g_yx = g_yx.permute(0, 2, 3, 1)
  return g_yx

def get_keypoint_data_from_feature_map(feature_map, gauss_std):
  """Returns keypoint information from a feature map.
  Args:
    feature_map: [B, H, W, K] Tensor, should be activations from a convnet.
    gauss_std: float, the standard deviation of the gaussians to be put around
      the keypoints.
  Returns:
    a dict with keys:
      'centers': A tensor of shape [B, K, 2] of the center locations for each
          of the K keypoints.
      'heatmaps': A tensor of shape [B, H, W, K] of gaussian maps over the
          keypoints.
  """
  gauss_mu = _get_keypoint_mus(feature_map)
  map_size = feature_map.shape[2:4]
  gauss_maps = _get_gaussian_maps(gauss_mu, map_size, 1.0 / gauss_std)

  return {
      "centers": gauss_mu,
      "heatmaps": gauss_maps,
  }

class KeyPointNet(nn.Module):
  def __init__(self,
               num_keypoints,
               gauss_std,
               keypoint_encoder,
               ):
    super(KeyPointNet, self).__init__()
    self._num_keypoints = num_keypoints
    self._gauss_std = gauss_std
    self._keypoint_encoder = keypoint_encoder

    self.conv = nn.Conv2d(
      in_channels=32,
      out_channels=num_keypoints,
      kernel_size=(1,1))

  def forward(self, image):
    image_features = self._keypoint_encoder(image)
    keypoint_features = self.conv(image_features)
    return get_keypoint_data_from_feature_map(
        keypoint_features, self._gauss_std)


