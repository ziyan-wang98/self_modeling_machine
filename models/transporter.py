import torch as th
import torch.nn as nn
import torch.nn.functional as F

class Transporter(nn.Module):
    def __init__(self, encoder,
      keypointer,
      decoder):
        super(Transporter, self).__init__()

        self._encoder = encoder
        self._decoder = decoder
        self._keypointer = keypointer

    def forward(self, image_a, image_b):
        image_a_features = self._encoder(image_a).detach()
        image_a_keypoints = self._keypointer(image_a)
        for key in image_a_keypoints:
            image_a_keypoints[key] = image_a_keypoints[key].detach()

        image_b_features = self._encoder(image_b)
        image_b_keypoints = self._keypointer(image_b)


        num_keypoints = image_a_keypoints["heatmaps"].shape[1]

        # print('heatmaps', image_a_keypoints["heatmaps"].shape, 'centers', image_a_keypoints["centers"].shape)
        transported_features = image_a_features
        for k in range(num_keypoints):
            mask_a = image_a_keypoints["heatmaps"][:, k:k+1]
            mask_b = image_b_keypoints["heatmaps"][:, k:k+1]

            # suppress features from image a, around both keypoint locations.

            # print(mask_a.shape, mask_b.shape, transported_features.shape)  # heatmaps torch.Size([2, 5, 42, 42]) centers torch.Size([2, 5, 42, 2])

            transported_features = (
                    (1 - mask_a) * (1 - mask_b) * transported_features)

            # copy features from image b around keypoints for image b.
            transported_features += (mask_b * image_b_features)

        reconstructed_image_b = self._decoder(
            transported_features)

        return {
            "reconstructed_image_b": reconstructed_image_b,
            "features_a": image_a_features,
            "features_b": image_b_features,
            "keypoints_a": image_a_keypoints,
            "keypoints_b": image_b_keypoints,
        }

    def get_keypoint(self, image):
        image_keypoints = self._keypointer(image)
        return image_keypoints



def reconstruction_loss(image, predicted_image, loss_type="l2"):
  """Returns the reconstruction loss between the image and the predicted_image.
  Args:
    image: target image tensor of shape [B, H, W, C]
    predicted_image: reconstructed image as returned by the model
    loss_type: `str` reconstruction loss, either `l2` (default) or `l1`.
  Returns:
    The reconstruction loss
  """

  if loss_type == "l2":
    return th.mean(th.square(image - predicted_image))
  elif loss_type == "l1":
    return th.mean(th.abs(image - predicted_image))
  else:
    raise ValueError("Unknown loss type: {}".format(loss_type))
