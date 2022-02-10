## loss.py

[code link here]()


latent code와 fixed c를 GAN에 넣어 GANgealing을 수행하고, perceptual loss를 계산하는 함수이다.

```python
def gangealing_loss(
  generator, stn, l1, loss_fn, resize_fake2stn,
  psi, batch, dim_latent, freeze_l1, device,
  sample_from_full_res = False, **stn_kwargs
):

  # GAN에서 unaligned img와 aligned img를 생성한다.
  unaligned_in, aligned_target = sample_gen_supervised_pairs(
    generator, l1, resize_fake2stn, psi, batch, dim_latent,
    freeze_l1, device, z=None
  )
  
  input_img_for_sampling = unaligned_in if sample_from_full_res else None
  
  # stn에서 unaligned_in를 aligned하게 만든다. 
  # aligned_pred : warping하 이미지
  # delta_flow : STN에서 grid generator부분에서 affine transformation의 matrix
  aligned_pred, delta_flow = stn(
    resize_fake2stn(unaligned_in), return_flow=True,
    input_img_for_sampling=input_img_for_sampling, **stn_kwargs
  )
  
  # perceptual loss 계산
  perceptual_loss = loss_fn(aligned_pred, aligned_target).mean()
  
  return perceptual_loss, delta_flow
```
