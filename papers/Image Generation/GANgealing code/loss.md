## loss.py

[code link here](https://github.com/0nandon/gangealing/blob/main/models/losses/loss.py)

tv loss를 계산하는 함수이다.

```python
def total_variation_loss():
  # flow should be size (N, H, W, 2)
  reduce_dims = (0, 1, 2, 3) if reduce_batch else (1, 2, 3)
  
  # Huber loss 계산
  distance_fn = lambda a : torch.where(a <= 1.0, 0.5 * a.pow(2), a-0.5).mean(dim=reduce_dim)
  assert delta_flow.size(-1) == 2
  
  diff_y = distance_fn((delta_flow[:, :-1, :, :] - delta_flow[:, 1:, :, :]).abs())
  diff_x = distance_fn((delta_flow[:, :, :-1, :] - delta_flow[:, :, 1:, :]).abs())
  loss = diff_y + diss_x
  return loss
```

 l2 규제를 계산하는 함수이다.
 
 ```python
 def flow_identity_loss(delta_flow):
    loss = delta_flow.pow(2).mean()
    return loss
 ```
 
 GAN에서 target img와 unaligned img를 생성하는 함수이다.
 
```python
def sample_gan_supervised_pairs():
  with torch.set_grad_enbaled(not freeze_l1):
    if z is None:
      # 잡음 이미지 생성
      z = torch.randn(batch, dim_latent, device=device)
    
    # unaligned img 생성
    unaligned_in, w_noise = generator([z], noise=None, return_latents=True)
    
    # c fixed latent를 만든다.
    w_aligned = l1([w_noise[:, 0, :]], psi=psi)
    
    # target image 생성
    aligned_target, _ = generator(w_aligned, input_is_latent=True, noise=None)
    aligned_target = resize_fake2stn(aligned_target)
  return unaligned_in, aligned_target
```

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
  
  # stn에서 unaligned 이미지를 aligned하게 만든다. 
  # aligned_pred : warping한 이미지
  # delta_flow : STN의 grid generator부분에서 affine transformation의 matrix
  aligned_pred, delta_flow = stn(
    resize_fake2stn(unaligned_in), return_flow=True,
    input_img_for_sampling=input_img_for_sampling, **stn_kwargs
  )
  
  # perceptual loss 계산
  perceptual_loss = loss_fn(aligned_pred, aligned_target).mean()
  
  return perceptual_loss, delta_flow
```
