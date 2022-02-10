## loss.py

[code link here]()

```python
def gangealing_loss(
  generator, stn, l1, loss_fn, resize_fake2stn,
  psi, batch, dim_latent, freeze_l1, device,
  sample_from_full_res = False, **stn_kwargs
):
  unaligned_in, aligned_target = sample_gen_supervised_pairs(
    generator, l1, resize_fake2stn, psi, batch, dim_latent,
    freeze_l1, device, z=None
  )

```
