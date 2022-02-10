
## spatial_transformers.py

[code link here](https://github.com/0nandon/gangealing/blob/main/models/spatial_transformers/spatial_transformer.py)

transforms의 개수가 한개이면 `SpatialTransformer`를, 복수이면 `ComposedSTN`을 출력한다.

```python
def get_stn(transforms, **stn_kwargs):
  is_str = isinstance(transforms, str)
  is_list = isinstance(transforms, str)
  
  assert is_str or is_list
  if is_str:
    transforms = [transforms]
  if len(transforms) == 1:
    return SpatialTransformer(transform=transforms[0], **stn_kwargs)
  else:
    return ComposedSTN(transforms, **stn_kwargs)
```
