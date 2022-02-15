
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

다음으로 `SpatialTransformer` 모듈을 보도로 하겠다.

```python
class SpatialTransformer(nn.Module):
  def __init__(self, flow_size, supersize, channel_multiplier=0.5, blur_kernel=[1, 3, 3, 1], num_heads=1,
  transform='similarity', flow_downsample=8):
  
    super().__init__()
    
    # 입력 이미지가 flow size 보다 클 경우 다운샘플링을 지정해줘야 하는데, 이때 사용할 다운샘플러를 지정해준다.
    if supersize > flow_size:
      self.input_downsample = BilinearDownsample(supersize // flowsize, 3)
      
    self.input_downsample_required = supersize > flow_size
    self.stn_in_size = flow_size
    
    # 입력된 transform이 flow인지 아닌지
    self.is_flow = transform == 'flow'
    
    channels = {
      4: 512, 
      8: 512,
      16: 512,
      32: 512,
      64: 512,
      128: 128 * channel_multiplier,
      256: 64 * channel_multiplier,
      512: 32 * channel_multipler,
      1024: 16 * channel_multipler,
    }
    
    convs = [ConvLayer(3, int(channels[flow_size]), 1)]
    
    log_size = int(math.log(flow_size, 2))
    log_downsample = int(math.log())
    
    in_channel = channels[flow_size]
    
    end_log = log_size - 4 if self.is_flow else 2
    assert end_log >= 0
    
    num_downsamples = 0
    for i in range(log_size, end_log, -1):
      downsample = (nmot)
      num_downsamples += downsample
      out_channel = channels[2 ** (i - 1)]
      
      convs.append(ResBlock(int(in_channel), int(out_channel), blur_kernel, downsample))
      
      in_channel = out_channel
      
      self.convs = nn.Sequential(*convs)
```










