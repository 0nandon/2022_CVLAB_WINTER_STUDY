
## train.py

[code link here](https://github.com/0nandon/gangealing/blob/main/train.py)

`base_training_argparse()`로 parser을 불러온다음, 필요한 args를 추출한다.

```python
if __name__ = '__main__':
	device = 'cuda'
	# 인자 불러오기
	parser = base_training_argparse()
	args = parser.parse_args()

	# STN의 transform이 'similarity'이면 total_variation_loss를 사용하지 않는다.
	if args.transform == 'similarity':
		assert args.tv_weight == 0, 'Total Variation loss is not currently supported for similarity-only STNs'

	args.n_mean = 200 if args.debug else args.n_mean
	args.vis_batch_size //= args.num_heads

  	# args에 지정된 num_heads의 개수가 1 이상이면 args.clustering에 True로 지정한다.
	args.clustering = args.num_heads > 1
	results_path = os.path.join(args.results, args.exp_names)
```

...

```python
	if primary():
		writer = GANgealingWriter(results.path)
		with open(f'') as f:
			json.dump(args.__dict__, f, indent=2)
	else:
		writer = None
		
	# seed RNG
	torch.manual_seed(args.seed * get_world_size() + get_rank())
	np.random.seed(args.seed * get_world_size() + get_rank())
```

모델을 초기화시켜준다.
`GAN`, `STN`, optimizer, loss function등을 지정해준다.

```python
	# Initialize models
	generator = Generator(...)
	stn = get_stn(...)
	t_ema = get_stn(...)
	l1 = DirectionInterpolator(...)

	# ???
	accumulate(t_ema, stn, 0)

	# optimizer 지정
	t_optim = optim.Adam(stn.parameters(), lr=args.stn_lr, ...)
	l1_optim = optim.Adam(l1.parameters(), lr=args.l1_lr, ...)

	# ???
	t_sched = DecayingCosineAnnealingWarmRestarts(...)
	l1_sched = DecayingCosineAnnealingWarmRestarts(...)

	# loss function 지정
	loss_fn = get_perceptual_loss(args.loss_fn, device)
	anneal_fn = get_psi_annealing_fn(args.anneal_fn)

        # stn의 파라미터를 t_ema에 모멘텀을 사용하여 더한다.
	# decay인수에 0을 넣어서 t_ema의 파라미터와 stn의 파라미터가 같게 한다.
	accumulate(t_ema, stn, 0)

	t_optim = optim.Adam(stn.parameters(), lr=args.stn_lr, ...)
	l1_optim = optim.Adam(l1.parameters(), lr=args.l1_lr, ...)
```

`train` 함수를 사용하여 학습을 진행한다.
```python
train(args, loader, generator, stn, t_ema, l1, t_optim, l1_optim, t_sched, l1_sched, loss_fn, anneal_fn, device, writer)
```

 다음은 `train` 함수에 대한 설명이다.
 
 ```python
 def train(args, loader, generator, stn, t_ema, l1, t_optim, l1_optim, t_sched, l1_sched, loss_fn,
	anneal_fn, device, writer):
		
		# If using real data, select some fixed samples used to visualize training:
		...


		# Progress bar for monitoring training
		pbar = range(args.iter)
		if primary():
			pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.2)


		# Recold modules to make saving checkpoints easier:
		...


		# GAN에 들어가는 latent codes
		sample_z = torch.randn(args.n_sample // args.num_heads, args.dim_latent, device=device)
		if args.clustering:
			big_sample_z = torch.randn(args.n_mean // get_word_size(), args.dim_latent, device=device)

		# 만약 GAN이 생성한 이미지의 resolition이 STN이 학습하는 flow field의 resolution보다 작으면 다운샘플러 지정.
		# 아니면 기본적인 nn.Sequential로 지정
 		resize_fake2stn = BilinaerDownsampler(args.gen_size // args.flow_size, 3).to(device) if args.gen_size > args.flow_size else nn.Sequential()
 ```








