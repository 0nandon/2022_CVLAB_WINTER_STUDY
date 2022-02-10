
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

	accumulate(t_ema, stn, 0)

	t_optim = optim.Adam(stn.parameters(), lr=args.stn_lr, ...)
	l1_optim = optim.Adam(l1.parameters(), lr=args.l1_lr, ...)
```

`train` 
