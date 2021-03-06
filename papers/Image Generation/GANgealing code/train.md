
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

	# accumulate 함수의 decay 인자에 0을 넣어줘서, t_ema와 stn의 파라미터 값이 동일하게 한다.
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
```

`train` 함수를 사용하여 학습을 진행한다.
```python
train(args, loader, generator, stn, t_ema, l1, t_optim, l1_optim, t_sched, l1_sched, loss_fn, anneal_fn, device, writer)
```

다음은 `train` 함수에 대한 설명이다.
 
* GAN에 들어갈 노이즈 이미지(latent code)를 생성한다.
* args에 입력된 GAN이 생성하는 이미지의 해상도와, STN이 학습하는 flow field의 해상도를 비교하여, 적절한 다운샘플러를 지정해준다.

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
	    resize_fake2stn = BilinearDownsampler(args.gen_size // args.flow_size, 3).to(device) if args.gen_size > args.flow_size else nn.Sequential()
 ```
 
 학습을 할 때는 pretrained된 GAN을 사용하기 때문에, requires_grad를 freeze해준다.
 
```python
	generator.eval()
	requires_grad(generator, False) # GAN is frozen throughout this entire process
	requires_grad(stn, True)
	requires_grad(l1, True)
```
 
 학습에 필요한 여러 변수들을 초기화시켜준다.
 
 이외에 learning scheduler에 따른 모델의 저장 체크포인트를 조정해준다.

```python
	# A model checkpoint will be saved whenever the learning rate is zero
	...

	# Initialize various training variables and constants
	zero = torch.tensor(0.0, device='cuda')
	accum = 0.5 ** (32 / (10 * 1000))
	psi = 1.0

	# create initial training visualizations
	...
```

이제 지정된 iteration에 따라 반복학습을 수행한다.

```python
	# main training loop
	for idx in pbar:
		i = idx + args.start_iter + 1
		if i <= args.anneal_psi:
			psi = anneal_fn(i, 1.0, 0.0, args.anneal_psi).item()
			psi_is_fixed = False
		else:
			psi = 0.0
			psi_is_fixed = True

		if i > args.iter:
			print('Done!')
			break
```

본결적을 학습을 수행한다.
* GANgearling 논문에서 언급했듯, loss는 perceptual loss, tv loss, flow identity loss를 모두 합해서 계산한다.
* t_ema에 stn 파라미터를 모멘텀 형식으로 더한다.

```python
		# Train STN and LL #

		if args.clustering or args.flips:
			perceptual_loss, delta_flow = gangealing_cluster_loss(...)
		else:
			perceptual_loss, delta_flow = gangealing_loss(...)

		tv_loss = total_variation_loss(delta_flow) if args.tv_weight > 0 else zero
		flow_idty_loss = flow_identity_loss(delta_flow) if args.flow_identity_weight > 0 else zero

		loss_dict = {'p': perceptual_loss, 'tv': tv_loss, 'f': flow_idty_loss}


		stn.zero_grad()
		l1.zero_grad()

		# loss를 모두 더해준다.
		full_stn_loss = perceptual_loss + tv_loss + flow_idty_loss
		full_stn_loss.backward()
		t_optim.step()

		if not args.freeze_l1:
			l1_optim.step()
		if psi_is_fixed:
			epoch = max(0, (i - args.anneal_psi) / args.period)

		# 학습된 stn 모듈을 t_ema에 모멘텀 기법을 활용하여 더해준다.
		# accum인자는 모멘텀 정도를 지정한다.
		accumulate(t_ema, t_module, accum)

		# Aggregate loss information across GPUs
		loss_reduced = reduce_loss_dict(loss_dict)

		...
```

또한 마지막으로, 모든 모델의 상태를 저장하는 함수 `save_state_dict()`를 선언해준다.

```python
def save_state_dict(ckpt_name, generator, t_module, t_ema, t_optim, t_sched, l1_module, l1_optim, l1_sched, args):
	ckpt_dict = {
		"g_ema": generator.state_dict(), "t": t_module.state_dict(),
		"t_ema": t_sched.state_dict(), "t_optim": t_optim.state_dict(),
		"t_sched": t_sched.state_dict(), "l1": l1_module.state_dict(),
		"l1_optim": l1_optim.state_dict(), "l1_sched": l1_sched.state_dict(),
		"args": args
	}
	torch.save(ckpt_dict, f'{results_path}/checkpoints/{ckpt_name}.pt')
```


