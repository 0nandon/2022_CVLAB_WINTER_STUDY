
## train.py

[code link here](https://github.com/0nandon/gangealing/blob/main/train.py)

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
