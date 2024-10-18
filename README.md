# FastKAN 실습 정리

## 목적
- RBF를 사용한 KAN 구현체 공부
- Torch, Flax로 구현 및 간단한 비교 실험 진행

## 알게된 사실
- Conv layer에서, torch는 `NCHW`를 사용하는 반면 jax/flax는 `NHWC`를 사용함
- jax와 torch로 동시에 작업하는 과정에서, DataLoader num_workers > 0 일 때 런타임 에러 발생. jax 엔진과 호환되지 않는 것으로 보임

## Citation
```python
@article{li2024kolmogorovarnold,
      title={Kolmogorov-Arnold Networks are Radial Basis Function Networks}, 
      author={Ziyao Li},
      year={2024},
      eprint={2405.06721},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```