# ADR-0001: Scope и принципы MVP

## Решение

- **CPU-only**, без CUDA/OpenCL.
- **DType**: `Float`, `Double` (параметризованный `Tensor a`).
- **Плотные, strided тензоры**, row-major; вьюхи без копий через stride/offset.
- **Динамические размеры** (типобезопасные размеры через DataKinds/TypeLits — позже).
- **Операции MVP**: поэлементные (+, -, \*, neg), редукции sum/mean, reshape/transpose/permute/slice/select/expand/contiguous,
  broadcast, matmul/mm, relu/sigmoid/tanh, logSoftmax, crossEntropy.
- **Autograd**: reverse-mode, динамический граф (на следующем этапе).
- **BLAS**: наивный `mm` + опционально `with-blas` для OpenBLAS/MKL.
- **API-слои**: низкий (`PrimMonad m => ... -> m ...`), поверх — удобный `IO`.

## Не в scope MVP

- Свёртки, GPU, смешанная точность, распределёнка, JIT/граф-компилятор.

## DoD Stage 0

- Собирается `cabal build`.
- Есть ADR и STYLE.
- Модули/тесты — заглушки на местах.
