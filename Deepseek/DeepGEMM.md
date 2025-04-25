### 1. 构建矩阵A， B， Scale_A, Scale_B

在tests/test_core.py construct函数中，通过torch.randn分别构建(m=4096, n=4096, k=7168)的A，B矩阵

```text
def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out
```

在该代码中，通过per_token_cast_to_fp8以 1 x 128 tile的方式将A压缩到fp8， 得到A(fp8)和Scale_A(fp32), 通过per_block_cast_to_fp8 以 128 x 128 block的方式将B压缩到fp8, 得到B(fp8)和Scale_B(fp32)

压缩方式为 tile/max(abs(tile))∗448.0 _或_ block/max(abs(block))∗448.0 ， 将title和block的值尽量分散到整个fp8值域中（448.0 为fp8 e4m3 的最大值， 对应的二进制为01111110， 2^8 * 1.75=448）