import triton
import triton.language as tl
import torch

BLOCK_SIZE = 16


@triton.jit
def matrix_transpose_kernel(
    input_ptr,
    output_ptr,
    rows,
    cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program instance handles one BLOCK_M x BLOCK_N tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets within the tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load tile from input[offs_m, offs_n] (row-major: input[m, n])
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
    input_ptrs = input_ptr + offs_m[:, None] * cols + offs_n[None, :]
    tile = tl.load(input_ptrs, mask=mask, other=0.0)

    # Store transposed tile to output[offs_n, offs_m] (row-major: output[n, m])
    out_mask = (offs_n[:, None] < cols) & (offs_m[None, :] < rows)
    output_ptrs = output_ptr + offs_n[:, None] * rows + offs_m[None, :]
    tl.store(output_ptrs, tl.trans(tile), mask=out_mask)


def solve(input: torch.Tensor) -> torch.Tensor:
    rows, cols = input.shape
    output = torch.empty((cols, rows), device=input.device, dtype=input.dtype)

    grid = (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
    matrix_transpose_kernel[grid](
        input, output, rows, cols, BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE
    )
    return output
