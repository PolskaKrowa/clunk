; Clunk Example: matrix_mul
; A triple-nested-loop matrix multiplication kernel.
; This is a prime target for superoptimisation:
;   - Loop tiling / blocking for cache
;   - Register tiling for accumulators
;   - Strength reduction (mul by power-of-2 -> shift)
;   - Memory access pattern optimisation
;   - GPU shared memory usage on CUDA targets

define void @matrix_mul(i32* %A, i32* %B, i32* %C, i32 %N) {
entry:
  br label %outer

outer:
  %i = phi i32 [0, %entry], [%i_next, %outer_end]
  %i_next = add i32 %i, 1
  %cond_i = icmp slt i32 %i_next, %N
  br i1 %cond_i, label %inner, label %exit

inner:
  %j = phi i32 [0, %outer], [%j_next, %inner_end]
  %j_next = add i32 %j, 1
  %cond_j = icmp slt i32 %j_next, %N
  br i1 %cond_j, label %compute, label %outer_end

compute:
  %k = phi i32 [0, %inner], [%k_next, %compute_end]
  %acc = phi i32 [0, %inner], [%new_acc, %compute_end]
  %k_next = add i32 %k, 1
  %cond_k = icmp slt i32 %k_next, %N
  br i1 %cond_k, label %body, label %store

body:
  %idx_a = mul i32 %i, %N
  %offset_a = add i32 %idx_a, %k
  %ptr_a = getelementptr i32, i32* %A, i32 %offset_a
  %val_a = load i32, i32* %ptr_a
  %idx_b = mul i32 %k, %N
  %offset_b = add i32 %idx_b, %j
  %ptr_b = getelementptr i32, i32* %B, i32 %offset_b
  %val_b = load i32, i32* %ptr_b
  %prod = mul i32 %val_a, %val_b
  %new_acc = add i32 %acc, %prod
  br label %compute_end

compute_end:
  br i1 %cond_k, label %compute, label %store

store:
  %idx_c = mul i32 %i, %N
  %offset_c = add i32 %idx_c, %j
  %ptr_c = getelementptr i32, i32* %C, i32 %offset_c
  store i32 %acc, i32* %ptr_c
  br label %inner_end

inner_end:
  br i1 %cond_j, label %inner, label %outer_end

outer_end:
  br i1 %cond_i, label %outer, label %exit

exit:
  ret void
}
