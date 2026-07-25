; Test IR: foldable arithmetic that the peephole miner should find.
define i32 @fold_mul_one(i32 %x) {
entry:
  %a = mul i32 %x, 1
  ret i32 %a
}

define i32 @fold_add_zero(i32 %x) {
entry:
  %a = add i32 %x, 0
  ret i32 %a
}

define i32 @fold_xor_zero(i32 %x) {
entry:
  %a = xor i32 %x, 0
  ret i32 %a
}
