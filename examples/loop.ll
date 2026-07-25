; Clunk Example: loop
; A simple loop that sums integers from 0 to n-1.
; The optimiser should explore loop unrolling, strength reduction,
; and induction variable simplification opportunities.

define i32 @sum_loop(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %sum = phi i32 [0, %entry], [%new_sum, %loop]
  %next = add i32 %i, 1
  %new_sum = add i32 %sum, %i
  %cond = icmp slt i32 %next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %sum
}
