; Clunk Example: simple_add
; A trivial addition function — the "hello world" of superoptimisation.
; The optimiser should recognise that this is already minimal.

define i32 @simple_add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}
