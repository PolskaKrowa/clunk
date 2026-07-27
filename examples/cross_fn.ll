; Test module for end-to-end cross-function + bounded-unrolling verification.
; Built to exercise:
;   - IPCP (both callers pass the same constant)
;   - DFE (helper is never called)
;   - Multi-block inliner (compute has 3 blocks)
;   - SMT bounded unrolling (loop has constant trip count)

define internal i32 @double(i32 %x) {
entry:
  %r = shl i32 %x, 1
  ret i32 %r
}

; Multi-block callee: double if x>=0 else 0
define internal i32 @double_if_pos(i32 %x) {
entry:
  %cmp = icmp sge i32 %x, 0
  br i1 %cmp, %then, %else
then:
  %r = call i32 @double(i32 %x)
  ret i32 %r
else:
  ret i32 0
}

; Both callers pass the constant 5 → IPCP should specialise.
define i32 @caller_a() {
entry:
  %c = call i32 @double_if_pos(i32 5)
  ret i32 %c
}

define i32 @caller_b() {
entry:
  %c = call i32 @double_if_pos(i32 5)
  ret i32 %c
}

; Dead function — never called.
define internal i32 @dead_helper(i32 %x) {
entry:
  %r = add i32 %x, 1
  ret i32 %r
}

define i32 @main() {
entry:
  %a = call i32 @caller_a()
  %b = call i32 @caller_b()
  %s = add i32 %a, %b
  ret i32 %s
}
