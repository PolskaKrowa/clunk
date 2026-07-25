; Clunk Example: vector-intrinsic synthesis targets.
; Scalar idioms over vector values that the VectorSynth pass should
; recognise and re-express as vector instructions + clunk.vector.reduce.*
; intrinsics (SMT-proved, cost-model-gated).

; Fully scalarised dot product: 8 extracts + 4 muls + 3 adds.
; Expected: %v = mul <4 x i32> %a, %b ; call @clunk.vector.reduce.add.v4i32(%v)
define i32 @dot4(<4 x i32> %a, <4 x i32> %b) {
entry:
  %a0 = extractelement <4 x i32> %a, i32 0
  %a1 = extractelement <4 x i32> %a, i32 1
  %a2 = extractelement <4 x i32> %a, i32 2
  %a3 = extractelement <4 x i32> %a, i32 3
  %b0 = extractelement <4 x i32> %b, i32 0
  %b1 = extractelement <4 x i32> %b, i32 1
  %b2 = extractelement <4 x i32> %b, i32 2
  %b3 = extractelement <4 x i32> %b, i32 3
  %m0 = mul i32 %a0, %b0
  %m1 = mul i32 %a1, %b1
  %m2 = mul i32 %a2, %b2
  %m3 = mul i32 %a3, %b3
  %s0 = add i32 %m0, %m1
  %s1 = add i32 %s0, %m2
  %s2 = add i32 %s1, %m3
  ret i32 %s2
}

; Horizontal add: expected to become call @clunk.vector.reduce.add.v4i32(%v).
define i32 @hadd4(<4 x i32> %v) {
entry:
  %e0 = extractelement <4 x i32> %v, i32 0
  %e1 = extractelement <4 x i32> %v, i32 1
  %e2 = extractelement <4 x i32> %v, i32 2
  %e3 = extractelement <4 x i32> %v, i32 3
  %t0 = add i32 %e0, %e1
  %t1 = add i32 %t0, %e2
  %t2 = add i32 %t1, %e3
  ret i32 %t2
}

; Shuffle-of-shuffle: expected to compose into a single shufflevector.
define <4 x i32> @swizzle(<4 x i32> %a, <4 x i32> %b) {
entry:
  %s1 = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %s2 = shufflevector <4 x i32> %s1, <4 x i32> %s1, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  ret <4 x i32> %s2
}
