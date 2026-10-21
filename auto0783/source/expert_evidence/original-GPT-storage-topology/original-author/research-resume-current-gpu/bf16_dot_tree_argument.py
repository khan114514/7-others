"""Exact scalar BF16-product/FP32-sum counterexample, no GPU simulation.

Same3584ordered operands and real dot product can round differently solely
through legal parenthesization. This is not an explanation of actual108CUDA
kernel execution: no kernel tree has been observed there.
"""
from fractions import Fraction
from pathlib import Path
import struct,json

def f32(x):return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def bf16_bits(x):
    bits=struct.unpack('<I',struct.pack('<f',float(x)))[0]
    return ((bits+0x7fff+((bits>>16)&1))>>16)&0xffff
def bf16_value(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]

a=[Fraction(1),Fraction(1,256),Fraction(1,2**25),Fraction(1,2**25),Fraction(1,2**25)]+[Fraction(0)]*(3584-5)
b=[Fraction(1)]*3584
assert all(Fraction(bf16_value(bf16_bits(x)))==x for x in a+b)
products=[f32(x*y) for x,y in zip(a,b)]
assert all(Fraction(value)==x*y for value,x,y in zip(products,a,b))
seq=f32(0)
for value in products:seq=f32(seq+value)
prefix=f32(0)
for value in products[:2]:prefix=f32(prefix+value)
tail=f32(0)
for value in products[2:]:tail=f32(tail+value)
segmented=f32(prefix+tail)
exact=sum((x*y for x,y in zip(a,b)),Fraction(0))
assert seq==1+1/256 and segmented==seq+2**-23
seq_bits=bf16_bits(seq);seg_bits=bf16_bits(segmented)
assert seq_bits==0x3f80 and seg_bits==0x3f81
report={'scope':'Constructive scalar arithmetic counterexample using BF16 operands, exact products, FP32 additions and RNE BF16 conversion. No CUDA measurements.','dot_features':3584,'ordered_a_first_five':['1','1/256','1/2^25','1/2^25','1/2^25'],'remaining_a':'3579zeros','b':'3584ones','all_inputs_exactly_BF16':True,'all_products_exactly_FP32':True,'real_dot':str(exact),'left_associative_FP32':seq,'two_segment_FP32_preserving_global_operand_order':segmented,'BF16_outputs':[bf16_value(seq_bits),bf16_value(seg_bits)],'BF16_bits':[hex(seq_bits),hex(seg_bits)],'only_changed_operation':'parenthesization/partial accumulation, not operand order/coefficients/bias','conclusion':'Same coefficients and ordered features do not guarantee same BF16 output when GEMM tiling/tree changes. A smallFP32delta can become a fullBF16ULP, then propagate through all28layers/CFG/reveal.','relevance_to_108':'108changed QK/V widths and compiler boundary; the counterexample supplies a possible numerical mechanism, not an actual cause. No observed108kernel tree or error localization is available.','limits':['Scalar IEEE model does not represent tensorcore scheduling/partial blocks or flush modes.','Nothing here claims actual checkpoint operands contain this constructed pattern.','Dev validation remains decisive; do not revise weights, backend flags or tolerance.']}
Path('research-resume-current-gpu/bf16-dot-tree-argument.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
