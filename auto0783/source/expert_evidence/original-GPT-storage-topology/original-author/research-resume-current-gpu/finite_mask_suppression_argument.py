"""Conditional finite-min masking lemma; no checkpoint score bounds assumed."""
import json,math,struct
from fractions import Fraction
from pathlib import Path

def f32(x):return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def bf16(x):
    bits=struct.unpack('<I',struct.pack('<f',float(x)))[0]
    rounded=((bits+0x7fff+((bits>>16)&1))>>16)&0xffff
    return struct.unpack('<f',struct.pack('<I',rounded<<16))[0]
M=(Fraction(2)-Fraction(1,128))*2**127
C=Fraction(2**118)
mask=-float(M)
assert struct.unpack('<I',struct.pack('<f',mask))[0]==0xff7f0000
# This conservative bound is strictly less than half oneBF16ULP at the
# sentinel. Every score in the whole interval is therefore rounded back to
# the sentinel when added inBF16; the examples verify selected endpoints.
assert C<Fraction(2**119)
examples=[]
for s in [-C,Fraction(-1),Fraction(0),Fraction(1),C]:
    masked=bf16(f32(mask+float(s)))
    assert masked==mask
    examples.append({'score':str(s),'masked_BF16':masked})
# For a non-fully-masked row, its max allowed logit is >=-C. Hence every
# suppressed mask entry differs from max by at most-M+C, astronomically
# below the FP32subnormalexp floor. This is a conditional lemma, not a
# verified bound on the actual checkpoint logits or a CUDAsoftmax model.
max_masked_minus_max_allowed=-M+C
threshold=Fraction(-104)
assert max_masked_minus_max_allowed<threshold
assert f32(math.exp(-104))==0.0
# Fully finite-min query rows remain well-defined, since equalfinite values
# subtract to zero. They read all values uniformly; not a zero output.
uniform_full=f32(1/556);uniform_text=f32(1/300)
assert uniform_full!=uniform_text
report={'scope':'IEEEscalarRNE/FP32exp reasoning, conditional on finite scaled-logit bound. No actualscorebounds/CUDAmeasurements.','BF16_mask_bits':'0xff7f','finite_mask_magnitude':str(M),'sufficient_scaled_logit_absolute_bound':str(C),'strict_half_ulp_limit':str(2**119),'mask_addition_preserved_for_entire_interval':'nearestBF16sentinel unchanged by all |s|<=2^118 because interval lies strictly inside its rounding cell','selected_scalar_examples':examples,'masked_minus_max_allowed_upper_bound':str(max_masked_minus_max_allowed),'FP32_exp_rounded_zero_at_minus104':True,'fully_masked_query':'allfinite-min logits become equal, FP32softmaxuniform over all556columns; candidate text-only attention uniform over300columns. Padding rows can legitimately differ.','uniform_full556':uniform_full,'uniform_text300':uniform_text,'actual_checkpoint_bound_established':False,'manifest_limit':'Published tensor manifest has shapes/dtypes/digests/offsets, no numerical coefficient maxima; do not invent a loadedweight/statistical bound.','next_decisions':['Public-maskv3 graph includes dense influence edges for fully finite-min padding queries; those never appear as readablekeys to validtext/images.','Zero-errorDevsupports actual disclosed prediction preservation but not a formal allinputlogit bound or universal bitwiseequivalence.','Do not hardcode public25/32visible text counts: hidden lengths unknown; catalog has no data-dependent gather/compaction/slice bounds.','A new graph numerical cache optimization needs an independent structural/numerical argument and unchanged gates, not this publicpaddingpattern alone.']}
Path('research-resume-current-gpu/finite-attention_suppression-argument.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='selected_scalar_examples'},indent=2))
