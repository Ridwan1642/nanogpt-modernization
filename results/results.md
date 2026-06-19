# nanoGPT modernization ablations
device=cuda, seed=1337, iters=5000, batch=64, block=256, lr=0.0003

### run: baseline | cfg={'rms': False, 'swiglu': False, 'rope': False} | params=10.789M
  step 0: train 4.2223, val 4.2306
  step 500: train 1.7332, val 1.8890
  step 1000: train 1.3899, val 1.6006
  step 1500: train 1.2678, val 1.5330
  step 2000: train 1.1896, val 1.5049
  step 2500: train 1.1251, val 1.5001
  step 3000: train 1.0736, val 1.5015
  step 3500: train 1.0217, val 1.5047
  step 4000: train 0.9668, val 1.5320
  step 4500: train 0.9126, val 1.5448
  step 4999: train 0.8592, val 1.5810

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8593 | 1.5793 | 1.5001 | 98,523 | 13.9 |

### run: rmsnorm | cfg={'rms': True, 'swiglu': False, 'rope': False} | params=10.784M
  step 0: train 4.2204, val 4.2283
  step 500: train 1.7354, val 1.8904
  step 1000: train 1.3873, val 1.6008
  step 1500: train 1.2682, val 1.5337
  step 2000: train 1.1872, val 1.5043
  step 2500: train 1.1263, val 1.5059
  step 3000: train 1.0732, val 1.5012
  step 3500: train 1.0251, val 1.4986
  step 4000: train 0.9769, val 1.5231
  step 4500: train 0.9186, val 1.5369
  step 4999: train 0.8658, val 1.5717

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8593 | 1.5793 | 1.5001 | 98,523 | 13.9 |
| rmsnorm | 10.784 | 0.8647 | 1.5645 | 1.4986 | 88,982 | 15.3 |

### run: swiglu | cfg={'rms': False, 'swiglu': True, 'rope': False} | params=10.777M
  step 0: train 4.3150, val 4.3111
  step 500: train 1.6526, val 1.8336
  step 1000: train 1.3140, val 1.5517
  step 1500: train 1.1938, val 1.5005
  step 2000: train 1.1051, val 1.4995
  step 2500: train 1.0247, val 1.5187
  step 3000: train 0.9418, val 1.5462
  step 3500: train 0.8657, val 1.5982
  step 4000: train 0.7711, val 1.6568
  step 4500: train 0.6872, val 1.7235
  step 4999: train 0.6131, val 1.7763

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8593 | 1.5793 | 1.5001 | 98,523 | 13.9 |
| rmsnorm | 10.784 | 0.8647 | 1.5645 | 1.4986 | 88,982 | 15.3 |
| swiglu | 10.777 | 0.6091 | 1.7772 | 1.4995 | 93,631 | 14.6 |

### run: rope | cfg={'rms': False, 'swiglu': False, 'rope': True} | params=10.691M
  step 0: train 4.2200, val 4.2186
  step 500: train 1.4819, val 1.6839
  step 1000: train 1.2884, val 1.5196
  step 1500: train 1.2052, val 1.4867
  step 2000: train 1.1366, val 1.4694
  step 2500: train 1.0796, val 1.4659
  step 3000: train 1.0297, val 1.4754
  step 3500: train 0.9806, val 1.4849
  step 4000: train 0.9289, val 1.5028
  step 4500: train 0.8764, val 1.5178
  step 4999: train 0.8238, val 1.5488

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8593 | 1.5793 | 1.5001 | 98,523 | 13.9 |
| rmsnorm | 10.784 | 0.8647 | 1.5645 | 1.4986 | 88,982 | 15.3 |
| swiglu | 10.777 | 0.6091 | 1.7772 | 1.4995 | 93,631 | 14.6 |
| rope | 10.691 | 0.8246 | 1.5536 | 1.4659 | 97,553 | 14.0 |

### run: all | cfg={'rms': True, 'swiglu': True, 'rope': True} | params=10.674M
  step 0: train 4.2632, val 4.2615
  step 500: train 1.3876, val 1.5930
  step 1000: train 1.2223, val 1.4863
  step 1500: train 1.1320, val 1.4645
  step 2000: train 1.0539, val 1.4696
  step 2500: train 0.9809, val 1.4838
  step 3000: train 0.9091, val 1.5211
  step 3500: train 0.8381, val 1.5591
  step 4000: train 0.7633, val 1.6062
  step 4500: train 0.6923, val 1.6536
  step 4999: train 0.6248, val 1.7169

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8593 | 1.5793 | 1.5001 | 98,523 | 13.9 |
| rmsnorm | 10.784 | 0.8647 | 1.5645 | 1.4986 | 88,982 | 15.3 |
| swiglu | 10.777 | 0.6091 | 1.7772 | 1.4995 | 93,631 | 14.6 |
| rope | 10.691 | 0.8246 | 1.5536 | 1.4659 | 97,553 | 14.0 |
| all | 10.674 | 0.6219 | 1.7239 | 1.4645 | 84,637 | 16.1 |

All runs done in 92.7 min total.
