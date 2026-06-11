# nanoGPT modernization ablations
device=cuda, seed=1337, iters=5000, batch=64, block=256, lr=0.0003

### run: baseline | cfg={'rms': False, 'swiglu': False, 'rope': False} | params=10.789M
  step 0: train 4.2221, val 4.2306
  step 500: train 1.7491, val 1.9071
  step 1000: train 1.3914, val 1.6089
  step 1500: train 1.2707, val 1.5293
  step 2000: train 1.1892, val 1.5100
  step 2500: train 1.1204, val 1.4933
  step 3000: train 1.0706, val 1.4831
  step 3500: train 1.0193, val 1.5094
  step 4000: train 0.9610, val 1.5108
  step 4500: train 0.9092, val 1.5306
  step 4999: train 0.8583, val 1.5566

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8595 | 1.5669 | 1.4831 | 119,196 | 11.5 |

### run: rmsnorm | cfg={'rms': True, 'swiglu': False, 'rope': False} | params=10.784M
  step 0: train 4.2202, val 4.2283
  step 500: train 1.7450, val 1.8999
  step 1000: train 1.3918, val 1.6045
  step 1500: train 1.2705, val 1.5271
  step 2000: train 1.1918, val 1.5060
  step 2500: train 1.1245, val 1.4944
  step 3000: train 1.0737, val 1.4817
  step 3500: train 1.0273, val 1.5097
  step 4000: train 0.9662, val 1.5226
  step 4500: train 0.9211, val 1.5365
  step 4999: train 0.8658, val 1.5626

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8595 | 1.5669 | 1.4831 | 119,196 | 11.5 |
| rmsnorm | 10.784 | 0.8655 | 1.5690 | 1.4817 | 120,176 | 11.4 |

### run: swiglu | cfg={'rms': False, 'swiglu': True, 'rope': False} | params=10.777M
  step 0: train 4.3154, val 4.3115
  step 500: train 1.6657, val 1.8285
  step 1000: train 1.3191, val 1.5695
  step 1500: train 1.1900, val 1.5112
  step 2000: train 1.1088, val 1.4927
  step 2500: train 1.0343, val 1.5134
  step 3000: train 0.9452, val 1.5403
  step 3500: train 0.8631, val 1.5900
  step 4000: train 0.7850, val 1.6628
  step 4500: train 0.7049, val 1.7280
  step 4999: train 0.6239, val 1.7772

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8595 | 1.5669 | 1.4831 | 119,196 | 11.5 |
| rmsnorm | 10.784 | 0.8655 | 1.5690 | 1.4817 | 120,176 | 11.4 |
| swiglu | 10.777 | 0.6212 | 1.7707 | 1.4927 | 117,827 | 11.6 |

### run: rope | cfg={'rms': False, 'swiglu': False, 'rope': True} | params=10.691M
  step 0: train 4.2202, val 4.2194
  step 500: train 1.4820, val 1.6691
  step 1000: train 1.2985, val 1.5326
  step 1500: train 1.2079, val 1.4834
  step 2000: train 1.1440, val 1.4791
  step 2500: train 1.0880, val 1.4625
  step 3000: train 1.0365, val 1.4737
  step 3500: train 0.9828, val 1.4886
  step 4000: train 0.9343, val 1.5029
  step 4500: train 0.8828, val 1.5364
  step 4999: train 0.8379, val 1.5406

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8595 | 1.5669 | 1.4831 | 119,196 | 11.5 |
| rmsnorm | 10.784 | 0.8655 | 1.5690 | 1.4817 | 120,176 | 11.4 |
| swiglu | 10.777 | 0.6212 | 1.7707 | 1.4927 | 117,827 | 11.6 |
| rope | 10.691 | 0.8403 | 1.5424 | 1.4625 | 117,265 | 11.6 |

### run: all | cfg={'rms': True, 'swiglu': True, 'rope': True} | params=10.674M
  step 0: train 4.2623, val 4.2633
  step 500: train 1.3917, val 1.5932
  step 1000: train 1.2312, val 1.4940
  step 1500: train 1.1367, val 1.4676
  step 2000: train 1.0600, val 1.4792
  step 2500: train 0.9878, val 1.4853
  step 3000: train 0.9184, val 1.5166
  step 3500: train 0.8375, val 1.5692
  step 4000: train 0.7696, val 1.6117
  step 4500: train 0.6974, val 1.6612
  step 4999: train 0.6395, val 1.7176

## Summary (so far)

| variant | params (M) | final train | final val | best val | tok/s | train min |
|---|---|---|---|---|---|---|
| baseline | 10.789 | 0.8595 | 1.5669 | 1.4831 | 119,196 | 11.5 |
| rmsnorm | 10.784 | 0.8655 | 1.5690 | 1.4817 | 120,176 | 11.4 |
| swiglu | 10.777 | 0.6212 | 1.7707 | 1.4927 | 117,827 | 11.6 |
| rope | 10.691 | 0.8403 | 1.5424 | 1.4625 | 117,265 | 11.6 |
| all | 10.674 | 0.6358 | 1.7256 | 1.4676 | 116,622 | 11.7 |

All runs done in 77.7 min total.
