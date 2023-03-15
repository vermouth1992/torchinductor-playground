# TorchInductor Playground

Investigate the compatibility of torchinductor with custom ops.

## Build
Build a custom sigmoid
```bash
cd custom_cuda_op; python3 setup.py build; cd ..;
```
Verify the op
```bash
python3 verify_custom_cpp_ops.py
```

Verify various ways the op is integrated into existing torch.compile framework
```bash
python3 test_custom_op.py
```

