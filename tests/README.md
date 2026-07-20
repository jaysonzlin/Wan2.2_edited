
Put all your models (Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B) in a folder and specify the max GPU number you want to use.

```bash
bash ./tests/test.sh <local model dir> <gpu number>
```

Run the local point-cloud workflow tests with:

```bash
python -m pytest tests/test_pc_config.py tests/test_pc_dataset.py \
  tests/test_pc_flow.py tests/test_pc_pipeline.py tests/test_pc_flow_model.py \
  tests/test_pc_visualization.py tests/test_train_pc.py -q
```
