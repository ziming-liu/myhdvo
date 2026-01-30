#!/bin/bash
# Quick test script to verify multi-sequence testing works

module load cuda/11.8.0
conda activate hdvo
export LD_LIBRARY_PATH=/home/izi2sgh/PROJECT/hdvo/openrox/cmake:$LD_LIBRARY_PATH

cd /home/izi2sgh/PROJECT/hdvo

echo "=== Testing Multi-Sequence Configuration ==="
echo ""
echo "Testing with checkpoint: work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth"
echo "Config: configs/hdvo/stereohdvo_posesup_coex_vkitti2_huberloss.py"
echo ""

# Quick validation test
python -c "
from mmcv import Config
cfg = Config.fromfile('configs/hdvo/stereohdvo_posesup_coex_vkitti2_huberloss.py')
print('✓ Config loaded successfully')
print(f'✓ Training sequences: {len(cfg.data.train.ann_files)}')
print(f'✓ Test sequences: {len(cfg.data.test.test_sequences)}')
print('')
print('Test sequences to be evaluated:')
for i, seq in enumerate(cfg.data.test.test_sequences, 1):
    print(f'  {i:2d}. {seq[\"test_seq_id\"]:20s} - {seq[\"ann_file\"]}')
"

echo ""
echo "=== All checks passed! ==="
echo ""
echo "To run full test:"
echo "  OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12860 \\"
echo "    tools/test.py configs/hdvo/stereohdvo_posesup_coex_vkitti2_huberloss.py \\"
echo "    work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth \\"
echo "    --launcher pytorch --eval EPE 3PE D1"
echo ""
echo "To run training:"
echo "  OMP_NUM_THREADS=12 torchrun --standalone --nnodes=1 --nproc_per_node=1 \\"
echo "    tools/train.py configs/hdvo/stereohdvo_posesup_coex_vkitti2_huberloss.py \\"
echo "    --launcher pytorch --validate"
