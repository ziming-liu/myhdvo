#!/usr/bin/env python3
import os
from vkitti2_annotation import save

vkitti2_root = '/home/izi2sgh/PROJECT/hdvo/data_sets/vkitti2'
save_path = os.path.join(os.path.dirname(__file__), '..', '..', 'annotations', 'vkitti2')
save_path = os.path.normpath(save_path)

# Scenes and variations present in repository
scenes = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
variations = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']

# Generate 2-frame annotations for all scene/variation combos
for scene in scenes:
    for var in variations:
        name = f"vkitti2_{scene.lower()}_{var}"
        print(f"Generating {name}")
        try:
            save(save_path=save_path, name=name, scenes=[scene], variations=[var], data_prefix=vkitti2_root, seq_len=2)
        except Exception as e:
            print(f"Failed {name}: {e}")

print('Done')
