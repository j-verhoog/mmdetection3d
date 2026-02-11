import pandas as pd
from nuscenes.nuscenes import NuScenes
from pathlib import Path

# Load mini dataset
nusc = NuScenes(version='v1.0-mini', dataroot='/home/jolle/mmdet/datasets/v1.0-mini', verbose=True)
mini_scene_tokens = {s['token'] for s in nusc.scene}

# Load original Excel
original_excel = Path('/home/jolle/mmdet/mmdetection3d/projects/subsets_creation/scene_domains_summary.xlsx')
df = pd.read_excel(original_excel, engine='openpyxl')

# Filter to only mini scenes
df_mini = df[df['scene_token'].isin(mini_scene_tokens)]

# Save filtered Excel
mini_excel = Path('/home/jolle/mmdet/mmdetection3d/projects/subsets_creation/scene_domains_summary_mini.xlsx')
df_mini.to_excel(mini_excel, index=False)

print(f"Original scenes: {len(df)}")
print(f"Mini scenes: {len(df_mini)}")
print(f"Filtered Excel saved to: {mini_excel}")