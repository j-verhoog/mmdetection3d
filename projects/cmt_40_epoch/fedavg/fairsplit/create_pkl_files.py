import pickle
import os

# Set your network data root
NETWORK_DATA_ROOT = "/tudelft.net/staff-umbrella/MscThesisjverhoog/datasets/nuscenes_cmt_full"
ann_file = os.path.join(NETWORK_DATA_ROOT, 'nuscenes_infos_train.pkl')

print(f"Loading original dataset from: {ann_file}")
with open(ann_file, 'rb') as f:
    data = pickle.load(f)

# Define your models and their starting offsets
models = {
    'ModelA': 0,
    'ModelB': 2,
    'ModelC': 4,
    'ModelD': 6,
    'ModelE': 8
}

interval = 10

for model_name, offset in models.items():
    client_data = data.copy()
    # Slice the infos list using the offset and interval
    client_data['infos'] = data['infos'][offset::interval]
    
    # Format the output name to exactly match your bash script
    out_filename = f'nuscenes_infos_train_uniformfedavg_{model_name}.pkl'
    out_file = os.path.join(NETWORK_DATA_ROOT, out_filename)
    
    with open(out_file, 'wb') as f:
        pickle.dump(client_data, f)
    
    print(f"Saved {model_name} (offset {offset}) with {len(client_data['infos'])} samples to {out_filename}")