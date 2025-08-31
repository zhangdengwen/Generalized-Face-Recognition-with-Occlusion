import torch
checkpoint = torch.load('/home/srp/face_recognition/out_dir/Epoch_19.pt', map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
print(state_dict.keys())
