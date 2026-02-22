import os.path
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_BtoA = os.path.join(opt.dataroot, opt.phase + 'BtoA')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.BtoA_paths = make_dataset(self.dir_BtoA)
        
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.BtoA_paths = sorted(self.BtoA_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

        # Build lookup from stem -> A path (clean images) for paired ground truth.
        # When which_direction=BtoA, generator input is from B (distorted) and
        # the metric loss needs the matching A (clean) as target.
        self.a_by_stem = {}
        for p in self.A_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            self.a_by_stem[stem] = p

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        BtoA_path = self.BtoA_paths[index_B]

        # Paired ground truth: the clean A image matching B's stem.
        # When which_direction=BtoA, B is the generator input (distorted) and
        # the metric loss needs the matching A (clean) as target.
        B_stem = os.path.splitext(os.path.basename(B_path))[0]
        paired_gt_path = self.a_by_stem.get(B_stem, A_path)

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        BtoA_img = Image.open(BtoA_path).convert('RGB')
        PairedGT_img = Image.open(paired_gt_path).convert('RGB')

        A = self.transform(A_img)

        # Save RNG state before B transform so PairedGT gets the same
        # random crop/flip — needed for pixel-aligned metric loss.
        torch_rng_state = torch.random.get_rng_state()
        py_rng_state = random.getstate()
        B = self.transform(B_img)

        BtoA = self.transform(BtoA_img)

        # Replay the same random crop/flip for PairedGT
        torch.random.set_rng_state(torch_rng_state)
        random.setstate(py_rng_state)
        PairedGT = self.transform(PairedGT_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path,
                'BtoA': BtoA, 'PairedGT': PairedGT
                }

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
