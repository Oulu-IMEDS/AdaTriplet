import solt
import solt.transforms as slt

def test_transformation():
    test_transforms = solt.Stream([slt.Crop((700, 700), crop_mode='c'),
                                   slt.Resize(resize_to=(280, 280)),
                                   slt.Crop((256, 256), crop_mode='c'),
                                   ])
    return test_transforms

def train_transformation():
    train_transforms = solt.Stream([slt.Crop((700, 700), crop_mode='c'),
                                    slt.Resize(resize_to=(280, 280)),
                                    slt.Noise(p=0.5, gain_range=0.3),
                                    slt.Rotate(angle_range=(-10, 10)),
                                    slt.Crop((256, 256), crop_mode='r'),
                                    slt.GammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
                                    ])
    return train_transforms

def train_chest_transformation():
    train_transforms = solt.Stream([slt.Resize(resize_to=(280, 280)),
                                    slt.Noise(p=0.5, gain_range=0.3),
                                    slt.Rotate(angle_range=(-10, 10)),
                                    slt.Crop((256, 256), crop_mode='r'),
                                    slt.GammaCorrection(p=0.5, gamma_range=(0.5, 1.5)),
                                    ])
    return train_transforms

def test_chest_transformation():
    test_transforms = solt.Stream([slt.Resize(resize_to=(280, 280)),
                                   slt.Crop((256, 256), crop_mode='c'),
                                   ])
    return test_transforms
