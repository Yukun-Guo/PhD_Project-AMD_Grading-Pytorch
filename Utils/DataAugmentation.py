import numpy as np


class RandomCrop2D(object):
    """Randomly crop a volume in a sample. volume is a 2D numpy array with channel (channel,height,width)

    Args:
        output_size (tuple or int): Desired output size. If int, cube crop
            is made.
        padding (int or tuple, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a tuple of
            length 3 is provided this is used to pad left, right, top, bottom,
            front, back borders. If a tuple of length 6 is provided this is
            used to pad the left, right, top, bottom, front, back borders
            respectively. Padding with non-constant mode needs the image to be
            padded to the borders if the padding is larger than the image.
            example: (1, 1, 1) or ((1, 1),(1, 1), (1, 1))
        pad_if_needed (bool, optional): It will pad the image if smaller than
            the desired size to avoid raising an exception
        fill (int or tuple or float or str, optional): Pixel fill value for
            constant fill.  default is 0.
        padding_mode (str, optional): Type of padding. Should be: constant,
            edge, reflect or symmetric. Default is constant.
        sample (dict): {'mnv': mnv, 'fluid': fluid, 'ga': ga, 'drusen': drusen, 'label': label}

    """

    def __init__(
        self,
        output_size,  # (height, width)
        padding=None,
        pad_if_needed=True,
        fill=0,
        padding_mode="constant",
    ):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.size = output_size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def get_params2d(self, img, output_size, mask=None):
        h, w = img.shape[-2:]
        th, tw = output_size

        range_w = 1 if w == tw else w - tw
        range_h = 1 if h == th else h - th

        i = np.random.randint(0, range_h)
        j = np.random.randint(0, range_w)
        return i, j, th, tw

    def __call__(self, sample):
        mnv, fluid, ga, drusen, label = sample["mnv"], sample["fluid"], sample["ga"], sample["drusen"], sample["label"]
        if self.padding is not None:
            mnv = np.pad(mnv, self.padding, self.fill, self.padding_mode)
            fluid = np.pad(fluid, self.padding, self.fill, self.padding_mode)
            ga = np.pad(ga, self.padding, self.fill, self.padding_mode)
            drusen = np.pad(drusen, self.padding, self.fill, self.padding_mode)
        # pad the height if needed
        size = mnv.shape[-2:]  # [h, w]
        pad_h = (
            (
                int(np.round((self.size[0] - size[0]) / 2)),
                int(self.size[0] - size[0] -
                    np.round((self.size[0] - size[0]) / 2)),
            )
            if self.pad_if_needed and size[0] < self.size[0]
            else (0, 0)
        )
        pad_w = (
            (
                int(np.round((self.size[1] - size[1]) / 2)),
                int(self.size[1] - size[1] -
                    np.round((self.size[1] - size[1]) / 2)),
            )
            if self.pad_if_needed and size[1] < self.size[1]
            else (0, 0)
        )

        mnv = np.pad(mnv,((0, 0), pad_h, pad_w),constant_values=self.fill,mode=self.padding_mode)
        fluid = np.pad(fluid,((0, 0), pad_h, pad_w),constant_values=self.fill,mode=self.padding_mode)
        ga = np.pad(ga,((0, 0), pad_h, pad_w),constant_values=self.fill,mode=self.padding_mode)
        drusen = np.pad(drusen,((0, 0), pad_h, pad_w),constant_values=self.fill,mode=self.padding_mode)
        
        i, j, h, w = self.get_params2d(mnv, self.size)

        # crop the image
        mnv = mnv[:, i: i + h, j: j + w].copy()
        fluid = fluid[:, i: i + h, j: j + w].copy()
        ga = ga[:, i: i + h, j: j + w].copy()
        drusen = drusen[:, i: i + h, j: j + w].copy()

        return {"mnv": mnv, "fluid": fluid, "ga": ga, "drusen": drusen, "label": label}
    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding
        )


class RandomCrop3D(object):
    """Randomly crop a volume in a sample. volume is a 3D numpy array with channel (channel,height,width,depth)

    Args:
        output_size (tuple or int): Desired output size. If int, cube crop
            is made.
        padding (int or tuple, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a tuple of
            length 3 is provided this is used to pad left, right, top, bottom,
            front, back borders. If a tuple of length 6 is provided this is
            used to pad the left, right, top, bottom, front, back borders
            respectively. Padding with non-constant mode needs the image to be
            padded to the borders if the padding is larger than the image.
            example: (1, 1, 1, 1) or ((1, 1),(1, 1), (1, 1), (1, 1))
        pad_if_needed (bool, optional): It will pad the image if smaller than
            the desired size to avoid raising an exception
        fill (int or tuple or float or str, optional): Pixel fill value for
            constant fill.  default is 0.
        padding_mode (str, optional): Type of padding. Should be: constant,
            edge, reflect or symmetric. Default is constant.
        sample (dict): {'img': img, 'mask': mask}

    """

    def __init__(
        self,
        output_size,  # (height, width, depth)
        padding=None,
        pad_if_needed=True,
        fill=0,
        padding_mode="constant",
    ):
        assert isinstance(output_size, (int, tuple,list))
        if isinstance(output_size, int):
            self.size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.size = output_size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def get_params3d(self, img, output_size, mask=None):
        h, w, d = img.shape[-3:]
        th, tw, td = output_size

        range_w = 1 if w == tw else w - tw
        range_h = 1 if h == th else h - th
        range_d = 1 if d == td else d - td

        i = np.random.randint(0, range_h)
        j = np.random.randint(0, range_w)
        k = np.random.randint(0, range_d)
        return i, j, k, th, tw, td

    def __call__(self, sample):
        img, label = sample["mat"], sample["label"]
        if self.padding is not None:
            img = np.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the height if needed
        size = img.shape[-3:]  # [h, w, d]
        pad_h = (
            (
                int(np.round((self.size[0] - size[0]) / 2)),
                int(self.size[0] - size[0] -
                    np.round((self.size[0] - size[0]) / 2)),
            )
            if self.pad_if_needed and size[0] < self.size[0]
            else (0, 0)
        )
        pad_w = (
            (
                int(np.round((self.size[1] - size[1]) / 2)),
                int(self.size[1] - size[1] -
                    np.round((self.size[1] - size[1]) / 2)),
            )
            if self.pad_if_needed and size[1] < self.size[1]
            else (0, 0)
        )
        pad_d = (
            (
                int(np.round((self.size[2] - size[2]) / 2)),
                int(self.size[2] - size[2] -
                    np.round((self.size[2] - size[2]) / 2)),
            )
            if self.pad_if_needed and size[2] < self.size[2]
            else (0, 0)
        )

        img = np.pad(
            img,
            ((0, 0), pad_h, pad_w, pad_d),
            constant_values=self.fill,
            mode=self.padding_mode,
        )

        i, j, k, h, w, d = self.get_params3d(img, self.size)

        # crop the image
        img = img[:, i: i + h, j: j + w, k: k + d].copy()

        return {"mat": img, "label": label}

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding
        )


class GrayJitter(object):
    """Randomly change the brightness and contrast of the image. the image can be 2D, 3D or 4D.
    Args:
        bright_range (tuple): range of brightness change.
        contrast_range (tuple): range of contrast change.
        max_value (int): max value of the image.
        sample (dict): {'mnv': mnv, 'fluid': fluid, 'ga': ga, 'drusen': drusen, 'label': label,'fname': filename}
    """

    def __init__(self, bright_range=(0, 40), contrast_range=(0.5, 1.5), max_value=255):
        self.bright_range = bright_range
        self.contrast_range = contrast_range
        self.max_value = max_value

    def __call__(self, sample):
        mnv, fluid, ga, drusen, label = sample["mnv"], sample["fluid"], sample["ga"], sample["drusen"], sample["label"]
        bright_scale = np.random.uniform(
            self.bright_range[0], self.bright_range[1])
        contrast_scale = np.random.uniform(
            self.contrast_range[0], self.contrast_range[1]
        )
        meanv = np.mean(mnv)
        mnv = (mnv - meanv) * contrast_scale + meanv
        mnv = mnv + bright_scale
        mnv = np.clip(mnv, 0, self.max_value)
        meanv = np.mean(fluid)
        fluid = (fluid - meanv) * contrast_scale + meanv
        fluid = fluid + bright_scale
        fluid = np.clip(fluid, 0, self.max_value)
        meanv = np.mean(ga)
        ga = (ga - meanv) * contrast_scale + meanv
        ga = ga + bright_scale
        ga = np.clip(ga, 0, self.max_value)
        meanv = np.mean(drusen)
        drusen = (drusen - meanv) * contrast_scale + meanv
        drusen = drusen + bright_scale
        drusen = np.clip(drusen, 0, self.max_value)
        return {"mnv": mnv, "fluid": fluid, "ga": ga, "drusen": drusen, "label": label}
    def __repr__(self):
        return (
            self.__class__.__name__
            + "(bright_range={0}, contrast_range={1})".format(
                self.bright_range, self.contrast_range
            )
        )


class AddGaussianNoise(object):
    """Add Gaussian noise to the image. the image can be 2D, 3D or 4D.

    Args:
        mean (float): mean of the Gaussian distribution.
        std (float): standard deviation of the Gaussian distribution.
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, mean=0.0, std=5.0):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        img, mask = sample["img"], sample["mask"]
        img = np.clip(img + np.random.rand(*img.shape)
                      * self.std + self.mean, 0, 255)
        return {"img": img, "mask": mask}

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class RandomRotate90n(object):
    """Rotate the image by 90, 180, 270 degrees randomly. the image can be 2D or 3D.

    Args:
        axes (tuple): axes to rotate. Default: (0, 1) for 2D, (1, 2) for 3D ?
        sample (dict): {'img': img, 'mask': mask}
    """

    def __init__(self, axes=0):
        self.axes = axes

    def __call__(self, sample):
        image, mask = sample["img"], sample["mask"]
        degree = np.random.randint(0, 3)
        image = np.rot90(image, degree, axes=self.axes)
        mask = np.rot90(mask, degree, axes=self.axes)
        return {"img": image, "mask": mask}


class RandomFlip(object):
    """Horizontally/Vertical flip the given Image randomly with a given probability. the image can be 2D or 3D.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        axis (int): axis to flip. 0 for vertical flip, 1 for horizontal flip. Default value is 0.
        sample (dict): {'mnv': mnv, 'fluid': fluid, 'ga': ga, 'drusen': drusen, 'label': label}
    """

    def __init__(self, p=0.5, axis=0):
        self.p = p
        self.axis = axis

    def __call__(self, sample):
        mnv, fluid, ga, drusen, label = sample["mnv"], sample["fluid"], sample["ga"], sample["drusen"], sample["label"]
        if np.random.random() < self.p:
            mnv = np.flip(mnv, axis=self.axis+1)
            fluid = np.flip(fluid, axis=self.axis+1)
            ga = np.flip(ga, axis=self.axis+1)
            drusen = np.flip(drusen, axis=self.axis+1)
        return {"mnv": mnv, "fluid": fluid, "ga": ga, "drusen": drusen, "label": label}

class RandomFlip3D(object):
    """Horizontally/Vertical flip the given Image randomly with a given probability. the image can be 2D or 3D.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        axis (int): axis to flip. 0 for vertical flip, 1 for horizontal flip. Default value is 0.
        sample (dict): {'mat': mat, 'label': label}
    """

    def __init__(self, p=0.5, axis=0):
        self.p = p
        self.axis = axis

    def __call__(self, sample):
        mat, label = sample["mat"], sample["label"]
        if np.random.random() < self.p:
            mat = np.flip(mat, axis=self.axis+1)
        return {"mat": mat, "label": label}


if __name__ == "__main__":
    img = np.random.randint(0, 255, (1, 128, 128, 128))
    mask = np.random.randint(0, 255, (1, 128, 128, 128))

    a = AddGaussianNoise()(sample={"mat": img, "mask": mask})
    print(a["mat"].shape)
