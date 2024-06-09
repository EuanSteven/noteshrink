# MIT License
# 
# Copyright (c) 2023, Matt Zucker & Euan Steven
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -*- encoding: utf-8 -*-
# ========== noteshrink.py ==========
# Author : Matt Zucker & Euan Steven
# Date Created : 06/09/2016
# Date Modified : 20/12/2023
# Version : 1.2
# License : Apache 2.0
# Description : Convert JPG to PNG
# =============================

# ========== Module Imports ==========

from __future__ import print_function

print("========== NoteShrink ==========" '\n')
print("Importing Modules...")

# Built-in Modules
import time
import sys
import os
import re
import subprocess
import shlex
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

# Third Party Modules
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans, vq


print("Modules Imported." '\n')

# ========== Quantize Images ==========

def quantize(image, bits_per_channel=None):
    print("Quantizing Images...")
    if bits_per_channel is None:
        bits_per_channel = 6

    assert image.dtype == np.uint8

    shift = 8 - bits_per_channel
    halfbin = (1 << shift) >> 1

    print("Quantizing Images Done." '\n')
    return ((image.astype(int) >> shift) << shift) + halfbin

# ========== RGB to HSV ==========

def pack_rgb(rgb):
    print("Packing RGB...")
    orig_shape = None

    if isinstance(rgb, np.ndarray):
        assert rgb.shape[-1] == 3
        orig_shape = rgb.shape[:-1]
    else:
        assert len(rgb) == 3
        rgb = np.array(rgb)

    rgb = rgb.astype(int).reshape((-1, 3))

    packed = (rgb[:, 0] << 16 | rgb[:, 1] << 8 | rgb[:, 2])

    print("Packing RGB Done." '\n')
    if orig_shape is None:
        return packed
    else:
        return packed.reshape(orig_shape)
    
# ========== HSV to RGB ==========    

def unpack_rgb(packed):
    print("Unpacking RGB...")
    orig_shape = None

    if isinstance(packed, np.ndarray):
        assert packed.dtype == int
        orig_shape = packed.shape
        packed = packed.reshape((-1, 1))

    rgb = ((packed >> 16) & 0xff, (packed >> 8) & 0xff, (packed) & 0xff)

    print("Unpacking RGB Done." '\n')
    if orig_shape is None:
        return rgb
    else:
        return np.hstack(rgb).reshape(orig_shape + (3,))

# ========== Get Background Color ==========

def get_bg_color(image, bits_per_channel=None):
    print("Getting Background Color...")
    assert image.shape[-1] == 3

    quantized = quantize(image, bits_per_channel).astype(int)
    packed = pack_rgb(quantized)

    unique, counts = np.unique(packed, return_counts=True)

    packed_mode = unique[counts.argmax()]

    print("Getting Background Color Done." '\n')
    return unpack_rgb(packed_mode)

# ========== RGB to SV ==========

def rgb_to_sv(rgb):
    print("RGB to SV...")
    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)

    axis = len(rgb.shape) - 1
    cmax = rgb.max(axis=axis).astype(np.float32)
    cmin = rgb.min(axis=axis).astype(np.float32)
    delta = cmax - cmin

    saturation = delta.astype(np.float32) / cmax.astype(np.float32)
    saturation = np.where(cmax == 0, 0, saturation)

    value = cmax / 255.0

    print("RGB to SV Done." '\n')
    return saturation, value

# ========== Postprocess ==========

def postprocess(output_filename, options):
    print("Postprocessing...")
    assert options.postprocess_cmd

    base, _ = os.path.splitext(output_filename)
    post_filename = base + options.postprocess_ext

    cmd = options.postprocess_cmd
    cmd = cmd.replace('%i', output_filename)
    cmd = cmd.replace('%o', post_filename)
    cmd = cmd.replace('%e', options.postprocess_ext)

    subprocess_args = shlex.split(cmd)

    if os.path.exists(post_filename):
        os.unlink(post_filename)

    try:
        result = subprocess.call(subprocess_args)
        before = os.stat(output_filename).st_size
        after = os.stat(post_filename).st_size
    except OSError:
        result = -1

    print("Postprocessing Done." '\n')

# ========== Percent ==========

def percent(string):
    return float(string) / 100.0

# ========== Argument Parser ==========

def get_argument_parser():
    parser = ArgumentParser(
        description='Convert Images to Clear PNGs')

    show_default = ' (default %(default)s)'

    parser.add_argument('filenames', metavar='IMAGE', nargs='+',
                        help='Files to Convert')

    parser.add_argument('-q', dest='quiet', action='store_true',
                        default=False,
                        help='Reduce Program Output')

    parser.add_argument('-b', dest='basename', metavar='BASENAME',
                        default='page',
                        help='Output PNG Filename Base' + show_default)

    parser.add_argument('-v', dest='value_threshold', metavar='PERCENT',
                        type=percent, default='25',
                        help='Background Value Threshold %%' + show_default)

    parser.add_argument('-s', dest='sat_threshold', metavar='PERCENT',
                        type=percent, default='20',
                        help='Background Saturation '
                             'Threshold %%' + show_default)

    parser.add_argument('-n', dest='num_colors', type=int,
                        default='8',
                        help='Number of Output Colors ' + show_default)

    parser.add_argument('-p', dest='sample_fraction',
                        metavar='PERCENT',
                        type=percent, default='5',
                        help='%% of Pixels to Sample' + show_default)

    parser.add_argument('-w', dest='white_bg', action='store_true',
                        default=False, help='Make Background White')

    parser.add_argument('-g', dest='global_palette',
                        action='store_true', default=False,
                        help='Use Global Palette')

    parser.add_argument('-S', dest='saturate', action='store_false',
                        default=True, help='Do Not Saturate Palette')

    parser.add_argument('-K', dest='sort_numerically',
                        action='store_false', default=True,
                        help='Keep Filenames Ordered as Specified')

    parser.add_argument('-P', dest='postprocess_cmd', default=None,
                        help='Set Post Processing Command (see -O, -C, -Q)')

    parser.add_argument('-e', dest='postprocess_ext',
                        default='_post.png',
                        help='Filename Suffix/Extension for '
                             'Post Processing Command')

    parser.add_argument('-O', dest='postprocess_cmd',
                        action='store_const',
                        const='optipng -silent %i -out %o',
                        help='Same as -P "%(const)s"')

    parser.add_argument('-C', dest='postprocess_cmd',
                        action='store_const',
                        const='pngcrush -q %i %o',
                        help='Same as -P "%(const)s"')

    parser.add_argument('-Q', dest='postprocess_cmd',
                        action='store_const',
                        const='pngquant --ext %e %i',
                        help='Same as -P "%(const)s"')

    return parser

# ========== Get Filenames ==========

def get_filenames(options):
    print("Getting Filenames...")
    if not options.sort_numerically:
        return options.filenames

    filenames = []

    for filename in options.filenames:
        basename = os.path.basename(filename)
        root, _ = os.path.splitext(basename)
        matches = re.findall(r'[0-9]+', root)
        if matches:
            num = int(matches[-1])
        else:
            num = -1
        filenames.append((num, filename))

    print("Getting Filenames Done." '\n')
    return [fn for (_, fn) in sorted(filenames)]

# ========== Load ==========

def load(input_filename):
    print("Loading...")
    try:
        pil_img = Image.open(input_filename)
    except IOError:
        sys.stderr.write('Error : Cannot Open {}\n'.format(
            input_filename))
        return None, None

    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    if 'dpi' in pil_img.info:
        dpi = pil_img.info['dpi']
    else:
        dpi = (300, 300)

    img = np.array(pil_img)

    print("Loading Done." '\n')
    return img, dpi

# ========== Sample Pixels ==========

def sample_pixels(img, options):
    print("Sampling Pixels...")
    pixels = img.reshape((-1, 3))
    num_pixels = pixels.shape[0]
    num_samples = int(num_pixels * options.sample_fraction)

    idx = np.random.choice(num_pixels, size=num_samples, replace=False)
    print("Sampling Pixels Done." '\n')
    return pixels[idx]

# ========== Get Background Color ==========

def get_fg_mask(bg_color, samples, options):
    print("Getting Foreground Mask...")
    s_bg, v_bg = rgb_to_sv(bg_color)
    s_samples, v_samples = rgb_to_sv(samples)

    s_diff = np.abs(s_bg - s_samples)
    v_diff = np.abs(v_bg - v_samples)

    fg_mask = (v_diff >= options.value_threshold) | (s_diff >= options.sat_threshold)

    print("Getting Foreground Mask Done." '\n')
    return fg_mask

# ========== Get Palette ==========

def get_palette(samples, options, return_mask=False, kmeans_iter=40):
    print("Getting Palette...")
    MAX_SAMPLES_FOR_KMEANS = 10000

    bg_color = get_bg_color(samples, 6)

    fg_mask = get_fg_mask(bg_color, samples, options)

    subsampled_samples = samples[fg_mask].astype(np.float32)
    if subsampled_samples.shape[0] > MAX_SAMPLES_FOR_KMEANS:
        idx = np.random.choice(subsampled_samples.shape[0], size=MAX_SAMPLES_FOR_KMEANS, replace=False)
        subsampled_samples = subsampled_samples[idx]

    centers, _ = kmeans(subsampled_samples,
                        options.num_colors - 1,
                        iter=kmeans_iter)

    palette = np.vstack((bg_color, centers)).astype(np.uint8)

    print("Getting Palette Done." '\n')
    if not return_mask:
        return palette
    else:
        return palette, fg_mask

# ========== Apply Palette ==========

def apply_palette(img, palette, options):
    print("Applying Palette...")

    bg_color = palette[0]

    fg_mask = get_fg_mask(bg_color, img, options)

    orig_shape = img.shape

    pixels = img.reshape((-1, 3))
    fg_mask = fg_mask.flatten()

    num_pixels = pixels.shape[0]

    labels = np.zeros(num_pixels, dtype=np.uint8)

    labels[fg_mask], _ = vq(pixels[fg_mask], palette)

    print("Applying Palette Done." '\n')
    return labels.reshape(orig_shape[:-1])

# ========== Save ==========

def save(output_filename, labels, palette, dpi, options):
    print("Saving...")

    if options.saturate:
        palette = palette.astype(np.float32)
        pmin = palette.min()
        pmax = palette.max()
        palette = 255 * (palette - pmin) / (pmax - pmin)
        palette = palette.astype(np.uint8)

    if options.white_bg:
        palette = palette.copy()
        palette[0] = (255, 255, 255)

    output_img = Image.fromarray(labels, 'P')
    output_img.putpalette(palette.flatten())
    output_img.save(output_filename, dpi=dpi)

    print("Saving Done." '\n')

# ========== Get Global Palette ==========

def get_global_palette(filenames, options):
    print("Getting Global Palette...")
    input_filenames = []
    all_samples = []

    for input_filename in filenames:
        img, _ = load(input_filename)
        if img is None:
            continue

        samples = sample_pixels(img, options)
        input_filenames.append(input_filename)
        all_samples.append(samples)

    num_inputs = len(input_filenames)

    all_samples = [s[:int(round(float(s.shape[0]) / num_inputs))]
                   for s in all_samples]

    all_samples = np.vstack(tuple(all_samples))

    global_palette = get_palette(all_samples, options)

    print("Getting Global Palette Done." '\n')

    return input_filenames, global_palette

# ========== Process Image ==========

def process_image(input_filename, options, global_palette=None, outputs=[]):
    print("Processing Image...")
    img, dpi = load(input_filename)
    if img is None:
        return None

    output_filename = '{}{:04d}.png'.format(
        options.basename, len(outputs))

    samples = sample_pixels(img, options)

    if not options.global_palette:
        palette = get_palette(samples, options)
    else:
        palette = global_palette

    labels = apply_palette(img, palette, options)

    save(output_filename, labels, palette, dpi, options)

    if options.postprocess_cmd:
        post_filename = postprocess(output_filename, options)
        if post_filename:
            output_filename = post_filename

    print("Processing Image Done." '\n')
    return output_filename

# ========== Apply Palette ==========

def apply_palette(img, palette, options):
    print("Applying Palette...")

    bg_color = palette[0]

    fg_mask = get_fg_mask(bg_color, img, options)

    orig_shape = img.shape

    pixels = img.reshape((-1, 3))
    fg_mask = fg_mask.flatten()

    num_pixels = pixels.shape[0]

    labels = np.zeros(num_pixels, dtype=np.uint8)

    labels[fg_mask], _ = vq(pixels[fg_mask], palette)

    print("Applying Palette Done." '\n')
    return labels.reshape(orig_shape[:-1])

# ========== Main ==========

def notescan_main(options):
    filenames = get_filenames(options)
    outputs = []
    global_palette = None

    if options.global_palette and len(filenames) > 1:
        _, global_palette = get_global_palette(filenames, options)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, filename, options, global_palette) for filename in filenames]

        for future in futures:
            output_filename = future.result()
            if output_filename:
                outputs.append(output_filename)

# ========== Run Main ==========

if __name__ == '__main__':
    startTime = time.time()
    print("Running Main...")
    notescan_main(options=get_argument_parser().parse_args())
    print("Running Main Done." '\n')
    endTime = time.time()
    totalTime = endTime - startTime
    totalTimeFormatted = time.strftime("%H:%M:%S", time.gmtime(totalTime))
    print("Total Time: " + totalTimeFormatted)