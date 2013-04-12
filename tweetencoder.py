from __future__ import print_function

import string
import cairo

import itertools as it
import functools as ft

import random

import numpy as np


"""
    encode(and decode) an image with 140 ascii printable characters(32-126) i.e a tweet
    as good as possible

    solution, genetic programming for the encoding into a defined language that
    simply be decoded

"""

"""
language definition: 1tweet = 152word (word=6bit) = 25lines+4bit = 912bit
    
    #start with only thick lines
    color = 6bit (EGA color table, from the enhanced graphics adapter)
    xa = 6bit \in range(2,256,4)
    ya = 6bit \in range(2,256,4)
    xb = 6bit \in range(2,256,4)
    yb = 6bit \in range(2,256,4)
    width = 6bit \in range(2,256,4)

    #the junk-dna (i.e the offset) is always last and can be threaded as such.
resulting image: 256x256

"""

WIDTH = 256
HEIGHT= WIDTH
BGCOLOR = (0,0,0,0)

#http://upload.wikimedia.org/wikipedia/commons/d/df/EGA_Table.PNG
colors = dict(enumerate( #not in order but unique
    map(
        lambda rgb: rgb+(0.9,),
        it.product(
            [
                float(0x00)/0xFF, 
                float(0x55)/0xFF, 
                float(0xAA)/0xFF, 
                float(0xFF)/0xFF
            ], 
            repeat=3
        )
    )
))

grid = dict(enumerate(
    range(2,WIDTH,4)
))

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)]*n
    return it.izip_longest(
        *args, 
        fillvalue=fillvalue
    )

def binarray2int(binarray):
    return int(
        ''.join(
            map(str, binarray)
        ),
        2
    )

def renderImage(dna):
    #renderDNA(dna)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)

    def renderLine(ctx, line):
        if(line[-1] is not None): #if not padding
            colorId, xa, ya, xb, yb, width = map(
                binarray2int,
                grouper(
                    6, 
                    line
                )
            )
            ctx.move_to(grid[xa], grid[ya])
            ctx.line_to(grid[xb], grid[yb])
            ctx.set_source_rgba(*colors[colorId])
            ctx.set_line_width(grid[width])
            ctx.stroke()

    list(map(
        ft.partial(
            renderLine, 
            ctx
        ),
        grouper(
            36, 
            dna
        )
    ))
    #surface.write_to_png("tmp/tmp.png")
    return surface

surfaceTrain = cairo.ImageSurface.create_from_png("training.png")

def imagesurface2tensor(imgsurface):
    return (
        np.cast['float64'](
            np.frombuffer(
                imgsurface.get_data(), 
                dtype=np.uint8
            )
        )/255.0
    ).reshape(
        imgsurface.get_width(), 
        imgsurface.get_height(),
        len("RGBA")
    )

def fitness2ref(refsurface, imgsurface):
    ref, img = map(imagesurface2tensor, [refsurface, imgsurface])
    sdiff = np.square(ref-img)
    return np.add.reduce(np.add.reduce(np.add.reduce(sdiff))) #if np.version>=1.7.1 => np.add.reduce(sdiff, axis=(0,1,2))

fitnessDNA = ft.partial(fitness2ref, surfaceTrain)

def renderDNA(dna):
    def renderLine(line):
        def renderData(data):
            return ''.join(map(str, data))
        return ' '.join(map(renderData,grouper(6, line)))
    list(map(print, 
        map(
            renderLine, 
            filter(lambda row: row[-1] is not None,
                grouper(6*6, dna)
            )
        )
    ))

def rndmDNA(length=912):
    def rndmBool(dummy=None):
        return random.randint(0,1)
    return map(rndmBool, range(length)) 

def mutateOrder(dna):
    gens = list(grouper(6*6, dna))
    nonLineGens = filter(lambda gen: gen[-1] is None, gens)
    lineGens = filter(lambda gen: gen[-1] is not None, gens)
    swapIndexA, swapIndexB = random.sample(range(len(lineGens)), 2)
    print(swapIndexA, swapIndexB)
    lineGens[swapIndexB], lineGens[swapIndexA] = lineGens[swapIndexA], lineGens[swapIndexB] #a bit unpure
    return list(it.chain(*(lineGens+nonLineGens)))



