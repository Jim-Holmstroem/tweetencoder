from __future__ import print_function

import string
import Image, ImageDraw #NOTE requires PIL version of >=1.1.6 
                        #for draw.line(,width) to work properly
import itertools as it
import functools as ft

import random

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

resulting image: 256x256

"""

WIDTH, HEIGHT = 256, 256
BGCOLOR = (0,0,0,1)

#http://upload.wikimedia.org/wikipedia/commons/d/df/EGA_Table.PNG
colors = dict(enumerate( #not in order but unique
    map(lambda cc: "#{hexc}".format(
        hexc=''.join(cc)
        ),
        it.product(["0","5","A","F"], repeat=3)
    )
))

grid = dict(enumerate(
    range(2,256,4)
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
    renderDNA(dna)
    img = Image.new('RGBA', (WIDTH, HEIGHT), BGCOLOR) #blank image
    draw = ImageDraw.Draw(img)
    def renderLine(draw, line):
        if(line[-1] is not None): #if not padding
            colorId, xa, ya, xb, yb, width = map(
                binarray2int,
                grouper(
                    6, 
                    line
                )
            )
            draw.line(
                [
                    (grid[xa], grid[ya]),
                    (grid[xb], grid[yb])
                ],
                width=grid[width],
                fill=colors[colorId]
            )

    list(map(
        ft.partial(
            renderLine, 
            draw
        ),
        grouper(
            36, 
            dna
        )
    ))
    del draw
    return img

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


