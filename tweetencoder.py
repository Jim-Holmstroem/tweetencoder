from __future__ import print_function

import string
from copy import copy

import cairo

import itertools as it
import functools as ft
import operator as op

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
resujlting image: 256x256

"""

WIDTH = 256
HEIGHT= WIDTH
BGCOLOR = (0, 0, 0, 0)
START_POPULATION = 1024
MUTATION_RATE = 10.0/1024
DNA_LENGTH = 912

def composition(f, *g):
    if(g):
        return lambda *x: f(composition(*g)(*x))
    else:
        return f

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

def baseChange(a, bfrom, bto):
    """lowest first"""
    value = sum(
        map(
            lambda (n, v): v*(bfrom**n),
            enumerate(a)
        )
    )
    if value==0:
        return [0]
    n = 1
    answer = []
    while value!=0:
        v = value%(bto**n)
        value-=v
        answer.append(v//(bto**(n-1)))
        n+=1
    return answer

def dna2tweet(dna):
    """ change base from 2->94 then convert to ascii"""
    return ''.join(
        map(
            lambda t: unichr(32+t),
            baseChange(
                filter(lambda x: x is not None, dna), 
                2, 
                94
            )
        )
    )

def tweet2dna(tweet):
    dna = baseChange(
        map(
            lambda t: ord(t)-32,
            tweet
        ),
        94,
        2
    )
    assert(len(dna)<=DNA_LENGTH)
    return dna+[0]*(DNA_LENGTH-len(dna))

grid = dict(enumerate(
    range(2, WIDTH, 4)
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
    return -np.add.reduce(sdiff, axis=(0,1,2))

fitnessDNA = composition(ft.partial(fitness2ref, surfaceTrain), renderImage)

def rndmFilename():
    return '{}.png'.format(''.join(random.sample(string.letters, 16)))

def rndmDNA(length=DNA_LENGTH):
    def rndmBool(dummy=None):
        return random.randint(0,1)
    return map(rndmBool, range(length)) 

#def mutateOrder(dna):
#    gens = list(grouper(6*6, dna))
#    nonLineGens = filter(lambda gen: gen[-1] is None, gens)
#    lineGens = filter(lambda gen: gen[-1] is not None, gens)
#    swapIndexA, swapIndexB = random.sample(range(len(lineGens)), 2)
#    lineGens[swapIndexB], lineGens[swapIndexA] = lineGens[swapIndexA], lineGens[swapIndexB] #a bit unpure
#    return list(it.chain(*(lineGens+nonLineGens)))

def mutate(dna):
    index = random.randrange(DNA_LENGTH)
    dnap = copy(dna.data)
    dnap[index] = int(not dnap[index])
    return DNA(dnap)

def crossover(dna, dnb):
    """crossovers 1-4 time(s) doesn't modify dna,dnb"""
    minLength = min(
        map(
            composition(
                len,
                lambda dn: filter(
                    lambda x: x is not None, 
                    dn
                )
            ),
            [
                dna.data, 
                dnb.data
            ]
        )
    )
    indeces = random.sample(
        range(minLength),
        random.randint(1, 4)
    )
    def crossoverAt(dna, dnb, ats):
        """``ats'' is a list of indices to crossover, modifies input"""
        if(len(ats)):
            at=ats[0]
            swp = dna[at:]
            dna[at:] = dnb[at:]
            dnb[at:] = swp
            return crossoverAt(dna, dnb, list(it.islice(ats, 1, None)))
        else:
            return dna, dnb
    return map(DNA, crossoverAt(copy(dna.data), copy(dnb.data), indeces))
    
def evaluate(population, p0 = 1.0/START_POPULATION):
    #replot distribution
    rawValue = np.array(map(op.methodcaller('fitness'), population))
    value = rawValue/rawValue.sum()
    valueBaseline = value + p0
    return valueBaseline/valueBaseline.sum()

class DNA(object):
    def __init__(self, data=None):
        self.__fitness = None
        self.__image = None
        self.__tweet = None
        if(data is None):
            self.data = rndmDNA()
        else:
            if(isinstance(data, basestring)):
                self.data = tweet2dna(data)
            else:
                self.data = data

    def fitness(self):
        if self.__fitness is None:
            self.__fitness = fitness2ref(
                surfaceTrain,
                self.image()
            )
        return self.__fitness

    def image(self):
        if self.__image is None:
            self.__image = renderImage(self.data) 
        return self.__image

    def tweet(self):
        if self.__tweet is None:
            self.__tweet = dna2tweet(self.data)
        return self.__tweet 
    def __len__(self):
        return len(self.data)
    def __str__(self):
        def renderLine(line):
            def renderData(data):
                return ''.join(map(str, filter(lambda x: x is not None,data)))
            return '  {}'.format(' '.join(map(renderData,grouper(6, line))))
        data = '\n'.join(
            map(
                renderLine, 
                grouper(6*6, self.data)
            )
        )
        return "DNA{{\n{}\n}}".format(data)
    def __repr__(self):
        return "DNA(id={})".format(id(self))

def combat4(combatants):
    """combat and procreate and die etc"""
    assert(len(combatants)==4)
    winnerA, winnerB, loserA, loserB = sorted(combatants, key=op.methodcaller('fitness'))
    children = crossover(winnerA, winnerB)
    return winnerA, winnerB, children[0], children[1] 

def life(population):
    """maintains polpulation size, inplace life"""
    fitnesses = evaluate(population)
    chosenIndexs = np.random.choice(
        range(len(population)), 
        4, 
        replace=False, 
        p=fitnesses
    )
    after = combat4(op.itemgetter(*chosenIndexs)(population)) 
    #a bit ugly, but I think it's the easiest way to do this
    for i,j  in enumerate(chosenIndexs):
        population[j] = after[i] #put them back

    #mutations
    for i in range(len(population)):
        if(np.random.random()<MUTATION_RATE/2):
            print('mutate')
            population[i] = mutate(population[i])
    #for i in range(len(population)):
    #    if(np.random.random()<MUTATION_RATE/2):
    #        print('mutateOrder')
    #        population[i] = mutateOrder(population[i])

population = [DNA() for i in range(START_POPULATION)]
generation = 0
while True:
    life(population)

    if(generation%100==0):
        population[
            np.argmax(
                np.array(
                    evaluate(
                        population
                    )
                )
            )
        ].image().write_to_png(
            'tmp/best{}.png'.format(generation)
        )
     
    generation+=1
    print(generation)

