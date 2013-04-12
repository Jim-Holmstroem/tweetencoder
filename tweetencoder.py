
import string
import Image, ImageDraw #NOTE requires PIL version of >=1.1.6 
                        #for draw.line(,width) to work properly
import itertools as it
import functools as ft

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


def hexs2rgba(hexcolors):
    """
    http://upload.wikimedia.org/wikipedia/commons/d/df/EGA_Table.PNG
    """
    rgb = tuple(map(
        lambda c: int(c, 16)/255.0,
        hexcolors
    ))
    return rgb+(1.0,) 

colors = dict(enumerate(
    map(
        hexs2rgba, #not in the right order
        it.product(["00","55","AA","FF"], repeat=3)
    )
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

def render_image(bindata):
    img = Image.new('RGBA', (100, 100), (0,0,0,0)) #blank image
    draw = ImageDraw.Draw(img)
    def render_line(draw, line):
        draw.line(fill=colors

    map(ft.partial(render_line, draw), grouper(6, bindata))
