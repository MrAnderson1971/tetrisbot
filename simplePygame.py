import pygame

def printTextCenter(text, font, surface, x, y, colour, aliasing = True, bg = None):
    '''Draws text on the screen'''
    
    textobj = font.render(text, aliasing, colour, bg)
    textrect = textobj.get_rect()
    textrect.centerx, textrect.centery = (x, y)
    surface.blit(textobj, textrect)
    return textrect
        

def printTextLeft(text, font, surface, x, y, colour, aliasing = True, bg = None):
    '''Draws text on the screen'''

    textobj = font.render(text, aliasing, colour, bg)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)
    return textrect


def printTextRight(text, font, surface, x, y, colour, aliasing = True, bg = None):
    '''Draws text on the screen'''

    textobj = font.render(text, aliasing, colour, bg)
    textrect = textobj.get_rect()
    textrect.bottomright = (x, y)
    surface.blit(textobj, textrect)
    return textrect

def printLeftBlank(text, font, surface, x, y, colour, aliasing = True, bg = None):
    '''Prints the text blank'''

    textobj = font.render(text, aliasing, colour, bg)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    return (textobj, textrect)

def printTopRightBlank(text, font, surface, x, y, colour, aliasing = True, bg = None):
    '''Prints the text blank'''

    textobj = font.render(text, aliasing, colour, bg)
    textrect = textobj.get_rect()
    textrect.topright = (x, y)
    return (textobj, textrect)

