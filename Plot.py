__author__ = 'satya'

import numpy as np

import matplotlib . pyplot as plt
import matplotlib . cm as cm
from LoadFaces import loadFaces


#########################
X = loadFaces()
########################

def create_font ( fontname = 'Tahoma ' , fontsize =10) :
    return {  'fontname' : fontname ,  'fontsize ': fontsize }

def subplot ( title , images , rows , cols , sptitle = " subplot " , sptitles =[] , colormap = cm .
    gray , ticks_visible = True , filename = None ) :
    fig = plt . figure ()
    # main title
    fig . text (.5 , .95 , title , horizontalalignment = 'center')
    for i in xrange ( len ( images ) ) :
        ax0 = fig . add_subplot ( rows , cols ,( i +1) )
        plt . setp ( ax0 . get_xticklabels () , visible = False)
        plt . setp ( ax0 . get_yticklabels () , visible = False)
        if len ( sptitles ) == len ( images ) :
            plt . title ( " % s #% s " % ( sptitle , str ( sptitles [ i ]) ) , create_font ( 'Tahoma' ,10) )
        else :
            plt . title ( " % s #% d " % ( sptitle , ( i +1) ) , create_font (  'Tahoma' ,10) )
        plt . imshow ( np . asarray ( images [ i ]) , cmap = colormap )

    print 'I WILL SHOW'
    plt . show ()
    #else :
        #fig . savefig ( filename )


def project (W , X , mu = None ) :
    if mu is None :
        return np . dot (X , W )
    return np . dot ( X - mu , W )

def reconstruct (W , Y , mu = None ) :
    if mu is None :
        return np . dot (Y , W . T )
    return np . dot (Y , W . T ) + mu

###################################################################
def normalize (X , low , high , dtype = None ) :
    X = np . asarray ( X )
    minX , maxX = np . min ( X ) , np . max ( X )
    # normalize to [0...1].
    X = X - float ( minX )
    X = X / float (( maxX - minX ) )
    # scale to [ low ... high ].
    X = X * ( high - low )
    X = X + low
    if dtype is None :
        return np . asarray ( X )
    return np . asarray (X , dtype = dtype )






####################################################################

print 'I AM HERE'
print 'LENGTH OF X----->',len(X)
steps =[ i for i in xrange (10 ,  320 , 20) ]

E = []
print 'I AM HERE TOO'
for i in xrange (16) :
    'INSIDE FOR LOOP'
    numEvs = steps [ i ]
    P = project ( W [: ,0: numEvs ] , X [0]. reshape (1 , -1) , mu )
    R = reconstruct ( W [: ,0: numEvs ] , P , mu )
    # reshape and append to plots
    R = R . reshape ( X [0]. shape )
    E . append ( normalize (R ,0 ,255) )
    print 'E------------------>', E
# plot them and store the plot to " p y t h o n _ r e c o n s t r u c t i o n . pdf "
subplot ( title = " Reconstruction AT & T Facedatabase " , images =E , rows =4 , cols =4 , sptitle = "Eigenvectors" ,sptitles = steps , colormap = cm . gray , filename = "python_pca_reconstruction.png" )