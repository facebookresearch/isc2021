# Lear's GIST implementation, version 1.1, (c) INRIA 2009, Licence: PSFL


all: compute_gist compute_gist_stream

WFFTINC=
WFFTLIB=
# WFFTINC=-I/scratch/adonis/mordelet/install/include
# WFFTLIB=-L/scratch/adonis/mordelet/install/lib


# gist.c: ../../descriptors/gist.c
#	cp $< $@


gist.o: gist.c gist.h standalone_image.h
	gcc -c -Wall -O3 -g $< $(WFFTINC) -DUSE_GIST -DSTANDALONE_GIST

standalone_image.o: standalone_image.c standalone_image.h
	gcc -c -Wall -O3 -g $<

compute_gist: compute_gist.c gist.o standalone_image.o
	gcc -Wall -O3 -g -o $@ $^ $(WFFTLIB) -lfftw3f -lm

compute_gist_stream: compute_gist_stream.c gist.o standalone_image.o
	gcc -Wall -g -O3 -o $@ $^ $(WFFTLIB) -lfftw3f -lm



clean:
	rm -f *.o compute_gist
