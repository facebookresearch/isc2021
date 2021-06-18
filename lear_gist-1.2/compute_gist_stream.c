/* @nolint Lear's GIST implementation, version 1.1, (c) INRIA 2009, Licence: PSFL */
/* Copyright (c) Facebook, Inc. All Rights Reserved */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


#include "gist.h"



static color_image_t *load_ppm(FILE *f) {
  int px,width,height,maxval;
  if(fscanf(f,"P%d %d %d %d",&px,&width,&height,&maxval)!=4 ||
     maxval!=255 || (px!=6 && px!=5)) {
      if(feof(f)) {
          /* fprintf(stderr,"End of input stream\n"); */
          return NULL;
      }
      fprintf(stderr,"Error: input not a raw PGM/PPM with maxval 255\n");
      exit(1);
  }
  fgetc(f); /* eat the newline */
  color_image_t *im=color_image_new(width,height);

  int i;
  for(i=0;i<width*height;i++) {
    im->c1[i]=fgetc(f);
    if(px==6) {
      im->c2[i]=fgetc(f);
      im->c3[i]=fgetc(f);
    } else {
      im->c2[i]=im->c1[i];
      im->c3[i]=im->c1[i];
    }
  }

  return im;
}


static void usage(void) {
  fprintf(stderr,"compute_gist options... [infilename]\n"
          "infile is a PPM raw file\n"
          "options:\n"
          "[-nblocks nb] use a grid of nb*nb cells (default 4)\n"
          "[-orientationsPerScale o_1,..,o_n] use n scales and compute o_i orientations for scale i\n"
          "[-o out.fvecs] output file\n"
          );

  exit(1);
}



int main(int argc,char **args) {

  const char *infilename="/dev/stdin";
  int nblocks=4;
  int n_scale=3;
  int n_orientation[50]={8,8,4};
  const char *outfilename="/dev/null";

  while(*++args) {
    const char *a=*args;

    if(!strcmp(a,"-h")) usage();
    else if(!strcmp(a,"-nblocks")) {
      if(!sscanf(*++args,"%d",&nblocks)) {
        fprintf(stderr,"could not parse %s argument",a);
        usage();
      }
    } else if(!strcmp(a,"-orientationsPerScale")) {
      char *c;
      n_scale=0;
      for(c=strtok(*++args,",");c;c=strtok(NULL,",")) {
        if(!sscanf(c,"%d",&n_orientation[n_scale++])) {
          fprintf(stderr,"could not parse %s argument",a);
          usage();
        }
      }
    } else if(!strcmp(a, "-o")) {
        outfilename = *++args;
        if(!outfilename) {
            fprintf(stderr,"no output file set\n");
            usage();
        }
    } else {
      infilename=a;
    }
  }

  FILE *f=fopen(infilename,"r");
  if(!f) {
    perror("could not open infile");
    exit(1);
  }

  FILE *out_f = fopen(outfilename, "w");
  if(!f) {
    perror("could not open outfile");
    exit(1);
  }

  int width = -1, height = -1;
  image_list_t *G = NULL;
  int tot_oris;

  int imno = 0;
  for(;;) {
      printf("image %d\n", imno);
      color_image_t *img = load_ppm(f);
      if (!img) break;
      if (width == -1) {
          width = img->width;
          height = img->height;
          printf("  width=%d height=%d\n", width, height);
          assert(width >= 8 && height >= 8);
          tot_oris=0;
          int i;
          for(i=0;i<n_scale;i++) tot_oris+=n_orientation[i];
          G = create_gabor(n_scale, n_orientation, width, height);
      } else {
          assert(width == img->width && height == img->height);
      }
      //   float *desc=color_gist_scaletab(im,nblocks,n_scale,norientation);

      color_prefilt(img, 4);

      float *g = color_gist_gabor(img, nblocks, G);

      int descsize = tot_oris*nblocks*nblocks*3;

      fwrite(&descsize, 1, sizeof(descsize), out_f);
      fwrite(g, descsize, sizeof(*g), out_f);
      free(g);

      color_image_delete(img);
      imno++;
  }
  fclose(out_f);
  fclose(f);
  image_list_delete(G);


  return 0;
}
