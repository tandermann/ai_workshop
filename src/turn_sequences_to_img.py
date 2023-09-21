#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:53:10 2020

@author: Tobias Andermann (tobiasandermann88@gmail.com)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import collections
from matplotlib import cm
import os

def count_kmers(sequence, k):
    d = collections.defaultdict(int)
    for i in range(len(sequence)-(k-1)):
        d[sequence[i:i+k]] +=1
    for key in d.keys():
        if "N" in key:
            del d[key]
    return d

def probabilities(sequence, kmer_count, k):
    probabilities = collections.defaultdict(float)
    N = len(sequence)
    for key, value in kmer_count.items():
        probabilities[key] = float(value) / (N - k + 1)
    return probabilities

def chaos_game_representation(probabilities, k):
    array_size = int(np.sqrt(4**k))
    chaos = np.zeros([array_size,array_size])
    maxx = array_size
    maxy = array_size
    posx = 1
    posy = 1
    for key, value in probabilities.items():
        for char in key:
            if char == "T":
                posx += maxx / 2
            elif char == "C":
                posy += maxy / 2
            elif char == "G":
                posx += maxx / 2
                posy += maxy / 2
            maxx = maxx / 2
            maxy /= 2
        chaos[int(posy)-1][int(posx)-1] = value
        maxx = array_size
        maxy = array_size
        posx = 1
        posy = 1
    return chaos


# read all the sequences
input_file = '/Users/xhofmt/GitHub/dna_sequence_nn/data/sequence.fasta'
genus_sequence_dict = {}
for record in SeqIO.parse(input_file,'fasta'):
    genus = record.description.replace(record.id+' ','').split(' ')[0]
    if not genus.isupper():
        pass
    genus_sequence_dict.setdefault(genus,[])
    genus_sequence_dict[genus].append(str(record.seq))


# select the best taxa (those with most sequences)
produce_n_images = 30
n_seqs = np.array([len(genus_sequence_dict[i]) for i in genus_sequence_dict])
taxa = np.array([i for i in genus_sequence_dict])
#n_seqs[n_seqs>=produce_n_images]
selected_taxa = taxa[n_seqs>=produce_n_images]
#index_order = np.argsort(n_seqs)[::-1]
#taxa_sorted = np.array(taxa)[index_order]
final_taxon_selection = [i for i in selected_taxa if i not in ['Homo', 'UNVERIFIED:']]


# create DNA images for selected taxa
out_folder = '/Users/xhofmt/GitHub/dna_sequence_nn/data/images'
train_out_folder = os.path.join(out_folder,'train')
validation_out_folder = os.path.join(out_folder,'validation')
kmer_size = 4
for taxon in final_taxon_selection:
    seqs = np.random.choice(genus_sequence_dict[taxon],produce_n_images,replace=False)
    for i,dna_seq in enumerate(seqs):
        if i < int(np.round(0.66*produce_n_images)):        
            taxon_out = os.path.join(train_out_folder,taxon)
        else:
            taxon_out = os.path.join(validation_out_folder,taxon)
        if not os.path.exists(taxon_out):
            os.makedirs(taxon_out)
        dna_seq = dna_seq.replace('N','')
        kmer_count = count_kmers(dna_seq,kmer_size)
        kmer_probs = probabilities(dna_seq, kmer_count, kmer_size)
        cgr_array = chaos_game_representation(kmer_probs, kmer_size)
        f = plt.figure(figsize=[5,5])
        plt.imshow(cgr_array, interpolation='nearest', cmap=cm.gray_r)      
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(os.path.join(taxon_out,'%04d.png'%i), bbox_inches = 'tight',pad_inches = 0)
        f.clear()
        plt.close(f)

