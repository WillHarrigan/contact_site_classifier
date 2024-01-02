# Load Libraries
from Bio import SeqIO
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import warnings

cos = nn.CosineSimilarity(dim=0, eps=1e-6)
import esm

import matplotlib.pyplot as plt
import seaborn as sns

from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import IUPACData
warnings.simplefilter('ignore', PDBConstructionWarning)

from collections import defaultdict
import os
import sys
import urllib.request
import Bio.PDB
import random
import glob
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import GridSearchCV

# Set directories

base_dir = '/Users/williamharrigan/Desktop/Github/contact_site_classifier/attention_classifier/data_files/'
desktop = '/Users/williamharrigan/Desktop/'
fasta_file = base_dir + 'rcsb_pdb_3KYN.fasta'
pdb_filename = base_dir + '3kyn.pdb'
structure_dir = base_dir +'structure_files/'
fasta_dir = base_dir +'fasta_files/'
casp_dir = '/Users/williamharrigan/Desktop/UH/Year_2/Research/contact_site_classifier/casp7/' 
casp_95 = casp_dir + 'training_95'


model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
print(model.eval())

warnings.simplefilter('ignore', PDBConstructionWarning)
parser = PDBParser()
    
# Set model to use cuda GPU
if torch.cuda.is_available():
    model = model.cuda()

# Get amino acid signature from 3 letter amino acid abbreviation

def simple_aa(three_letter_code):
    return IUPACData.protein_letters_3to1.get(three_letter_code.capitalize())


def parse_casp7_file(file_path):
    data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[ID]'):
                sequence_id = next(file).strip()
            elif line.startswith('[PRIMARY]'):
                sequence = next(file).strip()
                data_dict[sequence_id] = sequence

    return data_dict


def download_pdb(pdb_id, structure_dir, downloadurl="http://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
        Note that the unencrypted HTTP protocol is used by default
        to avoid spurious OpenSSL errors...
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    pdbfn = pdb_id + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(structure_dir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return pdbfn
    except Exception as err:
        # all sorts of things could have gone wrong...
        print(str(err), file=sys.stderr)
        return None


def download_fasta(pdb_id, fasta_dir, downloadurl="https://www.rcsb.org/fasta/entry/"):
    pdbfn = pdb_id
    url = downloadurl + pdbfn
    outfnm = os.path.join(fasta_dir, F'{pdbfn}.fasta')
    try:
        urllib.request.urlretrieve(url, outfnm)
        return pdbfn
    except Exception as err:
        # all sorts of things could have gone wrong...
        print(str(err), file=sys.stderr)
        return None


## Take pdb_ids that occur only once in CASP7 dataset and generate fasta files 
# of sequences from pdb
# Only sequences in first chain are taken to keep things simple down the line

def generate_fastas(ids):
    
    protein_data = {}

    for pdb_id in ids:
        fasta_file = fasta_dir + pdb_id + '.fasta'
        for record in SeqIO.parse(fasta_file, "fasta"):
#             print(record.id.split('|')[0].split('_')[0])
#             print(len(str(record.seq)))
            protein_data[record.id.split('|')[0].split('_')[0]] = str(record.seq)
            break
    return protein_data

def load_fastas(fasta_dir):

    protein_data = {}

    for i in glob.glob(f'{fasta_dir}/*'):
    #     pdb_id = i.split('/')[9].split('.')[0]
        fasta_file = i
        for record in SeqIO.parse(fasta_file, "fasta"):
#             print(record.id.split('|')[0].split('_')[0])
#             print(len(str(record.seq)))
            protein_data[record.id.split('|')[0].split('_')[0]] = str(record.seq)
            break
    return protein_data

def check_sequences(pdb_id, protein_data):  
    # Parse pdb file and save as structure. The pdb file is where we are getting CA coordinates from.
    structure = parser.get_structure(pdb_id, f"{structure_dir+pdb_id}.pdb")

    # Extract desired protein structure from PDB structure (typically only 1 structure to choose from)
    protein_structure = structure[0]

    residue_position = 0
    mismatches = 0
#     print(pdb_id)
    if 'A' in protein_structure:
        for residue in protein_structure['A']:
            if 'CA' in residue:
                if residue_position < len(protein_data[pdb_id]):
                    if simple_aa(residue.resname) != protein_data[pdb_id][residue_position]:
#                         print(residue.id[1], simple_aa(residue.resname), protein_data[pdb_id][residue_position])
                        mismatches+=1
            residue_position+=1
        if mismatches == 0:
#             same_sequence_ids.append(pdb_id)
            return pdb_id

def check_casp_pdb_seqs(protein_data):
    same_sequence_ids = []
    iterations = 0
    for pdb_id, protein_sequence in list(protein_data.items()):
#         print("Iterations: ", iterations)
#         print('Sequence ID: ', pdb_id)
#         check_sequences(pdb_id)
#         print('No mismatches in pdb sequence and pulled sequence: ', check_sequences(pdb_id))
        if check_sequences(pdb_id, protein_data) == None:
            continue
        else:
            same_sequence_ids.append(pdb_id)
        iterations+=1
    return same_sequence_ids


def calc_contact_sites(pdb_id, protein_data, in_contact_sites, non_contact_sites, subset_non_contact_sites):
    structure = parser.get_structure(pdb_id, f"{structure_dir}/{pdb_id}.pdb")  # Ensure correct path joining
    protein_structure = structure[0]
    chain = protein_structure['A']

    # Initialize count variable
    count = 0

    for i, residue1 in enumerate(chain):
        for j, residue2 in enumerate(chain):
            if i <= j:
                continue # Avoids redundant comparisons and self-comparison
            if residue1.id[1] > len(protein_data[pdb_id]) or residue2.id[1] > len(protein_data[pdb_id]):
                continue
            try:
                distance = abs(residue1['CA'] - residue2['CA'])
            except KeyError:
                continue
            aa_distance = abs(residue1.id[1] - residue2.id[1])
            if distance < 5:
                if aa_distance > 2:
    #                 print(residue1.id[1], residue1.resname, residue2.id[1], residue2.resname, distance)
                    in_contact_sites[pdb_id].append({
                        'res_1': residue1.id[1], 
                        'res_2': residue2.id[1], 
                        'sig_1': simple_aa(residue1.resname), 
                        'sig_2': simple_aa(residue2.resname),
                        'aa_dist': aa_distance,
                        'arn_dist': distance,
                        'in_contact': True
                    })
                    count += 1
            else:
                if aa_distance > 2:
                    non_contact_sites[pdb_id].append({
                        'res_1': residue1.id[1], 
                        'res_2': residue2.id[1], 
                        'sig_1': simple_aa(residue1.resname), 
                        'sig_2': simple_aa(residue2.resname),
                        'aa_dist': aa_distance,
                        'arn_dist': distance,
                        'in_contact': False
                    })

    if non_contact_sites[pdb_id]:
        subset_non_contact_sites[pdb_id] = random.sample(non_contact_sites[pdb_id], min(len(non_contact_sites[pdb_id]), len(in_contact_sites[pdb_id])))

    # Optionally print or process the results
    return f"Total contacts found {pdb_id}: {count}"
def contacts_per_pdb(same_sequence_ids, protein_data):

    in_contact_sites = defaultdict(list)
    non_contact_sites = defaultdict(list)
    subset_non_contact_sites = defaultdict(list)

    iterations = 0

    for pdb_id in same_sequence_ids:
        calc_contact_sites(pdb_id, protein_data, in_contact_sites, non_contact_sites, subset_non_contact_sites)
#         print(calc_contact_sites(pdb_id, in_contact_sites, non_contact_sites, subset_non_contact_sites))
#         print(len(in_contact_sites[pdb_id]), len(subset_non_contact_sites[pdb_id]))
#         print("Iterations: ", iterations)
        iterations+=1
        
    return in_contact_sites, non_contact_sites, subset_non_contact_sites

# Initialize contact_data as a defaultdict of lists

def generate_contact_data(in_contact_sites, subset_non_contact_sites):
    # Initialize contact_data as a defaultdict of lists
    contact_data = defaultdict(list)

    # Add data from in_contact_sites
    for pdb_id, contacts in in_contact_sites.items():
        for contact in contacts:
            contact_data[pdb_id].append({
                'res_1': contact['res_1'],
                'res_2': contact['res_2'],
                'sig_1': contact['sig_1'],
                'sig_2': contact['sig_2'],
                'aa_dist': contact['aa_dist'],
                'arn_dist': contact['arn_dist'],
                'in_contact': contact['in_contact']
            })

    # Add data from subset_non_contact_sites
    for pdb_id, non_contacts in subset_non_contact_sites.items():
        for non_contact in non_contacts:
            contact_data[pdb_id].append({
                'res_1': non_contact['res_1'],
                'res_2': non_contact['res_2'],
                'sig_1': non_contact['sig_1'],
                'sig_2': non_contact['sig_2'],
                'aa_dist': contact['aa_dist'],
                'arn_dist': contact['arn_dist'],
                'in_contact': non_contact['in_contact']
            })

# contact_data is now a defaultdict containing all the data from both dictionaries

    return contact_data


# Index protein sequence as sequence 0 (next sequence would be indexed as 1)

def generate_embeddings(pdb_id, protein_data):
    protein_sequence = protein_data[pdb_id]
    esm_input_data = [(0, protein_sequence)]
    # print('Data: ', esm_input_data, '\n')

    # Prepare variables to input sequence into ESM-2 model 
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(esm_input_data)
    batch_tokens = batch_tokens.cuda() if torch.cuda.is_available() else batch_tokens

    # print('batch_tokens: ', '\n\n', batch_tokens, '\n')

    # 4. Input prepared sequence information into model and output as results (contact predictions are included in embedding)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    return results['attentions']

# Extract attentions from all heads and layers for given amino acid residues

def get_x_y(attention_data, res_1, res_2):
    vectors = []
    for layer in range(0,33):
        for head in range(0,20):
            vectors.append(attention_data[0][layer][head][res_1][res_2])

    return vectors


def output_x_y(sequence_ids, contact_data, protein_data):
    X = []
    y = []
    iterations = 0

    for pdb_id in sequence_ids:
        structure = parser.get_structure(pdb_id, f"{structure_dir}/{pdb_id}.pdb")  # Ensure correct path joining
        protein_structure = structure[0]
        chain = protein_structure['A']
        first_residue = list(chain.get_residues())[0].id[1]
        print('Iteration: ', iterations)
        iterations+=1
        if first_residue == 1:
            attention_data = generate_embeddings(pdb_id, protein_data)
            for i in contact_data[pdb_id]:
                    X.append(get_x_y(attention_data, i['res_1'], i['res_2']))
                    y.append(i['in_contact'])
            else:
                continue
    return X,y



