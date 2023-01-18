# import packages

# general tools
import numpy as np
from scipy import stats
from chembl_structure_pipeline.standardizer import standardize_mol, get_parent_mol

# RDkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Pytorch
import torch
from torch_geometric.data import Data as GeometricData







# auxiliary functions for data saving/cleaning/preprocessig

def smiles_standardisation(smiles_list, get_parent_smiles = True):

    standardised_smiles_list = list(range(len(smiles_list)))

    for (k, smiles) in enumerate(smiles_list):

        try:

            # convert smiles to mol object
            mol = Chem.MolFromSmiles(smiles)

            # standardise mol object
            standardised_mol = standardize_mol(mol, check_exclusion = True)

            if get_parent_smiles == True:

                # systematically remove salts, solvents and isotopic information to get parent mol
                (standardised_mol, exclude) = get_parent_mol(standardised_mol,
                                                             neutralize = True,
                                                             check_exclusion=True,
                                                             verbose = False)

            # convert mol object back to smiles
            standardised_smiles = Chem.MolToSmiles(standardised_mol)

            # replace smiles with standardised parent smiles
            standardised_smiles_list[k] = standardised_smiles

        except:

            # leave smiles unchanged if it generates an exception
            standardised_smiles_list[k] = smiles

    return np.array(standardised_smiles_list)


# task-independent molecular featurisations (which only depend on the smiles string of a molecule)

def rdkit_mol_descriptors_from_smiles(smiles_string, descriptor_list = None):

    """Generates a vector of rdkit molecular descriptors. Per default, we use the same 200 descriptors as in the papers
    "Analyzing Learned Molecular Representations for Property Prediction" and
    "Molecular representation learning with languagemodels and domain-relevant auxiliary tasks". """

    if descriptor_list == None:

        # 200 descriptors
        descriptor_list = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
                           'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11',
                           'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
                           'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
                           'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
                           'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge',
                           'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge',
                           'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount',
                           'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
                           'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
                           'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
                           'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
                           'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
                           'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
                           'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3',
                           'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1',
                           'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
                           'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1',
                           'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6',
                           'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert',
                           'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
                           'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O',
                           'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                           'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',
                           'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene',
                           'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide',
                           'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine',
                           'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                           'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
                           'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole',
                           'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
                           'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
                           'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
                           'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']

    mol_descriptor_calculator = MolecularDescriptorCalculator(descriptor_list)


    mol = Chem.MolFromSmiles(smiles_string)
    descriptor_vals = mol_descriptor_calculator.CalcDescriptors(mol)

    return np.array(descriptor_vals)



def circular_fps_from_smiles(smiles_string,
                             radius = 2,
                             bitstring_length = 2**10,
                             use_features = False,
                             use_chirality = False):

    """Extended-connectivity fingerprints (ECFP) and functional-connectivity fingerprints (FCFP)."""

    molecule = Chem.MolFromSmiles(smiles_string)
    feature_list = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule,
                                                                       radius = radius,
                                                                       nBits = bitstring_length,
                                                                       useFeatures = use_features,
                                                                       useChirality = use_chirality)

    return np.array(feature_list)
















# conversion of smiles strings into molecular graphs

# functions to convert smiles string to molecular graph arrays (node feature matrix, adjacency matrix, edge feature tensor)

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding


def get_atom_features(atom, use_chirality = True, hydrogens_implicit = True):

    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I',
                                'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn',
                                'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry = True):

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE,
                                    Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE,
                                    Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def create_molecular_graph_arrays_from_smiles(smiles):
    """
    Input: SMILES-string of Molecule

    Output: a tuple of three numpy arrays (x, a, e)

    x is the atom-feature-matrix of the molecular graph with dimensions (n_nodes, n_node_features).
    a is the adjacency matrix of the molecular graph, only contains 0s and 1s, and has the dimensions (n_nodes, n_nodes).
    e is the edge-feature-tensor of the molecular graph and has dimensions (n_nodes, n_nodes, n_edge_features).

    """

    # convert smiles to rdkit mol object
    mol = Chem.MolFromSmiles(smiles)

    # get dimensions
    n_nodes = mol.GetNumAtoms()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

    # construct x
    x = np.zeros((n_nodes, n_node_features))

    for atom in mol.GetAtoms():
        row_index = atom.GetIdx()
        x[row_index,:] = get_atom_features(atom)

    # construct a
    a = GetAdjacencyMatrix(mol)
    assert a.shape == (n_nodes, n_nodes)

    # construct e
    e = np.zeros((n_nodes, n_nodes, n_edge_features))
    (rows, cols) = np.nonzero(a)

    for (i,j) in zip(rows, cols):
        e[i,j,:] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))

    return (x, a, e)


def tensorise_smiles_for_batch_mode(x_smiles, n_nodes_max = None):

    # convert smiles to molecular graph arrays x,a,e which can still have different dimensions n_nodes
    x_mol_graph_arrays = []

    for (k, smiles) in enumerate(x_smiles):

        (x, a, e) = create_molecular_graph_arrays_from_smiles(smiles)
        x_mol_graph_arrays.append((x, a, e))

    # define dimensional parameters
    n_batch = len(x_smiles)
    n_node_features = x_mol_graph_arrays[0][0].shape[1]
    n_edge_features = x_mol_graph_arrays[0][2].shape[2]

    if n_nodes_max == None:
        n_nodes_max = max([x.shape[0] for (x, a, e) in x_mol_graph_arrays])

    # perform zero padding n_nodes -> n_nodes_max for batch mode
    X_mol_graphs = np.zeros((n_batch, n_nodes_max, n_node_features))
    A_mol_graphs = np.zeros((n_batch, n_nodes_max, n_nodes_max))
    E_mol_graphs = np.zeros((n_batch, n_nodes_max, n_nodes_max, n_edge_features))

    for (k, (x, a, e)) in enumerate(x_mol_graph_arrays):

        n_nodes = x.shape[0]

        X_mol_graphs[k, 0:n_nodes, 0:n_node_features] = x
        A_mol_graphs[k, 0:n_nodes, 0:n_nodes] = a
        E_mol_graphs[k, 0:n_nodes, 0:n_nodes, 0:n_edge_features] = e

    return (X_mol_graphs, A_mol_graphs, E_mol_graphs)


def tensorise_mmps_for_batch_mode(X_smiles_mmps, n_nodes_max = None):

    x_smiles_1 = X_smiles_mmps[:,0]
    x_smiles_2 = X_smiles_mmps[:,1]

    all_smiles = set(list(x_smiles_1) + list(x_smiles_2))

    if n_nodes_max == None:
        n_nodes_max = max([Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in all_smiles])

    (X_mol_graphs_1, A_mol_graphs_1, E_mol_graphs_1) = tensorise_smiles_for_batch_mode(x_smiles_1, n_nodes_max = n_nodes_max)
    (X_mol_graphs_2, A_mol_graphs_2, E_mol_graphs_2) = tensorise_smiles_for_batch_mode(x_smiles_2, n_nodes_max = n_nodes_max)

    return (X_mol_graphs_1, A_mol_graphs_1, E_mol_graphs_1, X_mol_graphs_2, A_mol_graphs_2, E_mol_graphs_2)



def create_pytorch_geometric_data_set_from_smiles_and_targets(x_smiles, y):
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        
        # convert smiles to rdkit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct target tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
        
        # construct geometric data object and append to data list
        data_list.append(GeometricData(x = X, edge_index = E, edge_attr = EF, y = y_tensor))

    return data_list
















# auxiliary functions for mmp analysis

def extract_matched_molecular_series_of_each_core(X_smiles_mmps, x_smiles_mmp_cores):

    x_smiles_mms_dict = defaultdict(list)
    distinct_mmp_cores = list(set(x_smiles_mmp_cores))

    for (k, [smiles_1, smiles_2]) in enumerate(X_smiles_mmps):

        x_smiles_mms_dict[x_smiles_mmp_cores[k]].extend([smiles_1, smiles_2])

    for mmp_core in distinct_mmp_cores:
        x_smiles_mms_dict[mmp_core] = list(set(x_smiles_mms_dict[mmp_core]))

    return x_smiles_mms_dict


def extract_matched_molecular_pairs_of_each_molecule(X_smiles_mmps):

    x_smiles_mmp_dict = defaultdict(list)
    list_of_all_smiles = list(set(np.ndarray.flatten(X_smiles_mmps)))

    for [smiles_1, smiles_2] in X_smiles_mmps:

        x_smiles_mmp_dict[smiles_1].append([smiles_1, smiles_2])
        x_smiles_mmp_dict[smiles_2].append([smiles_1, smiles_2])

    return x_smiles_mmp_dict












# task-independent mmp featurisations (which only depend on a pair of SMILES strings of two molecules)

def circular_fp_abs_diff_from_mmps(mmp,
                                   radius = 2,
                                   bitstring_length = 2**10,
                                   use_features = False,
                                   use_chirality = False):

    """
    Extended-connectivity fingerprints (ECFP) and functional-connectivity fingerprints (FCFP). We featurise
    a pair of molecules by computing the difference of two ECFPs/FCFPs and then taking the absolute value.
    """

    smiles_string_1 = mmp[0]
    smiles_string_2 = mmp[1]

    molecule_1 = Chem.MolFromSmiles(smiles_string_1)
    molecule_2 = Chem.MolFromSmiles(smiles_string_2)

    feature_list_1 = np.array(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule_1,
                                                                         radius = radius,
                                                                         nBits = bitstring_length,
                                                                         useFeatures = use_features,
                                                                         useChirality = use_chirality))

    feature_list_2 = np.array(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule_2,
                                                                         radius = radius,
                                                                         nBits = bitstring_length,
                                                                         useFeatures = use_features,
                                                                         useChirality = use_chirality))

    circular_fp_abs_diff =  np.abs(feature_list_1 - feature_list_2)

    return circular_fp_abs_diff


def circular_fp_abs_diff_from_mmps_array(X_smiles_mmps,
                                         radius = 2,
                                         bitstring_length = 2**10,
                                         use_features = False,
                                         use_chirality = False):
    """
    Extended-connectivity fingerprints (ECFP) and functional-connectivity fingerprints (FCFP). We featurise
    an array consisting of pairs of molecules by computing the difference of two ECFPs/FCFPs and
    then taking the absolute value.
    """

    dict_smiles_to_fps = {}
    all_unique_smiles = set(X_smiles_mmps.flatten())

    n_mmps = len(X_smiles_mmps)
    X_fcfp_abs_diff = np.zeros((n_mmps, bitstring_length))

    for smiles in all_unique_smiles:

        mol = Chem.MolFromSmiles(smiles)
        dict_smiles_to_fps[smiles] = np.array(Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                                                                  radius = radius,
                                                                                                  nBits = bitstring_length,
                                                                                                  useFeatures = use_features,
                                                                                                  useChirality = use_chirality))

    for k in range(n_mmps):

        smiles_1 = X_smiles_mmps[k][0]
        smiles_2 = X_smiles_mmps[k][1]

        fp_1 = dict_smiles_to_fps[smiles_1]
        fp_2 = dict_smiles_to_fps[smiles_2]

        X_fcfp_abs_diff[k,:] = np.absolute(fp_1 - fp_2)

    return X_fcfp_abs_diff


def rdkit_mol_descriptors_abs_diff_from_mmps_array(X_smiles_mmps, descriptor_list = None):
    """
    RDKit descriptor vectors. We featurise an array consisting of pairs of molecules by computing
    the difference of two RDKit descriptor vectors and then taking the absolute value.
    """

    dict_smiles_to_rdkit_vectors = {}
    all_unique_smiles = set(X_smiles_mmps.flatten())
    n_mmps = len(X_smiles_mmps)

    if descriptor_list == None:
        n_features = 200
    else:
        n_features = len(descriptor_list)

    X_rdkit_abs_diff = np.zeros((n_mmps, n_features))

    for smiles in all_unique_smiles:

        dict_smiles_to_rdkit_vectors[smiles] = rdkit_mol_descriptors_from_smiles(smiles,
                                                                                 descriptor_list = descriptor_list)

    for k in range(n_mmps):

        smiles_1 = X_smiles_mmps[k][0]
        smiles_2 = X_smiles_mmps[k][1]

        rdkit_vector_1 = dict_smiles_to_rdkit_vectors[smiles_1]
        rdkit_vector_2 = dict_smiles_to_rdkit_vectors[smiles_2]

        X_rdkit_abs_diff[k,:] = np.absolute(rdkit_vector_1 - rdkit_vector_2)

    return X_rdkit_abs_diff
