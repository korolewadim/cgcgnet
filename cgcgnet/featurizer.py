import os
import uuid
import numpy as np

import torch
import dgl

from pathlib import Path
from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core.structure import Element
from matminer.utils.data import MatscholarElementData
from openbabel.pybel import readfile
from gensim.models import word2vec

# the word2vec model and accompanied code are taken from repository https://github.com/samoturk/mol2vec
GENSIM_MODEL = word2vec.Word2Vec.load(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models/mol2vec_300dim.pkl"
    )
)

Z_NODES_MOF = [
    z for z in range(1, 103) if Element.from_Z(z).is_metal or z in {32, 51, 84}
]
Z_NODES_COF = [
    z for z in range(1, 103) if Element.from_Z(z).is_metal or z in {5, 14, 32, 51, 84}
]


def add_charges(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    problems = Chem.DetectChemistryProblems(mol)
    if not problems:
        Chem.SanitizeMol(mol)
        return mol
    for problem in problems:
        if problem.GetType() == "AtomValenceException":
            atom = mol.GetAtomWithIdx(problem.GetAtomIdx())
            if (
                atom.GetAtomicNum() == 5
                and atom.GetFormalCharge() == 0
                and atom.GetExplicitValence() == 4
            ):
                atom.SetFormalCharge(1)
            if (
                atom.GetAtomicNum() == 7
                and atom.GetFormalCharge() == 0
                and atom.GetExplicitValence() == 4
            ):
                atom.SetFormalCharge(1)
    Chem.SanitizeMol(mol)
    return mol


def get_supra_graph(structure, feat_scheme):
    atomic_numbers = np.array(structure.atomic_numbers)
    if feat_scheme == "mof":
        idx_nodes = [i for i, z in enumerate(atomic_numbers) if z in Z_NODES_MOF]
    elif feat_scheme == "сof":
        idx_nodes = [i for i, z in enumerate(atomic_numbers) if z in Z_NODES_COF]
    else:
        raise ValueError("unknown featurization scheme")
    atomic_numbers_nodes = atomic_numbers[idx_nodes]

    Path("temp").mkdir(parents=True, exist_ok=True)
    reduced_structure = structure.copy()
    reduced_structure.remove_sites(idx_nodes)
    reduced_structure_path = os.path.join("temp", f"{uuid.uuid4()}.cif")
    reduced_structure.to(reduced_structure_path, "cif")

    *_, molecules = readfile("cif", reduced_structure_path, opt={"p": True})
    molecules_path = os.path.join("temp", f"{uuid.uuid4()}.txt")
    molecules.write(
        "can", molecules_path, overwrite=True, opt={"h": True, "i": True, "n": True}
    )
    smiles_ = set(Path(molecules_path).read_text().rstrip("\n").split("."))

    os.remove(reduced_structure_path)
    os.remove(molecules_path)

    smiles_linkers = []

    for smiles in smiles_:
        mol = add_charges(smiles)
        if mol is not None:
            mol = Chem.rdmolops.AddHs(mol, explicitOnly=True)
            smiles_linkers.append(Chem.MolToSmiles(mol))
    atomic_numbers_nodes = sorted(set(atomic_numbers_nodes))
    smiles_linkers = sorted(set(smiles_linkers))

    linker_src, node_dst = [], []

    for i in range(len(smiles_linkers)):
        for j in range(len(atomic_numbers_nodes)):
            linker_src.append(i)
            node_dst.append(j)
    return atomic_numbers_nodes, smiles_linkers, linker_src, node_dst


def mol2alt_sentence(mol, radius):
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element
    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])
    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def matscholar_featurize(atomic_numbers):
    el_data = MatscholarElementData()
    elements = list(map(Element.from_Z, atomic_numbers))
    all_attributes = []

    for attr in el_data.prop_names:
        elem_data = list(
            map(partial(el_data.get_elemental_property, property_name=attr), elements)
        )
        all_attributes.append(elem_data)
    return np.array(all_attributes).T


def mol2vec_featurize(smiles_linkers, model):
    def sentences2vec(sentences, model, unseen=None):
        keys = set(model.wv.key_to_index.keys())
        vec = []
        if unseen:
            unseen_vec = model.wv.get_vector(unseen)
        for sentence in sentences:
            if unseen:
                vec.append(
                    sum(
                        [
                            model.wv.get_vector(y)
                            if y in set(sentence) & keys
                            else unseen_vec
                            for y in sentence
                        ]
                    )
                )
            else:
                vec.append(
                    sum(
                        [
                            model.wv.get_vector(y)
                            for y in sentence
                            if y in set(sentence) & keys
                        ]
                    )
                )
        return np.array(vec)

    def featurize(model, mol):
        sentence = mol2alt_sentence(mol, 1)
        feature = sentences2vec([sentence], model, "UNK")[0]
        return feature

    features = [featurize(model, Chem.MolFromSmiles(smi)) for smi in smiles_linkers]
    return np.vstack(features)


def get_2cg_inputs(structure, feat_scheme="mof"):
    atomic_numbers_nodes, smiles_linkers, linker_src, node_dst = get_supra_graph(
        structure, feat_scheme
    )
    linker_features = torch.tensor(
        mol2vec_featurize(smiles_linkers, GENSIM_MODEL).tolist(), dtype=torch.float32
    )
    node_features = torch.tensor(
        matscholar_featurize(atomic_numbers_nodes), dtype=torch.float32
    )
    graph_data = {
        ("l", "l2n", "n"): (torch.tensor(linker_src), torch.tensor(node_dst)),
        ("n", "n2l", "l"): (torch.tensor(node_dst), torch.tensor(linker_src)),
    }
    g = dgl.heterograph(graph_data)
    g.nodes["l"].data["feat"] = linker_features
    g.nodes["n"].data["feat"] = node_features
    return g
