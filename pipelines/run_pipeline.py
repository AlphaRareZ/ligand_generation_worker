import os
import random
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, rdMolDescriptors, Draw, rdmolfiles
from rdkit.Chem import BRICS, rdChemReactions
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Geometry import Point3D

from .download_service import download_file,alpha_fold_link


try:
    import sascorer
    has_sascorer = True
except Exception:
    has_sascorer = False

# Biopython for pocket reading
from Bio.PDB import PDBParser

random.seed(time.time_ns())
# -------------------------
# Utility
# -------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def safe_sanitize(mol):
    """
    Try to sanitize mol. If any sanitization step fails, return None.
    This protects from RingInfo / kekulize / valence exceptions.
    """
    if mol is None:
        return None
    try:
        m = Chem.Mol(mol)
        Chem.SanitizeMol(m)
        return m
    except Exception:

        try:
            m2 = Chem.Mol(mol)
            try:
                Chem.Kekulize(m2, clearAromaticFlags=True)
            except Exception:
                pass
            Chem.SanitizeMol(m2, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            Chem.SanitizeMol(m2)
            return m2
        except Exception:
            return None

# -------------------------
# Pocket utilities & hotspot detection (k-means)
# -------------------------
def load_pocket_coords(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pocket", pdb_file)
    coords = []
    for atom in structure.get_atoms():
        coords.append(atom.get_coord())
    coords = np.array(coords)
    center_vals = coords.mean(axis=0)
    center = np.array(center_vals)
    return coords, center

def pocket_hotspots(coords, n_clusters=4, iterations=30):
    if len(coords) == 0:
        return []
    n = min(n_clusters, len(coords))
    idx = np.random.choice(len(coords), n, replace=False)
    centers = coords[idx].astype(float)
    for it in range(iterations):
        dists = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = []
        for k in range(n):
            members = coords[labels == k]
            if len(members) == 0:
                new_centers.append(centers[k])
            else:
                new_centers.append(members.mean(axis=0))
        new_centers = np.array(new_centers)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < 1e-3:
            break
    return [centers[i] for i in range(len(centers))]

# -------------------------
# RDKit properties and proxies
# -------------------------
def compute_properties_rdkit(mol):
    props = {}
    try: props['SMILES'] = Chem.MolToSmiles(mol, isomericSmiles=True)
    except: props['SMILES'] = ""
    try: props['MolWt'] = Descriptors.MolWt(mol)
    except: props['MolWt'] = None
    try: props['LogP'] = Crippen.MolLogP(mol)
    except: props['LogP'] = None
    try: props['QED'] = QED.qed(mol)
    except: props['QED'] = None
    try: props['TPSA'] = rdMolDescriptors.CalcTPSA(mol)
    except: props['TPSA'] = None
    try: props['RotatableBonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    except: props['RotatableBonds'] = None
    if has_sascorer:
        try: props['SA'] = sascorer.calculateScore(mol)
        except: props['SA'] = None
    else:
        try:
            heavy = mol.GetNumHeavyAtoms()
            rings = Chem.GetSSSR(mol)
            props['SA'] = float(heavy)/max(1.0,float(rings)) if rings>0 else float(heavy)/1.0
        except: props['SA'] = None
    return props

def proxy_score(props):
    qed = float(props.get('QED') or 0.0)
    sa = float(props.get('SA') or 10.0)
    logp = float(props.get('LogP') or 0.0)
    mw = float(props.get('MolWt') or 300.0)
    rot = float(props.get('RotatableBonds') or 0.0)
    qed_n = qed
    sa_n = 1.0 - min(max(sa/10.0,0.0),1.0)
    logp_n = 1.0 - min(abs(logp-2.5)/5.0,1.0)
    mw_n = 1.0 - min(max((mw-200)/400.0,0.0),1.0)
    rot_n = 1.0 - min(rot/10.0,1.0)
    score = (
            0.35 * qed_n +
            0.20 * sa_n +
            0.15 * logp_n +
            0.10 * mw_n +
            0.05 * rot_n +
            0.15 * props.get("HingeScore", 0)
    )
    return score

# -------------------------
# Fingerprint / novelty utils
# -------------------------
# def mol_fp(mol, nBits=2048, radius=2):
#     try:
#         return GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
#     except Exception:
#         return None
def mol_fp(mol, nBits=2048, radius=2):
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=nBits
        )
        return generator.GetFingerprint(mol)
    except Exception:
        return None
def novelty_score(mol, reference_fps):
    fp = mol_fp(mol)
    if fp is None:
        return 0.0
    if not reference_fps:
        return 1.0
    sims = [TanimotoSimilarity(fp, r) for r in reference_fps]
    max_sim = max(sims) if len(sims)>0 else 0.0
    novelty = 1.0 - max_sim
    if max_sim > 0.75:
        novelty *= 0.05
    if max_sim > 0.90:
        return 0.0
    return novelty

# -------------------------
# Conformer strain energy (UFF) - returns energy-per-heavy-atom penalty (0..1, lower is better)
# -------------------------
def conformer_strain_penalty(mol):
    if mol is None:
        return 1.0
    try:
        m_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m_h, AllChem.ETKDG()) != 0:
            # embedding can return nonzero but still produce coords; proceed cautiously
            pass
        try:
            ff = AllChem.UFFGetMoleculeForceField(m_h)
        except Exception:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(m_h, AllChem.MMFFGetMoleculeProperties(m_h))
            except Exception:
                return 1.0
        ff.Minimize(maxIts=200)
        energy = ff.CalcEnergy()
        heavy = mol.GetNumHeavyAtoms() if mol.GetNumHeavyAtoms() > 0 else 1
        pen = min(1.0, float(abs(energy)) / (6.0 * heavy))
        return float(pen)
    except Exception:
        return 1.0

# -------------------------
# Pocket fit scoring (fraction inside + proximity to hotspots)
# -------------------------
def pocket_fit_score(mol, pocket_center, pocket_radius, hotspots):
    try:
        conf = mol.GetConformer()
    except Exception:
        return 0.0
    heavy_positions = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(a.GetIdx())
        heavy_positions.append(np.array([pos.x, pos.y, pos.z]))
    if len(heavy_positions) == 0:
        return 0.0
    heavy_positions = np.array(heavy_positions)
    d_to_center = np.linalg.norm(heavy_positions - pocket_center[None, :], axis=1)
    frac_inside = float(np.sum(d_to_center <= pocket_radius)) / len(d_to_center)
    if len(hotspots) > 0:
        d_to_hot = np.min(np.linalg.norm(heavy_positions[:, None, :] - np.array(hotspots)[None, :, :], axis=2), axis=1)
        mean_dhot = float(np.mean(d_to_hot))
        dhot_n = 1.0 - (mean_dhot / (mean_dhot + 4.0))
    else:
        dhot_n = 0.0
    return 0.6 * frac_inside + 0.4 * dhot_n

# -------------------------
# Fragment library
# -------------------------
AML_WEIGHTED_FRAGS = [
    # FLT3 hinge binders (critical)
    "n1ccc(N)nc1",
    "n1ccnc(N)c1",
    "c1ncnc(N)c1",
    "c1nc(N)nc(N)c1",
    "c1ncc(-C#N)nc1",

    # quizartinib / crenolanib hybrid cores
    "c1cc(-n2cnc3ccccc23)ccc1",

    # hydrophobic deep-pocket aromatic systems
    "c1ccc(C#N)cc1",
    "c1ccc(CF3)cc1",

    # IDH inhibitors
    "CC(C)(C)C(=O)Nc1ncccc1",
]

DEFAULT_FRAG_SMILES = [
    # --- Privileged heterocycles ---
    "c1ncccc1", "c1ccncc1", "c1ncccn1",              # pyridine class
    "c1cncnc1", "c1ncncc1",                          # pyrimidine/pyridazine
    "c1cnc2ncccc2c1",                                # purine-like
    "c1cc2ccccn2c1",                                 # quinoline
    "c1ccc2ncccc2c1",                                # isoquinoline
    "c1c[nH]c2ccccc12",                              # indole
    "c1cc2ccncc2cc1",                                # quinazoline

    # --- FLT3 / IDH relevant motifs ---
    "c1ccc(C#N)cc1",                                 # benzonitrile
    "c1cc(CF3)ccc1",                                 # CF3 aromatic
    "c1ccc(OCF3)cc1",
    "c1ccc(CN)cc1",                                  # benzylamine
    "c1cccc(CN)1",                                   # cyclic benzylamine
    "NCCc1ccccc1",                                   # amino-aryl linker
    "O=S(=O)N",                                      # sulfonamide core
    "O=S(=O)NCc1ccccc1",
    "c1cc(-n2nccc2)ccc1",                 # type II hinge binder
    "c1ccc(-c2n[nH]cc2)cc1",              # aza-heterocycle hinge binder
    "Nc1nccc(-c2ccccc2)n1",               # quizartinib-like
    "COc1ccc(-c2n[nH]c3ccccc23)cc1",      # crenolanib-like
    "CCC(C)(C)C(=O)N",
    "C1CCC(CC1)C(=O)N",
    "c1ccc(C(=O)N(C)C)cc1",

    # --- Drug-likeness privileged moieties ---
    "C1CCN(CC1)C",                                   # dimethyl piperazine
    "C1CCNCC1",                                      # piperazine
    "C1CCCN1",                                       # piperidine
    "C1COCC1",                                      # THF
    "C1CNCO1",                                       # morpholine

    # Halogens tune lipophilicity and binding pocket complementarity
    "c1cc(Cl)ccc1",
    "c1cc(Br)ccc1",
    "c1ccc(F)c(F)c1",
    "c1ccc(Cl)c(Cl)c1",
    "c1ccc(I)cc1",            # for deep hydrophobic pockets

    # --- Functional groups ---
    "CC(=O)N",                                      # acetamide
    "NC(=O)C",                                      # amide
    "NCCO",                                         # amino-alcohol
    "CCOC(=O)N",                                    # carbamate
    "CNC(=O)O",                                     # carbamate alt.
    # IDH1 inhibitor
    "CCCC(=O)Nc1ccc(CF3)cc1",            # enasidenib-like
    "CC(C)(C)C(=O)Nc1ncccc1",            # vorasidenib core

    # --- Aromatic drug fragments ---
    "c1ccc(Cl)cc1",                                 # chloro-phenyl (safe)
    "c1ccc(F)cc1",                                  # fluoro-phenyl
    "c1ccc(OC)cc1",                                 # methoxy-phenyl
    "c1ccc(O)cc1",                                  # hydroxy-phenyl
    "c1cc(N)ccc1",                                  # aniline
    "c1ccc(C(=O)N)cc1",
    "c1cc(C(=O)NC)ccc1",
    "c1nc(N)cnc1",
    "C1CCN(CC1)C(=O)",                     # piperazine urea
    #FLT3 inhibitors
    "n1ccc(N)nc1",            # FLT3 hinge core (quizartinib-like)
    "n1ccnc(N)c1",            # crenolanib hinge binder
    "c1ncnc(N)c1",            # midostaurin-like
    "c1nc(N)nc(N)c1",         # sorafenib-type hinge anchor
    "c1ncc(-C#N)nc1",         # nitrile hinge binder (potent for FLT3)
    "c1cc2ncccc2c1",          # isoquinoline
    "c1ccc2ncncc2c1",         # quinazoline
    "c1nccc2ccccc2n1",        # quinoline-like hinge
    "c1cc2ccncc2cc1",         # bicyclic hinge chemotype
    "c1ncnc2cccc2c1",         # purine-like fused rin

# Adaptive linkers
    "NCC(=O)N",               # urea linker
    "NCC(=O)O",               # carbamate linker
    "NC(C)(C)C(=O)",          # tert-butyl carbamate
    "CNC(=O)N",               # secondary urea
    "NCCS(=O)2",              # sulfonyl linker, IDH inhibitors
    "OCCNC",                  # solubilizing flexible linker

    # --- AML-specific privileged kernels (broken down fragments) ---
    "c1cc(-n2ccc(N)nc2)ccc1",                       # mini-FLT3 core
    "c1cc(-n2cnc3ccccc23)ccc1",                     # fused N-hetero
]

def is_druglike_fragment(m):
    if m is None:
        return False
    try:
        mw = Descriptors.MolWt(m)
        ha = m.GetNumHeavyAtoms()
        if ha < 6:
            return False
        if mw < 90 or mw > 350:
            return False
        smi = Chem.MolToSmiles(m)
        if smi in ["CCCBr", "CCCl", "Brc1ccccc1","Fc1ccccc1","c1ccccc1Cl"]:
            return False
        return True
    except:
        return False


def build_fragment_mols(smiles_list=None):
    if smiles_list is None:
        smiles_list = smiles_list + AML_WEIGHTED_FRAGS * 3
    frags = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        m = safe_sanitize(m)
        if not is_druglike_fragment(m):
            continue
        smi = Chem.MolToSmiles(m)
        if smi in ["c1ccccc1", "Clc1ccccc1", "Fc1ccccc1", "Ic1ccccc1"]:
            continue
        m = safe_sanitize(m)
        if m is None:
            continue
        try:
            m_h = Chem.AddHs(m)
            AllChem.EmbedMolecule(m_h, AllChem.ETKDG())
            AllChem.UFFOptimizeMolecule(m_h, maxIters=200)
            m3 = Chem.RemoveHs(m_h)
            m3 = safe_sanitize(m3)
            if m3 is not None and is_druglike_fragment(m3):
                frags.append(m3)
        except Exception:
            # fallback to sanitized heavy-atom fragment
            frags.append(m)
    # unique by canonical smiles
    unique = {}
    for f in frags:
        try:
            s = Chem.MolToSmiles(f, isomericSmiles=True)
            if s not in unique:
                unique[s] = f
        except Exception:
            continue
    return list(unique.values())

# -------------------------
# 3D rotation-aware alignment (Rodrigues rotation)
# -------------------------
def rodrigues_rotation_matrix(axis, theta):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    a = math.cos(theta)
    b = math.sin(theta)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = a * np.eye(3) + b * K + (1 - a) * np.outer(axis, axis)
    return R

def translate_mol(mol, vec):
    new = Chem.Mol(mol)
    try:
        conf_new = new.GetConformer()
    except Exception:
        return new
    for i in range(new.GetNumAtoms()):
        p = conf_new.GetAtomPosition(i)
        conf_new.SetAtomPosition(i, (p.x + float(vec[0]), p.y + float(vec[1]), p.z + float(vec[2])))
    return new

def align_fragment_to_vector(mol, target_point):
    # safe: if fragment has no conformer, try to embed
    if mol is None:
        return None
    m = Chem.Mol(mol)
    conf = None
    try:
        conf = m.GetConformer()
    except Exception:
        try:
            m_h = Chem.AddHs(m)
            AllChem.EmbedMolecule(m_h, AllChem.ETKDG())
            m = Chem.RemoveHs(m_h)
            conf = m.GetConformer()
        except Exception:
            return m
    heavy_idx = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy_idx) == 0:
        return m
    coords = np.array([list(conf.GetAtomPosition(i)) for i in heavy_idx])
    com = coords.mean(axis=0)
    # find two farthest heavy atoms to define axis
    maxd = -1.0
    pair = (heavy_idx[0], heavy_idx[0])
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            d = np.linalg.norm(coords[i] - coords[j])
            if d > maxd:
                maxd = d
                pair = (heavy_idx[i], heavy_idx[j])
    p1 = np.array(conf.GetAtomPosition(pair[0]))
    p2 = np.array(conf.GetAtomPosition(pair[1]))
    frag_vec = p2 - p1
    if np.linalg.norm(frag_vec) < 1e-6:
        # only translate
        return translate_mol(m, target_point - com)
    frag_vec_u = frag_vec / np.linalg.norm(frag_vec)
    targ_vec = np.array(target_point) - com
    if np.linalg.norm(targ_vec) < 1e-6:
        return translate_mol(m, target_point - com)
    targ_vec_u = targ_vec / np.linalg.norm(targ_vec)
    cross = np.cross(frag_vec_u, targ_vec_u)
    dot = np.clip(np.dot(frag_vec_u, targ_vec_u), -1.0, 1.0)
    angle = math.acos(dot)
    if np.linalg.norm(cross) < 1e-8 or abs(angle) < 1e-6:
        R = np.eye(3)
    else:
        axis = cross / (np.linalg.norm(cross) + 1e-12)
        R = rodrigues_rotation_matrix(axis, angle)
    # apply rotation around COM then translate to target_point
    new = Chem.Mol(m)
    conf_new = new.GetConformer()
    for a in range(new.GetNumAtoms()):
        p = np.array(conf_new.GetAtomPosition(a)) - com
        p_rot = R.dot(p)
        new_p = p_rot + target_point
        conf_new.SetAtomPosition(a, (float(new_p[0]), float(new_p[1]), float(new_p[2])))
    return new

# -------------------------
# Reaction SMARTS collection & application (reaction-first linking)
# -------------------------
REACTION_SMARTS = [
    # 1. Amide formation (core of FLT3 inhibitors)
    "[N:1].[C:2](=O)O[CH3]>>[N:1]-[C:2](=O)",

    # 2. Urea formation (very common in AML inhibitors)
    "[N:1].[N:2]C(=O)Cl>>[N:1]-C(=O)-[N:2]",

    # 3. Reductive amination (benzylamine + aldehyde)
    "[N:1].[C:2](=O)[H]>>[N:1]-[C:2]",

    # 4. Suzuki-like C–C couplings (aryl–aryl)
    "[c:1][B].[c:2]Br>>[c:1]-[c:2]",

    # 5. Sulfonamide formation (IDH inhibitors)
    "[N:1].[S:2](=O)(=O)Cl>>N-S(=O)(=O)-[C,N]",
]


REACTIONS = []
for sm in REACTION_SMARTS:
    try:
        REACTIONS.append(rdChemReactions.ReactionFromSmarts(sm))
    except Exception:
        pass

def try_reaction_linking(molA, molB):
    for a,b in [(molA,molB),(molB,molA)]:
        if a is None or b is None:
            continue
        for rxn in REACTIONS:
            try:
                prods = rxn.RunReactants((a,b))
            except Exception:
                continue
            for prod_tuple in prods:
                p = prod_tuple[0]
                p_s = safe_sanitize(p)
                if p_s is None:
                    continue
                try:
                    p_h = Chem.AddHs(p_s)
                    AllChem.EmbedMolecule(p_h, AllChem.ETKDG())
                    AllChem.UFFOptimizeMolecule(p_h, maxIters=200)
                    p_s = Chem.RemoveHs(p_h)
                    p_s = safe_sanitize(p_s)
                    if p_s is not None:
                        return p_s
                except Exception:
                    if p_s is not None:
                        return p_s
    return None

# -------------------------
# BRICS-aware crossover and safe merge fallback
# -------------------------
def brics_fragments_from_mol(mol):
    if mol is None:
        return []
    try:
        smi = Chem.MolToSmiles(mol)
        frags = BRICS.BRICSDecompose(smi)
    except Exception:
        return []
    frag_mols = []
    for fsmi in frags:
        try:
            fm = Chem.MolFromSmiles(fsmi)
            fm = safe_sanitize(fm)
            if fm is not None:
                frag_mols.append(fm)
        except Exception:
            continue
    return frag_mols

def safe_valence_allows_bond(atom):
    try:
        pt = Chem.GetPeriodicTable()
        max_val = pt.GetDefaultValence(atom.GetAtomicNum())
    except Exception:
        max_val = 8
    try:
        v = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])
    except Exception:
        v = atom.GetDegree()
    return v < max_val

def merge_two_mols_safe(molA, molB, max_link_dist=4.5):
    """
    Merge using CombineMols but check valence before adding bond.
    If sanitization fails, discard and return None.
    """
    if molA is None or molB is None:
        return None
    combo = Chem.CombineMols(molA, molB)
    # ensure conformer exists (try embed)
    try:
        combo_h = Chem.AddHs(combo)
        AllChem.EmbedMolecule(combo_h, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(combo_h, maxIters=150)
        combo = Chem.RemoveHs(combo_h)
    except Exception:
        # if embedding fails, still proceed (positions may be missing and conf ops will error)
        pass
    try:
        rw = Chem.RWMol(combo)
    except Exception:
        return None
    try:
        conf = rw.GetConformer()
    except Exception:
        # try embedding again robustly
        try:
            combo_h = Chem.AddHs(combo)
            if AllChem.EmbedMolecule(combo_h, AllChem.ETKDG()) == 0:
                AllChem.UFFOptimizeMolecule(combo_h, maxIters=150)
                combo = Chem.RemoveHs(combo_h)
                rw = Chem.RWMol(combo)
                conf = rw.GetConformer()
            else:
                return None
        except Exception:
            return None
    na = molA.GetNumAtoms()
    nb = molB.GetNumAtoms()
    min_d = 1e9
    pair = None
    for i in range(na):
        ai = rw.GetAtomWithIdx(i)
        if ai.GetAtomicNum() == 1:
            continue
        pi = np.array(conf.GetAtomPosition(i))
        for j in range(na, na+nb):
            aj = rw.GetAtomWithIdx(j)
            if aj.GetAtomicNum() == 1:
                continue
            pj = np.array(conf.GetAtomPosition(j))
            d = np.linalg.norm(pi - pj)
            if d < min_d:
                min_d = d
                pair = (i, j)
    if pair is None or min_d > max_link_dist:
        return None
    at1 = rw.GetAtomWithIdx(pair[0])
    at2 = rw.GetAtomWithIdx(pair[1])
    if not (safe_valence_allows_bond(at1) and safe_valence_allows_bond(at2)):
        return None
    try:
        rw.AddBond(pair[0], pair[1], order=Chem.rdchem.BondType.SINGLE)
    except Exception:
        return None
    mol = rw.GetMol()
    mol = safe_sanitize(mol)
    if mol is None:
        return None
    try:
        m_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m_h, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(m_h, maxIters=200)
        mol = Chem.RemoveHs(m_h)
        mol = safe_sanitize(mol)
    except Exception:
        mol = safe_sanitize(mol)
    return mol

def brics_crossover_safe(molA, molB):
    fa = brics_fragments_from_mol(molA)
    fb = brics_fragments_from_mol(molB)
    if not fa or not fb:
        return None
    a_choice = random.choice(fa)
    b_choice = random.choice(fb)
    prod = try_reaction_linking(a_choice, b_choice)
    if prod is not None:
        return prod
    merged = merge_two_mols_safe(a_choice, b_choice)
    return merged

# -------------------------
# Evolutionary search class (safe)
# -------------------------
class EvoDesigner:
    def __init__(self, fragments, pocket_center, pocket_radius, hotspots,
                 reference_fps=None, pop_size=60, generations=40, seed=42,
                 strain_weight=0.2, novelty_weight=0.15, pocket_weight=0.35, proxy_weight=0.45):
        self.fragments = fragments
        self.pocket_center = pocket_center
        self.pocket_radius = pocket_radius
        self.hotspots = hotspots
        self.reference_fps = reference_fps or []
        self.pop_size = pop_size
        self.generations = generations
        self.seed = seed
        set_seed(seed)
        self.strain_weight = strain_weight
        self.novelty_weight = novelty_weight
        self.pocket_weight = pocket_weight
        self.proxy_weight = proxy_weight
    history = {
        "generation": [],
        "proxy": [],
        "pocket_fit": [],
        "novelty": [],
        "strain": [],
        "sa": [],
        "qed": [],
        "logp": [],
        "mw": [],
        "fp": [],
        "smiles": [],
        "fragment_counts": [],
    }

    def _clean_and_unique(self, mols):
        """Sanitize, drop None, compute canonical smiles, keep unique molecules."""
        cleaned = []
        seen = set()
        for m in mols:
            if m is None:
                continue
            m_s = safe_sanitize(m)
            if m_s is None:
                continue
            try:
                smi = Chem.MolToSmiles(m_s, isomericSmiles=True)
            except Exception:
                continue
            if smi in seen:
                continue
            seen.add(smi)
            cleaned.append(m_s)
        return cleaned

    def initialize_population(self):
        pop = []
        # seed with sanitized unique fragments
        frag_samples = list(self.fragments)
        random.shuffle(frag_samples)
        for f in frag_samples:
            f_copy = Chem.Mol(f)
            f_s = safe_sanitize(f_copy)
            if f_s is None:
                continue
            # embed if missing conformer
            try:
                _ = f_s.GetConformer()
            except Exception:
                try:
                    fh = Chem.AddHs(f_s)
                    AllChem.EmbedMolecule(fh, AllChem.ETKDG())
                    AllChem.UFFOptimizeMolecule(fh, maxIters=100)
                    f_s = Chem.RemoveHs(fh)
                    f_s = safe_sanitize(f_s)
                except Exception:
                    pass
            pop.append(f_s)
            if len(pop) >= max(2, self.pop_size // 3):
                break
        # create merges anchored to hotspots until population is filled
        attempts = 0
        while len(pop) < self.pop_size and attempts < self.pop_size * 10:
            attempts += 1
            a, b = random.sample(self.fragments, 2)
            ha = random.choice(self.hotspots) if len(self.hotspots) > 0 else self.pocket_center
            hb = random.choice(self.hotspots) if len(self.hotspots) > 0 else (self.pocket_center + np.array([1.0,1.0,1.0]))
            try:
                a_al = align_fragment_to_vector(a, ha)
                b_al = align_fragment_to_vector(b, hb)
            except Exception:
                a_al, b_al = a, b
            prod = try_reaction_linking(a_al, b_al)
            if prod is None:
                prod = brics_crossover_safe(a_al, b_al)
            if prod is None:
                prod = merge_two_mols_safe(a_al, b_al)
            if prod is not None:
                pop.append(prod)
        # clean and unique
        pop = self._clean_and_unique(pop)
        # if still short, fill with random fragments repeated
        while len(pop) < self.pop_size:
            f = random.choice(self.fragments)
            f_s = safe_sanitize(f)
            if f_s:
                pop.append(f_s)
            else:
                pop.append(f)
        return pop[:self.pop_size]

    def fitness(self, mol):
        if mol is None:
            return -100.0, {}
        m = safe_sanitize(mol)
        if m is None:
            return -100.0, {}
        try:
            props = compute_properties_rdkit(m)
        except Exception:
            return -100.0, {}
        p_score = proxy_score(props)
        pf_score = pocket_fit_score(m, self.pocket_center, self.pocket_radius, self.hotspots)
        nov = novelty_score(m, self.reference_fps)
        strain = conformer_strain_penalty(m)
        sa = props.get('SA') or 10.0
        sa_pen = max(0.0, (sa - 6.0) / 10.0)
        comp_proxy = p_score
        comp_pocket = pf_score
        comp_nov = nov
        comp_strain = 1.0 - min(1.0, strain)
        final = (self.proxy_weight * comp_proxy +
                 self.pocket_weight * comp_pocket +
                 self.novelty_weight * comp_nov +
                 self.strain_weight * comp_strain)
        final = final - 0.05 * sa_pen
        meta = {'proxy': comp_proxy, 'pocket_fit': comp_pocket, 'novelty': comp_nov, 'strain': strain, 'sa': sa}
        return float(final), meta

    def mutate(self, mol):
        if mol is None:
            return None
        op = random.random()
        # 1) fragment injection / reaction linking
        if op < 0.4:
            frag = random.choice(self.fragments)
            hotspot = random.choice(self.hotspots) if self.hotspots else self.pocket_center
            try:
                frag_al = align_fragment_to_vector(frag, hotspot)
            except Exception:
                frag_al = frag
            prod = try_reaction_linking(mol, frag_al)
            if prod is not None:
                return prod
            prod = brics_crossover_safe(mol, frag_al)
            if prod is not None:
                return prod
            merged = merge_two_mols_safe(mol, frag_al)
            return merged if merged is not None else mol
        # 2) conformer tweak / minimization
        elif op < 0.75:
            try:
                m_h = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(m_h, AllChem.ETKDG()) == 0:
                    AllChem.UFFOptimizeMolecule(m_h, maxIters=150)
                m2 = Chem.RemoveHs(m_h)
                m2 = safe_sanitize(m2)
                return m2 if m2 is not None else mol
            except Exception:
                return mol
        # 3) scaffold hop via BRICS fragment replacement
        else:
            frags = brics_fragments_from_mol(mol)
            if not frags:
                return mol
            replace = random.choice(frags)
            donor = random.choice(self.fragments)
            prod = try_reaction_linking(replace, donor)
            if prod:
                # attempt to merge product back with parent safely
                merged = merge_two_mols_safe(mol, prod)
                return merged if merged is not None else mol
            else:
                return mol

    def crossover(self, molA, molB):
        if molA is None or molB is None:
            return None
        prod = brics_crossover_safe(molA, molB)
        if prod is not None:
            return prod
        prod = try_reaction_linking(molA, molB)
        if prod is not None:
            return prod
        # fallback: mutate a parent
        return self.mutate(random.choice([molA, molB]))

    def select_population(self, population, scores, k):
        n_elite = max(1, int(0.05 * len(population)))
        idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        new_pop = [population[i] for i in idx_sorted[:n_elite]]
        attempts = 0
        while len(new_pop) < k and attempts < k * 10:
            attempts += 1
            a, b = random.sample(range(len(population)), 2)
            winner = population[a] if scores[a] > scores[b] else population[b]
            fp_w = mol_fp(winner)
            if fp_w is None or not self.reference_fps:
                new_pop.append(winner)
            else:
                sims = [TanimotoSimilarity(fp_w, r) for r in self.reference_fps]
                if max(sims) < 0.95 or random.random() < 0.1:
                    new_pop.append(winner)
                else:
                    # inject a random fragment to preserve novelty
                    new_pop.append(random.choice(self.fragments))
        # ensure size
        if len(new_pop) < k:
            for _ in range(k - len(new_pop)):
                new_pop.append(random.choice(self.fragments))
        # sanitize/unique
        new_pop = self._clean_and_unique(new_pop)
        # pad if needed
        while len(new_pop) < k:
            new_pop.append(random.choice(self.fragments))
        return new_pop[:k]

    def save_history_csv(self, path="run_history.csv"):
        """Save evolutionary history to CSV."""
        df = pd.DataFrame(self.history)

        # Convert fingerprints (lists) to strings for CSV
        if "fp" in df.columns:
            df["fp"] = df["fp"].apply(
                lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else x
            )

        df.to_csv(path, index=False)
        print(f"[Evo] History saved to {path}")
    def run(self):
        pop = self.initialize_population()
        pop = pop[:self.pop_size]

        best_record = None

        # reset history each run
        self.history = {
            "generation": [],
            "proxy": [],
            "pocket_fit": [],
            "novelty": [],
            "strain": [],
            "sa": [],
            "qed": [],
            "logp": [],
            "mw": [],
            "fp": [],
            "smiles": [],
            "fragment_counts": [],
        }

        for gen in range(self.generations):
            # ensure all population members are sanitized & unique before evaluation
            pop = self._clean_and_unique(pop)

            scores = []
            metas = []

            # ==============================
            #   Evaluate population
            # ==============================
            for ind in pop:
                sc, meta = self.fitness(ind)
                scores.append(sc)
                metas.append(meta)

                # ----- LOG HISTORY FOR EACH INDIVIDUAL -----
                try:
                    smi = Chem.MolToSmiles(ind)
                except:
                    smi = None

                fp = mol_fp(ind)
                props = compute_properties_rdkit(ind)

                self.history["generation"].append(gen)
                self.history["proxy"].append(meta.get("proxy"))
                self.history["pocket_fit"].append(meta.get("pocket_fit"))
                self.history["novelty"].append(meta.get("novelty"))
                self.history["strain"].append(meta.get("strain"))
                self.history["sa"].append(meta.get("sa"))
                self.history["qed"].append(props.get("QED"))
                self.history["logp"].append(props.get("LogP"))
                self.history["mw"].append(props.get("MW"))
                self.history["fp"].append(fp)
                self.history["smiles"].append(smi)
                self.history["fragment_counts"].append(len(brics_fragments_from_mol(ind)))
                # --------------------------------------------

            # If all molecules collapsed → restart half of them
            if all([s <= -90 for s in scores]):
                print("[Evo] All candidates invalid at gen", gen, "- reinitializing some individuals")
                for i in range(len(pop)//2):
                    pop[i] = random.choice(self.fragments)
                continue

            # best molecule
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]

            if best_record is None or best_score > best_record['score']:
                best_record = {
                    'score': best_score,
                    'mol': pop[best_idx],
                    'meta': metas[best_idx],
                    'gen': gen
                }

            if gen % max(1, self.generations // 10) == 0:
                print(f"[Evo] Gen {gen}/{self.generations} best={best_score:.4f} mean={np.mean(scores):.4f}")

            # ==============================
            #  Selection
            # ==============================
            new_pop = self.select_population(pop, scores, self.pop_size)

            # ==============================
            #  Reproduction
            # ==============================
            children = []
            while len(children) < self.pop_size - len(new_pop):
                if random.random() < 0.6:
                    a, b = random.sample(new_pop, 2)
                    child = self.crossover(a, b)
                else:
                    parent = random.choice(new_pop)
                    child = self.mutate(parent)

                child_s = safe_sanitize(child)
                if child_s is None:
                    child_s = random.choice(self.fragments)
                children.append(child_s)

            pop = new_pop + children

            # novelty injection
            if gen % 3 == 0:
                pop[random.randint(0, len(pop)-1)] = random.choice(self.fragments)

        # Final scoring phase
        pop = self._clean_and_unique(pop)
        final_scores = []
        final_metas = []
        for ind in pop:
            sc, meta = self.fitness(ind)
            final_scores.append(sc)
            final_metas.append(meta)

        idx_sorted = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        final_sorted = [(pop[i], final_scores[i], final_metas[i]) for i in idx_sorted]

        self.save_history_csv("run_history.csv")

        return final_sorted, self.history, best_record

# -------------------------
# Top-10 visualizer
# -------------------------
def save_top10_grid(mols_with_meta, out_path, molsPerRow=5, subImgSize=(300,300)):
    topn = min(10, len(mols_with_meta))
    top = mols_with_meta[:topn]
    top_mols = [t[0] for t in top]
    legends = []
    for i, (m, score, meta) in enumerate(top):
        q = meta.get('proxy') if meta and 'proxy' in meta else 0.0
        pf = meta.get('pocket_fit') if meta and 'pocket_fit' in meta else 0.0
        nov = meta.get('novelty') if meta and 'novelty' in meta else 0.0
        legends.append(f"#{i+1} score={score:.3f}\nproxy={q:.2f} pf={pf:.2f} nov={nov:.2f}")
    img = Draw.MolsToGridImage(top_mols, molsPerRow=molsPerRow, subImgSize=subImgSize, legends=legends)
    img.save(out_path)
    print("[Visualizer] Saved Top-10 grid:", out_path)
    return out_path

# -------------------------
# Main pipeline
# -------------------------
def process_de_novo_advanced_safe(pocket_file, work_dir,
                             fragment_smiles=None,
                             reference_smiles_list=None,
                             pop_size=60, generations=40,
                             hotspot_clusters=4,
                             seed=42):
    work_dir = Path(work_dir)
    ensure_dir(work_dir)
    set_seed(seed)

    print("[Pipeline] Loading pocket coordinates ...")
    coords, center = load_pocket_coords(pocket_file)
    padding = 3.0
    radius = float(np.max(np.linalg.norm(coords - center[None, :], axis=1))) + padding

    print("[Pipeline] Computing pocket hotspots ...")
    hotspots = pocket_hotspots(coords, n_clusters=hotspot_clusters, iterations=30)
    print(f"[Pipeline] Detected {len(hotspots)} hotspots.")

    print("[Pipeline] Building fragment library ...")
    fragments = build_fragment_mols(fragment_smiles)

    print("[Pipeline] Building reference fingerprints for novelty (if provided) ...")
    ref_fps = []
    if reference_smiles_list:
        for s in reference_smiles_list:
            m = Chem.MolFromSmiles(s)
            if m:
                fp = mol_fp(m)
                if fp:
                    ref_fps.append(fp)

    print("[Pipeline] Initializing evolutionary designer ...")
    designer = EvoDesigner(fragments, center, radius, hotspots, reference_fps=ref_fps,
                           pop_size=pop_size, generations=generations, seed=seed)

    print("[Pipeline] Running evolutionary search ... (this can be slow)")
    start = time.time()
    final_sorted, history, best_record = designer.run()
    elapsed = time.time() - start
    if best_record is None:
        print("[Pipeline] Evolution produced no valid candidates.")
    else:
        print(f"[Pipeline] Evolution finished in {elapsed/60.0:.2f} min. Best score: {best_record['score']:.4f} at gen {best_record['gen']}")

    # outputs
    records = []
    sdf_dir = work_dir / "sdf"
    pdb_dir = work_dir / "pdb"
    ensure_dir(sdf_dir)
    ensure_dir(pdb_dir)

    for idx, (mol, score, meta) in enumerate(final_sorted, start=1):
        m_s = safe_sanitize(mol)
        if m_s is None:
            continue
        props = compute_properties_rdkit(m_s)
        rec = {
            "ligand_id": idx,
            "smiles": props.get('SMILES',''),
            "MolWt": props.get('MolWt'),
            "LogP": props.get('LogP'),
            "QED": props.get('QED'),
            "TPSA": props.get('TPSA'),
            "RotatableBonds": props.get('RotatableBonds'),
            "SA": props.get('SA'),
            "final_score": float(score),
            "proxy": meta.get('proxy') if meta else None,
            "pocket_fit": meta.get('pocket_fit') if meta else None,
            "novelty": meta.get('novelty') if meta else None,
            "strain": meta.get('strain') if meta else None
        }
        records.append(rec)
        if idx <= 200:
            try:
                sdf_path = sdf_dir / f"ligand_{idx}.sdf"
                w = Chem.SDWriter(str(sdf_path))
                w.write(m_s)
                w.close()
                pdb_path = pdb_dir / f"ligand_{idx}.pdb"
                rdmolfiles.MolToPDBFile(m_s, str(pdb_path))
            except Exception:
                pass

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    out_csv = work_dir / f"{Path(pocket_file).stem}_ranked_ligands_advanced_safe.csv"
    df.to_csv(out_csv, index=False)
    print("[Pipeline] Saved ranked CSV:", out_csv)

    top_mols_meta = [(final_sorted[i][0], final_sorted[i][1], final_sorted[i][2]) for i in range(min(10, len(final_sorted)))]
    out_img = work_dir / f"{Path(pocket_file).stem}_top10_grid_advanced_safe.png"
    if top_mols_meta:
        save_top10_grid(top_mols_meta, out_img)

    return out_csv, out_img, pdb_dir, df




# -------------------------
# Run example
# -------------------------
def main(pdb_path,pop_size = 40, generations = 20,seed = 42):
    pocket_file = pdb_path
    work_root = r"Output"
    ensure_dir(work_root)

    known_smiles = [] 

    csv_out, img_out, lig_dir, df = process_de_novo_advanced_safe(
        pocket_file,
        work_root,
        fragment_smiles=DEFAULT_FRAG_SMILES,
        reference_smiles_list=known_smiles,
        pop_size=pop_size,          # start smaller for testing 40
        generations=generations,       # start smaller for testing 20
        hotspot_clusters=4,
        seed=seed # default seed = 42
    )

    print("DONE.")
    print("CSV:", csv_out)
    print("Top-10 image:", img_out)
    print("PDB files saved in:", lig_dir)
    print("Pop Size",pop_size)
    print("Generations",generations)
# DELETE
if __name__ == "__main__":
    path = download_file(alpha_fold_link("A0A796"))
    main(pdb_path=path, pop_size=40, generations=20, seed=42)