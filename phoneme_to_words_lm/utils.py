import numpy as np
import re
import os
import pickle
from pathlib import Path


# HuggingFace cache directory — used for both LLM model downloads and LM file downloads.
HF_CACHE_DIR = '~/brand/huggingface'

# Constants
PHONE_DEF_SIL = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', 'SIL'
]

PHONE_DEF_SIL_BLANK = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    'SIL', 'BLANK',
]

SIL_DEF = ['SIL']

LOGIT_PHONE_DEF = [
    'BLANK', 'SIL',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

# Load cmu_dict from package data
_CMU_DICT_PATH = Path(__file__).parent / 'cmu_dict.pkl'
with open(_CMU_DICT_PATH, 'rb') as f:
    cmu_dict = pickle.load(f)


# Convert text to phonemes
def phonemize_sentence(
        thisTranscription,
        phone_def = PHONE_DEF_SIL,
        sil_def = SIL_DEF,
        diphone_def = None,
        maxSeqLen = 500,
        g2p = None,
        correct_phonemes = True,
        print_corrected = False,
        return_seq = False,
        verbosity = True
        ):

    from g2p_en import G2p
    if g2p is None:
        g2p = G2p()

    # Initialize variables
    seqClassIDs = np.zeros([maxSeqLen]).astype(np.int32)

    # Remove punctuation
    thisTranscription = remove_punctuation(thisTranscription)

    # if the last character of any word is ' remove it
    words = thisTranscription.split()
    words = [w[:-1] if w.endswith("'") else w for w in words]
    thisTranscription = ' '.join(words)

    # Change 'a' to 'ay' if we are in spelling mode to correct phonemization
    if all(len(word) == 1 for word in thisTranscription.split()):
        thisTranscription = thisTranscription.replace('a','ay')

    # Convert to phonemes
    phonemes = []
    if len(thisTranscription) == 0:
        phonemes = sil_def
    else:
        if diphone_def is not None:
            #add one SIL symbol at the beginning so there's one at the beginning of each word
            phonemes.append('SIL')

        for p in g2p(thisTranscription):
            if p==' ':
                phonemes.append('SIL')

            p = re.sub(r'[0-9]', '', p)  # Remove stress
            if re.match(r'[A-Z]+', p):  # Only keep phonemes
                phonemes.append(p)

        # add a SIL to the end
        phonemes.append('SIL')

        # replace phoneme sequences for words where g2p_en differs from cmudict
        if correct_phonemes:
            phonemes_by_word = [p.strip() for p in ' '.join(phonemes[:-1]).split('SIL')]
            for w, p in zip(thisTranscription.split(), phonemes_by_word):
                if w in cmu_dict and p.split() not in cmu_dict[w]:
                    phonemes_by_word[phonemes_by_word.index(p)] = ' '.join(cmu_dict[w][0])
                    if verbosity:
                        print(f'Corrected phonemization of "{w}" from "{p}" to "{" ".join(cmu_dict[w][0])}"')

            # convert back to list of phonemes with 'SIL' between words
            phonemes = []
            for p in phonemes_by_word:
                phonemes += p.split(' ')
                phonemes.append('SIL')

        # convert to diphones
        if diphone_def is not None:
            diphones = []
            for i in range(len(phonemes)-1):
                diphones.append(phonemes[i] + '->' + phonemes[i])
                diphones.append(phonemes[i] + '->' + phonemes[i+1])
            phonemes = diphones

    # remove any empty phonemes
    phonemes = [p for p in phonemes if p != '']

    # remove duplicate phonemes
    phonemes = [phonemes[i] for i in range(len(phonemes)) if i == 0 or phonemes[i] != phonemes[i-1]]

    # return
    if not return_seq:
        return phonemes

    else:
        # Segment phonemes
        seqLen = len(phonemes)
        if diphone_def is not None:
            seqClassIDs[0:seqLen] = [diphone_def.index(p) + 1 for p in phonemes]
        else:
            seqClassIDs[0:seqLen] = [phone_def.index(p) + 1 for p in phonemes]

        return phonemes, seqClassIDs, seqLen


def remove_punctuation(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
    sentence = sentence.replace('- ', ' ').lower()
    sentence = sentence.replace('--', '').lower()
    sentence = sentence.replace(" '", "'").lower()

    sentence = sentence.strip()
    sentence = ' '.join([word for word in sentence.split() if word != ''])

    return sentence


_WORD_REPLACEMENTS = {
    'ok': 'okay',

    # colour/color
    'colour': 'color',
    'colours': 'colors',
    'coloured': 'colored',
    'colouring': 'coloring',

    # theatre/theater
    'theatre': 'theater',
    'theatres': 'theaters',

    # favourite/favorite
    'favourite': 'favorite',
    'favourites': 'favorites',

    # centre/center
    'centre': 'center',
    'centres': 'centers',

    # metre/meter
    'metre': 'meter',
    'metres': 'meters',

    # litre/liter
    'litre': 'liter',
    'litres': 'liters',

    # defence/defense
    'defence': 'defense',
    'defences': 'defenses',

    # offence/offense
    'offence': 'offense',
    'offences': 'offenses',

    # honour/honor
    'honour': 'honor',
    'honours': 'honors',
    'honoured': 'honored',
    'honouring': 'honoring',

    # favour/favor
    'favour': 'favor',
    'favours': 'favors',
    'favoured': 'favored',
    'favouring': 'favoring',

    # behaviour/behavior
    'behaviour': 'behavior',
    'behaviours': 'behaviors',

    # neighbour/neighbor
    'neighbour': 'neighbor',
    'neighbours': 'neighbors',
    'neighbouring': 'neighboring',

    # labour/labor
    'labour': 'labor',
    'labours': 'labors',
    'laboured': 'labored',
    'labouring': 'laboring',

    # humour/humor
    'humour': 'humor',
    'humours': 'humors',
    'humoured': 'humored',
    'humouring': 'humoring',

    # catalogue/catalog
    'catalogue': 'catalog',
    'catalogues': 'catalogs',
    'catalogued': 'cataloged',
    'cataloguing': 'cataloging',

    # grey/gray
    'grey': 'gray',
    'greys': 'grays',

    # recognise/recognize
    'recognise': 'recognize',
    'recognises': 'recognizes',
    'recognised': 'recognized',
    'recognising': 'recognizing',

    # realise/realize
    'realise': 'realize',
    'realises': 'realizes',
    'realised': 'realized',
    'realising': 'realizing',

    # organise/organize
    'organise': 'organize',
    'organises': 'organizes',
    'organised': 'organized',
    'organising': 'organizing',

    # apologise/apologize
    'apologise': 'apologize',
    'apologises': 'apologizes',
    'apologised': 'apologized',
    'apologising': 'apologizing',

    # analyse/analyze
    'analyse': 'analyze',
    'analyses': 'analyzes',
    'analysed': 'analyzed',
    'analysing': 'analyzing',

    # licence/license
    'licence': 'license',
    'licences': 'licenses',
    'licensed': 'licensed',
    'licencing': 'licensing',

    # traveller/traveler, travelling/traveling
    'traveller': 'traveler',
    'travellers': 'travelers',
    'travelling': 'traveling',

    # contractions
    "lets": "let's",
    'dont': "don't",
    'cant': "can't",
    'wont': "won't",
    'doesnt': "doesn't",
    'didnt': "didn't",
    'isnt': "isn't",
    'wasnt': "wasn't",
    'thats': "that's",
    'whats': "what's",
    'im': "i'm",
    "masters": "master's",
    "masters'": "master's",

    # abbreviations
    'mr': 'mister',
    'mr.': 'mister',
    'mrs': 'misses',
    'mrs.': 'misses',
    'dr': 'doctor',
    'dr.': 'doctor',
}


def replace_words(sentence: str) -> str:
    """Normalize British spellings to American and expand common contractions.

    Operates on lowercase, punctuation-free text (apostrophes allowed).
    Uses a precomputed lookup dict for O(n) word-level replacement.

    Args:
        sentence: Lowercase space-separated word string.

    Returns:
        Sentence with normalized spellings.
    """
    words = sentence.split()
    get = _WORD_REPLACEMENTS.get
    return ' '.join(get(w, w) for w in words)
