from collections import OrderedDict
from typing import Callable
from unittest import TestCase, mock

import pytest
from nltk import PunktSentenceTokenizer
from nltk.tokenize.api import TokenizerI

from services import nltk2json
from . import mocks


class TestNltk(TestCase):
    @mock.patch('settings.version')
    def test_process(self, version):
        version.return_value = '0.1'
        text = "Autonomous cars from the countryside of France shift insurance liability toward manufacturers. People are afraid that they will crash."
        actual = nltk2json.process(text, lang='en', dependencies='malt')
        expected = OrderedDict([('DC.conformsTo', 0.1), ('DC.source', 'NLTK 3.4'), ('DC.created', '2019-01-25T17:04:34'), ('DC.date', '2019-01-25T17:04:34'), ('DC.creator', ''), ('DC.publisher', ''), ('DC.title', ''), ('DC.description', ''), ('DC.identifier', ''), ('DC.language', 'en'), ('conll', {}), ('documents', [OrderedDict([('text', 'Autonomous cars from the countryside of France shift insurance liability toward manufacturers. People are afraid that they will crash.'), ('tokenList', [{'id': 1, 'text': 'Autonomous', 'lemma': 'Autonomous', 'stem': 'autonom', 'xpos': 'JJ', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'ADJ', 'synsets': [{'wordnetId': 'autonomous.s.01', 'definition': '(of political bodies) not controlled by outside forces', 'synonyms': ['independent', 'self-governing', 'sovereign'], 'examples': ['an autonomous judiciary', 'a sovereign state']}, {'wordnetId': 'autonomous.s.02', 'definition': 'existing as an independent entity', 'examples': ['the partitioning of India created two separate and autonomous jute economies']}, {'wordnetId': 'autonomous.s.03', 'definition': '(of persons) free from external control and constraint in e.g. action and judgment', 'synonyms': ['self-directed', 'self-reliant']}]}, {'id': 2, 'text': 'cars', 'lemma': 'car', 'stem': 'car', 'xpos': 'NNS', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'NOUN', 'synsets': [{'wordnetId': 'car.n.01', 'definition': 'a motor vehicle with four wheels; usually propelled by an internal combustion engine', 'synonyms': ['auto', 'automobile', 'machine', 'motorcar'], 'hypernyms': ['motor_vehicle', 'automotive_vehicle'], 'hyponyms': ['ambulance', 'beach_wagon', 'station_wagon', 'wagon', 'estate_car', 'beach_waggon', 'station_waggon', 'waggon', 'bus', 'jalopy', 'heap', 'cab', 'hack', 'taxi', 'taxicab', 'compact', 'compact_car', 'convertible', 'coupe', 'cruiser', 'police_cruiser', 'patrol_car', 'police_car', 'prowl_car', 'squad_car', 'electric', 'electric_automobile', 'electric_car', 'gas_guzzler', 'hardtop', 'hatchback', 'horseless_carriage', 'hot_rod', 'hot-rod', 'jeep', 'landrover', 'limousine', 'limo', 'loaner', 'minicar', 'minivan', 'Model_T', 'pace_car', 'racer', 'race_car', 'racing_car', 'roadster', 'runabout', 'two-seater', 'sedan', 'saloon', 'sport_utility', 'sport_utility_vehicle', 'S.U.V.', 'SUV', 'sports_car', 'sport_car', 'Stanley_Steamer', 'stock_car', 'subcompact', 'subcompact_car', 'touring_car', 'phaeton', 'tourer', 'used-car', 'secondhand_car'], 'examples': ['he needs a car to get to work']}, {'wordnetId': 'car.n.02', 'definition': 'a wheeled vehicle adapted to the rails of railroad', 'synonyms': ['railcar', 'railway_car', 'railroad_car'], 'hypernyms': ['wheeled_vehicle'], 'hyponyms': ['baggage_car', 'luggage_van', 'cabin_car', 'caboose', 'club_car', 'lounge_car', 'freight_car', "guard's_van", 'handcar', 'mail_car', 'passenger_car', 'coach', 'carriage', 'slip_coach', 'slip_carriage', 'tender', 'van'], 'examples': ['three cars had jumped the rails']}, {'wordnetId': 'car.n.03', 'definition': 'the compartment that is suspended from an airship and that carries personnel and the cargo and the power plant', 'synonyms': ['gondola'], 'hypernyms': ['compartment']}, {'wordnetId': 'car.n.04', 'definition': 'where passengers ride up and down', 'synonyms': ['elevator_car'], 'hypernyms': ['compartment'], 'examples': ['the car was on the top floor']}, {'wordnetId': 'cable_car.n.01', 'definition': 'a conveyance for passengers or freight on a cable railway', 'synonyms': ['car'], 'hypernyms': ['compartment'], 'examples': ['they took a cable car to the top of the mountain']}]}, {'id': 3, 'text': 'from', 'lemma': 'from', 'stem': 'from', 'xpos': 'IN', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'ADP'}, {'id': 4, 'text': 'the', 'lemma': 'the', 'stem': 'the', 'xpos': 'DT', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'DET'}, {'id': 5, 'text': 'countryside', 'lemma': 'countryside', 'stem': 'countrysid', 'xpos': 'NN', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'NOUN', 'synsets': [{'wordnetId': 'countryside.n.01', 'definition': 'rural regions', 'hypernyms': ['country', 'rural_area']}]}, {'id': 6, 'text': 'of', 'lemma': 'of', 'stem': 'of', 'xpos': 'IN', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'ADP'}, {'id': 7, 'text': 'France', 'lemma': 'France', 'stem': 'franc', 'xpos': 'NNP', 'entity_iob': 'B', 'features': {'Overt': 'Yes'}, 'upos': 'NOUN', 'entity': 'GPE', 'synsets': [{'wordnetId': 'france.n.01', 'definition': 'a republic in western Europe; the largest country wholly in Europe', 'synonyms': ['French_Republic']}, {'wordnetId': 'france.n.02', 'definition': 'French writer of sophisticated novels and short stories (1844-1924)', 'synonyms': ['Anatole_France', 'Jacques_Anatole_Francois_Thibault']}]}, {'id': 8, 'text': 'shift', 'lemma': 'shift', 'stem': 'shift', 'xpos': 'NN', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'NOUN', 'synsets': [{'wordnetId': 'shift.n.01', 'definition': 'an event in which something is displaced without rotation', 'synonyms': ['displacement'], 'hypernyms': ['translation'], 'hyponyms': ['amplitude', 'luxation']}, {'wordnetId': 'transformation.n.01', 'definition': 'a qualitative change', 'synonyms': ['transmutation', 'shift'], 'hypernyms': ['change', 'alteration', 'modification'], 'hyponyms': ['conversion', 'transition', 'changeover', 'degeneration', 'retrogression', 'improvement', 'betterment', 'advance', 'population_shift', 'pyrolysis', 'sea_change', 'strengthening', 'sublimation', 'tin_pest', 'tin_disease', 'tin_plague', 'weakening']}, {'wordnetId': 'shift.n.03', 'definition': 'the time period during which you are at work', 'synonyms': ['work_shift', 'duty_period'], 'hypernyms': ['hours'], 'hyponyms': ['day_shift', 'evening_shift', 'swing_shift', 'go', 'spell', 'tour', 'turn', 'night_shift', 'graveyard_shift', 'split_shift', 'trick', 'watch']}, {'wordnetId': 'switch.n.07', 'definition': 'the act of changing one thing or position for another', 'synonyms': ['switching', 'shift'], 'hypernyms': ['change'], 'hyponyms': ['switcheroo'], 'examples': ['his switch on abortion cost him the election']}, {'wordnetId': 'shift.n.05', 'definition': 'the act of moving from one place to another', 'synonyms': ['shifting'], 'hypernyms': ['motion', 'movement', 'move'], 'examples': ['his constant shifting disrupted the class']}, {'wordnetId': 'fault.n.04', 'definition': "(geology) a crack in the earth's crust resulting from the displacement of one side with respect to the other", 'synonyms': ['faulting', 'geological_fault', 'shift', 'fracture', 'break'], 'hypernyms': ['crack', 'cleft', 'crevice', 'fissure', 'scissure'], 'hyponyms': ['inclined_fault', 'strike-slip_fault'], 'examples': ['they built it right over a geological fault', "he studied the faulting of the earth's crust"]}, {'wordnetId': 'shift.n.07', 'definition': 'a crew of workers who work for a specific period of time', 'hypernyms': ['gang', 'crew', 'work_party'], 'hyponyms': ['day_shift', 'day_watch', 'evening_shift', 'night_shift', 'graveyard_shift', 'relay']}, {'wordnetId': 'shift_key.n.01', 'definition': 'the key on the typewriter keyboard that shifts from lower-case letters to upper-case letters', 'synonyms': ['shift'], 'hypernyms': ['key']}, {'wordnetId': 'chemise.n.01', 'definition': "a woman's sleeveless undergarment", 'synonyms': ['shimmy', 'shift', 'slip', 'teddy'], 'hypernyms': ['undergarment', 'unmentionable']}, {'wordnetId': 'chemise.n.02', 'definition': 'a loose-fitting dress hanging straight from the shoulders without a waist', 'synonyms': ['sack', 'shift'], 'hypernyms': ['dress', 'frock']}]}, {'id': 9, 'text': 'insurance', 'lemma': 'insurance', 'stem': 'insur', 'xpos': 'NN', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'NOUN', 'synsets': [{'wordnetId': 'insurance.n.01', 'definition': 'promise of reimbursement in the case of loss; paid to people or companies so concerned about hazards that they have made prepayments to an insurance company', 'hypernyms': ['security', 'protection'], 'hyponyms': ['assurance', 'automobile_insurance', 'car_insurance', 'business_interruption_insurance', 'coinsurance', 'fire_insurance', 'group_insurance', 'hazard_insurance', 'health_insurance', 'liability_insurance', 'life_insurance', 'life_assurance', 'malpractice_insurance', 'reinsurance', 'self-insurance', 'term_insurance']}, {'wordnetId': 'policy.n.03', 'definition': 'written contract or certificate of insurance', 'synonyms': ['insurance_policy', 'insurance'], 'hypernyms': ['contract'], 'hyponyms': ['floater', 'floating_policy'], 'examples': ['you should have read the small print on your policy']}, {'wordnetId': 'indemnity.n.01', 'definition': 'protection against future loss', 'synonyms': ['insurance'], 'hypernyms': ['protection', 'shelter']}]}, {'id': 10, 'text': 'liability', 'lemma': 'liability', 'stem': 'liabil', 'xpos': 'NN', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'NOUN', 'synsets': [{'wordnetId': 'liability.n.01', 'definition': 'the state of being legally obliged and responsible', 'hypernyms': ['susceptibility', 'susceptibleness'], 'hyponyms': ['ratability', 'rateability', 'taxability']}, {'wordnetId': 'indebtedness.n.01', 'definition': 'an obligation to pay money to another party', 'synonyms': ['liability', 'financial_obligation'], 'hypernyms': ['obligation'], 'hyponyms': ['account_payable', 'payable', 'arrears', 'debt', 'limited_liability', 'scot_and_lot']}, {'wordnetId': 'liability.n.03', 'definition': 'the quality of being something that holds you back', 'hypernyms': ['bad', 'badness'], 'hyponyms': ['disadvantage', 'weak_point'], 'antonyms': ['asset']}]}, {'id': 11, 'text': 'toward', 'lemma': 'toward', 'stem': 'toward', 'xpos': 'IN', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'ADP'}, {'id': 12, 'text': 'manufacturers.', 'lemma': 'manufacturers.', 'stem': 'manufacturers.', 'xpos': 'JJ', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'ADJ'}, {'id': 13, 'text': 'People', 'lemma': 'People', 'stem': 'peopl', 'xpos': 'NNS', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'NOUN', 'synsets': [{'wordnetId': 'people.n.01', 'definition': '(plural) any group of human beings (men or women or children) collectively', 'hypernyms': ['group', 'grouping'], 'hyponyms': ['age_group', 'age_bracket', 'cohort', 'ancients', 'baffled', 'blind', 'blood', 'brave', 'business_people', 'businesspeople', 'chosen_people', 'class', 'stratum', 'social_class', 'socio-economic_class', 'clientele', 'patronage', 'business', 'coevals', 'contemporaries', 'generation', 'damned', 'dead', 'deaf', 'defeated', 'discomfited', 'disabled', 'handicapped', 'doomed', 'lost', 'enemy', 'episcopacy', 'episcopate', 'folk', 'folks', 'common_people', 'free', 'free_people', 'homebound', 'initiate', 'enlightened', 'living', 'lobby', 'mentally_retarded', 'retarded', 'developmentally_challenged', 'migration', 'nation', 'land', 'country', 'nationality', 'network_army', 'peanut_gallery', 'peoples', 'pocket', 'poor_people', 'poor', 'populace', 'public', 'world', 'population', 'rank_and_file', 'retreated', 'rich_people', 'rich', 'sick', 'smart_money', 'timid', 'cautious', 'tradespeople', 'unconfessed', 'unemployed_people', 'unemployed', 'uninitiate', 'womankind', 'wounded', 'maimed'], 'examples': ['old people', 'there were at least 200 people in the audience']}, {'wordnetId': 'citizenry.n.01', 'definition': 'the body of citizens of a state or country', 'synonyms': ['people'], 'hypernyms': ['group', 'grouping'], 'hyponyms': ['Achaean', 'Arcado-Cyprians', 'Aeolian', 'country_people', 'countryfolk', 'Dorian', 'electorate', 'governed', 'Ionian'], 'examples': ['the Spanish people']}, {'wordnetId': 'people.n.03', 'definition': 'members of a family line', 'hypernyms': ['family', 'family_line', 'folk', 'kinfolk', 'kinsfolk', 'sept', 'phratry'], 'examples': ['his people have been farmers for generations', 'are your people still alive?']}, {'wordnetId': 'multitude.n.03', 'definition': 'the common people generally', 'synonyms': ['masses', 'mass', 'hoi_polloi', 'people', 'the_great_unwashed'], 'hypernyms': ['group', 'grouping'], 'hyponyms': ['audience', 'following', 'followers', 'laity', 'temporalty'], 'examples': ['separate the warriors from the mass', 'power to the people']}]}, {'id': 14, 'text': 'are', 'lemma': 'be', 'stem': 'are', 'xpos': 'VBP', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'VERB', 'synsets': [{'wordnetId': 'be.v.01', 'definition': 'have the quality of being; (copula, used with an adjective or a predicate noun)', 'hyponyms': ['abound', 'accept', 'take', 'account', 'account_for', 'act', 'answer', 'appear', 'seem', 'bake', 'broil', 'balance', 'be_well', 'beat', 'begin', 'begin', 'start', 'belong', 'belong', 'belong', 'belong', 'breathe', 'buy', 'clean', 'cohere', 'come_in_for', 'come_in_handy', 'compact', 'pack', 'compare', 'confuse', 'throw', 'fox', 'befuddle', 'fuddle', 'bedevil', 'confound', 'discombobulate', 'connect', 'consist', 'consist', 'comprise', 'contain', 'contain', 'take', 'hold', 'continue', 'cost', 'be', 'count', 'matter', 'weigh', 'count', 'cover', 'cut', 'cut_across', 'deck', 'adorn', 'decorate', 'grace', 'embellish', 'beautify', 'depend', 'deserve', 'merit', 'disagree', 'disaccord', 'discord', 'distribute', 'diverge', 'draw', 'end', 'terminate', 'fall', 'come', 'fall', 'feel', 'figure', 'enter', 'fit', 'gape', 'yawn', 'yaw', 'go', 'gravitate', 'hail', 'come', 'hang', 'head', 'head_up', 'hold', 'hoodoo', 'hum', 'buzz', 'seethe', 'impend', 'incarnate', 'body_forth', 'embody', 'substantiate', 'iridesce', 'jumble', 'mingle', 'kill', 'lend', 'let_go', 'lie', 'litter', 'loiter', 'lounge', 'footle', 'lollygag', 'loaf', 'lallygag', 'hang_around', 'mess_about', 'tarry', 'linger', 'lurk', 'mill_about', 'mill_around', 'look', 'appear', 'seem', 'look', 'lubricate', 'make', 'make_sense', 'add_up', 'measure', 'mope', 'moon_around', 'moon_about', 'object', 'osculate', 'owe', 'pay', 'point', 'press', 'promise', 'prove', 'turn_out', 'turn_up', 'put_out', 'rage', 'range', 'run', 'rank', 'rate', 'recognize', 'relate', 'interrelate', 'remain', 'represent', 'rest', 'retard', 'run', 'go', 'run_into', 'encounter', 'rut', 'seem', 'seethe', 'boil', 'sell', 'sell', 'sell', 'shine', 'shine', 'sparkle', 'scintillate', 'coruscate', 'specify', 'define', 'delineate', 'delimit', 'delimitate', 'squat', 'stagnate', 'stagnate', 'stand', 'stand', 'stand_by', 'stick_by', 'stick', 'adhere', 'stay', 'remain', 'rest', 'stay', 'stay_on', 'continue', 'remain', 'stick', 'stink', 'subtend', 'delimit', 'suck', 'suffer', 'hurt', 'suffer', 'suit', 'swim', 'swim', 'drown', 'swing', 'tend', 'be_given', 'lean', 'incline', 'run', 'test', 'total', 'number', 'add_up', 'come', 'amount', 'translate', 'transplant', 'trim', 'underlie', 'want', 'need', 'require', 'wash', 'wind', 'twist', 'curve', 'work'], 'examples': ['John is rich', 'This is not a good answer']}, {'wordnetId': 'be.v.02', 'definition': 'be identical to; be someone or something', 'examples': ['The president of the company is John Smith', 'This is my house']}, {'wordnetId': 'be.v.03', 'definition': 'occupy a certain position or area; be somewhere', 'hyponyms': ['attend', 'go_to', 'belong', 'go', 'center_on', 'come', 'cover', 'continue', 'extend', 'extend', 'poke_out', 'reach_out', 'face', 'follow', 'go', 'lead', 'inhabit', 'lie', 'lie', 'rest', 'occupy', 'fill', 'populate', 'dwell', 'live', 'inhabit', 'reach', 'extend_to', 'touch', 'run', 'go', 'pass', 'lead', 'extend', 'sit', 'sit_around', 'sit', 'stand_back', "keep_one's_eyes_off", "keep_one's_distance", "keep_one's_hands_off", 'stay_away', 'straddle', 'stretch', 'stretch_along'], 'examples': ['Where is my umbrella?" "The toolshed is in the back', 'What is behind this behavior?']}, {'wordnetId': 'exist.v.01', 'definition': 'have an existence, be extant', 'synonyms': ['be'], 'hyponyms': ['coexist', 'come', 'distribute', 'dwell', 'consist', 'lie', 'lie_in', 'dwell', 'inhabit', 'endanger', 'jeopardize', 'jeopardise', 'menace', 'threaten', 'imperil', 'peril', 'flow', 'indwell', 'kick_around', 'knock_about', 'kick_about', 'preexist', 'prevail', 'hold', 'obtain'], 'examples': ['Is there a God?']}, {'wordnetId': 'be.v.05', 'definition': 'happen, occur, take place; this was during the visit to my parents\' house"', 'examples': ['I lost my wallet', 'There were two hundred people at his funeral', 'There was a lot of noise in the kitchen']}, {'wordnetId': 'equal.v.01', 'definition': 'be identical or equivalent to', 'synonyms': ['be'], 'hyponyms': ['equate', 'correspond', 'match', 'fit', 'correspond', 'check', 'jibe', 'gibe', 'tally', 'agree', 'represent', 'stand_for', 'correspond', 'translate'], 'examples': ['One dollar equals 1,000 rubles these days!'], 'antonyms': ['differ']}, {'wordnetId': 'constitute.v.01', 'definition': 'form or compose', 'synonyms': ['represent', 'make_up', 'comprise', 'be'], 'hyponyms': ['compose', 'fall_into', 'fall_under', 'form', 'constitute', 'make', 'make', 'present', 'pose', 'range', 'straddle', 'supplement'], 'examples': ['This money is my only income', 'The stone wall was the backdrop for the performance', 'These constitute my entire belonging', 'The children made up the chorus', 'This sum represents my entire income for a year', 'These few men comprise his entire army']}, {'wordnetId': 'be.v.08', 'definition': 'work in a specific place, with a specific subject, or in a specific function', 'synonyms': ['follow'], 'hyponyms': ['cox', 'vet'], 'examples': ['He is a herpetologist', 'She is our resident philosopher']}, {'wordnetId': 'embody.v.02', 'definition': 'represent, as of a character on stage', 'synonyms': ['be', 'personify'], 'hypernyms': ['typify', 'symbolize', 'symbolise', 'stand_for', 'represent'], 'hyponyms': ['body', 'personify', 'exemplify', 'represent'], 'examples': ['Derek Jacobi was Hamlet']}, {'wordnetId': 'be.v.10', 'definition': 'spend or use time', 'hypernyms': ['take', 'occupy', 'use_up'], 'examples': ['I may be an hour']}, {'wordnetId': 'be.v.11', 'definition': 'have life, be alive', 'synonyms': ['live'], 'examples': ['Our great leader is no more', 'My grandfather lived until the end of war']}, {'wordnetId': 'be.v.12', 'definition': 'to remain unmolested, undisturbed, or uninterrupted -- used only in infinitive form', 'hypernyms': ['stay', 'remain', 'rest'], 'examples': ['let her be']}, {'wordnetId': 'cost.v.01', 'definition': 'be priced at', 'synonyms': ['be'], 'hypernyms': ['be'], 'hyponyms': ['set_back', 'knock_back', 'put_back'], 'examples': ['These shoes cost $100']}]}, {'id': 15, 'text': 'afraid', 'lemma': 'afraid', 'stem': 'afraid', 'xpos': 'JJ', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'ADJ', 'synsets': [{'wordnetId': 'afraid.a.01', 'definition': 'filled with fear or apprehension', 'examples': ['afraid even to turn his head', 'suddenly looked afraid', 'afraid for his life', 'afraid of snakes', 'afraid to ask questions'], 'antonyms': ['unafraid']}, {'wordnetId': 'afraid.s.02', 'definition': 'filled with regret or concern; used often to soften an unpleasant statement', 'examples': ["I'm afraid I won't be able to come", 'he was afraid he would have to let her go', "I'm afraid you're wrong"]}, {'wordnetId': 'afraid.s.03', 'definition': 'feeling worry or concern or insecurity', 'examples': ['She was afraid that I might be embarrassed', 'terribly afraid of offending someone', 'I am afraid we have witnessed only the first phase of the conflict']}, {'wordnetId': 'afraid.s.04', 'definition': 'having feelings of aversion or unwillingness', 'examples': ['afraid of hard work', 'afraid to show emotion']}]}, {'id': 16, 'text': 'that', 'lemma': 'that', 'stem': 'that', 'xpos': 'IN', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'ADP'}, {'id': 17, 'text': 'they', 'lemma': 'they', 'stem': 'they', 'xpos': 'PRP', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'PRON'}, {'id': 18, 'text': 'will', 'lemma': 'will', 'stem': 'will', 'xpos': 'MD', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'VERB', 'synsets': [{'wordnetId': 'volition.n.01', 'definition': 'the capability of conscious choice and decision and intention', 'synonyms': ['will'], 'hypernyms': ['faculty', 'mental_faculty', 'module'], 'hyponyms': ['velleity'], 'examples': ['the exercise of their volition we construe as revolt"- George Meredith']}, {'wordnetId': 'will.n.02', 'definition': 'a fixed and persistent intent or purpose', 'hypernyms': ['purpose', 'intent', 'intention', 'aim', 'design'], 'examples': ["where there's a will there's a way"]}, {'wordnetId': 'will.n.03', 'definition': "a legal document declaring a person's wishes regarding the disposal of their property when they die", 'synonyms': ['testament'], 'hypernyms': ['legal_document', 'legal_instrument', 'official_document', 'instrument'], 'hyponyms': ['devise', 'New_Testament', 'Old_Testament']}]}, {'id': 19, 'text': 'crash', 'lemma': 'crash', 'stem': 'crash', 'xpos': 'VB', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': 'VERB', 'synsets': [{'wordnetId': 'crash.v.01', 'definition': 'fall or come down violently', 'hypernyms': ['descend', 'fall', 'go_down', 'come_down'], 'examples': ['The branch crashed down on my car', 'The plane crashed in the sea']}, {'wordnetId': 'crash.v.02', 'definition': 'move with, or as if with, a crashing noise', 'hypernyms': ['move'], 'examples': ['The car crashed through the glass door']}, {'wordnetId': 'crash.v.03', 'definition': 'undergo damage or destruction on impact', 'synonyms': ['ram'], 'hypernyms': ['collide', 'clash'], 'examples': ['the plane crashed into the ocean', 'The car crashed into the lamp post']}, {'wordnetId': 'crash.v.04', 'definition': 'move violently as through a barrier', 'hypernyms': ['pass', 'go_through', 'go_across'], 'examples': ['The terrorists crashed the gate']}, {'wordnetId': 'crash.v.05', 'definition': 'break violently or noisily; smash; ', 'synonyms': ['break_up', 'break_apart'], 'hypernyms': ['disintegrate']}, {'wordnetId': 'crash.v.06', 'definition': 'occupy, usually uninvited', 'hypernyms': ['occupy', 'reside', 'lodge_in'], 'examples': ["My son's friends crashed our house last weekend"]}, {'wordnetId': 'crash.v.07', 'definition': 'make a sudden loud sound', 'hypernyms': ['sound', 'go'], 'examples': ['the waves crashed on the shore and kept us awake all night']}, {'wordnetId': 'barge_in.v.01', 'definition': 'enter uninvited; informal', 'synonyms': ['crash', 'gate-crash'], 'hypernyms': ['intrude', 'irrupt'], 'examples': ["let's crash the party!"]}, {'wordnetId': 'crash.v.09', 'definition': 'cause to crash', 'hypernyms': ['collide'], 'hyponyms': ['ditch', 'prang', 'wrap'], 'examples': ['The terrorists crashed the plane into the palace', 'Mother crashed the motorbike into the lamppost']}, {'wordnetId': 'crash.v.10', 'definition': 'hurl or thrust violently', 'synonyms': ['dash'], 'hypernyms': ['hurl', 'hurtle', 'cast'], 'examples': ['He dashed the plate against the wall', 'Waves were dashing against the rock']}, {'wordnetId': 'crash.v.11', 'definition': 'undergo a sudden and severe downturn', 'hypernyms': ['change'], 'examples': ['the economy crashed', 'will the stock market crash again?']}, {'wordnetId': 'crash.v.12', 'definition': 'stop operating', 'synonyms': ['go_down'], 'hypernyms': ['fail', 'go_bad', 'give_way', 'die', 'give_out', 'conk_out', 'go', 'break', 'break_down'], 'examples': ['My computer crashed last night', 'The system goes down at least once a week']}, {'wordnetId': 'doss.v.01', 'definition': 'sleep in a convenient place', 'synonyms': ['doss_down', 'crash'], 'hypernyms': ['bed_down', 'bunk_down'], 'examples': ["You can crash here, though it's not very comfortable"]}]}, {'id': 20, 'text': '.', 'lemma': '.', 'stem': '.', 'xpos': '.', 'entity_iob': 'O', 'features': {'Overt': 'Yes'}, 'upos': '.'}]), ('clauses', []), ('sentences', [{'id': '0', 'text': 'Autonomous cars from the countryside of France shift insurance liability toward manufacturers.', 'tokenFrom': 1, 'tokenTo': 13, 'tokens': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}, {'id': '1', 'text': 'People are afraid that they will crash.', 'tokenFrom': 13, 'tokenTo': 21, 'tokens': [13, 14, 15, 16, 17, 18, 19, 20]}]), ('paragraphs', []), ('dependenciesBasic', [{'governor': 14, 'dependent': 13, 'label': 'nn'}, {'governor': 12, 'dependent': 14, 'label': 'root'}, {'governor': 14, 'dependent': 15, 'label': 'prep'}, {'governor': 24, 'dependent': 16, 'label': 'det'}, {'governor': 15, 'dependent': 24, 'label': 'pobj'}, {'governor': 24, 'dependent': 17, 'label': 'nn'}, {'governor': 24, 'dependent': 18, 'label': 'nn'}, {'governor': 24, 'dependent': 19, 'label': 'nn'}, {'governor': 24, 'dependent': 20, 'label': 'nn'}, {'governor': 24, 'dependent': 21, 'label': 'nn'}, {'governor': 24, 'dependent': 22, 'label': 'nn'}, {'governor': 24, 'dependent': 23, 'label': 'nn'}, {'governor': 27, 'dependent': 21, 'label': 'nn'}, {'governor': 20, 'dependent': 27, 'label': 'root'}, {'governor': 27, 'dependent': 22, 'label': 'nn'}, {'governor': 27, 'dependent': 23, 'label': 'nn'}, {'governor': 27, 'dependent': 24, 'label': 'nn'}, {'governor': 27, 'dependent': 25, 'label': 'nn'}, {'governor': 27, 'dependent': 26, 'label': 'nn'}, {'governor': 27, 'dependent': 28, 'label': 'punct'}]), ('dependenciesEnhanced', []), ('coreferences', []), ('constituents', []), ('expressions', [])])])])
        assert expected == actual, actual

    def test_get_wordnet_pos(self):
        actual = nltk2json.get_wordnet_pos('JJ')
        expected = nltk2json.wordnet.ADJ
        assert expected == actual, actual
        actual = nltk2json.get_wordnet_pos('VV')
        expected = nltk2json.wordnet.VERB
        assert expected == actual, actual
        actual = nltk2json.get_wordnet_pos('RR')
        expected = nltk2json.wordnet.ADV
        assert expected == actual, actual
        actual = nltk2json.get_wordnet_pos('anything else')
        expected = nltk2json.wordnet.NOUN
        assert expected == actual, actual

    def test_get_sentence_tokenizer(self):
        t = nltk2json.get_sentence_tokenizer()
        assert isinstance(t, PunktSentenceTokenizer), t

    def test_get_tokenize(self):
        t = nltk2json.get_tokenizer('punkt')
        assert isinstance(t, TokenizerI), t
        t = nltk2json.get_tokenizer('treebank')
        assert isinstance(t, TokenizerI), t
        t = nltk2json.get_tokenizer('')
        assert isinstance(t, TokenizerI), t
        with pytest.raises(ModuleNotFoundError):
            nltk2json.get_tokenizer('inconceivable!')

    def test_get_lemmatizer(self):
        t = nltk2json.get_lemmatizer()
        assert isinstance(t, Callable), t

    def test_get_stemmer(self):
        t = nltk2json.get_stemmer()
        assert isinstance(t, Callable), t

    def test_get_tag_mapper(self):
        t = nltk2json.get_tag_mapper('en')
        assert isinstance(t, dict), t
        assert len(t) > 0, len(t)
        t = nltk2json.get_tag_mapper('marian')
        assert isinstance(t, dict), t
        assert len(t) == 0, len(t)

    def test_get_dependency_parser(self):
        t = nltk2json.get_dependency_parser('malt', 'martian')
        assert t is None, t
        t = nltk2json.get_dependency_parser('perfect', 'martian')
        assert t is None, t
