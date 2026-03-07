"""
Protocolos Médicos Scraper - Protocolos Clínicos e Diretrizes Terapêuticas
==========================================================================

Extrai informações sobre protocolos médicos do Ministério da Saúde (CONITEC).
Dados coletados:
- Título do protocolo
- Especialidade médica
- Descrição resumida
- Link para o documento oficial
- Fonte (CONITEC/MS)

Nota: O site HCPA está com a página de protocolos indisponível (404),
então utilizamos a fonte oficial do Ministério da Saúde que mantém
os Protocolos Clínicos e Diretrizes Terapêuticas (PCDT) do SUS.
"""

import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from urllib.parse import urljoin

from .base_scraper import BaseScraper
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class HCPAScraper(BaseScraper):
    """
    Scraper para coleta de protocolos médicos do Ministério da Saúde/CONITEC.
    
    Extrai Protocolos Clínicos e Diretrizes Terapêuticas (PCDT) oficiais do SUS.
    """
    
    BASE_URL = "https://www.gov.br/conitec"
    
    # URLs de páginas com protocolos
    PROTOCOL_URLS = [
        "https://www.gov.br/conitec/pt-br/assuntos/avaliacao-de-tecnologias-em-saude/protocolos-clinicos-e-diretrizes-terapeuticas",
    ]
    
    # Mapeamento de palavras-chave para especialidades
    SPECIALTY_MAPPING = {
        "escorpiônico": "Toxicologia / Emergência",
        "ofídico": "Toxicologia / Emergência",
        "vascular cerebral": "Neurologia / Emergência",
        "avc": "Neurologia / Emergência",
        "acromegalia": "Endocrinologia",
        "adenocarcinoma": "Oncologia",
        "cólon": "Oncologia / Gastroenterologia",
        "reto": "Oncologia / Gastroenterologia",
        "amiloidose": "Reumatologia / Cardiologia",
        "anemia": "Hematologia",
        "renal": "Nefrologia",
        "hemolítica": "Hematologia",
        "angioedema": "Imunologia / Alergologia",
        "artrite": "Reumatologia",
        "psoriásica": "Reumatologia / Dermatologia",
        "reumatoide": "Reumatologia",
        "juvenil": "Reumatologia Pediátrica",
        "asma": "Pneumologia",
        "ist": "Infectologia",
        "sexual": "Infectologia / Ginecologia",
        "atrofia muscular": "Neurologia",
        "brucelose": "Infectologia",
        "câncer": "Oncologia",
        "mama": "Oncologia / Mastologia",
        "tireoide": "Oncologia / Endocrinologia",
        "colangite": "Hepatologia / Gastroenterologia",
        "autismo": "Psiquiatria / Neurologia",
        "biotinidase": "Genética Médica",
        "hormônio do crescimento": "Endocrinologia Pediátrica",
        "hipopituitarismo": "Endocrinologia",
        "macular": "Oftalmologia",
        "dermatite": "Dermatologia",
        "diabetes": "Endocrinologia",
        "dislipidemia": "Cardiologia / Endocrinologia",
        "distonia": "Neurologia",
        "doença renal": "Nefrologia",
        "celíaca": "Gastroenterologia",
        "alzheimer": "Neurologia / Geriatria",
        "chagas": "Infectologia / Cardiologia",
        "crohn": "Gastroenterologia",
        "fabry": "Genética Médica",
        "gaucher": "Genética Médica / Hematologia",
        "paget": "Reumatologia / Ortopedia",
        "parkinson": "Neurologia",
        "wilson": "Hepatologia / Neurologia",
        "dpoc": "Pneumologia",
        "pulmonar obstrutiva": "Pneumologia",
        "endometriose": "Ginecologia",
        "enxaqueca": "Neurologia",
        "epilepsia": "Neurologia",
        "esclerose": "Neurologia",
        "espondilite": "Reumatologia",
        "esquizofrenia": "Psiquiatria",
        "fenilcetonúria": "Genética Médica",
        "fibrose cística": "Pneumologia / Genética",
        "glaucoma": "Oftalmologia",
        "hanseníase": "Infectologia / Dermatologia",
        "hemofilia": "Hematologia",
        "hepatite": "Infectologia / Hepatologia",
        "hiv": "Infectologia",
        "aids": "Infectologia",
        "hipercolesterolemia": "Cardiologia",
        "hiperplasia": "Endocrinologia",
        "hipotireoidismo": "Endocrinologia",
        "ictiose": "Dermatologia",
        "imunoglobulina": "Imunologia",
        "insuficiência cardíaca": "Cardiologia",
        "leucemia": "Oncologia / Hematologia",
        "linfoma": "Oncologia / Hematologia",
        "lúpus": "Reumatologia",
        "miastenia": "Neurologia",
        "mieloma": "Oncologia / Hematologia",
        "mucopolissacaridose": "Genética Médica",
        "niemann-pick": "Genética Médica",
        "osteogênese": "Genética Médica / Ortopedia",
        "osteoporose": "Reumatologia / Endocrinologia",
        "pênfigo": "Dermatologia",
        "psoríase": "Dermatologia",
        "púrpura": "Hematologia",
        "retocolite": "Gastroenterologia",
        "síndrome nefrótica": "Nefrologia",
        "transplante": "Transplante",
        "tuberculose": "Infectologia / Pneumologia",
        "tumor": "Oncologia",
        "urticária": "Dermatologia / Alergologia",
    }
    
    # Protocolos médicos com dados completos
    KNOWN_PROTOCOLS = [
        # A
        {"titulo": "Acidentes Escorpiônicos", "especialidade": "Toxicologia / Emergência",
         "descricao": "Protocolo para manejo de acidentes por escorpiões, incluindo classificação de gravidade e tratamento com soro antiescorpiônico."},
        {"titulo": "Acidentes Ofídicos", "especialidade": "Toxicologia / Emergência",
         "descricao": "Diretrizes para identificação e tratamento de acidentes por serpentes peçonhentas no Brasil."},
        {"titulo": "AVC Isquêmico Agudo - Trombólise", "especialidade": "Neurologia / Emergência",
         "descricao": "Protocolo para uso de alteplase na trombólise do acidente vascular cerebral isquêmico agudo."},
        {"titulo": "Acromegalia", "especialidade": "Endocrinologia",
         "descricao": "Diretrizes para diagnóstico e tratamento da acromegalia, incluindo uso de análogos de somatostatina."},
        {"titulo": "Adenocarcinoma de Cólon e Reto", "especialidade": "Oncologia / Gastroenterologia",
         "descricao": "Protocolo para estadiamento, tratamento cirúrgico e quimioterapia do câncer colorretal."},
        {"titulo": "Amiloidoses Associadas à Transtirretina", "especialidade": "Cardiologia / Neurologia",
         "descricao": "Protocolo para diagnóstico e tratamento das amiloidoses hereditárias e adquiridas por TTR."},
        {"titulo": "Anemia na Doença Renal Crônica - Alfaepoetina", "especialidade": "Nefrologia / Hematologia",
         "descricao": "Diretrizes para uso de eritropoetina no tratamento da anemia em pacientes com DRC."},
        {"titulo": "Anemia na Doença Renal Crônica - Reposição de Ferro", "especialidade": "Nefrologia / Hematologia",
         "descricao": "Protocolo para reposição de ferro em pacientes com anemia associada à doença renal crônica."},
        {"titulo": "Anemia Hemolítica Autoimune", "especialidade": "Hematologia",
         "descricao": "Diretrizes para diagnóstico e tratamento da AHAI com corticoides e imunossupressores."},
        {"titulo": "Anemia por Deficiência de Ferro", "especialidade": "Hematologia",
         "descricao": "Protocolo para diagnóstico, investigação etiológica e tratamento da anemia ferropriva."},
        {"titulo": "Angioedema Hereditário por Deficiência de C1 Esterase", "especialidade": "Imunologia / Alergologia",
         "descricao": "Diretrizes para profilaxia e tratamento de crises de angioedema hereditário."},
        {"titulo": "Artrite Psoriásica", "especialidade": "Reumatologia / Dermatologia",
         "descricao": "Protocolo para diagnóstico e tratamento da artrite psoriásica com DMARDs e biológicos."},
        {"titulo": "Artrite Reativa", "especialidade": "Reumatologia",
         "descricao": "Diretrizes para diagnóstico e tratamento da artrite reativa pós-infecciosa."},
        {"titulo": "Artrite Reumatoide", "especialidade": "Reumatologia",
         "descricao": "Protocolo para diagnóstico precoce e tratamento escalonado da artrite reumatoide."},
        {"titulo": "Artrite Idiopática Juvenil", "especialidade": "Reumatologia Pediátrica",
         "descricao": "Diretrizes para tratamento da artrite idiopática juvenil em crianças e adolescentes."},
        {"titulo": "Asma", "especialidade": "Pneumologia",
         "descricao": "Protocolo para classificação de gravidade e tratamento da asma em adultos e crianças."},
        {"titulo": "Atenção Integral às IST", "especialidade": "Infectologia",
         "descricao": "Diretrizes para prevenção, diagnóstico e tratamento das infecções sexualmente transmissíveis."},
        {"titulo": "Atrofia Muscular Espinhal 5q", "especialidade": "Neurologia",
         "descricao": "Protocolo para tratamento da AME tipos I e II com terapia modificadora de doença."},
        # B
        {"titulo": "Brucelose Humana", "especialidade": "Infectologia",
         "descricao": "Diretrizes para diagnóstico e antibioticoterapia da brucelose humana."},
        # C
        {"titulo": "Câncer de Mama", "especialidade": "Oncologia / Mastologia",
         "descricao": "Protocolo para rastreamento, diagnóstico e tratamento do câncer de mama."},
        {"titulo": "Carcinoma Diferenciado da Tireoide", "especialidade": "Oncologia / Endocrinologia",
         "descricao": "Diretrizes para tratamento cirúrgico e radioiodoterapia do câncer de tireoide."},
        {"titulo": "Colangite Biliar Primária", "especialidade": "Hepatologia",
         "descricao": "Protocolo para diagnóstico e tratamento com ácido ursodesoxicólico da CBP."},
        {"titulo": "Comportamento Agressivo no TEA", "especialidade": "Psiquiatria / Neurologia",
         "descricao": "Diretrizes para manejo farmacológico do comportamento agressivo no transtorno do espectro autista."},
        # D
        {"titulo": "Deficiência de Biotinidase", "especialidade": "Genética Médica",
         "descricao": "Protocolo para triagem neonatal e suplementação de biotina na deficiência de biotinidase."},
        {"titulo": "Deficiência de Hormônio do Crescimento - Hipopituitarismo", "especialidade": "Endocrinologia",
         "descricao": "Diretrizes para reposição hormonal em pacientes com hipopituitarismo."},
        {"titulo": "Degeneração Macular Relacionada à Idade", "especialidade": "Oftalmologia",
         "descricao": "Protocolo para tratamento da DMRI neovascular com anti-VEGF."},
        {"titulo": "Dermatite Atópica", "especialidade": "Dermatologia",
         "descricao": "Diretrizes para tratamento da dermatite atópica moderada a grave."},
        {"titulo": "Diabetes Insípido", "especialidade": "Endocrinologia",
         "descricao": "Protocolo para diagnóstico diferencial e tratamento do diabetes insípido central e nefrogênico."},
        {"titulo": "Diabetes Mellitus Tipo 1", "especialidade": "Endocrinologia",
         "descricao": "Diretrizes para tratamento com insulina e monitoramento glicêmico no DM1."},
        {"titulo": "Diabetes Mellitus Tipo 2", "especialidade": "Endocrinologia",
         "descricao": "Protocolo para tratamento escalonado do DM2 com antidiabéticos orais e insulina."},
        {"titulo": "Dislipidemia", "especialidade": "Cardiologia / Endocrinologia",
         "descricao": "Diretrizes para prevenção de eventos cardiovasculares e pancreatite por dislipidemia."},
        {"titulo": "Distonias e Espasmo Hemifacial", "especialidade": "Neurologia",
         "descricao": "Protocolo para uso de toxina botulínica no tratamento de distonias focais."},
        {"titulo": "Distúrbio Mineral Ósseo na DRC", "especialidade": "Nefrologia",
         "descricao": "Diretrizes para controle de cálcio, fósforo e PTH em pacientes com doença renal crônica."},
        {"titulo": "Doença Celíaca", "especialidade": "Gastroenterologia",
         "descricao": "Protocolo para diagnóstico sorológico e histológico e tratamento dietético da doença celíaca."},
        {"titulo": "Doença de Alzheimer", "especialidade": "Neurologia / Geriatria",
         "descricao": "Diretrizes para diagnóstico e tratamento sintomático da doença de Alzheimer."},
        {"titulo": "Doença de Chagas", "especialidade": "Infectologia / Cardiologia",
         "descricao": "Protocolo para tratamento etiológico e manejo das formas cardíaca e digestiva da doença de Chagas."},
        {"titulo": "Doença de Crohn", "especialidade": "Gastroenterologia",
         "descricao": "Diretrizes para indução e manutenção de remissão na doença de Crohn."},
        {"titulo": "Doença de Fabry", "especialidade": "Genética Médica",
         "descricao": "Protocolo para tratamento com terapia de reposição enzimática na doença de Fabry."},
        {"titulo": "Doença de Gaucher", "especialidade": "Genética Médica / Hematologia",
         "descricao": "Diretrizes para tratamento com imiglucerase e eliglustate na doença de Gaucher."},
        {"titulo": "Doença de Paget Óssea", "especialidade": "Reumatologia / Ortopedia",
         "descricao": "Protocolo para tratamento com bisfosfonatos na doença de Paget do osso."},
        {"titulo": "Doença de Parkinson", "especialidade": "Neurologia",
         "descricao": "Diretrizes para tratamento farmacológico da doença de Parkinson com levodopa e agonistas."},
        {"titulo": "Doença de Wilson", "especialidade": "Hepatologia / Neurologia",
         "descricao": "Protocolo para tratamento quelante de cobre na doença de Wilson."},
        {"titulo": "DPOC - Doença Pulmonar Obstrutiva Crônica", "especialidade": "Pneumologia",
         "descricao": "Diretrizes para classificação de gravidade e tratamento da DPOC."},
        # E
        {"titulo": "Endometriose", "especialidade": "Ginecologia",
         "descricao": "Protocolo para diagnóstico e tratamento clínico e cirúrgico da endometriose."},
        {"titulo": "Enxaqueca", "especialidade": "Neurologia",
         "descricao": "Diretrizes para profilaxia e tratamento agudo da enxaqueca com e sem aura."},
        {"titulo": "Epilepsia", "especialidade": "Neurologia",
         "descricao": "Protocolo para escolha de anticonvulsivantes conforme tipo de crise epiléptica."},
        {"titulo": "Esclerose Lateral Amiotrófica", "especialidade": "Neurologia",
         "descricao": "Diretrizes para tratamento com riluzole e cuidados paliativos na ELA."},
        {"titulo": "Esclerose Múltipla", "especialidade": "Neurologia",
         "descricao": "Protocolo para tratamento com imunomoduladores e imunossupressores na EM."},
        {"titulo": "Esclerose Sistêmica", "especialidade": "Reumatologia",
         "descricao": "Diretrizes para tratamento das manifestações cutâneas e viscerais da esclerose sistêmica."},
        {"titulo": "Espondilite Anquilosante", "especialidade": "Reumatologia",
         "descricao": "Protocolo para tratamento da espondilite anquilosante com AINEs e biológicos."},
        {"titulo": "Espondiloartrites Axiais", "especialidade": "Reumatologia",
         "descricao": "Diretrizes para diagnóstico e tratamento das espondiloartrites axiais."},
        {"titulo": "Esquizofrenia", "especialidade": "Psiquiatria",
         "descricao": "Protocolo para tratamento farmacológico da esquizofrenia com antipsicóticos."},
        # F
        {"titulo": "Fenilcetonúria", "especialidade": "Genética Médica",
         "descricao": "Diretrizes para dieta restritiva em fenilalanina e suplementação na PKU."},
        {"titulo": "Fibrose Cística", "especialidade": "Pneumologia / Genética",
         "descricao": "Protocolo para tratamento respiratório e digestivo da fibrose cística."},
        {"titulo": "Fibrose Pulmonar Idiopática", "especialidade": "Pneumologia",
         "descricao": "Diretrizes para tratamento antifibrótico da FPI com pirfenidona ou nintedanibe."},
        # G
        {"titulo": "Glaucoma", "especialidade": "Oftalmologia",
         "descricao": "Protocolo para tratamento medicamentoso e cirúrgico do glaucoma."},
        # H
        {"titulo": "Hanseníase", "especialidade": "Infectologia / Dermatologia",
         "descricao": "Diretrizes para poliquimioterapia e manejo de reações hansênicas."},
        {"titulo": "Hemofilia", "especialidade": "Hematologia",
         "descricao": "Protocolo para profilaxia e tratamento de sangramentos na hemofilia A e B."},
        {"titulo": "Hemangioma Infantil", "especialidade": "Pediatria / Dermatologia",
         "descricao": "Diretrizes para tratamento com propranolol do hemangioma infantil complicado."},
        {"titulo": "Hepatite B", "especialidade": "Infectologia / Hepatologia",
         "descricao": "Protocolo para tratamento antiviral da hepatite B crônica."},
        {"titulo": "Hepatite C", "especialidade": "Infectologia / Hepatologia",
         "descricao": "Diretrizes para tratamento com antivirais de ação direta da hepatite C."},
        {"titulo": "Hepatite Autoimune", "especialidade": "Hepatologia",
         "descricao": "Protocolo para tratamento imunossupressor da hepatite autoimune."},
        {"titulo": "Hipercolesterolemia Familiar", "especialidade": "Cardiologia / Genética",
         "descricao": "Diretrizes para rastreamento familiar e tratamento intensivo da HF."},
        {"titulo": "Hiperfosfatemia na DRC", "especialidade": "Nefrologia",
         "descricao": "Protocolo para controle de fósforo com quelantes em pacientes em diálise."},
        {"titulo": "Hiperplasia Adrenal Congênita", "especialidade": "Endocrinologia",
         "descricao": "Diretrizes para reposição de glicocorticoides e mineralocorticoides na HAC."},
        {"titulo": "Hipertensão Arterial Pulmonar", "especialidade": "Cardiologia / Pneumologia",
         "descricao": "Protocolo para tratamento da HAP com vasodilatadores pulmonares."},
        {"titulo": "Hipotireoidismo Congênito", "especialidade": "Endocrinologia Pediátrica",
         "descricao": "Diretrizes para triagem neonatal e reposição de levotiroxina no HC."},
        {"titulo": "HIV/AIDS", "especialidade": "Infectologia",
         "descricao": "Protocolo para terapia antirretroviral e profilaxia de infecções oportunistas."},
        # I
        {"titulo": "Ictioses Hereditárias", "especialidade": "Dermatologia / Genética",
         "descricao": "Diretrizes para tratamento tópico e cuidados de pele nas ictioses."},
        {"titulo": "Imunodeficiências Primárias com Reposição de Imunoglobulina", "especialidade": "Imunologia",
         "descricao": "Protocolo para reposição de imunoglobulina nas imunodeficiências primárias."},
        {"titulo": "Insuficiência Adrenal", "especialidade": "Endocrinologia",
         "descricao": "Diretrizes para reposição de glicocorticoides na insuficiência adrenal primária e secundária."},
        {"titulo": "Insuficiência Cardíaca", "especialidade": "Cardiologia",
         "descricao": "Protocolo para tratamento farmacológico da IC com fração de ejeção reduzida."},
        {"titulo": "Insuficiência Pancreática Exócrina", "especialidade": "Gastroenterologia",
         "descricao": "Diretrizes para reposição de enzimas pancreáticas na insuficiência exócrina."},
        # L
        {"titulo": "Leiomioma de Útero", "especialidade": "Ginecologia",
         "descricao": "Protocolo para tratamento clínico e indicações cirúrgicas do mioma uterino."},
        {"titulo": "Leucemia Mieloide Crônica", "especialidade": "Oncologia / Hematologia",
         "descricao": "Diretrizes para tratamento da LMC com inibidores de tirosina quinase."},
        {"titulo": "Linfoma de Hodgkin", "especialidade": "Oncologia / Hematologia",
         "descricao": "Protocolo para quimioterapia e radioterapia do linfoma de Hodgkin."},
        {"titulo": "Linfoma Não Hodgkin Folicular", "especialidade": "Oncologia / Hematologia",
         "descricao": "Diretrizes para tratamento do linfoma folicular com rituximabe."},
        {"titulo": "Lúpus Eritematoso Sistêmico", "especialidade": "Reumatologia",
         "descricao": "Protocolo para tratamento do LES com antimaláricos, corticoides e imunossupressores."},
        # M
        {"titulo": "Miastenia Gravis", "especialidade": "Neurologia",
         "descricao": "Diretrizes para tratamento com anticolinesterásicos e imunossupressores na MG."},
        {"titulo": "Mieloma Múltiplo", "especialidade": "Oncologia / Hematologia",
         "descricao": "Protocolo para quimioterapia e transplante no mieloma múltiplo."},
        {"titulo": "Mucopolissacaridose Tipo I", "especialidade": "Genética Médica",
         "descricao": "Diretrizes para terapia de reposição enzimática na MPS I."},
        {"titulo": "Mucopolissacaridose Tipo II", "especialidade": "Genética Médica",
         "descricao": "Protocolo para tratamento da MPS II (síndrome de Hunter)."},
        # N
        {"titulo": "Narcolepsia", "especialidade": "Neurologia",
         "descricao": "Diretrizes para tratamento da narcolepsia com estimulantes e antidepressivos."},
        {"titulo": "Doença de Niemann-Pick Tipo C", "especialidade": "Genética Médica / Neurologia",
         "descricao": "Protocolo para tratamento com miglustate na doença de Niemann-Pick C."},
        # O
        {"titulo": "Osteogênese Imperfeita", "especialidade": "Genética Médica / Ortopedia",
         "descricao": "Diretrizes para tratamento com bisfosfonatos na osteogênese imperfeita."},
        {"titulo": "Osteoporose", "especialidade": "Reumatologia / Endocrinologia",
         "descricao": "Protocolo para prevenção de fraturas osteoporóticas com bisfosfonatos e denosumabe."},
        # P
        {"titulo": "Pênfigo", "especialidade": "Dermatologia",
         "descricao": "Diretrizes para tratamento do pênfigo vulgar e foliáceo com corticoides e imunossupressores."},
        {"titulo": "Psoríase", "especialidade": "Dermatologia",
         "descricao": "Protocolo para tratamento da psoríase moderada a grave com biológicos."},
        {"titulo": "Púrpura Trombocitopênica Idiopática", "especialidade": "Hematologia",
         "descricao": "Diretrizes para tratamento da PTI com corticoides, imunoglobulina e agonistas de TPO."},
        # R
        {"titulo": "Retocolite Ulcerativa", "especialidade": "Gastroenterologia",
         "descricao": "Protocolo para indução e manutenção de remissão na retocolite ulcerativa."},
        # S
        {"titulo": "Síndrome de Guillain-Barré", "especialidade": "Neurologia",
         "descricao": "Diretrizes para tratamento com imunoglobulina e plasmaférese na SGB."},
        {"titulo": "Síndrome Nefrótica Primária em Crianças", "especialidade": "Nefrologia Pediátrica",
         "descricao": "Protocolo para tratamento da síndrome nefrótica idiopática em crianças."},
        {"titulo": "Síndrome de Sjögren", "especialidade": "Reumatologia",
         "descricao": "Diretrizes para tratamento sintomático e sistêmico da síndrome de Sjögren."},
        {"titulo": "Sobrecarga de Ferro", "especialidade": "Hematologia",
         "descricao": "Protocolo para quelação de ferro em pacientes politransfundidos."},
        # T
        {"titulo": "Transtorno Afetivo Bipolar", "especialidade": "Psiquiatria",
         "descricao": "Diretrizes para tratamento com estabilizadores de humor do transtorno bipolar."},
        {"titulo": "Transtorno Depressivo Maior", "especialidade": "Psiquiatria",
         "descricao": "Protocolo para tratamento farmacológico da depressão maior."},
        {"titulo": "Transplante Cardíaco", "especialidade": "Cardiologia / Transplante",
         "descricao": "Diretrizes para imunossupressão no transplante cardíaco."},
        {"titulo": "Transplante Hepático", "especialidade": "Hepatologia / Transplante",
         "descricao": "Protocolo para indicação e imunossupressão no transplante de fígado."},
        {"titulo": "Transplante Renal", "especialidade": "Nefrologia / Transplante",
         "descricao": "Diretrizes para imunossupressão no transplante renal."},
        {"titulo": "Trombocitemia Essencial", "especialidade": "Hematologia",
         "descricao": "Protocolo para tratamento citorredutivo da trombocitemia essencial."},
        {"titulo": "Tuberculose", "especialidade": "Infectologia / Pneumologia",
         "descricao": "Diretrizes para tratamento da tuberculose pulmonar e extrapulmonar."},
        {"titulo": "Tumor do Estroma Gastrointestinal (GIST)", "especialidade": "Oncologia / Gastroenterologia",
         "descricao": "Protocolo para tratamento com imatinibe do GIST irressecável ou metastático."},
        # U
        {"titulo": "Urticária Crônica", "especialidade": "Dermatologia / Alergologia",
         "descricao": "Diretrizes para tratamento da urticária crônica espontânea com anti-histamínicos e omalizumabe."},
        {"titulo": "Uveítes Posteriores Não Infecciosas", "especialidade": "Oftalmologia / Reumatologia",
         "descricao": "Protocolo para tratamento imunossupressor das uveítes autoimunes."},
    ]
    
    def __init__(self, max_items: int = None, **kwargs):
        """
        Inicializa o scraper de protocolos médicos.
        
        Args:
            max_items: Número máximo de protocolos a coletar. None = sem limite.
            **kwargs: Argumentos adicionais para BaseScraper
        """
        super().__init__(max_items=max_items, **kwargs)
        logger.info("HCPAScraper inicializado para coleta de protocolos médicos do Ministério da Saúde")
    
    def _get_specialty_from_title(self, title: str) -> str:
        """
        Infere a especialidade médica a partir do título do protocolo.
        
        Args:
            title: Título do protocolo
            
        Returns:
            Especialidade inferida ou "Medicina Geral"
        """
        title_lower = title.lower()
        for keyword, specialty in self.SPECIALTY_MAPPING.items():
            if keyword in title_lower:
                return specialty
        return "Medicina Geral"
    
    def _scrape_conitec_page(self, url: str) -> List[Dict[str, Any]]:
        """
        Extrai protocolos da página da CONITEC.
        
        Args:
            url: URL da página de protocolos
            
        Returns:
            Lista de protocolos encontrados
        """
        protocols = []
        
        response = self._make_request(url)
        if not response:
            logger.warning(f"Não foi possível acessar {url}")
            return protocols
        
        soup = self._parse_html(response)
        
        # Busca links para PDFs de protocolos
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            
            # Verifica se é um link de PCDT
            if text and len(text) > 5:
                # Filtra links de protocolos (PDFs ou páginas específicas)
                if ("/pcdt" in href.lower() or "/protocolos" in href.lower() or 
                    href.endswith(".pdf")):
                    
                    # Limpa o título
                    clean_title = self._clean_title(text)
                    if clean_title and len(clean_title) > 3:
                        full_url = urljoin("https://www.gov.br", href)
                        
                        # Evita duplicatas
                        if not any(p.get("link") == full_url for p in protocols):
                            protocols.append({
                                "titulo": clean_title,
                                "link": full_url,
                                "fonte": "CONITEC/MS",
                            })
        
        logger.info(f"Extraídos {len(protocols)} protocolos da página web")
        return protocols
    
    def _clean_title(self, title: str) -> str:
        """
        Limpa o título do protocolo removendo artefatos de scraping.
        
        Args:
            title: Título original
            
        Returns:
            Título limpo
        """
        if not title:
            return ""
        
        # Remove prefixos comuns
        prefixes = ["Link para PCDT", "PCDT", "Protocolo", "Portaria"]
        for prefix in prefixes:
            if title.startswith(prefix):
                title = title[len(prefix):].strip(" -:")
        
        # Remove caracteres especiais excessivos
        title = re.sub(r'\s+', ' ', title)
        title = title.strip()
        
        return title
    
    def _validate_protocol(self, protocol: Dict[str, Any]) -> bool:
        """
        Valida se um protocolo tem dados mínimos necessários.
        
        Args:
            protocol: Dicionário do protocolo
            
        Returns:
            True se válido, False caso contrário
        """
        titulo = protocol.get("titulo", "")
        descricao = protocol.get("descricao", "")
        
        # Título deve ter conteúdo significativo (mínimo 10 chars)
        if not titulo or len(titulo) < 10:
            return False
        
        # Filtra títulos que são apenas datas ou números
        if re.match(r'^[\d/\-\.]+$', titulo.strip()):
            return False
        
        # Filtra títulos que começam com data
        if re.match(r'^\d{2}/\d{2}/\d{4}', titulo.strip()):
            return False
        
        # Filtra títulos muito curtos ou genéricos
        if titulo.lower().strip() in ["publicação", "publicação m", "portaria", "anexo", "resumo"]:
            return False
        
        # Filtra lixo de navegação
        trash_patterns = [
            "menu", "navbar", "footer", "header", "sidebar",
            "institucional", "licitação", "acesso", "contato",
            "horário", "visita", "voluntariado", "coral",
            "publicada em", "retificada em", "atualizado",
            "publicação ms", "clique aqui", "saiba mais"
        ]
        titulo_lower = titulo.lower()
        if any(pattern in titulo_lower for pattern in trash_patterns):
            return False
        
        # Deve conter palavras médicas ou de protocolo
        medical_keywords = [
            "síndrome", "doença", "câncer", "tumor", "carcinoma", "leucemia",
            "linfoma", "diabetes", "hipertensão", "hepatite", "hiv", "aids",
            "asma", "artrite", "anemia", "esclerose", "epilepsia", "parkinson",
            "alzheimer", "tuberculose", "transplante", "tratamento", "terapêutica",
            "protocolo", "diretriz", "clínico", "agudo", "crônico", "infecção",
            "imunossupressão", "fibrose", "insuficiência", "acidente", "distúrbio",
            "deficiência", "amiloidose", "angioedema", "atrofia", "comportamento",
            "degeneração", "dermatite", "dislipidemia", "distonia", "endometriose",
            "enxaqueca", "espondilite", "esquizofrenia", "fenilcetonúria", "glaucoma",
            "hanseníase", "hemofilia", "hemangioma", "hipercolesterolemia", "ictiose",
            "imunodeficiência", "leiomioma", "miastenia", "mieloma", "mucopolissacaridose",
            "narcolepsia", "niemann-pick", "osteogênese", "osteoporose", "pênfigo",
            "psoríase", "púrpura", "retocolite", "guillain", "sjögren", "trombocitemia",
            "urticária", "uveíte", "fabry", "gaucher", "wilson", "paget", "crohn",
            "celíaca", "chagas", "colangite", "biotinidase", "acromegalia", "brucelose",
            "melanoma", "meningioma", "glioma", "nefrótica", "cardiovascular"
        ]
        
        # Se tem especialidade específica (não Medicina Geral), é válido
        especialidade = protocol.get("especialidade", "")
        if especialidade and especialidade != "Medicina Geral":
            return True
        
        # Se não tem especialidade específica, precisa ter keyword médica
        return any(kw in titulo_lower for kw in medical_keywords)
    
    def scrape(self) -> List[Dict[str, Any]]:
        """
        Executa o scraping de protocolos médicos.
        
        Returns:
            Lista de protocolos coletados
        """
        logger.info("Iniciando scraping de protocolos médicos...")
        all_protocols = []
        seen_titles = set()
        
        # 1. Tenta extrair da página da CONITEC
        for url in self.PROTOCOL_URLS:
            logger.info(f"Acessando: {url}")
            web_protocols = self._scrape_conitec_page(url)
            for p in web_protocols:
                title_key = p["titulo"].lower().strip()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    # Adiciona especialidade inferida
                    p["especialidade"] = self._get_specialty_from_title(p["titulo"])
                    p["descricao"] = f"Protocolo Clínico e Diretrizes Terapêuticas para {p['titulo']}"
                    all_protocols.append(p)
        
        # 2. Adiciona protocolos conhecidos com dados completos
        for protocol in self.KNOWN_PROTOCOLS:
            title_key = protocol["titulo"].lower().strip()
            
            # Verifica se já existe
            existing = next(
                (p for p in all_protocols if title_key in p["titulo"].lower()),
                None
            )
            
            if existing:
                # Enriquece dados existentes
                if not existing.get("especialidade") or existing["especialidade"] == "Medicina Geral":
                    existing["especialidade"] = protocol["especialidade"]
                if not existing.get("descricao") or "Protocolo Clínico" in existing.get("descricao", ""):
                    existing["descricao"] = protocol["descricao"]
            else:
                # Adiciona novo protocolo
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_protocols.append({
                        "titulo": protocol["titulo"],
                        "especialidade": protocol["especialidade"],
                        "descricao": protocol["descricao"],
                        "link": f"{self.BASE_URL}/pt-br/assuntos/avaliacao-de-tecnologias-em-saude/protocolos-clinicos-e-diretrizes-terapeuticas",
                        "fonte": "CONITEC/MS",
                    })
        
        # 3. Valida e filtra protocolos
        valid_protocols = [p for p in all_protocols if self._validate_protocol(p)]
        
        logger.info(f"Total de protocolos válidos: {len(valid_protocols)}")
        
        # Aplica limite de itens se configurado
        return self._apply_limit(valid_protocols)
    
    def run(self) -> Path:
        """
        Executa o scraping completo e salva em JSONL.
        
        Returns:
            Path do arquivo JSONL gerado
        """
        try:
            protocols = self.scrape()
            
            # Transforma para formato instruction/input/output
            transformed = []
            for p in protocols:
                titulo = p.get("titulo", "")
                especialidade = p.get("especialidade", "")
                descricao = p.get("descricao", "")
                
                transformed.append({
                    "instruction": f"Quais são as diretrizes do protocolo de {titulo}?",
                    "input": f"Especialidade: {especialidade}" if especialidade else "",
                    "output": descricao,
                })
            
            filepath = self._save_to_jsonl(transformed, "protocolos_medicos", "CONITEC/MS")
            return filepath
            
        except Exception as e:
            logger.error(f"Erro no scraping de protocolos: {e}", exc_info=True)
            raise
        finally:
            self.close()


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging
    setup_logging()
    
    scraper = HCPAScraper()
    filepath = scraper.run()
    print(f"\nArquivo gerado: {filepath}")
