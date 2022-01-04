from fastapi import FastAPI, Request, status, Depends
from fastapi.exceptions import HTTPException

import requests
import spacy
import json
import string
import urllib.parse
import urllib.request
#import nltk
from spacy.lang.en import English
from networkx.readwrite import json_graph
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from models import LongText, word

### Arango connection ##
from arango import ArangoClient
client = ArangoClient(hosts="http://10.0.0.12:8529")
arango_db = client.db('knowledgegraph', username='admin@knowledgegraph', password='KcHr4fPn')

### PyArango connection ###
from pyArango.connection import *
from pyArango.collection import Collection, Edges, Field
from pyArango.graph import Graph, EdgeDefinition

#conn = Connection(username="admin@knowledgegraph", password="KcHr4fPn")
conn = Connection(arangoURL='http://10.0.0.12:8529', username="admin@knowledgegraph", password="KcHr4fPn")
db = conn['knowledgegraph']


app = FastAPI()

def google_kg(word):
  text = []
  api_key = 'AIzaSyDT4JA_CvldekcDC4FtoF64Q5D-0ERe4uk'
  query = word
  service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
  params = {
      'query': query,
      'limit': 10,
      'indent': True,
      'key': api_key}
  url = service_url + '?' + urllib.parse.urlencode(params)
  response = json.loads(urllib.request.urlopen(url).read())
  for element in response['itemListElement']:
      text.append((element['result'].get('detailedDescription', {}).get('articleBody', '')))
  return text

def getSentences(text):
    nlp = English()
    #nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe('sentencizer')
    document = nlp(text)
    return [sent.text.strip() for sent in document.sents]
def printToken(token):
    (token.text, "->", token.dep_)
def appendChunk(original, chunk):
    return original + ' ' + chunk
def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)
def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)
def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
        printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''
    print (subject.strip(), ",", relation.strip(), ",", object.strip())
    subject = subject.strip().lower().replace(' ', '_')
    object = object.strip().lower().replace(' ', '_')
    return (subject, relation.strip(), object)
def processSentence(sentence, nlp_model):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)
def printGraph(triples):
    G = nx.Graph()
    for triple in triples:
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])
    pos = nx.spring_layout(G)
    #print('pos....',pos)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='skyblue', alpha=0.9,
            labels={node: node for node in G.nodes()})

    plt.axis('off')
    plt.show()

def create_graph_arango(triples):
    try:
        text_graph = db.graphs['kg_graph_staging']
        arango_node_collection = arango_db['entity']
        arango_edge_collection = arango_db['relation']
        nodes = []
        connections = triples
        for triple in triples:
            text1 = triple[0]
            if text1 != '' and text1 not in nodes:
                nodes.append(text1)
            text2 = triple[2]
            if text2 != '' and text2 not in nodes:
                nodes.append(text2)
        print(nodes)
        print(connections)
        for node in nodes:
            if arango_node_collection.has(node) == True:
                node_dict = arango_node_collection[node]
                if node_dict.get('occurance'):
                    node_dict['occurance'] += 1
                else:
                    node_dict['occurance'] = 1
                arango_node_collection.update(node_dict)
            else:
                text_graph.createVertex('entity', {"_key": node, 'name': node, 'occurance':1})
        for node1, node2, relation in connections:
            edge_key = node1 + '_' + node2
            if arango_edge_collection.has(edge_key):
                edge_dict = arango_edge_collection[edge_key]
                if node_dict.get('occurance'):
                    node_dict['occurance'] += 1
                else:
                    node_dict['occurance'] = 1
                arango_edge_collection.update(edge_dict)
            else:
                if node1 != '' and node2 != '':
                    text_graph.link('relation', db["entity"][node1], db["entity"][node2], {"connection": relation, 'occurance':1})
        return nodes
        '''for node in nodes:
            try:
                if db.collections['entity'][node]:
                    pass
            except Exception as error:
                text_graph.createVertex('entity', {"_key": node, 'name': node})
        for node1, relation, node2 in connections:
            if node1 != '' and node2 != '':
                text_graph.link('relation', db["entity"][node1], db["entity"][node2], {"connection": relation})
            else:
                pass
        return nodes'''
    except Exception as error:
        print(error)
        return nodes

def text_extraction(text):
    punct_char = string.punctuation.replace('.', '')
    def remove_punctuation(text):
        no_punct=[words for words in text if words not in punct_char]
        words_wo_punct=''.join(no_punct)
        return words_wo_punct
    new_text = remove_punctuation(text)
    sentences = getSentences(new_text)
    nlp_model = spacy.load('en_core_web_sm')
    triples = []
    for sentence in sentences:
      triples.append(processSentence(sentence, nlp_model))
    #print(triples)
    '''default_weight = 0
    graph = nx.Graph()
    for triple in triples:
        n0 = triple[0]
        n1 = triple[2]
        if graph.has_edge(n0,n1):
            graph[n0][n1]['weight'] += default_weight
        else:
            graph.add_edge(n0,n1, weight=default_weight)
    json_data = json_graph.node_link_data(graph)
    return json_data'''
    return triples
    #elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 0.5]
    #esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 0.5]
    #pos = nx.spring_layout(graph, seed=7)
    #plt.figure(figsize=(17,17))
    #nx.draw_networkx_nodes(graph, pos, node_size=500)
    #nx.draw_networkx_edges(graph, pos, width=2, style="dashed")
    #nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif")
    #plt.show()

def google_kg(word):
  text = []
  api_key = 'AIzaSyDT4JA_CvldekcDC4FtoF64Q5D-0ERe4uk'
  query = word
  service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
  params = {
      'query': query,
      'limit': 10,
      'indent': True,
      'key': api_key}
  url = service_url + '?' + urllib.parse.urlencode(params)
  response = json.loads(urllib.request.urlopen(url).read())
  for element in response['itemListElement']:
      text.append((element['result'].get('detailedDescription', {}).get('articleBody', '')))
  return text


@app.post('/knowledge_graph_from_sentence', tags=["text"])
async def knowledge_graph_from_sentence(long_text: LongText):
    graph_res = [{"v":None,"e":None}]
    word = long_text.text
    description = long_text.description
    if description == '' and word != '':
        description = google_kg(word)
    graph_model = text_extraction(description)
    nodes = create_graph_arango(graph_model)
    if nodes:
        graph_res = knowledge_graph(nodes)
    response = {'keyword': '', 'graph': graph_res}
    return response

@app.post('/knowledge_graph_content_push', tags=["text"])
async def knowledge_graph_content_push(text: word):
    description = text.description
    url = text.url
    graph_model = text_extraction(description)
    nodes =  create_graph_arango(graph_model)
    return {'message': 'added data in arango db'}


def knowledge_graph(nodes):
    graph_list = []
    for text in nodes:
        node="entity/"+text
        query = '''FOR v, e IN 0..20 ANY '{}' GRAPH "kg_graph_staging"'''.format(node) +" "+ ''' OPTIONS {bfs: true, uniqueVertices: 'global'} RETURN {v: v, e: e}'''
        queryResult = db.AQLQuery(query, rawResults=True)
        for result in list(queryResult):
            graph_list.append(result)
    
    return graph_list

