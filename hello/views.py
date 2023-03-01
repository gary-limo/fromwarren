from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from dotenv import load_dotenv
from django.core import serializers
import json
from django.conf import settings
from .models import Question
from .models import Person
from .models import Organization
from .models import Location
import pandas as pd
import openai
import numpy as np

 

import os

load_dotenv('.env')

 
openai.api_key =   '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'

COMPLETIONS_MODEL = "text-davinci-003"

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 150,
    "model": COMPLETIONS_MODEL,
}

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> tuple[str, str]:
    """
    Fetch relevant embeddings
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[df['title'] == section_index].iloc[0]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            space_left = MAX_SECTION_LEN - chosen_sections_len - len(SEPARATOR)
            chosen_sections.append(SEPARATOR + document_section.content[:space_left])
            chosen_sections_indexes.append(str(section_index))
            break

        chosen_sections.append(SEPARATOR + document_section.content)
        chosen_sections_indexes.append(str(section_index))

    header = """Answer truthfully as much as possible"""

     
    return (header + "".join(chosen_sections) +"\n\n\nQ: " + question + "\n\nA: "), ("".join(chosen_sections))

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
) -> tuple[str, str]:
    prompt, context = construct_prompt(
        query,
        document_embeddings,
        df
    )

    #print("===\n", prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
     
    

    return response["choices"][0]["text"].strip(" \n"), context



def get_google_maps_api_key(request):
    api_key = getattr(settings, 'GOOGLE_MAPS_API_KEY', None)
    if api_key:
        data = {'api_key': api_key}
    else:
        data = {'error': 'API key not found'}
    return JsonResponse(data)



def index(request):
    return render(request, "index.html", { "default_question": "What is The Minimalist Entrepreneur about?" })

@csrf_exempt
def ask(request):
    question_asked = request.POST.get("value", "")
   
    #question_asked= "Please write about " + "Henry Ford" + ". Please the prompt only. Please return result in suitable HTML format with bullet format"
    question_asked= "Scan everything about " + question_asked + ". Please use the above data only."
    print(question_asked)

    if not question_asked.endswith('?'):
        question_asked += '?'

    previous_question = Question.objects.filter(question=question_asked).first()
    
  
    df = pd.read_csv('book.pdf.pages.csv')
   
    document_embeddings = load_embeddings('book.pdf.embeddings.csv')
    answer, context = answer_query_with_context(question_asked, df, document_embeddings)
 

    return JsonResponse({  "answer": answer })




def person_chart_view(request):
    # Retrieve the data from the Person model
    persons = Person.objects.all()
    org = Organization.objects.all()
    location = Location.objects.all()

    # Format the data for use with Google Charts
    chart_data = []
    chart_data.append(['Person', 'Frequency'])
    for person in persons:
        chart_data.append([person.person_name, person.frequency])

    org_data = []
    org_data.append(['Organization', 'Frequency'])
    for organization in org:
        org_data.append([organization.name, organization.frequency])  
      

    location_data = []
    #location_data.append(['Location', 'Frequency'])
    for loc in location:
        location_data.append([loc.location, loc.frequency])  
      

    # Return the chart data as JSON
    return render(request, 'person.html', {'p_data': json.dumps(chart_data) , 'o_data': json.dumps(org_data) , 'l_data': json.dumps(location_data)})



@login_required
def db(request):
    questions = Question.objects.all().order_by('-ask_count')

    return render(request, "db.html", { "questions": questions })

def question(request, id):
    question = Question.objects.get(pk=id)
    return render(request, "index.html", { "default_question": question.question, "answer": question.answer})
