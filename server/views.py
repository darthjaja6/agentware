# In views.py
import os
from . import views
from django.urls import path
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import functools
import jwt
import time
from agentware.base import Knowledge
from server.vector_db_clients.command_vector_db import CommandsVectorStore
from server.vector_db_clients.knowledge_vector_db import KnowledgeVectorStore
from server.knowledge_graph_clients.knowledge_graph_client import KnowledgeGraphClient, Node
from server.db_clients.memory_db_client import DbClient

TOKEN_TIMEOUT = "Authentication error, token timeout"

config = None
with open("./server/configs/server_config.json", "r") as f:
    config = json.loads(f.read())

command_hub_client = CommandsVectorStore(
    config["vector_db_config"])
knowledge_base_client = KnowledgeVectorStore(
    config["vector_db_config"])
kg_client = KnowledgeGraphClient(config)
db_client = DbClient(config["redis_db_config"])

auth_secret_key = config["auth_secret_key"]


def verify_token(view_func):
    @functools.wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # Get the Authorization header
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')

        # Make sure the Authorization header is in the correct format
        auth_parts = auth_header.split()
        if len(auth_parts) != 2 or auth_parts[0].lower() != 'bearer':
            return JsonResponse({'error': 'Invalid Authorization header format. Expected "Bearer <token>"'}, status=401)

        # Verify the token (this example just checks if the token equals 'my-secret-token')
        try:
            decoded_payload = jwt.decode(
                auth_parts[1], auth_secret_key, algorithms=['HS256'])
            print("payload is", decoded_payload)
            if time.time() > decoded_payload["expires_at"]:
                raise TOKEN_TIMEOUT
            kwargs["user_id"] = decoded_payload["user_id"]
        except:
            return JsonResponse({'error': 'Unauthorized'}, status=401)
        # If the token is verified, call the view normally
        return view_func(request, *args, **kwargs)

    return _wrapped_view


@csrf_exempt
@require_http_methods("GET")
def get_token(request, api_key: str):
    try:
        # Check if token is valid in redis
        # Check the user that has this api key
        user_data = db_client.get_user(api_key)
        if not user_data:
            raise ValueError("Invalid api key")
        # Define the payload data for the JWT
        expires_at = time.time() + 3600 * 10 * 1000  # Expires after 10 hrs
        payload = {
            "user_id": user_data["id"],
            "expires_at": expires_at
        }

        # Generate the JWT
        result = {
            "token": jwt.encode(payload, auth_secret_key, algorithm='HS256')
        }
        return JsonResponse(result, safe=False)
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("PUT")
@verify_token
def create_agent(request, **kwargs):
    try:
        user_id = kwargs["user_id"]
        result = db_client.create_agent(user_id)
        return JsonResponse({'agent_id': result})
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("GET")
@verify_token
def get_checkpoint(request, agent_id, **kwargs):
    try:
        main_agent_config, helper_agent_configs, memory_units, knowledges, context = db_client.get_checkpoint(
            agent_id)
        result = dict()
        if main_agent_config:
            result["main_agent_config"] = main_agent_config
        if helper_agent_configs:
            result["helper_agent_configs"] = helper_agent_configs
        if memory_units:
            result["memory_units"] = memory_units
        if knowledges:
            result["knowledges"] = knowledges
        if context:
            result["context"] = context
        return JsonResponse(result, safe=False)
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("GET")
@verify_token
def list_agents(request, **kwargs):
    try:
        user_id = kwargs["user_id"]
        result = db_client.get_agents_of_user(user_id)
        return JsonResponse(result, safe=False)
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("GET")
@verify_token
def get_longterm_memory(request, agent_id: int, **kwargs):
    try:
        page_number = int(request.GET.get('page_number', "0"))
        page_size = int(request.GET.get('page_size', "10"))

        if page_number < 0:
            raise ValueError(f"Invalid page number {page_number}")
        if page_size <= 0:
            raise ValueError(f"Invalid page size {page_size}")
        result = db_client.get_longterm_memory(
            agent_id, page_number, page_size)
        return JsonResponse(result, safe=False)
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("PUT")
@verify_token
def update_longterm_memory(request, agent_id: int, **kwargs):
    try:
        data = json.loads(request.body)
        memory_data = data['memory_data']
        print("updating...")
        db_client.save_longterm_memory(agent_id, memory_data)
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("PUT")
@verify_token
def update_checkpoint(request, agent_id: int, **kwargs):
    try:
        user_id = kwargs["user_id"]
        data = json.loads(request.body)
        agent_config = data['agent_config']
        helper_agent_configs = data['helper_agent_configs']
        memory_data = data["memory_data"]
        knowledge = data['knowledge']
        context = data['context']
        db_client.update_checkpoint(
            agent_config, helper_agent_configs, memory_data, knowledge, context, user_id, agent_id)
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("PUT")
@verify_token
def save_knowledge(request, collection_name: str, **kwargs):
    try:
        data = json.loads(request.body)
        knowledges_json = data['knowledges']
        knowledges = [Knowledge.from_json(k) for k in knowledges_json]
        knowledge_base_client.insert_knowledge(
            collection_name, knowledges)
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("GET")
def search_commands(request, collection_name):
    try:
        data = json.loads(request.body)
        query_embeds = data['query_embeds']
        token_limit = data['token_limit']
        if not query_embeds:
            raise ValueError("Invalid query embeds")
        result = command_hub_client.search_command(
            collection_name, query_embeds, token_limit)
        return JsonResponse(result, safe=False)
    except Exception as e:
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("GET")
@verify_token
def search_knowledge(request, collection_name: str, **kwargs):
    try:
        # query_embeds = request.GET.get('query_embeds', [])
        data = json.loads(request.body)
        query_embeds = data["query_embeds"]
        token_limit = data["token_limit"]
        if not query_embeds:
            raise ValueError(f"query embeds value invalid")
        result = knowledge_base_client.search_knowledge(
            collection_name, query_embeds, token_limit)
        return JsonResponse(result, safe=False)
    except Exception as e:
        print(e)
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("GET")
@verify_token
def get_recent_knowledge(request, collection_name: str, **kwargs):
    try:
        # query_embeds = request.GET.get('query_embeds', [])
        data = json.loads(request.body)
        token_limit = data["token_limit"]
        result = knowledge_base_client.get_recent_knowledge(
            collection_name, token_limit)
        return JsonResponse(result, safe=False)
    except Exception as e:
        print(e)
        return HttpResponseBadRequest(str(e))


@csrf_exempt
@require_http_methods("GET")
@verify_token
def search_kg(request, collection_name: str, **kwargs):
    try:
        # query_embeds = request.GET.get('query_embeds', [])
        data = json.loads(request.body)
        query_embeds = data["query_embeds"]
        token_limit = data["token_limit"]
        if not query_embeds:
            raise ValueError(f"query embeds value invalid")
        result = kg_client.keyword_search(
            query_embeds, collection_name)
        return JsonResponse(result, safe=False)
    except Exception as e:
        print(e)
        return HttpResponseBadRequest(str(e))
