"""
URL configuration for server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from server import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("get_token/<str:api_key>", views.get_token, name='get_token'),
    path('create_agent', views.create_agent, name='create_agent'),
    path('list_agents', views.list_agents, name='list_agents'),
    path('get_longterm_memory/<int:agent_id>', views.get_longterm_memory,
         name='get_longterm_memory'),
    path('update_longterm_memory/<int:agent_id>', views.update_longterm_memory,
         name='update_longterm_memory'),
    path('update_checkpoint/<int:agent_id>',
         views.update_checkpoint, name='update_checkpoint'),
    path('get_checkpoint/<int:agent_id>/',
         views.get_checkpoint, name='get_checkpoint'),
    path('save_knowledge/<str:collection_name>',
         views.save_knowledge, name='save_knowledge'),
    path('search_commands/<str:collection_name>',
         views.search_commands, name='search_commands'),
    path('search_knowledge/<str:collection_name>',
         views.search_knowledge, name='search_knowledge'),
    path('get_recent_knowledge/<str:collection_name>',
         views.get_recent_knowledge, name='get_recent_knowledge'),
    path('search_kg/<str:collection_name>',
         views.search_kg, name='search_kg')
]